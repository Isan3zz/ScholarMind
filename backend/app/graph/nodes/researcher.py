from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from app.tools.search import search_tavily
from app.graph.state import AgentState, RetrievalAuditEntry
from app.rag.engine import apply_section_boost, children_only_search, global_parent_enrichment
from app.utils.llm import get_llm

llm = get_llm(model_type="smart")


@tool
def search_web(query: str) -> str:
    """Search the internet for information not found in local documents.
Use when local documents lack sufficient evidence for the user's question."""
    return search_tavily(query)


def _format_evidence_docs(docs):
    """格式化 evidence，按检索 query 分组展示。"""
    # 按 retrieval_query 分组
    groups: dict[str, list] = {}
    for doc in docs:
        meta = doc.metadata or {}
        rq = meta.get("retrieval_query", "(原始问题)")
        groups.setdefault(rq, []).append(doc)

    blocks = []
    for rq, group_docs in groups.items():
        blocks.append(f"## 检索: {rq}")
        for doc in group_docs:
            meta = doc.metadata or {}
            source = meta.get("source", "unknown")
            section = meta.get("section", "Unknown")
            score = meta.get("relevance_score", "")
            score_text = f", score={score:.2f}" if isinstance(score, (int, float)) else ""
            context = meta.get("parent_text") or doc.page_content
            blocks.append(
                f"[source: {source}, section: {section}{score_text}]\n{context}"
            )
        blocks.append("")
    return "\n\n".join(blocks)


def _retrieve_local_docs(query: str, sources: list[str] | None = None, target_sections: list[str] | None = None) -> list[Document]:
    """子块检索（不查父块），可选章节 boost。"""
    hits = children_only_search(query, top_k=5, sources=sources)
    if target_sections:
        hits = apply_section_boost(hits, target_sections)
    docs: list[Document] = []
    for hit in hits:
        metadata = hit.to_metadata()
        metadata["relevance_score"] = hit.score
        docs.append(Document(page_content=hit.content, metadata=metadata))
    return docs


def _build_local_retrieval_queries(query: str, plans: list[str], *, skip_raw_query: bool = False) -> list[tuple[str, int]]:
    """构建去重后的检索 query 列表，保留原始 plan 索引用于对齐 plan_sources。"""
    queries: list[tuple[str, int]] = []
    seen: set[str] = set()
    # 原始 query，索引为 -1（无对应 plan_sources）
    if not skip_raw_query:
        cleaned = str(query or "").strip()
        if cleaned:
            key = cleaned.casefold()
            if key not in seen:
                seen.add(key)
                queries.append((cleaned, -1))
    for i, plan in enumerate(plans or []):
        cleaned = str(plan or "").strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            queries.append((cleaned, i))
    return queries


def _retrieve_local_docs_for_queries(
    query: str,
    plans: list[str],
    plan_sources: list[list[str]] | None = None,
    plan_sections: list[list[str]] | None = None,
) -> tuple[list[Document], list[tuple[str, int]]]:
    # 有多论文路由时跳过原始 query（Planner 已按论文拆分了 query）
    has_routing = any(srcs for srcs in (plan_sources or []))
    retrieval_queries = _build_local_retrieval_queries(query, plans, skip_raw_query=has_routing)

    # 构建 (retrieval_query, sources, sections) 任务列表
    tasks: list[tuple[str, list[str] | None, list[str] | None]] = []
    for rq, plan_idx in retrieval_queries:
        srcs: list[str] | None = None
        secs: list[str] | None = None
        if plan_idx >= 0:
            if plan_sources and plan_idx < len(plan_sources) and plan_sources[plan_idx]:
                srcs = plan_sources[plan_idx]
            if plan_sections and plan_idx < len(plan_sections) and plan_sections[plan_idx]:
                secs = plan_sections[plan_idx]
        tasks.append((rq, srcs, secs))

    # 并行检索子块（不查父块）
    child_docs: list[Document] = []
    seen_chunks: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
        future_map = {
            pool.submit(_retrieve_local_docs, rq, sources=srcs, target_sections=secs): (rq, srcs)
            for rq, srcs, secs in tasks
        }
        for future in as_completed(future_map):
            rq, _ = future_map[future]
            for doc in future.result():
                meta = doc.metadata or {}
                identity = str(
                    meta.get("chunk_id")
                    or f"{meta.get('source')}:{meta.get('page')}:{meta.get('section')}:{doc.page_content}"
                )
                if identity in seen_chunks:
                    continue
                seen_chunks.add(identity)
                meta["retrieval_query"] = rq
                child_docs.append(doc)

    # 全局父块补充 + 去重 + 相邻合并
    from app.rag.hits import RetrievalHit

    # 建立 chunk_id → retrieval_query 的映射（用于合并后恢复）
    chunk_to_queries: dict[str, set[str]] = {}
    for doc in child_docs:
        cid = doc.metadata.get("chunk_id", "")
        rq = doc.metadata.get("retrieval_query", "")
        if cid:
            chunk_to_queries.setdefault(cid, set()).add(rq)

    child_hits = [
        RetrievalHit(
            chunk_id=doc.metadata.get("chunk_id", ""),
            content=doc.page_content,
            source=str(doc.metadata.get("source", "unknown")),
            page=doc.metadata.get("page", "?"),
            section=str(doc.metadata.get("section", "Unknown")),
            score=float(doc.metadata.get("relevance_score", 0.0)),
            retriever="rerank",
            metadata=dict(doc.metadata),
        )
        for doc in child_docs
    ]
    enriched_hits = global_parent_enrichment(child_hits, top_k=max(5, len(child_hits)))

    # 转换回 Document，恢复 retrieval_query
    docs: list[Document] = []
    for hit in enriched_hits:
        meta = hit.to_metadata()
        meta["relevance_score"] = hit.score
        # 从原始子块映射恢复 retrieval_query
        cid = hit.metadata.get("chunk_id", "")
        parent_id = hit.metadata.get("parent_chunk_id", "")
        queries = chunk_to_queries.get(cid, set()) | chunk_to_queries.get(parent_id, set())
        meta["retrieval_query"] = "; ".join(sorted(queries)) if queries else "(original)"
        docs.append(Document(page_content=hit.context_text, metadata=meta))

    return docs, retrieval_queries


def _build_audit_entry(
    docs: list[Document],
    retrieval_queries: list[tuple[str, int]],
    round_num: int,
) -> RetrievalAuditEntry:
    """Build a compact audit record from retrieval results."""
    # which queries returned hits
    hit_queries: set[str] = set()
    all_sources: set[str] = set()
    all_sections: set[str] = set()
    max_score: float = 0.0

    for doc in docs:
        meta = doc.metadata or {}
        rq = meta.get("retrieval_query", "")
        if rq:
            # rq can be "; "-joined from multiple queries
            for q in rq.split("; "):
                hit_queries.add(q.strip())
        src = meta.get("source", "")
        if src and src != "unknown":
            all_sources.add(str(src))
        sec = meta.get("section", "")
        if sec and sec != "Unknown":
            all_sections.add(str(sec))
        score = meta.get("relevance_score", 0.0)
        if isinstance(score, (int, float)) and score > max_score:
            max_score = float(score)

    all_query_texts = [q for q, _ in retrieval_queries]
    empty = [q for q in all_query_texts if q not in hit_queries]

    return {
        "round": round_num,
        "queries": all_query_texts,
        "sources_hit": sorted(all_sources),
        "sections_hit": sorted(all_sections),
        "max_relevance": max_score,
        "empty_queries": empty,
    }


def _with_audit_log(
    result: dict,
    state: AgentState,
    audit_entry: RetrievalAuditEntry | None,
) -> dict:
    """Append audit_entry to the existing retrieval_audit_log in state."""
    existing: list[RetrievalAuditEntry] = list(state.get("retrieval_audit_log") or [])
    if audit_entry is not None:
        existing.append(audit_entry)
    result["retrieval_audit_log"] = existing
    return result


def research_node(state: AgentState):

    mode = state.get("search_mode", "hybrid")
    query = state["query"]
    plans = state["plan"]
    plan_sources = state.get("plan_sources", [])
    plan_sections = state.get("plan_sections", [])
    results = []

    print(f"--- [Researcher] Starting | mode={mode} ---")

    # Step 1: Always search local docs first (code-driven, unchanged)
    local_content = ""
    round_num = state.get("revision_number", 0)
    audit_entry: RetrievalAuditEntry | None = None
    try:
        docs, retrieval_queries = _retrieve_local_docs_for_queries(
            query, plans, plan_sources=plan_sources, plan_sections=plan_sections
        )
        audit_entry = _build_audit_entry(docs, retrieval_queries, round_num)
        if docs:
            local_content = _format_evidence_docs(docs)
            results.append(f"### 📂 本地文档资料\n{local_content}\n")
            print(f"--- [Researcher] Local docs found | queries={len(retrieval_queries)} hits={len(docs)} max_score={audit_entry['max_relevance']:.2f} ---")
        else:
            print("--- [Researcher] No local docs found ---")
    except Exception as e:
        print(f"--- [Researcher] Local search error: {e} ---")

    # Step 2: Build tool set based on mode
    tools = [search_web] if mode == "hybrid" else []

    # Step 3: Build agent prompt
    if mode == "document":
        mode_instruction = (
            "You are in DOCUMENT-ONLY mode. You do NOT have access to any external tools.\n"
            "Evaluate whether the local documents are relevant to the user's question.\n"
            "If the documents are completely irrelevant, output EXACTLY 'INSUFFICIENT_EVIDENCE' on its own line, then explain why.\n"
            "If relevant, summarize the key evidence from the documents that answers the question."
        )
    else:
        mode_instruction = (
            "You are in HYBRID mode. You have a search_web tool.\n"
            "First assess whether local documents already contain sufficient evidence.\n"
            "If YES — do NOT call search_web. Just summarize the local evidence.\n"
            "If NO (documents are irrelevant, incomplete, or missing key details) — call search_web to find supplementary information.\n"
            "You may call search_web multiple times with different queries."
        )

    prompt = f"""{mode_instruction}

User question: {query}

Planned search angles: {', '.join(plans) if plans else 'N/A'}

Local document search results:
{local_content if local_content else '(No local documents matched the query)'}
"""

    # Step 4: Invoke LLM (with or without tools)
    if not tools:
        # Document-only: no tools, simple evaluation
        response = llm.invoke(prompt)
    else:
        # Hybrid: LLM with web search tool, decides autonomously
        llm_with_tools = llm.bind_tools(tools)
        messages = [HumanMessage(content=prompt)]
        response = llm_with_tools.invoke(messages)

        # Tool execution loop (max 3 rounds)
        for _ in range(3):
            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                if tc["name"] == "search_web":
                    web_query = tc["args"].get("query", query)
                    print(f"--- [Researcher] LLM chose web search: {web_query} ---")
                    web_result = search_tavily(web_query)
                    results.append(f"### 🌐 网络搜索结果 ({web_query})\n{web_result}\n")
                    messages.append(response)
                    messages.append(ToolMessage(content=web_result, tool_call_id=tc["id"]))

            response = llm_with_tools.invoke(messages)

    final = response.content

    # Step 5: Circuit breaker for document-only mode
    if mode == "document" and "INSUFFICIENT_EVIDENCE" in (final or ""):
        print("--- [Researcher] Document-only: docs irrelevant, aborting ---")
        return _with_audit_log(
            {"search_results": results + [final], "should_stop": True},
            state, audit_entry,
        )

    results.append(final)
    print(f"--- [Researcher] Done | result_blocks={len(results)} ---")
    return _with_audit_log({"search_results": results}, state, audit_entry)

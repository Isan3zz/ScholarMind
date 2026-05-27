from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from app.tools.search import search_tavily
from app.graph.state import AgentState
from app.rag.engine import hybrid_search
from app.utils.llm import get_llm

llm = get_llm(model_type="smart")


@tool
def search_web(query: str) -> str:
    """Search the internet for information not found in local documents.
Use when local documents lack sufficient evidence for the user's question."""
    return search_tavily(query)


def _format_evidence_docs(docs):
    blocks = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        section = meta.get("section", "Unknown")
        score = meta.get("relevance_score", "")
        score_text = f", score={score:.2f}" if isinstance(score, (int, float)) else ""
        context = meta.get("parent_text") or doc.page_content
        blocks.append(
            f"[source: {source}, section: {section}{score_text}]\n{context}"
        )
    return "\n\n".join(blocks)


def _retrieve_local_docs(query: str) -> list[Document]:
    docs: list[Document] = []
    for hit in hybrid_search(query, top_k=5):
        metadata = hit.to_metadata()
        metadata["relevance_score"] = hit.score
        docs.append(Document(page_content=hit.context_text, metadata=metadata))
    return docs


def _build_local_retrieval_queries(query: str, plans: list[str]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for candidate in [query, *(plans or [])]:
        cleaned = str(candidate or "").strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            queries.append(cleaned)
    return queries


def _retrieve_local_docs_for_queries(query: str, plans: list[str]) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()
    for retrieval_query in _build_local_retrieval_queries(query, plans):
        for doc in _retrieve_local_docs(retrieval_query):
            meta = doc.metadata or {}
            identity = str(
                meta.get("chunk_id")
                or f"{meta.get('source')}:{meta.get('page')}:{meta.get('section')}:{doc.page_content}"
            )
            if identity in seen:
                continue
            seen.add(identity)
            docs.append(doc)
    return docs


def research_node(state: AgentState):

    mode = state.get("search_mode", "hybrid")
    query = state["query"]
    plans = state["plan"]
    results = []

    print(f"--- [Researcher] Starting | mode={mode} ---")

    # Step 1: Always search local docs first (code-driven, unchanged)
    local_content = ""
    try:
        docs = _retrieve_local_docs_for_queries(query, plans)
        if docs:
            local_content = _format_evidence_docs(docs)
            results.append(f"### 📂 本地文档资料\n{local_content}\n")
            print("--- [Researcher] Local docs found ---")
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
        return {
            "search_results": results + [final],
            "should_stop": True,
        }

    results.append(final)
    print(f"--- [Researcher] Done | result_blocks={len(results)} ---")
    return {"search_results": results}

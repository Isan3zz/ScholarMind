import os
import shutil
import warnings
from typing import Any, List, Optional, Sequence
from copy import deepcopy

warnings.filterwarnings("ignore", message=".*ElasticsearchStore.*deprecated.*")
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import httpx
from app.rag.hits import RetrievalHit
from app.rag.rank_fusion import rrf_fusion


ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "scholarmind_knowledge_base")


class OpenAICompatibleRerank:
    """
    使用 OpenAI 兼容接口的 Rerank 实现。
    支持 DashScope 的 qwen3-rerank 等模型。
    """

    def __init__(
        self,
        model: str = "qwen3-rerank",
        top_n: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.top_n = top_n
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-api/v1"

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        if not documents:
            return []

        docs_content = []
        for doc in documents:
            content = doc.page_content
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            elif not isinstance(content, str):
                content = str(content)
            docs_content.append(content)

        docs_content = [c for c in docs_content if c.strip()]
        if not docs_content:
            print("[Rerank] 警告：没有有效的文档内容")
            return list(documents)[:self.top_n]

        query = str(query) if query is not None else ""

        payload = {
            "model": self.model,
            "query": query,
            "documents": docs_content,
            "parameters": {
                "top_n": min(self.top_n, len(docs_content)),
                "return_documents": False,
            }
        }

        debug_enabled = os.getenv("RERANKER_DEBUG", "").lower() in {"1", "true", "yes"}
        if debug_enabled:
            import json

            print(f"[Rerank Debug] Request URL: {self.base_url}/reranks")
            print(f"[Rerank Debug] Payload: {json.dumps(payload, ensure_ascii=True, indent=2)}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                f"{self.base_url}/reranks",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            if debug_enabled:
                print(f"[Rerank Debug] Response status: {response.status_code}")
                print(f"[Rerank Debug] Response body: {response.text.encode('ascii', errors='backslashreplace').decode('ascii')}")
            response.raise_for_status()
            result = response.json()

            compressed = []
            results = result.get("results") or result.get("output", {}).get("results", [])
            for item in results:
                idx = item.get("index", 0)
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
                    doc_copy.metadata["relevance_score"] = item.get("relevance_score", 0.0)
                    compressed.append(doc_copy)

            return compressed

        except Exception as e:
            print(f"[Rerank] 调用失败: {e}")
            return list(documents)[:self.top_n]


class RerankRetriever(BaseRetriever):
    """
    两阶段检索：
    1) ES 混合召回 (BM25 + 向量, RRF 融合) fetch_k 个候选
    2) qwen3-rerank 精排
    3) 返回 top_k
    """

    vectorstore: Any
    compressor: Any
    top_k: int = 5
    fetch_k: int = 20

    def _get_relevant_documents(self, query: str) -> list[Document]:
        candidates: list[Document] = self.vectorstore.similarity_search(query, k=self.fetch_k)
        if not candidates:
            return []
        return list(self.compressor.compress_documents(candidates, query))[:self.top_k]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")


def _build_embeddings():
    """
    根据环境变量构建 embedding 实例。
    Embedding 必须走 DashScope 原生 API（兼容模式不支持 /embeddings），凭证统一用 OPENAI_API_KEY。
    EMBEDDING_PROVIDER: dashscope (默认) 或 huggingface (本地)
    EMBEDDING_MODEL: 模型名称
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "dashscope")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model)
    else:
        return DashScopeEmbeddings(
            model=model,
            dashscope_api_key=os.getenv("OPENAI_API_KEY"),
        )


def _build_reranker():
    """
    根据环境变量构建 OpenAI 兼容 reranker，使用统一的 OPENAI_API_KEY。
    RERANKER_MODEL: 重排序模型名称，默认 qwen3-rerank
    """
    model = os.getenv("RERANKER_MODEL", "qwen3-rerank")
    return OpenAICompatibleRerank(
        model=model,
        top_n=int(os.getenv("RERANKER_TOP_N", "30")),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )


def _get_es_kwargs():
    """构建连接 ES 的关键字参数，统一从环境变量读取。"""
    kwargs = {
        "es_url": ES_URL,
    }
    es_user = os.getenv("ES_USER")
    es_password = os.getenv("ES_PASSWORD")
    if es_user:
        kwargs["es_user"] = es_user
    if es_password:
        kwargs["es_password"] = es_password
    return kwargs


def _invalidate_caches():
    """重置所有模块级缓存（索引重建后调用）。"""
    global _vectorstore
    _vectorstore = None


def _child_chunk_filter() -> dict:
    return {
        "bool": {
            "must_not": [
                {"terms": {"metadata.chunk_type.keyword": ["parent", "paper_metadata"]}}
            ]
        }
    }


embeddings = _build_embeddings()
_reranker = None
_es_client = None
_vectorstore = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = _build_reranker()
    return _reranker


def get_es_client():
    """获取 ES 客户端单例，复用 TCP 连接避免每次检索都重新握手。"""
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch

        _es_client = Elasticsearch(ES_URL)
    return _es_client


def _get_cached_vectorstore():
    """获取 ES 向量存储单例，复用同一个 LangChain ElasticsearchStore 实例。"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore


def _build_vectorstore():
    """创建新的 ES 向量存储实例（不缓存，仅供初始化或重建时使用）。"""
    from langchain_community.vectorstores.elasticsearch import ElasticsearchStore, ApproxRetrievalStrategy

    return ElasticsearchStore(
        index_name=ES_INDEX,
        embedding=embeddings,
        strategy=ApproxRetrievalStrategy(hybrid=True, rrf=False),
        distance_strategy="COSINE",
        **_get_es_kwargs(),
    )


def reset_knowledge_base():
    """
    重置知识库：删除 ES 索引并重建，同时清空上传目录。
    """
    if os.path.exists(UPLOAD_DIR):
        try:
            shutil.rmtree(UPLOAD_DIR)
        except Exception as e:
            print(f"--- [RAG] 清理上传目录警告: {e} ---")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    print("--- [RAG] 正在重置知识库数据... ---")
    es = get_es_client()
    try:
        es.indices.delete(index=ES_INDEX, ignore_unavailable=True)
        print(f"--- [RAG] ES 索引 {ES_INDEX} 已删除 ---")
    except Exception as e:
        print(f"--- [RAG] 删除索引时遇到非致命错误: {e} ---")

    _invalidate_caches()


# 论文章节检测模式
SECTION_PATTERNS = [
    re.compile(r, re.IGNORECASE) for r in [
        r"^abstract$",
        r"^\d+\.?\s*introduction$",
        r"^\d+\.?\s*related\s*work$",
        r"^\d+\.?\s*(background|preliminar)",
        r"^\d+\.?\s*(method|approach|framework|model)",
        r"^\d+\.?\s*(experiment|evaluation|result)",
        r"^\d+\.?\s*(discussion|analysis)",
        r"^\d+\.?\s*(conclusion|summary)",
        r"^\d+\.?\s*(limitation|future)",
        r"^\d+(\.\d+)*\s+.+",
    ]
]


def _detect_section(text: str) -> str:
    """检测一行文本是否为论文章节标题，返回章节名或空字符串。"""
    line = text.strip()
    if len(line) > 80 or len(line) < 3:
        return ""
    for pat in SECTION_PATTERNS:
        if pat.match(line):
            return line
    return ""


def _generate_summaries_batch(texts: List[str]) -> List[str]:
    """批量为多个文本块生成一句话摘要（仅 ENABLE_SUMMARY=true 时调用）。"""
    from app.utils.llm import get_llm
    from langchain_core.messages import HumanMessage

    if not texts:
        return []

    items = "\n\n---\n\n".join(f"[{i}] {t[:500]}" for i, t in enumerate(texts))
    prompt = (
        "为以下每个文本片段生成一句话摘要（不超过20字），严格按编号格式输出：\n"
        "[编号] 摘要\n\n"
        f"{items}\n\n"
        "只输出摘要列表："
    )

    llm = get_llm("fast")
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        summaries = {}
        for line in result.content.strip().split("\n"):
            line = line.strip()
            if line.startswith("[") and "] " in line:
                idx_str, summary = line.split("] ", 1)
                try:
                    summaries[int(idx_str[1:])] = summary.strip()
                except ValueError:
                    continue
        return [summaries.get(i, "") for i in range(len(texts))]
    except Exception:
        return [""] * len(texts)


def _build_paper_documents_from_pages(pages: list[tuple[str, int]], source: str) -> list[Document]:
    from app.rag.paper_chunker import build_paper_chunks
    from app.rag.paper_cleaner import clean_paper_pages
    from app.rag.paper_sections import assign_sections

    all_docs: list[Document] = []
    paper_title = source

    for cleaned, page in clean_paper_pages(pages):
        units = assign_sections(cleaned.body_text)
        docs = build_paper_chunks(
            units=units,
            source=source,
            paper_title=paper_title,
            page=page,
        )
        all_docs.extend(docs)

    return all_docs



def _build_paper_documents_from_mineru(file_path: str, source: str) -> list[Document]:
    """用 MinerU Cloud API 解析 PDF，转换为 LangChain Document 列表。"""
    from app.rag.paper_chunker import build_paper_chunks
    from app.rag.mineru_cloud_parser import parse_pdf_with_mineru

    units, title, authors = parse_pdf_with_mineru(file_path)
    return build_paper_chunks(
        units=units,
        source=source,
        paper_title=title,
        page=0,
        authors=authors,
    )


def _process_single_file(file_path: str) -> list[Document]:
    """处理单个文件：MinerU Cloud 优先，失败回退 PyPDFLoader。"""
    source = os.path.basename(file_path)

    # 1) 优先尝试 MinerU Cloud
    try:
        splits = _build_paper_documents_from_mineru(file_path, source=source)
        total = len(splits)
        print(f"  [MinerU] {source} -> {total} 个块 (子块+父块)")
        return splits
    except Exception as mineru_error:
        import traceback
        print(f"  [{source}] MinerU 失败，回退 PyPDFLoader: {mineru_error}")
        traceback.print_exc()

    # 2) 回退 PyPDFLoader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    pages = [(doc.page_content, int(doc.metadata.get("page", 0)) + 1) for doc in docs]
    splits = _build_paper_documents_from_pages(pages, source=source)
    total = len(splits)
    print(f"  [PyPDFLoader] {source} -> {total} 个块 (子块+父块)")
    return splits


def process_documents(file_paths: List[str]):
    """
    核心逻辑：并发解析 PDF -> 边处理边流式写入 ES（不累积全量chunk）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 复用缓存的 vectorstore 实例
    vector_store = _get_cached_vectorstore()

    total_chunks = 0
    batch_buffer: list[Document] = []
    BATCH_SIZE = 50

    def _flush():
        nonlocal batch_buffer
        if not batch_buffer:
            return
        try:
            vector_store.add_documents(batch_buffer)
            batch_buffer.clear()
        except Exception as e:
            print(f"  [ERROR] 写入批次失败: {e}")
            raise

    max_workers = min(len(file_paths), 3)
    print(f"--- [RAG] 并发处理 {len(file_paths)} 个文件 ({max_workers} 线程) ---")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_single_file, fp): fp for fp in file_paths}
        for i, future in enumerate(as_completed(futures), 1):
            fp = futures[future]
            try:
                splits = future.result()
            except Exception as e:
                print(f"[ERROR] 处理文件 {fp} 失败: {e}")
                continue
            for doc in splits:
                batch_buffer.append(doc)
                if len(batch_buffer) >= BATCH_SIZE:
                    _flush()
                    print(f"  [OK] 已写入当前批次 (含 {i}/{len(file_paths)} 个文件的chunk)")
            total_chunks += len(splits)

    _flush()
    print(f"--- [RAG] 完成，共 {total_chunks} 个块写入 {ES_INDEX} ---")
    return total_chunks


def get_retriever():
    """
    获取检索器：给 Agent 用的接口
    """
    es = get_es_client()
    if not es.indices.exists(index=ES_INDEX):
        return None
    return RerankRetriever(
        vectorstore=_get_cached_vectorstore(),
        compressor=get_reranker(),
        top_k=5,
        fetch_k=20,
    )


def _fetch_parent_texts(parent_ids: list[str]) -> dict[str, str]:
    """批量从 ES 查询父块的 page_content。"""
    if not parent_ids:
        return {}
    es = get_es_client()
    try:
        response = es.search(
            index=ES_INDEX,
            body={
                "query": {"terms": {"metadata.chunk_id.keyword": parent_ids}},
                "size": len(parent_ids),
                "_source": ["text", "metadata.chunk_id"],
            },
        )
        result: dict[str, str] = {}
        for item in response["hits"]["hits"]:
            src = item.get("_source", {})
            pid = src.get("metadata", {}).get("chunk_id", "")
            text = src.get("text") or ""
            if pid:
                result[pid] = text
        return result
    except Exception as e:
        print(f"[RAG] 父块查询失败: {e}")
        return {}


def _enrich_hits_with_parents(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """为子块命中补充 parent_text，使 context_text 返回父块文本。"""
    parent_ids: list[str] = []
    for hit in hits:
        ct = hit.metadata.get("chunk_type", "")
        if ct != "parent" and ct != "paper_metadata":
            pid = hit.metadata.get("parent_chunk_id")
            if pid:
                parent_ids.append(pid)

    if not parent_ids:
        return hits

    # 去重后批量查询
    unique_ids = list(dict.fromkeys(parent_ids))
    parent_texts = _fetch_parent_texts(unique_ids)

    for hit in hits:
        pid = hit.metadata.get("parent_chunk_id")
        if pid and pid in parent_texts:
            hit.metadata["parent_text"] = parent_texts[pid]

    return hits


def _es_hit_to_retrieval_hit(raw_hit: dict, retriever: str) -> RetrievalHit:
    source = raw_hit.get("_source", {})
    metadata = source.get("metadata", {}) or {}
    content = source.get("text") or source.get("page_content") or ""
    return RetrievalHit(
        chunk_id=str(metadata.get("chunk_id", "")),
        content=content,
        source=str(metadata.get("source", "unknown")),
        page=metadata.get("page", "?"),
        section=str(metadata.get("section", "Unknown")),
        score=float(raw_hit.get("_score", 0.0)),
        retriever=retriever,
        metadata=metadata,
    )


def _source_filter(sources: str | list[str] | None) -> dict:
    """构造 ES source 过滤条件。

    None / [] → 不做过滤
    str → 单论文 term filter
    list[str] → 多论文 terms filter（限在这几篇里搜）
    """
    if not sources:  # None 和 [] 都走 match_all
        return {"match_all": {}}
    if isinstance(sources, str):
        return {"term": {"metadata.source.keyword": sources}}
    return {"terms": {"metadata.source.keyword": sources}}


def bm25_search(query: str, top_k: int = 30, sources: str | list[str] | None = None, *, skip_enrichment: bool = False) -> list[RetrievalHit]:
    es = get_es_client()
    if not es.indices.exists(index=ES_INDEX):
        return []

    response = es.search(
        index=ES_INDEX,
        query={
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": [_child_chunk_filter(), _source_filter(sources)],
            }
        },
        size=top_k,
    )
    hits = [_es_hit_to_retrieval_hit(item, retriever="bm25") for item in response["hits"]["hits"]]
    if skip_enrichment:
        return hits
    return _enrich_hits_with_parents(hits)


def dense_search(query: str, top_k: int = 30, sources: str | list[str] | None = None, *, skip_enrichment: bool = False) -> list[RetrievalHit]:
    es = get_es_client()
    if not es.indices.exists(index=ES_INDEX):
        return []
    query_vector = embeddings.embed_query(query)
    response = es.search(
        index=ES_INDEX,
        query={
            "script_score": {
                "query": {
                    "bool": {
                        "filter": [_child_chunk_filter(), _source_filter(sources)],
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_vector},
                },
            }
        },
        size=top_k,
    )
    hits = [_es_hit_to_retrieval_hit(item, retriever="dense") for item in response["hits"]["hits"]]
    if skip_enrichment:
        return hits
    return _enrich_hits_with_parents(hits)


def rerank_hits(query: str, hits: list[RetrievalHit], top_k: int = 5) -> list[RetrievalHit]:
    if not hits:
        return []

    docs = [
        Document(
            page_content=hit.content,
            metadata={**hit.to_metadata(), "_hit_identity": hit.identity},
        )
        for hit in hits
    ]
    reranked_docs = list(get_reranker().compress_documents(docs, query))[:top_k]
    by_identity = {hit.identity: hit for hit in hits}

    reranked_hits: list[RetrievalHit] = []
    for doc in reranked_docs:
        identity = doc.metadata.get("_hit_identity")
        original = by_identity.get(identity)
        if original is None:
            continue
        reranked_hits.append(
            RetrievalHit(
                chunk_id=original.chunk_id,
                content=doc.page_content,
                source=original.source,
                page=original.page,
                section=original.section,
                score=float(doc.metadata.get("relevance_score", original.score)),
                retriever="rerank",
                metadata={k: v for k, v in doc.metadata.items() if k != "_hit_identity"},
            )
        )

    return reranked_hits


def _context_tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()))


def _context_overlap(left: str, right: str) -> float:
    left_tokens = _context_tokens(left)
    right_tokens = _context_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _select_diverse_parent_contexts(
    hits: list[RetrievalHit],
    top_k: int = 5,
    overlap_threshold: float = 0.75,
) -> list[RetrievalHit]:
    """保留 rerank 顺序，生成前按父块和上下文重叠做最终去重。"""
    selected: list[RetrievalHit] = []
    seen_parent_ids: set[str] = set()
    selected_contexts: list[str] = []

    for hit in hits:
        parent_id = str(hit.metadata.get("parent_chunk_id") or hit.chunk_id)
        if parent_id in seen_parent_ids:
            continue

        context = hit.context_text
        if any(_context_overlap(context, existing) >= overlap_threshold for existing in selected_contexts):
            continue

        selected.append(hit)
        seen_parent_ids.add(parent_id)
        selected_contexts.append(context)

        if len(selected) >= top_k:
            break

    return selected


def _search_pipeline_raw(query: str, top_k: int, fetch_k: int, sources: str | list[str] | None = None) -> list[RetrievalHit]:
    """管线核心（不含最终去重）：BM25 + Dense → RRF → Rerank → 父块补充。"""
    bm25_hits = bm25_search(query, top_k=fetch_k, sources=sources)
    dense_hits = dense_search(query, top_k=fetch_k, sources=sources)
    fused = rrf_fusion([bm25_hits, dense_hits], top_k=fetch_k)
    reranked = rerank_hits(query, fused, top_k=max(top_k * 3, fetch_k))
    return _enrich_hits_with_parents(reranked)


def _search_pipeline_children(query: str, top_k: int, fetch_k: int, sources: str | list[str] | None = None) -> list[RetrievalHit]:
    """子块管线（不查父块）：BM25 + Dense → RRF → Rerank。"""
    bm25_hits = bm25_search(query, top_k=fetch_k, sources=sources, skip_enrichment=True)
    dense_hits = dense_search(query, top_k=fetch_k, sources=sources, skip_enrichment=True)
    fused = rrf_fusion([bm25_hits, dense_hits], top_k=fetch_k)
    return rerank_hits(query, fused, top_k=max(top_k * 3, fetch_k))


def _search_pipeline(query: str, top_k: int, fetch_k: int, sources: str | list[str] | None = None) -> list[RetrievalHit]:
    """完整管线：_search_pipeline_raw + 最终多样性去重。"""
    enriched = _search_pipeline_raw(query, top_k=top_k, fetch_k=fetch_k, sources=sources)
    return _select_diverse_parent_contexts(enriched, top_k=top_k)


def hybrid_search(
    query: str,
    top_k: int = 5,
    fetch_k: int = 30,
    sources: list[str] | None = None,
) -> list[RetrievalHit]:
    """混合检索。

    sources=None  → 统一搜所有论文
    sources=["a.pdf"] → 只搜这篇 (ES term filter)
    sources=["a.pdf","b.pdf"] → 多篇论文限定范围 (ES terms filter)，单次管线统一搜
    """
    # 单论文 / 多论文 / 全量池：统一走 _search_pipeline，
    # sources 作为 ES terms filter 限定论文范围，dense 检索在限定论文内做语义匹配
    return _search_pipeline(query, top_k=top_k, fetch_k=fetch_k, sources=sources)


def apply_section_boost(
    hits: list[RetrievalHit],
    target_sections: list[str],
    boost: float = 0.05,
) -> list[RetrievalHit]:
    """对匹配目标章节的 chunk 做轻量加分，修正 reranker 对同论文内相似章节的区分不足。"""
    if not target_sections or not hits:
        return hits

    target_lower = {s.strip().lower() for s in target_sections}
    boosted = []
    for hit in hits:
        section = (hit.section or "").lower()
        subsection = (hit.metadata.get("subsection") or "").lower()
        matched = any(
            t in section or t in subsection or section.startswith(t) or f"{section} / {subsection}".startswith(t)
            for t in target_lower
        )
        if matched:
            hit.score += boost
        boosted.append(hit)

    boosted.sort(key=lambda h: -h.score)
    return boosted


def children_only_search(
    query: str,
    top_k: int = 5,
    fetch_k: int = 30,
    sources: list[str] | None = None,
) -> list[RetrievalHit]:
    """子块检索（不查父块、不做去重）。

    用于 Researcher 并行分发多条 query 后，在全局层统一做父块补充和去重。
    """
    return _search_pipeline_children(query, top_k=top_k, fetch_k=fetch_k, sources=sources)


def global_parent_enrichment(
    hits: list[RetrievalHit],
    top_k: int = 5,
    overlap_threshold: float = 0.75,
    merge_overlap_min: float = 0.15,
) -> list[RetrievalHit]:
    """全局父块补充 + 去重 + 相邻父块合并。

    1. 对所有子块统一查父块
    2. 按 parent_chunk_id 去重
    3. 同论文同 section 且内容有部分重叠的父块合并
    4. 高重叠 (>overlap_threshold) 的跳过
    5. 截断到 top_k
    """
    if not hits:
        return []

    # Step 1: 父块补充
    enriched = _enrich_hits_with_parents(hits)

    # Step 2: 去重 + 相邻合并
    selected: list[RetrievalHit] = []
    seen_parent_ids: set[str] = set()
    selected_contexts: list[tuple[str, str]] = []  # (source, context)

    for hit in enriched:
        parent_id = str(hit.metadata.get("parent_chunk_id") or hit.chunk_id)
        context = hit.context_text

        # 相同 parent_id → 跳过
        if parent_id in seen_parent_ids:
            continue

        # 同论文同 section 相邻父块 → 合并
        merged = False
        for i, (prev_source, prev_ctx) in enumerate(selected_contexts):
            same_paper = hit.source == prev_source
            same_section = hit.section == selected[i].section
            overlap = _context_overlap(context, prev_ctx)
            if same_paper and same_section and merge_overlap_min <= overlap < overlap_threshold:
                merged_text = prev_ctx + "\n\n" + context
                selected[i].metadata["parent_text"] = merged_text
                selected_contexts[i] = (prev_source, merged_text)
                seen_parent_ids.add(parent_id)
                merged = True
                break

        if merged:
            continue

        # 高重叠 → 视为重复，跳过
        if any(_context_overlap(context, existing) >= overlap_threshold for _, existing in selected_contexts):
            continue

        selected.append(hit)
        seen_parent_ids.add(parent_id)
        selected_contexts.append((hit.source, context))

        if len(selected) >= top_k:
            break

    return selected


def get_available_papers() -> list[dict]:
    """返回当前 ES 索引中所有论文的元信息列表。

    每项: {source, title, abstract}
    - source: 论文文件名 (metadata.source)
    - title: 论文标题 (from paper_metadata chunk)
    - abstract: 摘要正文前 500 字符 (from section=Abstract body chunks)
    """
    es = get_es_client()
    if not es.indices.exists(index=ES_INDEX):
        return []

    # 1. 用 terms aggregation 获取所有不重复论文 source
    agg_resp = es.search(
        index=ES_INDEX,
        query={"match_all": {}},
        aggs={
            "paper_sources": {
                "terms": {
                    "field": "metadata.source.keyword",
                    "size": 100,
                }
            }
        },
        size=0,
    )

    buckets = agg_resp.get("aggregations", {}).get("paper_sources", {}).get("buckets", [])

    papers: list[dict] = []
    for bucket in buckets:
        source = bucket["key"]

        # 2. 从 paper_metadata chunk 取标题
        title_resp = es.search(
            index=ES_INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"metadata.source.keyword": source}},
                        {"term": {"metadata.chunk_type.keyword": "paper_metadata"}},
                    ]
                }
            },
            size=1,
        )
        title = source  # fallback
        hits = title_resp.get("hits", {}).get("hits", [])
        if hits:
            md = hits[0].get("_source", {}).get("metadata", {}) or {}
            title = md.get("paper_title", source)

        # 3. 从 section=Abstract 的 body chunk 取摘要
        abstract_resp = es.search(
            index=ES_INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"metadata.source.keyword": source}},
                        {"term": {"metadata.section.keyword": "Abstract"}},
                    ],
                    "must_not": [
                        {"terms": {"metadata.chunk_type.keyword": ["parent", "paper_metadata"]}}
                    ],
                }
            },
            size=20,
        )
        abstract_parts = []
        for hit in abstract_resp.get("hits", {}).get("hits", []):
            text = hit.get("_source", {}).get("text", "")
            if text:
                # strip heading prefix like "[Paper Title | Abstract]"
                if "] " in text:
                    text = text.split("] ", 1)[-1]
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)[:500] if abstract_parts else ""

        papers.append({
            "source": source,
            "title": title,
            "abstract": abstract,
        })

    return papers

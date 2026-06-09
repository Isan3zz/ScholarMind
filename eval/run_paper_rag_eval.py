import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")


DATASET_PATH = Path(__file__).with_name("paper_rag_dataset.jsonl")
DEFAULT_PDF_CANDIDATES = [
    Path(r"D:\学术\越狱防御\2024-acl-SafeDecoding.pdf"),
    BACKEND / "app" / "rag" / "uploads" / "2024-acl-SafeDecoding.pdf",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "provide",
    "the",
    "this",
    "to",
    "used",
    "what",
    "which",
    "with",
}


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def compute_section_hit(sample: dict, hits: list[dict], k: int = 5) -> bool:
    expected = set(sample.get("expected_sections") or [])
    if not expected:
        return False
    for hit in hits[:k]:
        labels = _hit_section_labels(hit)
        if labels & expected:
            return True
    return False


def compute_mrr(sample: dict, hits: list[dict]) -> float:
    expected = set(sample.get("expected_sections") or [])
    if not expected:
        return 0.0
    for rank, hit in enumerate(hits, start=1):
        if _hit_section_labels(hit) & expected:
            return 1.0 / rank
    return 0.0


def compute_section_hit_rate(samples: list[dict], results: list[list[dict]], k: int = 5) -> float:
    answerable = [(sample, hits) for sample, hits in zip(samples, results) if sample.get("expected_sections")]
    if not answerable:
        return 0.0
    hits = sum(1 for sample, retrieved in answerable if compute_section_hit(sample, retrieved, k=k))
    return hits / len(answerable)


def _matched_expected_labels(sample: dict, hit: dict) -> set[str]:
    expected = set(sample.get("expected_sections") or [])
    return _hit_section_labels(hit) & expected


def compute_precision_recall_f1_at_k(sample: dict, hits: list[dict], k: int = 5) -> dict:
    expected = set(sample.get("expected_sections") or [])
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    retrieved = hits[:k]
    relevant_hits = sum(1 for hit in retrieved if _matched_expected_labels(sample, hit))
    matched_labels: set[str] = set()
    for hit in retrieved:
        matched_labels.update(_matched_expected_labels(sample, hit))

    precision = relevant_hits / k if k else 0.0
    recall = len(matched_labels) / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_mean_precision_recall_f1_at_k(
    samples: list[dict],
    results: list[list[dict]],
    k: int = 5,
) -> dict:
    answerable = [(sample, hits) for sample, hits in zip(samples, results) if sample.get("expected_sections")]
    if not answerable:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    per_case = [compute_precision_recall_f1_at_k(sample, hits, k=k) for sample, hits in answerable]
    return {
        name: sum(item[name] for item in per_case) / len(per_case)
        for name in ["precision", "recall", "f1"]
    }


def compute_mean_mrr(samples: list[dict], results: list[list[dict]]) -> float:
    answerable = [(sample, hits) for sample, hits in zip(samples, results) if sample.get("expected_sections")]
    if not answerable:
        return 0.0
    return sum(compute_mrr(sample, hits) for sample, hits in answerable) / len(answerable)


def compute_keyword_recall(sample: dict, hits: list[dict], k: int = 5) -> float:
    keywords = [str(item) for item in sample.get("answer_keywords") or []]
    if not keywords:
        return 0.0
    raw_context = "\n".join(str(hit.get("content", "")) for hit in hits[:k]).lower()
    normalized_context = _normalize_for_keyword_match(raw_context)
    matched = sum(
        1
        for keyword in keywords
        if keyword.lower() in raw_context
        or _normalize_for_keyword_match(keyword) in normalized_context
    )
    return matched / len(keywords)


def compute_mean_keyword_recall(samples: list[dict], results: list[list[dict]], k: int = 5) -> float:
    answerable = [(sample, hits) for sample, hits in zip(samples, results) if sample.get("answer_keywords")]
    if not answerable:
        return 0.0
    return sum(compute_keyword_recall(sample, hits, k=k) for sample, hits in answerable) / len(answerable)


def compute_refusal_rejection_rate(
    samples: list[dict],
    results: list[list[dict]],
    min_score: float = 0.5,
) -> float:
    refusal_cases = [(sample, hits) for sample, hits in zip(samples, results) if sample.get("should_refuse")]
    if not refusal_cases:
        return 0.0
    rejected = 0
    for _, hits in refusal_cases:
        if not hits or float(hits[0].get("score", 0.0)) < min_score:
            rejected += 1
    return rejected / len(refusal_cases)


def has_section_citation(answer: str) -> bool:
    return bool(re.search(r"\[source:\s*[^,\]]+,\s*section:\s*[^\]]+\]", answer or "", re.IGNORECASE))


def looks_like_refusal(answer: str) -> bool:
    text = (answer or "").lower()
    refusal_markers = [
        "does not contain",
        "not contain",
        "no evidence",
        "insufficient evidence",
        "cannot answer",
        "not discussed",
        "not provide",
        "没有",
        "未提到",
        "无法回答",
        "证据不足",
    ]
    return any(marker in text for marker in refusal_markers)


def compute_answer_keyword_recall(sample: dict, answer: str) -> float:
    keywords = [str(item) for item in sample.get("answer_keywords") or []]
    if not keywords:
        return 0.0
    raw_answer = (answer or "").lower()
    normalized_answer = _normalize_for_keyword_match(raw_answer)
    matched = sum(
        1
        for keyword in keywords
        if keyword.lower() in raw_answer
        or _normalize_for_keyword_match(keyword) in normalized_answer
    )
    return matched / len(keywords)


def compute_answer_metrics(samples: list[dict], answers: list[str]) -> dict:
    answerable = [(sample, answer) for sample, answer in zip(samples, answers) if not sample.get("should_refuse")]
    refusals = [(sample, answer) for sample, answer in zip(samples, answers) if sample.get("should_refuse")]
    return {
        "answer_keyword_recall": (
            sum(compute_answer_keyword_recall(sample, answer) for sample, answer in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "citation_format_rate": (
            sum(1 for _, answer in answerable if has_section_citation(answer)) / len(answerable)
            if answerable
            else 0.0
        ),
        "refusal_answer_rate": (
            sum(1 for _, answer in refusals if looks_like_refusal(answer)) / len(refusals)
            if refusals
            else 0.0
        ),
    }


JUDGE_FIELDS = [
    "answer_correctness",
    "faithfulness",
    "key_point_coverage",
    "citation_accuracy",
    "evidence_sufficiency",
]


def compute_judge_metrics(judgments: list[dict]) -> dict:
    if not judgments:
        return {f"judge_{field}": 0.0 for field in JUDGE_FIELDS}
    metrics = {}
    for field in JUDGE_FIELDS:
        metrics[f"judge_{field}"] = sum(float(item.get(field, 0.0)) for item in judgments) / len(judgments)
    return metrics


def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def deepseek_judge_config() -> dict:
    from dotenv import load_dotenv

    load_dotenv(BACKEND / ".env", override=False)
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Add DEEPSEEK_API_KEY to backend/.env before "
            "running --llm-judge, or use OPENAI_API_KEY with an OpenAI-compatible judge base URL."
        )
    return {
        "model": os.getenv("DEEPSEEK_JUDGE_MODEL") or os.getenv("SMART_LLM_MODEL", "deepseek-chat"),
        "base_url": os.getenv("DEEPSEEK_API_BASE") or os.getenv("OPENAI_API_BASE", "https://api.deepseek.com"),
        "api_key": api_key.strip(),
    }


def get_deepseek_judge_llm():
    from langchain_openai import ChatOpenAI

    config = deepseek_judge_config()
    return ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["base_url"],
        api_key=config["api_key"],
    )


def judge_answer_with_llm(sample: dict, hits: list[dict], answer: str, judge_llm=None) -> dict:
    from langchain_core.messages import HumanMessage

    if sample.get("should_refuse"):
        refused = looks_like_refusal(answer)
        score = 1.0 if refused else 0.0
        return {
            "answer_correctness": score,
            "faithfulness": score,
            "key_point_coverage": score,
            "citation_accuracy": score,
            "evidence_sufficiency": score,
            "notes": "Correct refusal for an out-of-scope question." if refused else "Expected a refusal for an out-of-scope question.",
        }

    judge_llm = judge_llm or get_deepseek_judge_llm()
    evidence = format_hits_for_answer(hits)
    prompt = f"""
You are a RAG evaluation judge for academic paper QA. Return only valid JSON.
Scores must be between 0.0 and 1.0.

=== CORE RULE: HONEST REFUSAL IS A VALID ANSWER ===
If the answer states that the evidence is insufficient to fully answer the question, and honestly explains what IS available and what IS missing, this is CORRECT behavior. Score it high on faithfulness and evidence_sufficiency (it is being truthful). Score it moderate on correctness and coverage (identifying a knowledge gap IS valuable, but the question is not fully resolved). ONLY give 0.0 if the answer fabricates claims or ignores obviously relevant evidence.

=== DIMENSION DEFINITIONS ===

answer_correctness (0-1):
- Does the answer say things that are factually consistent with the evidence?
- If the answer says "evidence does not contain X" and X is genuinely absent → correct statement, give ≥0.5.
- If the answer draws reasonable conclusions from the evidence → give ≥0.7.
- Score 0.0 ONLY for factually false claims or complete non-answers.

faithfulness (0-1):
- Are the factual claims in the answer verifiable in the retrieved evidence?
- An honest refusal with accurate description of available evidence → ≥0.8.
- Penalize ONLY fabricated claims that contradict the evidence.
- Synthesizing across multiple chunks is faithful if each piece is supported.

key_point_coverage (0-1):
- Does the answer address the key dimensions the question asks about?
- If the question asks about two papers and the answer discusses both → ≥0.5.
- If the answer identifies WHICH dimensions are covered vs. missing → ≥0.7.
- The question text itself defines what needs to be covered. There is no checklist.

citation_accuracy (0-1):
- Does the answer use [source: file, section: ...] markers?
- Are the cited sections relevant to the adjacent claims?
- If no citations are present but the answer is otherwise good → 0.3 (penalty, not fatal).
- If citations are present and point to correct sections → ≥0.7.

evidence_sufficiency (0-1):
- Does the answer make effective use of whatever evidence IS available?
- If the answer correctly extracts all available relevant information → ≥0.7.
- If the answer ignores obviously relevant evidence chunks → deduct.
- Do NOT deduct because "the evidence alone can't answer everything." That is normal for cross-paper questions. Deduct ONLY for ignoring available evidence or claiming unsupported facts.

=== SCORING GUIDE ===
0.0: answer is fabricated, contradicts evidence, or says nothing at all.
0.3-0.5: answer contains some true statements but misses major evidence or key question dimensions.
0.6-0.8: answer is mostly correct, uses most available evidence honestly, covers key dimensions. Some gaps are acceptable.
0.9-1.0: answer is comprehensive, uses all relevant evidence, correctly identifies what IS and IS NOT supported, well-cited.

=== CONTEXT ===
This evaluation involves cross-paper comparison questions. Evidence comes from multiple documents. Synthesizing across sources is expected and rewarded.

Question:
{sample.get("question", "")}

Retrieved evidence:
{evidence[:12000]}

Answer:
{answer}

Return JSON with exactly these keys:
{{
  "answer_correctness": 0.0,
  "faithfulness": 0.0,
  "key_point_coverage": 0.0,
  "citation_accuracy": 0.0,
  "evidence_sufficiency": 0.0,
  "notes": "short reason"
}}
"""
    raw = judge_llm.invoke([HumanMessage(content=prompt)]).content
    parsed = _extract_json_object(raw)
    result = {"notes": str(parsed.get("notes", ""))}
    for field in JUDGE_FIELDS:
        value = float(parsed.get(field, 0.0))
        result[field] = max(0.0, min(1.0, value))
    return result


def evaluate_answers_with_judge(
    samples: list[dict],
    retrieval_results: list[list[dict]],
    answers: list[str],
    judge_llm=None,
) -> tuple[list[dict], dict]:
    judgments = [
        judge_answer_with_llm(sample, hits, answer, judge_llm=judge_llm)
        for sample, hits, answer in zip(samples, retrieval_results, answers)
    ]
    return judgments, compute_judge_metrics(judgments)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-']+", (text or "").lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def _normalize_for_keyword_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _strip_section_prefix(name: str) -> str:
    """去掉 MinerU 输出的章节编号前缀，如 '6 Conclusion' → 'Conclusion'."""
    import re as _re
    # 匹配: 数字(可能带.) + 空格 或 大写字母(可能带数字/.) + 空格
    stripped = _re.sub(r"^(?:\d+(?:\.\d+)*|[A-Z](?:\d+)?(?:\.\d+)*)\s+", "", name)
    return stripped if stripped != name else name


def _hit_section_labels(hit: dict) -> set[str]:
    section = str(hit.get("section") or "").strip()
    subsection = str(hit.get("subsection") or "").strip()
    labels: set[str] = set()
    for raw in [section, subsection]:
        if raw:
            labels.add(raw)
            stripped = _strip_section_prefix(raw)
            if stripped != raw:
                labels.add(stripped)
    if section and subsection:
        labels.add(f"{section} / {subsection}")
        # 也加一份去前缀的复合标签
        s = _strip_section_prefix(section)
        u = _strip_section_prefix(subsection)
        if s != section or u != subsection:
            labels.add(f"{s} / {u}")
    return labels


def lexical_search(query: str, docs: list[dict], top_k: int = 5, min_score: float = 0.01) -> list[dict]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    query_terms = set(query_tokens)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    scored = []
    for doc in docs:
        content = str(doc.get("content", ""))
        scored_text = " ".join(
            [
                str(doc.get("section", "")),
                str(doc.get("subsection", "")),
                content,
            ]
        )
        doc_tokens = _tokens(scored_text)
        if not doc_tokens:
            continue
        doc_terms = set(doc_tokens)
        doc_bigrams = set(zip(doc_tokens, doc_tokens[1:]))
        term_overlap = len(query_terms & doc_terms)
        bigram_overlap = len(query_bigrams & doc_bigrams)
        score = term_overlap + (2.0 * bigram_overlap)
        hit = dict(doc)
        hit["score"] = score / math.sqrt(len(doc_terms))
        if hit["score"] < min_score:
            continue
        scored.append(hit)
    scored.sort(key=lambda item: (-float(item["score"]), item.get("section", ""), item.get("chunk_id", "")))
    return scored[:top_k]


def _hit_to_dict(hit) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "content": hit.context_text,
        "section": hit.section,
        "subsection": hit.metadata.get("subsection", ""),
        "chunk_type": hit.metadata.get("chunk_type", ""),
        "source": hit.source,
        "score": hit.score,
        "retriever": hit.retriever,
    }


def rerank_eval_hits(
    query: str,
    hits: list[dict],
    top_k: int = 5,
    rerank_fn=None,
) -> list[dict]:
    if not hits:
        return []

    from app.rag.hits import RetrievalHit

    candidates = [
        RetrievalHit(
            chunk_id=str(hit.get("chunk_id", "")),
            content=str(hit.get("content", "")),
            source=str(hit.get("source", "unknown")),
            page=hit.get("page", 0),
            section=str(hit.get("section", "Unknown")),
            score=float(hit.get("score", 0.0)),
            retriever=str(hit.get("retriever", "eval_lexical")),
            metadata={
                "subsection": hit.get("subsection", ""),
                "chunk_type": hit.get("chunk_type", ""),
            },
        )
        for hit in hits
    ]

    if rerank_fn is None:
        from app.rag.engine import rerank_hits

        rerank_fn = rerank_hits

    return [_hit_to_dict(hit) for hit in rerank_fn(query, candidates, top_k=top_k)]


def resolve_pdf_path(pdf_path: str | None = None) -> Path:
    candidates = []
    if pdf_path:
        candidates.append(Path(pdf_path))
    env_path = os.getenv("SCHOLARMIND_EVAL_PDF")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(DEFAULT_PDF_CANDIDATES)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    raise FileNotFoundError("SafeDecoding PDF not found. Pass --pdf or set SCHOLARMIND_EVAL_PDF.")


def build_docs_from_pdf(pdf_path: Path) -> list[dict]:
    from app.rag.paper_chunker import build_paper_chunks
    from app.rag.marker_parser import parse_pdf_with_marker

    units, title, authors = parse_pdf_with_marker(str(pdf_path))
    documents = build_paper_chunks(
        units=units,
        source=pdf_path.name,
        paper_title=title,
        page=0,
        authors=authors,
    )
    docs = []
    for doc in documents:
        meta = doc.metadata or {}
        chunk_type = meta.get("chunk_type", "")
        if chunk_type in {"paper_metadata", "parent"}:
            continue
        docs.append(
            {
                "chunk_id": meta.get("chunk_id", ""),
                "content": doc.page_content,
                "section": meta.get("section", "Unknown"),
                "subsection": meta.get("subsection", ""),
                "chunk_type": chunk_type,
                "source": meta.get("source", pdf_path.name),
            }
        )
    return docs


def summarize_corpus(docs: list[dict]) -> dict:
    sections = sorted({doc.get("section", "Unknown") for doc in docs})
    formula_chunks = sum(1 for doc in docs if "[Formula" in str(doc.get("content", "")))
    return {
        "chunks": len(docs),
        "sections": sections,
        "section_count": len(sections),
        "formula_chunks": formula_chunks,
    }


def evaluate_samples(
    samples: list[dict],
    docs: list[dict],
    top_k: int = 5,
    fetch_k: int = 20,
    min_score: float = 0.01,
    reject_threshold: float = 0.5,
    use_reranker: bool = True,
) -> tuple[list[list[dict]], dict]:
    results = []
    for sample in samples:
        candidates = lexical_search(sample["question"], docs, top_k=fetch_k, min_score=min_score)
        if use_reranker and candidates:
            results.append(rerank_eval_hits(sample["question"], candidates, top_k=top_k))
        else:
            results.append(candidates[:top_k])
    metrics = {
        "top1_accuracy": compute_section_hit_rate(samples, results, k=1),
        "section_hit@1": compute_section_hit_rate(samples, results, k=1),
        "section_hit@3": compute_section_hit_rate(samples, results, k=3),
        f"section_hit@{top_k}": compute_section_hit_rate(samples, results, k=top_k),
        "mrr": compute_mean_mrr(samples, results),
        f"keyword_recall@{top_k}": compute_mean_keyword_recall(samples, results, k=top_k),
        "refusal_rejection_rate": compute_refusal_rejection_rate(samples, results, min_score=reject_threshold),
    }
    prf = compute_mean_precision_recall_f1_at_k(samples, results, k=top_k)
    metrics[f"precision@{top_k}"] = prf["precision"]
    metrics[f"recall@{top_k}"] = prf["recall"]
    metrics[f"f1@{top_k}"] = prf["f1"]
    return results, metrics


def evaluate_with_search_fn(
    samples: list[dict],
    search_fn,
    top_k: int = 5,
    fetch_k: int = 20,
    reject_threshold: float = 0.5,
) -> tuple[list[list[dict]], dict]:
    results = []
    for sample in samples:
        paper = sample.get("paper")
        sources = None
        if isinstance(paper, str):
            sources = [paper]
        elif isinstance(paper, list) and paper:
            sources = paper
        try:
            results.append(search_fn(sample["question"], top_k=top_k, fetch_k=fetch_k, sources=sources))
        except TypeError:
            results.append(search_fn(sample["question"], top_k=top_k, fetch_k=fetch_k))
    metrics = {
        "top1_accuracy": compute_section_hit_rate(samples, results, k=1),
        "section_hit@1": compute_section_hit_rate(samples, results, k=1),
        "section_hit@3": compute_section_hit_rate(samples, results, k=3),
        f"section_hit@{top_k}": compute_section_hit_rate(samples, results, k=top_k),
        "mrr": compute_mean_mrr(samples, results),
        f"keyword_recall@{top_k}": compute_mean_keyword_recall(samples, results, k=top_k),
        "refusal_rejection_rate": compute_refusal_rejection_rate(samples, results, min_score=reject_threshold),
    }
    prf = compute_mean_precision_recall_f1_at_k(samples, results, k=top_k)
    metrics[f"precision@{top_k}"] = prf["precision"]
    metrics[f"recall@{top_k}"] = prf["recall"]
    metrics[f"f1@{top_k}"] = prf["f1"]
    return results, metrics


def formal_hybrid_search(query: str, top_k: int = 5, fetch_k: int = 20, sources: list[str] | None = None) -> list[dict]:
    from app.rag.engine import hybrid_search

    return [_hit_to_dict(hit) for hit in hybrid_search(query, top_k=top_k, fetch_k=fetch_k, sources=sources)]


def formal_candidate_search(query: str, fetch_k: int = 20, sources: list[str] | None = None) -> list[dict]:
    from app.rag.engine import bm25_search, dense_search
    from app.rag.rank_fusion import rrf_fusion

    bm25_hits = bm25_search(query, top_k=fetch_k, sources=sources)
    dense_hits = dense_search(query, top_k=fetch_k, sources=sources)
    return [_hit_to_dict(hit) for hit in rrf_fusion([bm25_hits, dense_hits], top_k=fetch_k)]


def global_rerank_agent_hits(query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    return rerank_eval_hits(query, hits, top_k=top_k)


def _preferred_labels_for_query(query: str) -> set[str]:
    q = (query or "").lower()
    labels: set[str] = set()

    # Map query intent → likely target sections
    intent_keywords = [
        # Contribution / summary
        (["contribution", "main idea", "summary", "overview", "key finding",
          "propos", "introduce", "novel", "what is"], {"Abstract", "Introduction", "Conclusion and Future Work", "6 Conclusion and Future Work"}),
        # Method / how
        (["method", "how does", "architecture", "approach", "framework",
          "mechanism", "algorithm", "strategy", "pipeline", "design",
          "training", "fine-tune", "expert model", "inference", "decoding",
          "sample space", "token distribution"], {"4 Safety-Aware Decoding: SafeDecoding", "3 Preliminaries", "Method", "Approach"}),
        # Experiments / setup
        (["experiment", "setup", "baseline", "benchmark", "dataset",
          "evaluat", "compared", "compare against", "defense method",
          "attack method", "model used"], {"5 Experiments", "A Detailed Experimental Setups", "Experimental Setup"}),
        # Results
        (["result", "performance", "score", "metric", "accuracy",
          "effective", "improve", "outperform", "reduce"], {"B More Results", "5 Experiments", "Abstract"}),
        # Limitations / future
        (["limitation", "future work", "drawback", "failure", "weakness",
          "challenge", "gap"], {"7 Limitations", "6 Conclusion and Future Work"}),
        # Ethics / impact
        (["ethic", "impact", "societal", "safety concern", "harm",
          "misuse"], {"8 Ethical Impact", "6 Conclusion and Future Work"}),
        # Formula / math
        (["formula", "equation", "math", "notation", "symbol",
          "sample space", "distribution"], {"4 Safety-Aware Decoding: SafeDecoding"}),
    ]
    for keywords, section_labels in intent_keywords:
        if any(term in q for term in keywords):
            labels.update(section_labels)

    return labels


def _apply_eval_label_boost(query: str, hits: list[dict], boost: float = 0.05) -> list[dict]:
    preferred = _preferred_labels_for_query(query)
    if not preferred:
        return hits
    boosted = []
    for hit in hits:
        copied = dict(hit)
        if _hit_section_labels(copied) & preferred:
            copied["score"] = float(copied.get("score", 0.0)) + boost
            copied["eval_label_boosted"] = True
        boosted.append(copied)
    return sorted(boosted, key=lambda item: (-float(item.get("score", 0.0)), str(item.get("chunk_id", ""))))


def plan_retrieval_queries(query: str) -> tuple[list[str], list[list[str]]]:
    from app.graph.nodes.planner import PLAN_PROMPT, PlanResult
    from app.rag.engine import get_available_papers
    from app.utils.llm import get_llm

    # Build paper catalog for the Planner prompt
    papers = get_available_papers()
    if papers:
        lines = []
        for i, p in enumerate(papers, 1):
            title = p.get("title", p["source"])
            abstract = p.get("abstract", "")
            abstract_short = abstract[:300] + ("..." if len(abstract) > 300 else "")
            lines.append(f"{i}. source: {p['source']}")
            lines.append(f"   title: {title}")
            if abstract_short:
                lines.append(f"   abstract: {abstract_short}")
        paper_catalog = "\n".join(lines)
    else:
        paper_catalog = "(No papers currently indexed in the knowledge base)"

    result = get_llm().with_structured_output(PlanResult).invoke(
        PLAN_PROMPT.format(query=query, critique="", short_memory_context="", paper_catalog=paper_catalog)
    )
    queries = []
    sections = []
    for qp in result.queries:
        if qp.query.strip():
            queries.append(qp.query.strip())
            sections.append([s.strip() for s in (qp.sections or []) if s.strip()])
    return queries, sections


def planned_retrieval_queries(
    query: str,
    plans: list[str],
    plan_sections: list[list[str]] | None = None,
    max_rewrites: int | None = None,
) -> list[tuple[str, list[str]]]:
    """返回 (query, sections) 列表，sections 为空列表表示无 boost。"""
    result: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    selected_plans = plans or []
    selected_sections = plan_sections or []
    if max_rewrites is not None:
        selected_plans = selected_plans[: max(0, max_rewrites)]
        selected_sections = selected_sections[: max(0, max_rewrites)]
    # 原始 query，无 sections
    cleaned = str(query or "").strip()
    if cleaned:
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append((cleaned, []))
    for i, plan in enumerate(selected_plans):
        cleaned = str(plan or "").strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            secs = selected_sections[i] if i < len(selected_sections) else []
            result.append((cleaned, secs))
    return result


def formal_agent_search(
    query: str,
    top_k: int = 5,
    fetch_k: int = 20,
    max_rewrites: int | None = None,
    planner_fn=None,
    candidate_search_fn=None,
    rerank_fn=None,
    paper_sources: list[str] | None = None,
) -> list[dict]:
    if planner_fn is None:
        planner_fn = plan_retrieval_queries
    if candidate_search_fn is None:
        candidate_search_fn = formal_candidate_search
    if rerank_fn is None:
        rerank_fn = global_rerank_agent_hits

    planner_queries, planner_sections = planner_fn(query)
    docs = []
    seen = set()
    from app.rag.engine import apply_section_boost
    from app.rag.hits import RetrievalHit

    for retrieval_query, target_sections in planned_retrieval_queries(query, planner_queries, planner_sections, max_rewrites=max_rewrites):
        hits = candidate_search_fn(retrieval_query, fetch_k=fetch_k, sources=paper_sources)
        # Section boost BEFORE rerank (Planner-driven, not eval-driven)
        if target_sections and hits:
            retrieval_hits = [
                RetrievalHit(
                    chunk_id=str(h.get("chunk_id", "")),
                    content=str(h.get("content", "")),
                    source=str(h.get("source", "unknown")),
                    page=h.get("page", 0),
                    section=str(h.get("section", "Unknown")),
                    score=float(h.get("score", 0.0)),
                    retriever=str(h.get("retriever", "")),
                    metadata={k: v for k, v in h.items() if k not in ("chunk_id", "content", "source", "page", "section", "score", "retriever")},
                )
                for h in hits
            ]
            retrieval_hits = apply_section_boost(retrieval_hits, target_sections)
            hits = [_hit_to_dict(hit) for hit in retrieval_hits]
        for hit in hits:
            identity = str(
                hit.get("chunk_id")
                or f"{hit.get('source')}:{hit.get('section')}:{hit.get('content')}"
            )
            if identity in seen:
                continue
            seen.add(identity)
            doc = dict(hit)
            doc["retrieval_query"] = retrieval_query
            docs.append(doc)
    rerank_k = min(len(docs), max(top_k, fetch_k))
    reranked = rerank_fn(query, docs, top_k=rerank_k)
    return reranked[:top_k]


def format_hits_for_answer(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        source = hit.get("source", "unknown")
        section = hit.get("section") or "Unknown"
        subsection = hit.get("subsection") or ""
        section_label = f"{section} / {subsection}" if subsection else section
        content = str(hit.get("content", "")).strip()
        blocks.append(f"[source: {source}, section: {section_label}]\n{content}")
    return "\n\n".join(blocks)


def generate_answer(question: str, hits: list[dict], reject_threshold: float = 0.5) -> str:
    from langchain_core.messages import HumanMessage
    from app.utils.llm import get_llm

    if not hits or float(hits[0].get("score", 0.0)) < reject_threshold:
        return "The provided paper does not contain sufficient evidence to answer this question."

    evidence = format_hits_for_answer(hits)
    prompt = f"""
Answer the question using only the provided paper evidence.
Every factual claim must include a citation in this format: [source: file, section: SectionName].
If the evidence is insufficient, say that the paper does not contain sufficient evidence.

Question:
{question}

Evidence:
{evidence}
"""
    return get_deepseek_judge_llm().invoke([HumanMessage(content=prompt)]).content.strip()


def evaluate_answers(
    samples: list[dict],
    retrieval_results: list[list[dict]],
    reject_threshold: float = 0.5,
) -> tuple[list[str], dict]:
    answers = [
        generate_answer(sample["question"], hits, reject_threshold=reject_threshold)
        for sample, hits in zip(samples, retrieval_results)
    ]
    return answers, compute_answer_metrics(samples, answers)


def run_mock_strategy(sample: dict, strategy: str) -> list[dict]:
    expected = sample.get("expected_sections") or []
    if strategy == "dense_only":
        return [{"section": "Introduction", "content": ""}]
    if strategy == "bm25_only":
        return [{"section": expected[0] if expected else "Unknown", "content": ""}]
    if strategy == "hybrid_rrf":
        return [{"section": expected[0] if expected else "Unknown", "content": ""}]
    return []


def safe_console_text(text: str) -> str:
    return str(text).encode("gbk", errors="backslashreplace").decode("gbk")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper RAG evaluation.")
    parser.add_argument(
        "--pipeline",
        choices=["offline", "formal", "formal-agent"],
        default="offline",
        help=(
            "offline: parse PDF in-memory and use lexical recall plus optional rerank; "
            "formal: query the existing Elasticsearch index via hybrid_search "
            "(BM25 + Dense + RRF + section boost + rerank); "
            "formal-agent: run planner query rewrite before formal retrieval."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the jsonl eval dataset. Defaults to paper_rag_dataset.jsonl.",
    )
    parser.add_argument("--pdf", default=None, help="Path to the SafeDecoding PDF.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument(
        "--max-rewrites",
        type=int,
        default=None,
        help="Limit planner rewrites per question for formal-agent eval; original query is always included.",
    )
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    parser.add_argument("--answer-eval", action="store_true", help="Generate answers and compute answer-level metrics.")
    parser.add_argument("--llm-judge", action="store_true", help="Use DeepSeek to judge answer correctness, faithfulness, coverage, citations, and evidence sufficiency.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable qwen3-rerank in eval.")
    parser.add_argument("--mock", action="store_true", help="Run the old mock strategy sanity check.")
    args = parser.parse_args()

    samples = load_dataset(Path(args.dataset))
    print(f"Loaded {len(samples)} paper RAG eval samples")

    if args.mock:
        strategies = ["dense_only", "bm25_only", "hybrid_rrf"]
        for strategy in strategies:
            results = [run_mock_strategy(sample, strategy) for sample in samples]
            print(
                f"{strategy}: "
                f"section_hit@{args.top_k}={compute_section_hit_rate(samples, results, k=args.top_k):.2f}, "
                f"mrr={compute_mean_mrr(samples, results):.2f}"
            )
        return

    if args.pipeline == "formal":
        print("Pipeline: formal hybrid_search (BM25 + Dense + RRF + qwen3-rerank + section boost)")
        results, metrics = evaluate_with_search_fn(
            samples,
            formal_hybrid_search,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            reject_threshold=args.reject_threshold,
        )
    elif args.pipeline == "formal-agent":
        print(
            "Pipeline: formal agent retrieval "
            "(planner rewrite + original query + BM25 + Dense + RRF + qwen3-rerank + section boost)"
        )
        results, metrics = evaluate_with_search_fn(
            samples,
            lambda query, top_k, fetch_k, sources=None: formal_agent_search(
                query,
                top_k=top_k,
                fetch_k=fetch_k,
                max_rewrites=args.max_rewrites,
                paper_sources=sources,
            ),
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            reject_threshold=args.reject_threshold,
        )
    else:
        pdf_path = resolve_pdf_path(args.pdf)
        print(f"PDF: {pdf_path}")
        docs = build_docs_from_pdf(pdf_path)
        corpus = summarize_corpus(docs)
        print(
            f"Corpus: chunks={corpus['chunks']}, sections={corpus['section_count']}, "
            f"formula_chunks={corpus['formula_chunks']}"
        )
        print("Sections:", ", ".join(corpus["sections"]))

        results, metrics = evaluate_samples(
            samples,
            docs,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            min_score=args.min_score,
            reject_threshold=args.reject_threshold,
            use_reranker=not args.no_rerank,
        )
        print(f"Reranker: {'disabled' if args.no_rerank else 'enabled'}")

    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")

    if args.answer_eval:
        print("\nAnswer eval:")
        answers, answer_metrics = evaluate_answers(
            samples,
            results,
            reject_threshold=args.reject_threshold,
        )
        for name, value in answer_metrics.items():
            print(f"{name}: {value:.2f}")
        if args.llm_judge:
            print("\nDeepSeek judge eval:")
            judgments, judge_metrics = evaluate_answers_with_judge(samples, results, answers)
            for name, value in judge_metrics.items():
                print(f"{name}: {value:.2f}")

    print("\nPer-case:")
    for sample, hits in zip(samples, results):
        top = hits[0] if hits else {}
        top_section = top.get("section", "REJECTED")
        top_score = top.get("score", 0.0)
        hit = compute_section_hit(sample, hits, k=args.top_k)
        keyword_recall = compute_keyword_recall(sample, hits, k=args.top_k)
        print(
            f"- {sample['id']}: top_section={top_section}, top_score={top_score:.2f}, "
            f"section_hit={hit}, keyword_recall={keyword_recall:.2f}"
        )
        if args.answer_eval:
            answer = answers[samples.index(sample)]
            print(f"  answer: {safe_console_text(answer[:300].replace(chr(10), ' '))}")
            if args.llm_judge:
                judgment = judgments[samples.index(sample)]
                print(
                    safe_console_text("  judge: "
                    + ", ".join(f"{field}={judgment[field]:.2f}" for field in JUDGE_FIELDS)
                    + f", notes={judgment.get('notes', '')[:160]}")
                )


if __name__ == "__main__":
    main()

from dataclasses import dataclass

from app.rag.hits import RetrievalHit


def infer_paper_intent(query: str) -> str:
    q = (query or "").lower()
    if any(x in q for x in ["方法", "模型", "架构", "method", "approach", "architecture"]):
        return "method"
    if any(x in q for x in ["实验", "结果", "指标", "数据集", "experiment", "result", "dataset", "metric"]):
        return "experiments"
    if any(x in q for x in ["相关工作", "已有工作", "related work", "prior work"]):
        return "related_work"
    if any(x in q for x in ["局限", "不足", "limitation", "future work"]):
        return "limitations"
    if any(x in q for x in ["总结", "贡献", "summary", "contribution"]):
        return "summary"
    return "general"


def sections_for_intent(intent: str) -> list[str]:
    mapping = {
        "method": ["Method", "Background"],
        "experiments": ["Experiments", "Results", "Discussion"],
        "related_work": ["Related Work", "Introduction"],
        "limitations": ["Limitations", "Discussion", "Conclusion"],
        "summary": ["Abstract", "Introduction", "Conclusion"],
        "general": [],
    }
    return mapping.get(intent, [])


@dataclass
class EvidenceDecision:
    status: str
    can_answer: bool
    reason: str
    top_score: float


def judge_evidence_sufficiency(
    hits: list[RetrievalHit],
    threshold: float = 0.5,
) -> EvidenceDecision:
    if not hits:
        return EvidenceDecision(
            status="empty",
            can_answer=False,
            reason="没有检索到本地论文证据",
            top_score=0.0,
        )

    top_score = max(hit.score for hit in hits)
    if top_score < threshold:
        return EvidenceDecision(
            status="insufficient",
            can_answer=False,
            reason=f"最高证据分数 {top_score:.2f} 低于阈值 {threshold:.2f}",
            top_score=top_score,
        )

    return EvidenceDecision(
        status="sufficient",
        can_answer=True,
        reason=f"最高证据分数 {top_score:.2f} 达到阈值 {threshold:.2f}",
        top_score=top_score,
    )

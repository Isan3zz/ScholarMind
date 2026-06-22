from dataclasses import dataclass

from app.rag.hits import RetrievalHit


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

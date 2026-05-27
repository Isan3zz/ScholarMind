from copy import deepcopy

from app.rag.hits import RetrievalHit


def rrf_fusion(
    result_lists: list[list[RetrievalHit]],
    rank_constant: int = 60,
    top_k: int = 20,
) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    best_hits: dict[str, RetrievalHit] = {}

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            identity = hit.identity
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rank_constant + rank)
            if identity not in best_hits or hit.score > best_hits[identity].score:
                best_hits[identity] = hit

    ranked_ids = sorted(scores, key=lambda item: (-scores[item], item))
    fused: list[RetrievalHit] = []
    for identity in ranked_ids[:top_k]:
        hit = deepcopy(best_hits[identity])
        hit.score = scores[identity]
        hit.retriever = "hybrid_rrf"
        fused.append(hit)
    return fused


def apply_section_boost(
    hits: list[RetrievalHit],
    preferred_sections: list[str],
    boost: float = 0.02,
) -> list[RetrievalHit]:
    preferred = set(preferred_sections or [])
    boosted: list[RetrievalHit] = []
    for hit in hits:
        copied = deepcopy(hit)
        if copied.section in preferred:
            copied.score += boost
            copied.metadata = dict(copied.metadata)
            copied.metadata["section_boosted"] = True
        boosted.append(copied)
    return sorted(boosted, key=lambda item: (-item.score, item.identity))

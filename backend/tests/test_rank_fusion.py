import unittest

from app.rag.hits import RetrievalHit
from app.rag.rank_fusion import apply_section_boost, rrf_fusion


def hit(chunk_id, retriever, score=1.0, section="Method"):
    return RetrievalHit(
        chunk_id=chunk_id,
        content=f"text {chunk_id}",
        source="paper.pdf",
        page=1,
        section=section,
        score=score,
        retriever=retriever,
    )


class RankFusionTest(unittest.TestCase):
    def test_rrf_fusion_promotes_hits_seen_by_multiple_retrievers(self):
        bm25 = [hit("a", "bm25"), hit("b", "bm25"), hit("c", "bm25")]
        dense = [hit("b", "dense"), hit("d", "dense"), hit("a", "dense")]

        fused = rrf_fusion([bm25, dense], rank_constant=60, top_k=3)

        self.assertEqual(fused[0].chunk_id, "b")
        self.assertEqual([h.chunk_id for h in fused], ["b", "a", "d"])
        self.assertEqual(fused[0].retriever, "hybrid_rrf")
        self.assertGreater(fused[0].score, 0)


class SectionBoostTest(unittest.TestCase):
    def test_apply_section_boost_promotes_preferred_sections(self):
        hits = [
            hit("intro", "hybrid_rrf", score=0.03, section="Introduction"),
            hit("method", "hybrid_rrf", score=0.02, section="Method"),
        ]

        boosted = apply_section_boost(hits, preferred_sections=["Method"], boost=0.02)

        self.assertEqual(boosted[0].chunk_id, "method")
        self.assertAlmostEqual(boosted[0].score, 0.04)


if __name__ == "__main__":
    unittest.main()

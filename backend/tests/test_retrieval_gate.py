import unittest

from app.rag.hits import RetrievalHit
from app.rag.retrieval import judge_evidence_sufficiency


def hit(score):
    return RetrievalHit(
        chunk_id="c1",
        content="text",
        source="paper.pdf",
        page=1,
        section="Method",
        score=score,
        retriever="rerank",
    )


class RetrievalGateTest(unittest.TestCase):
    def test_low_score_is_insufficient(self):
        decision = judge_evidence_sufficiency([hit(0.2)], threshold=0.5)

        self.assertEqual(decision.status, "insufficient")
        self.assertFalse(decision.can_answer)

    def test_high_score_is_sufficient(self):
        decision = judge_evidence_sufficiency([hit(0.8)], threshold=0.5)

        self.assertEqual(decision.status, "sufficient")
        self.assertTrue(decision.can_answer)


if __name__ == "__main__":
    unittest.main()

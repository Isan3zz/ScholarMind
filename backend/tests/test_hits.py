import unittest

from app.rag.hits import RetrievalHit


class RetrievalHitTest(unittest.TestCase):
    def test_retrieval_hit_uses_chunk_id_as_identity(self):
        hit = RetrievalHit(
            chunk_id="c1",
            content="method text",
            source="paper.pdf",
            page=3,
            section="Method",
            score=0.8,
            retriever="bm25",
        )

        self.assertEqual(hit.identity, "c1")
        self.assertEqual(hit.to_metadata()["source"], "paper.pdf")
        self.assertEqual(hit.to_metadata()["section"], "Method")

    def test_retrieval_hit_uses_parent_text_as_context_when_available(self):
        hit = RetrievalHit(
            chunk_id="c1",
            content="child paragraph",
            source="paper.pdf",
            page=3,
            section="Method",
            score=0.8,
            retriever="dense",
            metadata={"parent_text": "full subsection context"},
        )

        self.assertEqual(hit.context_text, "full subsection context")


if __name__ == "__main__":
    unittest.main()

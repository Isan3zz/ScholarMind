import unittest
from unittest.mock import patch

from app.rag.engine import (
    _child_chunk_filter,
    _es_hit_to_retrieval_hit,
    _select_diverse_parent_contexts,
    hybrid_search,
    rerank_hits,
)
from app.rag.hits import RetrievalHit


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


class HybridSearchPipelineTest(unittest.TestCase):
    def test_hybrid_search_fuses_bm25_and_dense_results(self):
        with (
            patch("app.rag.engine.bm25_search", return_value=[hit("a", "bm25"), hit("b", "bm25")]),
            patch("app.rag.engine.dense_search", return_value=[hit("b", "dense"), hit("c", "dense")]),
            patch("app.rag.engine.rerank_hits", side_effect=lambda query, hits, top_k=5: hits[:top_k]),
            patch("app.rag.engine._enrich_hits_with_parents", side_effect=lambda hits: hits),
        ):
            hits = hybrid_search("论文方法是什么", top_k=3)

        self.assertEqual(hits[0].chunk_id, "b")
        self.assertEqual(hits[0].retriever, "hybrid_rrf")
        self.assertEqual(len(hits), 3)

    def test_hybrid_search_applies_section_boost_after_rerank(self):
        reranked = [
            hit("intro", "rerank", score=0.90, section="Introduction"),
            hit("method", "rerank", score=0.89, section="Method"),
        ]

        with (
            patch("app.rag.engine.bm25_search", return_value=[hit("intro", "bm25", section="Introduction")]),
            patch("app.rag.engine.dense_search", return_value=[hit("method", "dense", section="Method")]),
            patch("app.rag.engine.infer_paper_intent", return_value="method"),
            patch("app.rag.engine.sections_for_intent", return_value=["Method"]),
            patch("app.rag.engine.rerank_hits", return_value=reranked),
            patch("app.rag.engine._enrich_hits_with_parents", side_effect=lambda hits: hits),
        ):
            hits = hybrid_search("what is the method?", top_k=2)

        self.assertEqual([item.chunk_id for item in hits], ["method", "intro"])
        self.assertTrue(hits[0].metadata["section_boosted"])


class ChildChunkFilterTest(unittest.TestCase):
    def test_child_chunk_filter_excludes_parent_and_metadata_chunks(self):
        self.assertEqual(
            _child_chunk_filter(),
            {
                "bool": {
                    "must_not": [
                        {"terms": {"metadata.chunk_type.keyword": ["parent", "paper_metadata"]}}
                    ]
                }
            },
        )


class ElasticsearchHitConvertTest(unittest.TestCase):
    def test_es_hit_to_retrieval_hit_preserves_metadata(self):
        raw = {
            "_score": 3.2,
            "_source": {
                "text": "method text",
                "metadata": {
                    "chunk_id": "c1",
                    "source": "paper.pdf",
                    "page": 4,
                    "section": "Method",
                },
            },
        }

        hit = _es_hit_to_retrieval_hit(raw, retriever="bm25")

        self.assertEqual(hit.chunk_id, "c1")
        self.assertEqual(hit.score, 3.2)
        self.assertEqual(hit.section, "Method")


class RerankHitsTest(unittest.TestCase):
    def test_rerank_hits_uses_existing_reranker(self):
        class FakeReranker:
            def compress_documents(self, docs, query):
                return [docs[1], docs[0]]

        hits = [
            hit("a", "hybrid_rrf", score=0.1),
            hit("b", "hybrid_rrf", score=0.2),
        ]

        with patch("app.rag.engine.get_reranker", return_value=FakeReranker()):
            reranked = rerank_hits("query", hits, top_k=2)

        self.assertEqual([item.chunk_id for item in reranked], ["b", "a"])
        self.assertEqual(reranked[0].retriever, "rerank")

    def test_rerank_hits_scores_child_chunks_before_parent_context_fill(self):
        captured_texts = []

        class FakeReranker:
            def compress_documents(self, docs, query):
                captured_texts.extend(doc.page_content for doc in docs)
                return docs

        hits = [
            RetrievalHit(
                chunk_id="c1",
                content="child paragraph",
                source="paper.pdf",
                page=1,
                section="Method",
                score=0.3,
                retriever="hybrid_rrf",
                metadata={"parent_text": "full subsection parent context"},
            )
        ]

        with patch("app.rag.engine.get_reranker", return_value=FakeReranker()):
            reranked = rerank_hits("query", hits, top_k=1)

        self.assertEqual(captured_texts, ["child paragraph"])
        self.assertEqual(reranked[0].context_text, "full subsection parent context")


class ParentContextSelectionTest(unittest.TestCase):
    def test_select_diverse_parent_contexts_keeps_one_hit_per_parent(self):
        hits = [
            RetrievalHit(
                chunk_id="c1",
                content="loss function child",
                source="paper.pdf",
                page=3,
                section="Method",
                score=0.9,
                retriever="rerank",
                metadata={"parent_chunk_id": "p1", "parent_text": "shared subsection parent"},
            ),
            RetrievalHit(
                chunk_id="c2",
                content="training detail child",
                source="paper.pdf",
                page=3,
                section="Method",
                score=0.8,
                retriever="rerank",
                metadata={"parent_chunk_id": "p1", "parent_text": "shared subsection parent"},
            ),
            RetrievalHit(
                chunk_id="c3",
                content="experiment child",
                source="paper.pdf",
                page=5,
                section="Experiments",
                score=0.7,
                retriever="rerank",
                metadata={"parent_chunk_id": "p2", "parent_text": "different experiment parent"},
            ),
        ]

        selected = _select_diverse_parent_contexts(hits, top_k=3)

        self.assertEqual([item.chunk_id for item in selected], ["c1", "c3"])

    def test_select_diverse_parent_contexts_skips_highly_overlapping_parent_text(self):
        hits = [
            RetrievalHit(
                chunk_id="c1",
                content="paragraph one",
                source="paper.pdf",
                page=1,
                section="Introduction",
                score=0.9,
                retriever="rerank",
                metadata={
                    "parent_chunk_id": "p1",
                    "parent_text": "alpha beta gamma delta epsilon",
                },
            ),
            RetrievalHit(
                chunk_id="c2",
                content="paragraph two",
                source="paper.pdf",
                page=1,
                section="Introduction",
                score=0.8,
                retriever="rerank",
                metadata={
                    "parent_chunk_id": "p2",
                    "parent_text": "alpha beta gamma delta epsilon zeta",
                },
            ),
            RetrievalHit(
                chunk_id="c3",
                content="paragraph three",
                source="paper.pdf",
                page=2,
                section="Introduction",
                score=0.7,
                retriever="rerank",
                metadata={
                    "parent_chunk_id": "p3",
                    "parent_text": "theta iota kappa lambda",
                },
            ),
        ]

        selected = _select_diverse_parent_contexts(hits, top_k=3, overlap_threshold=0.75)

        self.assertEqual([item.chunk_id for item in selected], ["c1", "c3"])


if __name__ == "__main__":
    unittest.main()

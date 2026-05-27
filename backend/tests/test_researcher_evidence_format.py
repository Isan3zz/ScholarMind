import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.graph.nodes.researcher import _format_evidence_docs, _retrieve_local_docs, research_node
from app.rag.hits import RetrievalHit


class ResearcherEvidenceFormatTest(unittest.TestCase):
    def test_format_evidence_docs_includes_source_section_and_score_without_page(self):
        docs = [
            Document(
                page_content="The method uses section-aware retrieval.",
                metadata={
                    "source": "iris.pdf",
                    "page": 3,
                    "section": "Method",
                    "relevance_score": 0.91,
                },
            )
        ]

        text = _format_evidence_docs(docs)

        self.assertIn("iris.pdf", text)
        self.assertNotIn("p. 3", text)
        self.assertIn("Method", text)
        self.assertIn("0.91", text)

    def test_format_evidence_docs_uses_parent_text_when_available(self):
        docs = [
            Document(
                page_content="child paragraph",
                metadata={
                    "source": "iris.pdf",
                    "page": 3,
                    "section": "Method",
                    "parent_text": "full subsection context",
                },
            )
        ]

        text = _format_evidence_docs(docs)

        self.assertIn("full subsection context", text)
        self.assertNotIn("child paragraph", text)

    def test_retrieve_local_docs_uses_hybrid_search_hits(self):
        hit = RetrievalHit(
            chunk_id="c1",
            content="child paragraph",
            source="iris.pdf",
            page=3,
            section="Method",
            score=0.87,
            retriever="rerank",
            metadata={"parent_text": "full context"},
        )

        with patch("app.graph.nodes.researcher.hybrid_search", return_value=[hit]):
            docs = _retrieve_local_docs("query")

        self.assertEqual(docs[0].page_content, "full context")
        self.assertEqual(docs[0].metadata["source"], "iris.pdf")
        self.assertEqual(docs[0].metadata["relevance_score"], 0.87)

    def test_research_node_uses_planner_queries_for_local_retrieval(self):
        doc = Document(
            page_content="method evidence",
            metadata={"source": "iris.pdf", "page": 1, "section": "Method"},
        )
        state = {
            "query": "How does the paper solve retrieval coverage?",
            "plan": [
                "method retrieval coverage",
                "experiment recall comparison",
                "limitations missing evidence",
            ],
            "search_mode": "document",
        }
        grader_response = Mock()
        grader_response.content = "YES"
        grader = Mock()
        grader.invoke.return_value = grader_response

        with (
            patch("app.graph.nodes.researcher._retrieve_local_docs", return_value=[doc]) as retrieve,
            patch("app.graph.nodes.researcher.llm", grader),
            patch("builtins.print"),
        ):
            research_node(state)

        self.assertEqual(
            [call.args[0] for call in retrieve.call_args_list],
            [
                "How does the paper solve retrieval coverage?",
                "method retrieval coverage",
                "experiment recall comparison",
                "limitations missing evidence",
            ],
        )


if __name__ == "__main__":
    unittest.main()

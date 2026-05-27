import unittest

from app.rag.engine import _build_paper_documents_from_pages


def _children(chunks):
    return [c for c in chunks if c.metadata.get("chunk_type") not in ("parent", "paper_metadata")]


def _parents(chunks):
    return [c for c in chunks if c.metadata.get("chunk_type") == "parent"]


class PaperIngestionPipelineTest(unittest.TestCase):
    def test_build_paper_documents_from_pages_skips_references_and_keeps_sections(self):
        pages = [
            ("Abstract\nThis paper proposes IRIS.\n\nReferences\n[1] Noise", 1),
        ]

        docs = _build_paper_documents_from_pages(pages, source="paper.pdf")

        self.assertEqual(docs[0].metadata["chunk_type"], "paper_metadata")
        self.assertEqual(docs[0].metadata["section"], "Metadata")
        children = _children(docs)
        self.assertEqual(len(children), 1)
        self.assertEqual(len(_parents(docs)), 1)
        self.assertEqual(children[0].metadata["section"], "Abstract")
        self.assertNotIn("[1] Noise", children[0].page_content)

    def test_build_paper_documents_from_pages_removes_repeated_headers_and_footers(self):
        pages = [
            (
                "IRIS: A Paper Assistant\n"
                "Abstract\nThis paper proposes IRIS.\n"
                "Proceedings of Test Conference\n1",
                1,
            ),
            (
                "IRIS: A Paper Assistant\n"
                "Introduction\nIRIS cleans noisy papers.\n"
                "Proceedings of Test Conference\n2",
                2,
            ),
        ]

        docs = _build_paper_documents_from_pages(pages, source="paper.pdf")
        body = "\n".join(doc.page_content for doc in docs)

        self.assertNotIn("IRIS: A Paper Assistant", body)
        self.assertNotIn("Proceedings of Test Conference", body)
        self.assertIn("This paper proposes IRIS.", body)
        self.assertIn("IRIS cleans noisy papers.", body)


if __name__ == "__main__":
    unittest.main()

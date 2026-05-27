import unittest

from app.rag.paper_cleaner import clean_paper_pages, clean_paper_text


class PaperCleanerTest(unittest.TestCase):
    def test_clean_paper_text_repairs_wrapped_words_and_lines(self):
        raw = "The trans-\nformer model is strong.\nIt uses attention.\n\nReferences\n[1] Some paper"

        cleaned = clean_paper_text(raw)

        self.assertIn("transformer model is strong. It uses attention.", cleaned.body_text)
        self.assertNotIn("trans-\nformer", cleaned.body_text)
        self.assertEqual(cleaned.references_text.strip(), "[1] Some paper")

    def test_clean_paper_pages_removes_repeated_headers_and_footers(self):
        pages = [
            (
                "IRIS: A Paper Assistant\n"
                "Abstract\nThis paper proposes a RAG system.\n"
                "Proceedings of Test Conference\n1",
                1,
            ),
            (
                "IRIS: A Paper Assistant\n"
                "Introduction\nThe system cleans PDF text.\n"
                "Proceedings of Test Conference\n2",
                2,
            ),
        ]

        cleaned_pages = clean_paper_pages(pages)
        combined = "\n".join(page.body_text for page, _ in cleaned_pages)

        self.assertNotIn("IRIS: A Paper Assistant", combined)
        self.assertNotIn("Proceedings of Test Conference", combined)
        self.assertNotIn("\n1", combined)
        self.assertNotIn("\n2", combined)
        self.assertIn("This paper proposes a RAG system.", combined)
        self.assertIn("The system cleans PDF text.", combined)


if __name__ == "__main__":
    unittest.main()

import unittest

from app.rag.paper_sections import assign_sections


class PaperSectionsTest(unittest.TestCase):
    def test_assign_sections_to_paragraphs(self):
        text = """Abstract
This paper studies retrieval.

1 Introduction
RAG systems need grounding.

3 Method
We propose a section-aware retriever.

References
[1] Other work"""

        units = assign_sections(text)

        self.assertEqual(units[0].section, "Abstract")
        self.assertEqual(units[1].section, "Introduction")
        self.assertEqual(units[2].section, "Method")
        self.assertEqual(units[3].section, "References")
        self.assertEqual(units[2].text, "We propose a section-aware retriever.")


if __name__ == "__main__":
    unittest.main()

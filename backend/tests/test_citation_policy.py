import unittest

from app.graph.nodes.reviewer import has_source_citation


class CitationPolicyTest(unittest.TestCase):
    def test_has_source_citation_detects_source_section_format(self):
        self.assertTrue(has_source_citation("The method uses RAG [source: iris.pdf, section: Method]"))
        self.assertFalse(has_source_citation("The method uses RAG [source: iris.pdf, p. 3]"))
        self.assertFalse(has_source_citation("The method uses RAG."))


if __name__ == "__main__":
    unittest.main()

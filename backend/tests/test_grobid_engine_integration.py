import unittest
from unittest.mock import patch

from app.rag.engine import _build_paper_documents_from_grobid


def _children(chunks):
    return [c for c in chunks if c.metadata.get("chunk_type") not in ("parent", "paper_metadata")]


def _parents(chunks):
    return [c for c in chunks if c.metadata.get("chunk_type") == "parent"]


TEI_SAMPLE = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>IRIS Paper</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename>Jane</forename><surname>Doe</surname></persName></author>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract><p>This is the abstract.</p></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1 Introduction</head>
        <p>Intro paragraph.</p>
      </div>
    </body>
  </text>
</TEI>"""


class GrobidEngineIntegrationTest(unittest.TestCase):
    def test_build_paper_documents_from_grobid_uses_tei_title_and_sections(self):
        with patch("app.rag.engine.GrobidClient") as client_cls:
            client_cls.return_value.process_fulltext_document.return_value = TEI_SAMPLE

            docs = _build_paper_documents_from_grobid("paper.pdf", source="paper.pdf")

        self.assertEqual(docs[0].metadata["paper_title"], "IRIS Paper")
        self.assertEqual(docs[0].metadata["authors"], ["Jane Doe"])
        self.assertEqual(docs[0].metadata["section"], "Metadata")
        children = _children(docs)
        self.assertEqual(len(children), 2)
        self.assertEqual(len(_parents(docs)), 2)
        self.assertEqual(children[0].metadata["section"], "Abstract")
        self.assertEqual(children[1].metadata["section"], "Introduction")


if __name__ == "__main__":
    unittest.main()

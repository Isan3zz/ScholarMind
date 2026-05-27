import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.rag.grobid_client import GrobidClient


class GrobidClientTest(unittest.TestCase):
    def test_process_fulltext_document_posts_pdf_to_grobid(self):
        response = Mock()
        response.text = "<TEI>ok</TEI>"
        response.raise_for_status = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            with patch("app.rag.grobid_client.httpx.post", return_value=response) as post:
                xml = GrobidClient(base_url="http://localhost:8070").process_fulltext_document(str(pdf_path))

        self.assertEqual(xml, "<TEI>ok</TEI>")
        self.assertEqual(post.call_args.args[0], "http://localhost:8070/api/processFulltextDocument")
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()

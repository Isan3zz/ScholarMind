import tempfile
import unittest
from pathlib import Path

from scripts.export_es_chunks import format_chunk, write_chunks


class ExportEsChunksTest(unittest.TestCase):
    def test_format_chunk_includes_metadata_and_text_without_vector(self):
        chunk = {
            "_id": "abc",
            "text": "Chunk body",
            "vector": [1, 2, 3],
            "metadata": {
                "chunk_id": "c1",
                "section": "Introduction",
            },
        }

        formatted = format_chunk(chunk, 1)

        self.assertIn("CHUNK 1", formatted)
        self.assertIn("ES_ID: abc", formatted)
        self.assertIn("SECTION: Introduction", formatted)
        self.assertIn("SUBSECTION:", formatted)
        self.assertIn("CHUNK_TYPE:", formatted)
        self.assertIn('"section": "Introduction"', formatted)
        self.assertIn("TEXT:\nChunk body", formatted)
        self.assertNotIn("[1, 2, 3]", formatted)

    def test_write_chunks_creates_text_file(self):
        chunks = [
            {
                "_id": "abc",
                "text": "Chunk body",
                "metadata": {"chunk_id": "c1", "section": "Introduction"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "chunks.txt"
            count = write_chunks(chunks, output)

            self.assertEqual(count, 1)
            self.assertTrue(output.exists())
            self.assertIn("Chunk body", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

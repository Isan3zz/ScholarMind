import asyncio
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

from app.api import routes


class UploadBehaviorTest(unittest.TestCase):
    def test_upload_resets_knowledge_base_before_indexing_files(self):
        calls = []

        def fake_reset():
            calls.append("reset")

        def fake_process(paths):
            calls.append("process")
            self.assertEqual(len(paths), 1)
            self.assertTrue(Path(paths[0]).exists())
            return 1

        with tempfile.TemporaryDirectory() as tmpdir:
            file = UploadFile(filename="paper.pdf", file=BytesIO(b"%PDF-1.4"))

            with (
                patch.object(routes, "UPLOAD_DIR", tmpdir),
                patch.object(routes, "reset_knowledge_base", side_effect=fake_reset),
                patch.object(routes, "process_documents", side_effect=fake_process),
            ):
                result = asyncio.run(routes.upload_files([file]))

        self.assertEqual(calls, ["reset", "process"])
        self.assertEqual(result["chunks_stored"], 1)


if __name__ == "__main__":
    unittest.main()

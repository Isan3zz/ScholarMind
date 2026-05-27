import os
from pathlib import Path

import httpx


class GrobidClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("GROBID_URL", "http://localhost:8070")).rstrip("/")
        self.timeout = timeout

    def process_fulltext_document(self, pdf_path: str) -> str:
        path = Path(pdf_path)
        with path.open("rb") as pdf_file:
            response = httpx.post(
                f"{self.base_url}/api/processFulltextDocument",
                files={"input": (path.name, pdf_file, "application/pdf")},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.text

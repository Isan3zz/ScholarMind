"""
Marker PDF test script.

Usage:
    cd backend
    ..\.venv\Scripts\python.exe scripts/test_marker.py path/to/paper.pdf
    ..\.venv\Scripts\python.exe scripts/test_marker.py path/to/paper.pdf --pages 3

Output:
    1. Timing
    2. Markdown (first 2000 chars)
    3. JSON block overview (type / text / section_hierarchy)
    4. PaperTextUnit mapping (first 20)
    5. Full results saved to scripts/output/
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("br", "p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "div"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "div", "table"):
            self._parts.append("\n")

    def get_text(self) -> str:
        import re
        text = "".join(self._parts).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    ex = _TextExtractor()
    ex.feed(html)
    return ex.get_text()


# ---------------------------------------------------------------------------
# Recursive traversal of Marker JSON
# ---------------------------------------------------------------------------

SKIP_TYPES = {"Page", "Document", "PageHeader", "PageFooter", "Figure", "Picture", "Form", "Handwriting"}


def collect_blocks(blocks: list[dict], depth: int = 0) -> list[dict]:
    result = []
    for b in blocks:
        bt = b.get("block_type", "?")
        if bt in SKIP_TYPES:
            children = b.get("children") or []
            if children:
                result.extend(collect_blocks(children, depth))
            continue

        html = b.get("html", "")
        text = html_to_text(html)[:200]
        sh = b.get("section_hierarchy", {})
        result.append({
            "depth": depth,
            "type": bt,
            "text": text,
            "section_hierarchy": sh,
            "html_len": len(html),
        })

        children = b.get("children") or []
        if children:
            result.extend(collect_blocks(children, depth + 1))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    max_pages = None
    args = sys.argv[1:]
    pdf_path = None
    i = 0
    while i < len(args):
        if args[i] == "--pages" and i + 1 < len(args):
            max_pages = int(args[i + 1])
            i += 2
        elif not pdf_path:
            pdf_path = args[i]
            i += 1
        else:
            i += 1

    if not pdf_path:
        candidates = ["backend/uploads", "test.pdf"]
        pdf_path = None
        for c in candidates:
            p = Path(c)
            if p.is_dir():
                pdfs = list(p.glob("*.pdf"))
                if pdfs:
                    pdf_path = str(pdfs[0])
                    print(f"Auto select: {pdf_path}")
                    break
            elif p.is_file():
                pdf_path = str(p)
                print(f"Auto select: {pdf_path}")
                break
        if not pdf_path:
            print("Usage: python scripts/test_marker.py <pdf_path>")
            sys.exit(1)

    pdf_path = str(Path(pdf_path).resolve())
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"File: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    print(f"{'='*60}\n")

    # --- Step 1: Load Marker ---
    print("Loading Marker models...")
    t0 = time.time()

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered

    config_dict = {"output_format": "json"}
    if max_pages:
        config_dict["page_range"] = f"0-{max_pages - 1}"
        print(f"Pages: first {max_pages} only")
    config = ConfigParser(config_dict)
    converter = PdfConverter(
        config=config.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config.get_processors(),
        renderer=config.get_renderer(),
    )
    t1 = time.time()
    print(f"Models loaded ({t1 - t0:.1f}s)\n")

    # --- Step 2: Parse ---
    print("Parsing PDF...")
    t2 = time.time()
    rendered = converter(pdf_path)
    t3 = time.time()
    print(f"Done ({t3 - t2:.1f}s, total {t3 - t0:.1f}s)\n")

    # --- Step 3: Markdown ---
    print(f"{'='*60}")
    print("MARKDOWN (first 2000 chars)")
    print(f"{'='*60}")
    md_text, _, _ = text_from_rendered(rendered)
    print(md_text[:2000])
    if len(md_text) > 2000:
        print(f"\n... ({len(md_text)} chars total, truncated)")

    # --- Step 4: JSON blocks ---
    data = rendered.model_dump() if hasattr(rendered, "model_dump") else rendered

    if isinstance(data, dict) and data.get("block_type") == "Document":
        pages = data.get("children", [])
    elif isinstance(data, list):
        pages = data
    else:
        pages = [data]

    print(f"\n{'='*60}")
    print(f"JSON BLOCKS - {len(pages)} pages")
    print(f"{'='*60}")

    all_blocks = []
    for i, page in enumerate(pages):
        page_children = page.get("children", [])
        blocks = collect_blocks(page_children)
        all_blocks.extend(blocks)
        print(f"\n--- Page {i+1} ({len(blocks)} blocks) ---")
        for b in blocks:
            sh_str = ",".join(f"h{k}" for k in sorted(b["section_hierarchy"].keys())) if b["section_hierarchy"] else "-"
            print(f"  [{b['type']:20s}] sh={sh_str:8s} | {b['text'][:120]}")

    # --- Step 5: PaperTextUnit ---
    print(f"\n{'='*60}")
    print(f"PaperTextUnit mapping (first 20)")
    print(f"{'='*60}")

    from app.rag.marker_parser import parse_pdf_with_marker

    t4 = time.time()
    units, title, authors = parse_pdf_with_marker(pdf_path)
    t5 = time.time()
    print(f"Done ({t5 - t4:.1f}s)\n")
    print(f"Title: {title}")
    print(f"Authors: {authors or '(not extracted)'}")
    print(f"Total units: {len(units)}\n")

    for i, u in enumerate(units[:20]):
        text_preview = u.text[:100].replace("\n", "\\n")
        print(f"  [{i:3d}] sec={u.section[:30]:30s} sub={u.subsection[:20]:20s} type={u.chunk_type:10s} | {text_preview}")

    if len(units) > 20:
        print(f"\n  ... ({len(units)} total, showing first 20)")

    # --- Step 6: Save ---
    out_dir = Path("scripts/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(pdf_path).stem
    (out_dir / f"{stem}.md").write_text(md_text, encoding="utf-8")
    (out_dir / f"{stem}_blocks.json").write_text(
        json.dumps(all_blocks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{stem}_units.json").write_text(
        json.dumps([{
            "text": u.text,
            "section": u.section,
            "subsection": u.subsection,
            "chunk_type": u.chunk_type,
        } for u in units], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"Results saved to scripts/output/")
    print(f"   {stem}.md           - Markdown")
    print(f"   {stem}_blocks.json  - JSON blocks")
    print(f"   {stem}_units.json   - PaperTextUnit list")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

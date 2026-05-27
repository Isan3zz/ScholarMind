import re
from dataclasses import dataclass


@dataclass
class CleanedPaperText:
    body_text: str
    references_text: str


def _non_empty_line_indices(lines: list[str]) -> list[int]:
    return [index for index, line in enumerate(lines) if line.strip()]


def _normalize_edge_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).lower()


def _is_page_number(line: str) -> bool:
    return bool(re.match(r"^(?:page\s*)?\d+(?:\s*/\s*\d+|\s+of\s+\d+)?$", line.strip(), re.I))


def _edge_line_indices(lines: list[str], edge_size: int = 2) -> set[int]:
    indices = _non_empty_line_indices(lines)
    return set(indices[:edge_size] + indices[-edge_size:])


def _find_repeated_edge_lines(raw_pages: list[str]) -> set[str]:
    counts: dict[str, int] = {}
    for raw_page in raw_pages:
        lines = raw_page.splitlines()
        seen_on_page = set()
        for index in _edge_line_indices(lines):
            normalized = _normalize_edge_line(lines[index])
            if normalized and not _is_page_number(lines[index]):
                seen_on_page.add(normalized)
        for normalized in seen_on_page:
            counts[normalized] = counts.get(normalized, 0) + 1
    return {line for line, count in counts.items() if count >= 2}


def _remove_repeated_headers_and_footers(text: str, repeated_edge_lines: set[str]) -> str:
    lines = text.splitlines()
    edge_indices = _edge_line_indices(lines)
    kept = []
    for index, line in enumerate(lines):
        normalized = _normalize_edge_line(line)
        is_edge_noise = index in edge_indices and (
            normalized in repeated_edge_lines or _is_page_number(line)
        )
        if not is_edge_noise:
            kept.append(line)
    return "\n".join(kept)


def _split_references(text: str) -> tuple[str, str]:
    match = re.search(r"(?im)^references\s*$", text)
    if not match:
        return text, ""
    return text[:match.start()], text[match.end():]


def _repair_wrapped_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _repair_line_breaks(text: str) -> str:
    section_break = "__SCHOLARMIND_SECTION_BREAK__"
    text = re.sub(
        r"(?im)^((?:\d+(?:\.\d+)*\.?\s+)?(?:abstract|introduction|related work|background|preliminaries|method|methods|approach|framework|model|experiments?|evaluation|results|discussion|conclusion|limitations?|ethics statement|acknowledgements|acknowledgments|references|appendix))\s*\n",
        rf"\1{section_break}",
        text,
    )
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = text.replace(section_break, "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_paper_text(raw_text: str) -> CleanedPaperText:
    body, references = _split_references(raw_text or "")
    body = _repair_wrapped_words(body)
    body = _repair_line_breaks(body)
    references = _repair_wrapped_words(references)
    references = _repair_line_breaks(references)
    return CleanedPaperText(body_text=body, references_text=references)


def clean_paper_pages(pages: list[tuple[str, int]]) -> list[tuple[CleanedPaperText, int]]:
    raw_pages = [raw_text or "" for raw_text, _ in pages]
    repeated_edge_lines = _find_repeated_edge_lines(raw_pages)

    cleaned_pages = []
    for raw_text, page in pages:
        text = _remove_repeated_headers_and_footers(raw_text or "", repeated_edge_lines)
        cleaned_pages.append((clean_paper_text(text), page))
    return cleaned_pages

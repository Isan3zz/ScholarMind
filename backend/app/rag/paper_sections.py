import re
from dataclasses import dataclass


SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "preliminaries": "Background",
    "method": "Method",
    "methods": "Method",
    "approach": "Method",
    "framework": "Method",
    "model": "Method",
    "experiments": "Experiments",
    "experiment": "Experiments",
    "evaluation": "Experiments",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "limitations": "Limitations",
    "limitation": "Limitations",
    "ethics statement": "Ethics Statement",
    "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
    "references": "References",
    "appendix": "Appendix",
}


@dataclass
class PaperTextUnit:
    text: str
    section: str
    subsection: str = ""
    chunk_type: str = "body"


def normalize_section_title(line: str) -> str:
    cleaned = re.sub(r"^\d+(\.\d+)*\.?\s+", "", line.strip()).strip()
    key = cleaned.lower()
    return SECTION_ALIASES.get(key, "")


def is_section_heading(line: str) -> str:
    line = line.strip()
    if not line or len(line) > 90:
        return ""
    normalized = normalize_section_title(line)
    if normalized:
        return normalized
    return ""


def assign_sections(text: str) -> list[PaperTextUnit]:
    units: list[PaperTextUnit] = []
    current_section = "Unknown"
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        paragraph = " ".join(buffer).strip()
        if paragraph:
            chunk_type = "reference" if current_section == "References" else "body"
            units.append(PaperTextUnit(text=paragraph, section=current_section, chunk_type=chunk_type))
        buffer.clear()

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush_buffer()
            continue

        heading = is_section_heading(line)
        if heading:
            flush_buffer()
            current_section = heading
            continue

        buffer.append(line)

    flush_buffer()

    return units

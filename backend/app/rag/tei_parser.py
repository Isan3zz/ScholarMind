from dataclasses import dataclass
import re
from xml.etree import ElementTree as ET

from app.rag.paper_sections import PaperTextUnit, normalize_section_title


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class ParsedPaper:
    title: str
    authors: list[str]
    units: list[PaperTextUnit]


def _text_content(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(text.strip() for text in element.itertext() if text and text.strip())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _remove_inline_figure_caption_noise(text: str) -> str:
    text = re.sub(
        r"\s*The words in red are GCG suffixes\..*?harmful query\.\s*",
        " ",
        text or "",
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?is)^\s*(?:fig(?:ure)?\.?\s*\d+|figure\s*\d+)[:.\s].*$",
        "",
        text.strip(),
    )
    return re.sub(r"\s+", " ", text).strip()


def _clean_heading(raw: str) -> str:
    normalized = normalize_section_title(raw)
    if normalized:
        return normalized
    return raw.strip()


def _head_number_level(number: str) -> int:
    cleaned = (number or "").strip()
    if not cleaned:
        return 0
    parts = [part for part in cleaned.split(".") if part]
    if not parts:
        return 0
    if not (parts[0].isdigit() or re.fullmatch(r"[A-Z]", parts[0], flags=re.IGNORECASE)):
        return 0
    if not all(part.isdigit() for part in parts[1:]):
        return 0
    return len(parts)


def _heading_number(raw_heading: str) -> str:
    match = re.match(r"^\s*([A-Z]|\d+)(?:\.\d+)*\.?(?=\s+)", raw_heading or "", flags=re.IGNORECASE)
    return match.group(0).rstrip(".") if match else ""


def _extract_authors(root: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in root.findall(".//tei:sourceDesc//tei:analytic/tei:author", TEI_NS):
        forenames = [
            _text_content(item)
            for item in author.findall(".//tei:forename", TEI_NS)
            if _text_content(item)
        ]
        surname = _text_content(author.find(".//tei:surname", TEI_NS))
        parts = [*forenames, surname]
        name = " ".join(part for part in parts if part).strip()
        if not name:
            name = _text_content(author)
        if name:
            authors.append(name)
    return authors


def _parse_div(
    div: ET.Element,
    units: list[PaperTextUnit],
    current_section: str = "Unknown",
    current_subsection: str = "",
    section_by_number: dict[str, str] | None = None,
) -> None:
    section_by_number = section_by_number if section_by_number is not None else {}
    head = div.find("tei:head", TEI_NS)
    heading = _text_content(head)
    if heading:
        cleaned_heading = _clean_heading(heading)
        head_number = (head.attrib.get("n", "") if head is not None else "") or _heading_number(heading)
        number_level = _head_number_level(head_number)
        if number_level == 1:
            current_section = cleaned_heading
            current_subsection = ""
            section_by_number[head_number.strip()] = current_section
        elif number_level > 1:
            parent_number = ".".join(head_number.strip().split(".")[:-1])
            parent_section = section_by_number.get(parent_number)
            if parent_section:
                current_section = parent_section
                current_subsection = cleaned_heading
            else:
                current_section = cleaned_heading
                current_subsection = ""
        elif current_section == "Unknown":
            current_section = cleaned_heading
            current_subsection = ""
        else:
            current_subsection = cleaned_heading

    pending_parts: list[str] = []

    def flush_pending() -> None:
        if not pending_parts:
            return
        units.append(
            PaperTextUnit(
                text="\n\n".join(pending_parts),
                section=current_section,
                subsection=current_subsection,
                chunk_type="body",
            )
        )
        pending_parts.clear()

    for child in div:
        tag = _local_name(child.tag)
        if tag == "head":
            continue
        if tag == "p":
            text = _remove_inline_figure_caption_noise(_text_content(child))
            if text:
                pending_parts.append(text)
            if len(pending_parts) == 1:
                flush_pending()
            continue
        if tag == "formula":
            formula = _text_content(child)
            if formula:
                formula_id = child.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "")
                label = f"Formula {formula_id}".strip()
                if not pending_parts and units:
                    previous = units.pop()
                    pending_parts.append(previous.text)
                pending_parts.append(f"[{label}] {formula}")
            continue
        if tag == "div":
            flush_pending()
            _parse_div(child, units, current_section, current_subsection, section_by_number)

    flush_pending()


def parse_tei_to_paper(tei_xml: str) -> ParsedPaper:
    root = ET.fromstring(tei_xml)
    title = _text_content(root.find(".//tei:titleStmt/tei:title", TEI_NS)) or "Untitled Paper"
    authors = _extract_authors(root)
    units: list[PaperTextUnit] = []

    for paragraph in root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS):
        text = _text_content(paragraph)
        if text:
            units.append(PaperTextUnit(text=text, section="Abstract", chunk_type="abstract"))

    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is not None:
        section_by_number: dict[str, str] = {}
        for div in body.findall("tei:div", TEI_NS):
            _parse_div(div, units, section_by_number=section_by_number)

    return ParsedPaper(title=title, authors=authors, units=units)

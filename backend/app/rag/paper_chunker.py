import hashlib
import re
from collections import defaultdict

from langchain_core.documents import Document

from app.rag.paper_sections import PaperTextUnit


def _stable_id(*parts: str) -> str:
    raw = "::".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _slug(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z一-鿿]+", "-", (text or "").strip().lower())
    return slug.strip("-") or "unknown"


def _build_parent_documents(
    children: list[Document],
    source: str,
) -> list[Document]:
    """按 parent_chunk_id 分组子块，为每组生成一个父块 Document。"""
    # 按 parent_chunk_id 分组
    groups: dict[str, list[int]] = defaultdict(list)
    for i, child in enumerate(children):
        pid = child.metadata["parent_chunk_id"]
        groups[pid].append(i)

    parents: list[Document] = []
    for pid, indices in groups.items():
        group = [children[i] for i in indices]
        rep = group[0].metadata

        if rep["subsection_id"]:
            # 有子节：同 subsection 所有子块拼接
            group.sort(key=lambda c: c.metadata["paragraph_index"])
            parent_text = "\n\n".join(c.page_content for c in group)
        else:
            # 无子节：当前块 + 前后各一个同节无子节的邻居
            child = group[0]
            gi = child.metadata["global_chunk_index"] - 1  # 转为 0-based
            section = rep["section"]

            neighbor_indices = [gi]
            # 左邻居
            left = gi - 1
            while left >= 0:
                c = children[left]
                if c.metadata["section"] == section and not c.metadata["subsection_id"]:
                    neighbor_indices.append(left)
                    break
                left -= 1
            # 右邻居
            right = gi + 1
            while right < len(children):
                c = children[right]
                if c.metadata["section"] == section and not c.metadata["subsection_id"]:
                    neighbor_indices.append(right)
                    break
                right += 1

            neighbor_indices.sort()
            parent_text = "\n\n".join(children[i].page_content for i in neighbor_indices)

        parents.append(
            Document(
                page_content=parent_text,
                metadata={
                    "source": source,
                    "paper_title": rep["paper_title"],
                    "authors": rep["authors"],
                    "page": rep["page"],
                    "section": rep["section"],
                    "section_id": rep["section_id"],
                    "subsection": rep["subsection"],
                    "subsection_id": rep["subsection_id"],
                    "chunk_type": "parent",
                    "chunk_id": pid,
                    "parent_chunk_id": pid,
                    "paragraph_index": rep["paragraph_index"],
                    "global_chunk_index": -1,
                },
            )
        )

    return parents


def build_paper_chunks(
    units: list[PaperTextUnit],
    source: str,
    paper_title: str,
    page: int = 0,
    authors: list[str] | None = None,
) -> list[Document]:
    authors = authors or []

    # 元数据块
    metadata_chunk_id = _stable_id(source, "paper_metadata", paper_title, ",".join(authors))
    metadata_chunk = Document(
        page_content=f"Paper Title: {paper_title}\nAuthors: {', '.join(authors) if authors else 'Unknown'}",
        metadata={
            "source": source,
            "paper_title": paper_title,
            "authors": authors,
            "page": page,
            "section": "Metadata",
            "section_id": "metadata",
            "subsection": "",
            "subsection_id": "",
            "chunk_type": "paper_metadata",
            "chunk_id": metadata_chunk_id,
            "parent_chunk_id": metadata_chunk_id,
            "paragraph_index": 0,
            "global_chunk_index": 0,
        },
    )

    # 过滤 References，构建索引映射
    indexed_units = [
        (global_index, unit)
        for global_index, unit in enumerate(units)
        if unit.section != "References"
    ]

    # 构建子块（不含 parent_text）
    section_counts: dict[str, int] = {}
    children: list[Document] = []
    for filtered_index, (global_index, unit) in enumerate(indexed_units):
        section_counts[unit.section] = section_counts.get(unit.section, 0) + 1
        local_index = section_counts[unit.section]
        section_id = _slug(unit.section)
        subsection_id = f"{section_id}::{_slug(unit.subsection)}" if unit.subsection else ""
        parent_scope = subsection_id or f"{section_id}::{max(0, local_index - 1)}"
        parent_chunk_id = _stable_id(source, "parent", parent_scope)
        chunk_id = _stable_id(source, unit.section, str(local_index), unit.text[:80])
        heading = f"{paper_title} | {unit.section}"
        if unit.subsection:
            heading = f"{heading} | {unit.subsection}"

        children.append(
            Document(
                page_content=f"[{heading}] {unit.text}",
                metadata={
                    "source": source,
                    "paper_title": paper_title,
                    "authors": authors,
                    "page": page,
                    "section": unit.section,
                    "section_id": section_id,
                    "subsection": unit.subsection,
                    "subsection_id": subsection_id,
                    "chunk_type": unit.chunk_type,
                    "chunk_id": chunk_id,
                    "parent_chunk_id": parent_chunk_id,
                    "paragraph_index": local_index,
                    "global_chunk_index": global_index + 1,
                },
            )
        )

    # 构建父块
    parents = _build_parent_documents(children, source=source)

    return [metadata_chunk] + children + parents

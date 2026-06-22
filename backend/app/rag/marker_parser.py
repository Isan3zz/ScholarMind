"""
MinerU → Marker 替换：使用 marker-pdf 解析 PDF，输出 PaperTextUnit 列表。

marker-pdf 是高精度的 PDF→Markdown/JSON 工具：
- 布局检测 (Surya) → OCR (SuryaOCR) → 启发式重建
- 可选 LLM 增强（Gemini / Ollama）用于表格/公式优化
- GPU / CPU / MPS 均可运行
- 输出 JSON 树结构，含 block_type + section_hierarchy + html

本模块的核心工作：
1. 调用 PdfConverter 将 PDF 转为 JSON
2. 遍历 JSON 树 → 提取文本 + 追踪 section 层级 → PaperTextUnit 列表
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any


# ---------------------------------------------------------------------------
# HTML → 纯文本提取器（stdlib，零依赖）
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """从 HTML 中提取纯文本，保留段落/换行结构。"""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("br", "p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "div"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "div", "table"):
            self._parts.append("\n")

    def get_text(self) -> str:
        return "".join(self._parts).strip()


def _html_to_text(html: str) -> str:
    """HTML → 纯文本，合并多余空白。"""
    if not html:
        return ""
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    # 合并连续换行为最多两个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 合并行内空白
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Marker JSON 遍历 & PaperTextUnit 生成
# ---------------------------------------------------------------------------

# Marker block_type → 跳过（非文本内容）
_SKIP_BLOCK_TYPES = frozenset({
    "Figure",
    "Picture",
    "PageHeader",
    "PageFooter",
    "Page",
    "Document",
    "FigureGroup",
    "PictureGroup",
    "Form",
    "Handwriting",
    "TableOfContents",
})

# Marker block_type → 视为 body 文本
_TEXT_BLOCK_TYPES = frozenset({
    "Text",
    "TextInlineMath",
    "ListItem",
    "Caption",
})

# 其他不跳过也不在 _TEXT_BLOCK_TYPES 中的，单独处理（Equation / Table / Code）


def _heading_level(section_hierarchy: dict[str, str] | None) -> int:
    """从 section_hierarchy 推断标题层级（1=h1, 2=h2, …）。

    section_hierarchy 的 key 是层级数字字符串，value 是对应 SectionHeader 的 ID。
    最高 key = 当前块的标题层级。
    """
    if not section_hierarchy:
        return 0
    try:
        return max(int(k) for k in section_hierarchy.keys())
    except (ValueError, TypeError):
        return 0


def _parse_marker_json(rendered: Any) -> list[dict[str, Any]]:
    """从 PdfConverter 返回的 Pydantic model 中提取页面 children 列表。

    兼容两种形态：
    - rendered 本身是 Document block（有 children 属性）
    - rendered.children 是页面列表
    """
    # Pydantic model → dict 便于遍历
    if hasattr(rendered, "model_dump"):
        data = rendered.model_dump()
    elif hasattr(rendered, "dict"):
        data = rendered.dict()
    elif isinstance(rendered, dict):
        data = rendered
    else:
        raise TypeError(f"Unsupported rendered type: {type(rendered)}")

    # 顶层可能是 Document block 或直接是 list
    if isinstance(data, list):
        pages = data
    elif isinstance(data, dict):
        block_type = data.get("block_type", "")
        if block_type == "Document":
            pages = data.get("children", [])
        elif block_type == "Page":
            pages = [data]
        else:
            pages = data.get("children", []) or [data]
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")

    return pages


def _traverse_blocks(
    blocks: list[dict[str, Any]],
    current_section: str,
    current_subsection: str,
) -> list["PaperTextUnit"]:
    """递归遍历 Marker JSON 块树，生成 PaperTextUnit 列表。"""
    from app.rag.paper_sections import PaperTextUnit

    units: list[PaperTextUnit] = []
    section = current_section
    subsection = current_subsection

    for block in blocks:
        block_type = block.get("block_type", "")

        # --- 跳过非内容块 ---
        if block_type in _SKIP_BLOCK_TYPES:
            # 仍然递归处理其 children（某些块如 FigureGroup 可能包裹内容）
            children = block.get("children") or []
            if children:
                units.extend(_traverse_blocks(children, section, subsection))
            continue

        # --- SectionHeader: 更新当前 section/subsection ---
        if block_type == "SectionHeader":
            level = _heading_level(block.get("section_hierarchy"))
            text = _html_to_text(block.get("html", ""))

            if not text:
                children = block.get("children") or []
                if children:
                    units.extend(_traverse_blocks(children, section, subsection))
                continue

            if level <= 1:
                section = text
                subsection = ""
            else:
                subsection = text

            # SectionHeader 本身也作为文本单元（可选，帮助检索定位）
            # 不单独生成单元——标题信息已体现在后续块的 section/subsection 中
            children = block.get("children") or []
            if children:
                units.extend(_traverse_blocks(children, section, subsection))
            continue

        # --- Footnote: 可选合并到上下文或跳过 ---
        if block_type == "Footnote":
            # 脚注一般干扰正文，跳过
            continue

        # --- 文本类块 ---
        if block_type in _TEXT_BLOCK_TYPES:
            text = _html_to_text(block.get("html", ""))
            if text:
                units.append(PaperTextUnit(
                    text=text,
                    section=section,
                    subsection=subsection,
                    chunk_type="body",
                ))
            # 继续递归 children
            children = block.get("children") or []
            if children:
                units.extend(_traverse_blocks(children, section, subsection))
            continue

        # --- Equation ---
        if block_type == "Equation":
            text = _html_to_text(block.get("html", ""))
            if text:
                units.append(PaperTextUnit(
                    text=text,
                    section=section,
                    subsection=subsection,
                    chunk_type="body",
                ))
            continue

        # --- Table: 保留 HTML 或提取文本 ---
        if block_type in ("Table", "TableGroup"):
            html = block.get("html", "")
            # 优先提取纯文本，若为空则保留 HTML
            text = _html_to_text(html)
            if not text and html:
                text = html  # fallback: 保留原始 HTML
            if text:
                units.append(PaperTextUnit(
                    text=text,
                    section=section,
                    subsection=subsection,
                    chunk_type="body",
                ))
            children = block.get("children") or []
            if children:
                units.extend(_traverse_blocks(children, section, subsection))
            continue

        # --- Code ---
        if block_type == "Code":
            text = _html_to_text(block.get("html", ""))
            if text:
                units.append(PaperTextUnit(
                    text=text,
                    section=section,
                    subsection=subsection,
                    chunk_type="body",
                ))
            continue

        # --- 兜底: 未知 block_type，尝试提取文本并递归 ---
        html = block.get("html", "")
        text = _html_to_text(html)
        if text:
            units.append(PaperTextUnit(
                text=text,
                section=section,
                subsection=subsection,
                chunk_type="body",
            ))
        children = block.get("children") or []
        if children:
            units.extend(_traverse_blocks(children, section, subsection))

    return units


# ---------------------------------------------------------------------------
# 模块级缓存：PdfConverter 和模型只初始化一次
# ---------------------------------------------------------------------------

_converter: Any = None
_converter_lock: Any = None


def _get_converter():
    """延迟初始化 PdfConverter + 模型下载（仅首次调用）。"""
    global _converter

    if _converter is not None:
        return _converter

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    config = ConfigParser({"output_format": "json"})
    _converter = PdfConverter(
        config=config.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config.get_processors(),
        renderer=config.get_renderer(),
    )
    return _converter


def parse_pdf_with_marker(file_path: str) -> tuple[list["PaperTextUnit"], str, list[str]]:
    """用 Marker 解析 PDF，返回 (PaperTextUnit 列表, 论文标题, 作者列表)。

    论文标题从第一个 SectionHeader (h1) 或文档 metadata 提取。
    作者信息 Marker 不直接输出——返回空列表，由下游容错。
    """
    from app.rag.paper_sections import PaperTextUnit

    converter = _get_converter()
    rendered = converter(str(file_path))

    pages = _parse_marker_json(rendered)

    all_units: list[PaperTextUnit] = []
    global_section = "Unknown"
    global_subsection = ""

    for page in pages:
        page_children = page.get("children") or []
        page_units = _traverse_blocks(page_children, global_section, global_subsection)

        if page_units:
            # 将最后一页的 section/subsection 状态传递到下一页
            last = page_units[-1]
            global_section = last.section
            global_subsection = last.subsection

        all_units.extend(page_units)

    # 提取标题：第一个非 Unknown 的 section 作为论文标题
    title = "Unknown Paper"
    for unit in all_units:
        if unit.section and unit.section != "Unknown":
            title = unit.section
            break

    # 若所有 section 都是 Unknown，尝试从 page children 的第一个 SectionHeader 取
    if title == "Unknown Paper":
        for page in pages:
            for child in (page.get("children") or []):
                if child.get("block_type") == "SectionHeader":
                    t = _html_to_text(child.get("html", ""))
                    if t:
                        title = t
                        break
            if title != "Unknown Paper":
                break

    return all_units, title, []

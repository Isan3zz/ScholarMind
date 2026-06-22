"""
MinerU Cloud API parser — 调用 mineru.net 云端服务解析 PDF。

通过 `mineru` SDK 提交 + 轮询，用 curl 下载结果（绕过 Python SSL 兼容问题）。
Token 从环境变量 MINERU_API_TOKEN 读取。

输出：
  - Markdown 全文
  - content_list (结构化块列表，含 text_level 标题层级)
  - 映射为 PaperTextUnit → build_paper_chunks → ES
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _ensure_curl_patch():
    """Monkey-patch mineru SDK 的 download 方法用 curl (Python SSL 不支持 CDN 的 TLS renegotiation)。"""
    try:
        from mineru._api import ApiClient as _ApiClientV1
        _orig = _ApiClientV1.download
        if getattr(_orig, "__curl_patched__", False):
            return
        def _curl_download(self, url):
            # 容器 OpenSSL 3.5.6 无法与 MinerU CDN 完成 TLS 握手，
            # 通过主机代理 (host.docker.internal:9876) 转发 HTTPS 请求。
            proxy_url = f"http://host.docker.internal:9876/{url}"
            r = subprocess.run(["curl", "-sL", proxy_url], capture_output=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(f"curl download failed (exit={r.returncode}): {r.stderr[:200]}")
            return r.stdout
        _curl_download.__curl_patched__ = True
        _ApiClientV1.download = _curl_download
    except (ImportError, AttributeError):
        pass


def parse_pdf_with_mineru(file_path: str) -> tuple[list["PaperTextUnit"], str, list[str]]:
    """用 MinerU Cloud API 解析 PDF。

    Returns:
        (PaperTextUnit 列表, 论文标题, 作者列表)
    """
    from app.rag.paper_sections import PaperTextUnit

    _ensure_curl_patch()

    token = (os.getenv("MINERU_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "MINERU_API_TOKEN not set. Get one at https://mineru.net/apiManage/docs"
        )

    from mineru import MinerU

    client = MinerU(token=token)
    print(f"  [MinerU Cloud] Submitting {os.path.basename(file_path)} ...")
    result = client.extract(str(file_path))

    # --- Markdown 做主路径，图片/表格自然穿插，阅读顺序不间断 ---
    if result.markdown:
        units, title = _markdown_to_units(result.markdown, Path(file_path).stem)
    else:
        units, title = [], "Unknown Paper"

    print(f"  [MinerU Cloud] Done — {len(units)} units, title='{title[:60]}'")
    return units, title, []


def _content_list_to_units(content_list: list[dict]) -> tuple[list["PaperTextUnit"], str]:
    """content_list → PaperTextUnit + 标题"""
    from app.rag.paper_sections import PaperTextUnit

    units: list[PaperTextUnit] = []
    current_section = "Unknown"
    current_subsection = ""
    paper_title = "Unknown Paper"

    for block in content_list:
        block_type = block.get("type", "text")
        text = (block.get("text") or "").strip()
        text_level = block.get("text_level", 0)

        # 跳过非内容块（在提取文本之前判断，因为这些块没有 text 字段）
        if block_type in ("image", "page_number", "footer", "page_footnote", "header", "page_header"):
            continue

        # table → 取 table_body HTML（text 字段通常为空）
        if block_type == "table":
            html = block.get("table_body", "")
            caption = " ".join(block.get("table_caption", []))
            content = f"{caption}\n{html}" if caption else html
            if content.strip():
                units.append(PaperTextUnit(
                    text=content, section=current_section,
                    subsection=current_subsection, chunk_type="body"))
            continue

        # chart → 提取 content（Markdown 表格）
        if block_type == "chart":
            chart_content = block.get("content", "")
            if chart_content.strip():
                units.append(PaperTextUnit(
                    text=chart_content, section=current_section,
                    subsection=current_subsection, chunk_type="body"))
            continue

        # equation → 保留 LaTeX
        if block_type == "equation":
            eq_text = block.get("text", "")
            if eq_text.strip():
                units.append(PaperTextUnit(
                    text=eq_text, section=current_section,
                    subsection=current_subsection, chunk_type="body"))
            continue

        # list → 合并 list_items
        if block_type == "list":
            items = block.get("list_items", [])
            merged = "\n".join(items) if items else text
            if merged.strip():
                units.append(PaperTextUnit(
                    text=merged, section=current_section,
                    subsection=current_subsection, chunk_type="body"))
            continue

        # text 类块需要 text 非空
        if not text:
            continue

        # 标题 (text_level 表示层级)
        if text_level >= 1:
            if text_level == 1:
                current_section = text
                current_subsection = ""
                if paper_title == "Unknown Paper":
                    paper_title = text
            else:
                current_subsection = text
            continue

        # 普通段落 / 公式 / 表格
        chunk_type = "body"
        units.append(PaperTextUnit(
            text=text,
            section=current_section,
            subsection=current_subsection,
            chunk_type=chunk_type,
        ))

    return units, paper_title


def _markdown_to_units(md: str, fallback_title: str) -> tuple[list["PaperTextUnit"], str]:
    """Markdown → PaperTextUnit — 保留图文穿插的阅读顺序，跳过纯格式行。"""
    import re
    from app.rag.paper_sections import PaperTextUnit

    units: list[PaperTextUnit] = []
    current_section = "Unknown"
    current_subsection = ""
    paper_title = fallback_title
    buffer: list[str] = []
    in_details = False  # 跳过 <details>...</details> HTML 块

    def flush():
        nonlocal buffer
        if not buffer:
            return
        t = "\n".join(buffer).strip()
        if t:
            # 检测内容类型
            if "<table" in t or "<tr>" in t or "<td" in t:
                ct = "table"
            elif re.match(r"^```\w*", t):
                ct = "code"
            elif t.startswith("•"):
                ct = "list"
            elif re.match(r"^(Figure|Table|Chart|Algorithm)\s*[\d:]", t):
                ct = "caption"
            else:
                # 公式和普通正文放一起，不单独拆分
                ct = "body"
            units.append(PaperTextUnit(
                text=t, section=current_section,
                subsection=current_subsection, chunk_type=ct))
        buffer = []

    for line in md.splitlines():
        stripped = line.strip()

        # HTML details 块跳过去
        if stripped.startswith("<details") or stripped == "<details>":
            in_details = True
            continue
        if stripped == "</details>":
            in_details = False
            continue
        if in_details:
            continue

        # 空行 → flush
        if not stripped:
            flush()
            continue

        # 标题 — MinerU 全部输出 #, 按编号推断层级
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            flush()
            heading = m.group(2)

            # 判断层级: 单级编号=section, 多级=subsection
            # 匹配 "2", "A", "2.1", "A.1", "IV", "C.2" 等
            num_match = re.match(r"^([A-Z]\d*(?:\.\d+)*|\d+(?:\.\d+)*)\.?\s+", heading)
            if num_match:
                num = num_match.group(1)
                dots = num.count(".")
                if dots == 0:
                    # 单级编号: section
                    current_section = heading
                    current_subsection = ""
                    if paper_title == fallback_title:
                        paper_title = heading
                else:
                    # 多级编号: subsection
                    current_subsection = heading
            elif heading in ("Abstract", "References", "Acknowledgement", "Appendix"):
                current_section = heading
                current_subsection = ""
            else:
                # 无编号标题: 按 level 区分
                level = len(m.group(1))
                if level == 1:
                    current_section = heading
                    current_subsection = ""
                    if paper_title == fallback_title:
                        paper_title = heading
                else:
                    current_subsection = heading
            continue

        # 跳过纯格式行
        if re.match(r"^[|\-:\s]+$", stripped):     # 表格分隔线
            continue
        if re.match(r"^!\[", stripped):            # 图片 ![desc](path)
            continue
        if re.match(r"^\|[\s\w\d%\.×]+$", stripped):  # 单列表格碎片 (如 |Sure|)
            continue

        buffer.append(line)

    flush()

    # 后处理：公式和上下文合并 (上文 + 公式们 + 下文)
    merged: list[PaperTextUnit] = []
    i = 0
    while i < len(units):
        u = units[i]
        # 孤立的公式块：合并 [i-1] + [i] + 后续连续公式 + 下文
        if (u.text.strip().startswith("$$")
            and merged
            and merged[-1].chunk_type in ("body", "caption")
            and merged[-1].section == u.section
            and merged[-1].subsection == u.subsection
        ):
            combined = merged[-1].text + "\n\n" + u.text
            # 连续吃掉后续的同 section 公式 + 短下文
            while i + 1 < len(units):
                nxt = units[i + 1]
                same_sec = (nxt.section == u.section and nxt.subsection == u.subsection)
                if not same_sec:
                    break
                is_eq = nxt.text.strip().startswith("$$")
                # 下文看起来是公式的延续（小写开头 / "where"）
                is_continuation = (
                    nxt.chunk_type in ("body", "caption")
                    and bool(re.match(r"^(where|[a-z\(])", nxt.text.strip()))
                )
                if is_eq or is_continuation:
                    combined += "\n\n" + nxt.text
                    i += 1
                else:
                    break
            merged[-1] = PaperTextUnit(
                text=combined,
                section=merged[-1].section,
                subsection=merged[-1].subsection,
                chunk_type="body",
            )
        else:
            merged.append(u)
        i += 1
    units = merged

    # 再合并相邻的同 section body/caption 块（图片截断的段落跨 caption 粘回来）
    merged2: list[PaperTextUnit] = []
    for u in units:
        if (merged2
            and merged2[-1].chunk_type in ("body", "caption")
            and u.chunk_type in ("body", "caption")
            and merged2[-1].section == u.section
            and merged2[-1].subsection == u.subsection
        ):
            merged2[-1] = PaperTextUnit(
                text=merged2[-1].text + "\n\n" + u.text,
                section=merged2[-1].section,
                subsection=merged2[-1].subsection,
                chunk_type="body",
            )
        else:
            merged2.append(u)
    # 过滤掉 References 章节
    units = [u for u in merged2 if u.section != "References"]
    return units, paper_title

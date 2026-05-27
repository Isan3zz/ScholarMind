"""
Trace 持久化：JSON 写入 / 读取。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.trace.collector import TraceCollector

# traces 目录放在 backend 根下，与 app/ 同级
_TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"


def ensure_traces_dir() -> Path:
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    return _TRACES_DIR


def save_trace(collector: TraceCollector) -> Path:
    """将 collector 的完整 trace 保存为 JSON 文件。"""
    ensure_traces_dir()
    path = _TRACES_DIR / f"{collector.thread_id}.json"
    payload = collector.to_dict()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def load_trace(thread_id: str) -> dict | None:
    """读取已有的 trace 文件。不存在则返回 None。"""
    path = _TRACES_DIR / f"{thread_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

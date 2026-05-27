"""
TraceCollector: 每个 thread_id 一个实例，跨 turn 累积。
在 routes.py 的 event_generator 中逐节点收集 state 快照，
SSE 结束后调用 finish_and_save() 写入 traces/<thread_id>.json。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


RAW_HEAD_MAX_CHARS = 200


@dataclass
class TraceTurn:
    turn: int
    query: str
    intent: str | None
    mode: str
    plan: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    report: str = ""
    review_status: str | None = None
    critique: str = ""
    revision_number: int = 0
    should_stop: bool = False
    nodes: list[str] = field(default_factory=list)

    # 放在 turn 级别方便看到每轮开销
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "query": self.query,
            "intent": self.intent,
            "mode": self.mode,
            "plan": self.plan,
            "evidence": self.evidence,
            "report": self.report,
            "review": {
                "status": self.review_status,
                "critique": self.critique,
                "revision": self.revision_number,
                "should_stop": self.should_stop,
            },
            "nodes": self.nodes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class TraceCollector:
    """按 thread_id 累积 trace，每轮 /chat 请求对应一个 turn。"""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.turns: list[TraceTurn] = []
        self._started_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 每轮开始时调用
    # ------------------------------------------------------------------
    def begin_turn(self, query: str, intent: str | None, mode: str) -> TraceTurn:
        turn = TraceTurn(
            turn=len(self.turns) + 1,
            query=query,
            intent=intent,
            mode=mode,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.turns.append(turn)
        return turn

    # ------------------------------------------------------------------
    # 每个节点执行后调用
    # ------------------------------------------------------------------
    def record_node(self, turn: TraceTurn, node_name: str, state: dict) -> None:
        turn.nodes.append(node_name)

        match node_name:
            case "router":
                turn.intent = turn.intent or str(state.get("intent", ""))

            case "planner":
                turn.plan = [str(p) for p in state.get("plan", [])]

            case "researcher":
                turn.evidence = _extract_evidence(state.get("search_results", []))

            case "writer" | "refiner":
                report = str(state.get("final_report", "") or "")
                if report:
                    turn.report = report

            case "reviewer":
                turn.review_status = str(state.get("review_status", ""))
                turn.critique = str(state.get("critique", ""))
                turn.revision_number = int(state.get("revision_number", 0))
                turn.should_stop = bool(state.get("should_stop", False))

    # ------------------------------------------------------------------
    # 流结束 / 异常时调用
    # ------------------------------------------------------------------
    def finish_turn(self, turn: TraceTurn) -> None:
        turn.finished_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "started_at": self._started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_turns": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
        }


# ======================================================================
# 内置助手：从 search_results 提取结构化 evidence
# ======================================================================
_SOURCE_RE = re.compile(
    r"\[source:\s*(?P<source>[^,\]]+),\s*section:\s*(?P<section>[^,\]]+)(?:,\s*score=(?P<score>[\d.]+))?\]",
    re.IGNORECASE,
)


def _extract_evidence(search_results: list[str]) -> list[dict]:
    """从 researcher 的 search_results 中提取结构化证据。

    识别三种块：
    1. 本地文档块（带 [source: file, section: ...] 头）
    2. LLM 摘要（文本，不含 source 头）
    3. 网络搜索块（### 🌐 网络搜索结果）
    """
    evidence: list[dict] = []
    llm_summary_buffer: list[str] = []

    for block in search_results:
        text = str(block or "").strip()
        if not text:
            continue

        # 网络搜索块
        if text.startswith("### 🌐 网络搜索结果"):
            evidence.append(_web_entry(text))
            continue

        # 本地文档 / LLM 摘要
        lines = text.split("\n")
        for line in lines:
            match = _SOURCE_RE.search(line)
            if match:
                # 之前的文本是 LLM 摘要
                if llm_summary_buffer:
                    evidence.append(_summary_entry("\n".join(llm_summary_buffer)))
                    llm_summary_buffer.clear()

                # 提取 source 行后面的原文
                content_start = line.find("]") + 1 if "]" in line else 0
                raw_text = line[content_start:].strip()
                evidence.append(
                    {
                        "type": "local",
                        "source": match.group("source").strip(),
                        "section": match.group("section").strip(),
                        "score": float(match.group("score")) if match.group("score") else None,
                        "summary": "",  # 后面由 LLM 摘要单独补充
                        "raw_head": raw_text[:RAW_HEAD_MAX_CHARS] if raw_text else "",
                    }
                )
                continue

            # 不以 [source: 开头的行，可能是 LLM 摘要
            if line.strip() and not line.startswith("###"):
                llm_summary_buffer.append(line)

    # 剩余的 LLM 摘要
    if llm_summary_buffer:
        evidence.append(_summary_entry("\n".join(llm_summary_buffer)))

    return evidence


def _web_entry(text: str) -> dict:
    return {
        "type": "web",
        "source": "tavily_search",
        "section": "Web",
        "score": None,
        "summary": "",
        "raw_head": text[:RAW_HEAD_MAX_CHARS],
    }


def _summary_entry(text: str) -> dict:
    return {
        "type": "summary",
        "source": "llm",
        "section": "Synopsis",
        "score": None,
        "summary": text,
        "raw_head": text[:RAW_HEAD_MAX_CHARS],
    }

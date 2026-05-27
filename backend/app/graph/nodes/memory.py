from copy import deepcopy
from typing import Any

from app.graph.state import AgentState, ShortMemory

MAX_MEMORY_CHARS = 420
MAX_CHANGE_LOG_ITEMS = 5


def empty_short_memory() -> ShortMemory:
    return {
        "topic": "",
        "report_summary": "",
        "change_log": [],
        "last_intent": "",
    }


def _clip(text: Any, limit: int = MAX_MEMORY_CHARS) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _existing_memory(state: AgentState) -> ShortMemory:
    memory = empty_short_memory()
    existing = deepcopy(state.get("short_memory") or {})
    memory.update({
        "topic": str(existing.get("topic") or ""),
        "report_summary": str(existing.get("report_summary") or ""),
        "change_log": list(existing.get("change_log") or []),
        "last_intent": str(existing.get("last_intent") or ""),
    })
    return memory


def fallback_report_summary(query: str, report: str) -> str:
    topic = query or "当前问题"
    excerpt = _clip(report, 180)
    if excerpt:
        return _clip(f"生成了关于“{topic}”的报告，主要内容大概是：{excerpt}")
    return _clip(f"生成了关于“{topic}”的报告。")


def fallback_change_summary(query: str) -> str:
    instruction = query or "用户修改指令"
    return _clip(f"根据用户指令“{instruction}”更新了当前报告。")


def update_short_memory(state: AgentState) -> dict[str, ShortMemory]:
    query = str(state.get("query") or "").strip()
    intent = str(state.get("intent") or "").strip().lower()
    report = str(state.get("final_report") or "").strip()
    memory_event = _clip(state.get("memory_event") or "")

    if intent == "new_topic":
        memory = empty_short_memory()
        memory["topic"] = _clip(query)
        memory["report_summary"] = memory_event or fallback_report_summary(query, report)
    else:
        memory = _existing_memory(state)
        if not memory["topic"] and query:
            memory["topic"] = _clip(query)
        if memory_event:
            memory["change_log"].append(memory_event)
        elif intent in {"edit_report", "augment_report"}:
            memory["change_log"].append(fallback_change_summary(query))

    memory["change_log"] = [_clip(item) for item in memory["change_log"] if str(item).strip()][-MAX_CHANGE_LOG_ITEMS:]
    memory["last_intent"] = intent
    return {"short_memory": memory}


def format_short_memory_for_prompt(memory: ShortMemory | None) -> str:
    if not memory:
        return "No short-term memory available."

    lines: list[str] = []
    if memory.get("topic"):
        lines.append(f"Current topic: {memory['topic']}")
    if memory.get("report_summary"):
        lines.append(f"Current report summary: {memory['report_summary']}")

    changes = memory.get("change_log") or []
    if changes:
        lines.append("Recent report changes:")
        lines.extend(f"- {item}" for item in changes)

    return "\n".join(lines) if lines else "No short-term memory available."

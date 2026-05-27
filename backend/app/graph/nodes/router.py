from enum import Enum

from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from app.graph.nodes.memory import format_short_memory_for_prompt
from app.graph.state import AgentState
from app.utils.llm import get_llm

router_llm = get_llm()


class Intent(str, Enum):
    NEW_TOPIC = "new_topic"
    AUGMENT_REPORT = "augment_report"
    EDIT_REPORT = "edit_report"


class RouteResult(BaseModel):
    intent: Intent


RESEARCH_REQUIRED_TRIGGERS = [
    "证据", "引用", "来源", "页码", "补充实验", "实验对比",
    "消融", "相关工作", "局限", "再查", "检索", "找更多",
    "evidence", "citation", "source", "page", "experiment",
    "comparison", "ablation", "benchmark", "related work",
    "limitation", "future work",
]

STYLE_EDIT_TRIGGERS = [
    "改", "润色", "优化", "扩写", "写详细", "更通俗", "更正式",
    "重写", "调整", "标题", "格式", "总结", "结论",
    "concise", "shorter", "rewrite", "polish", "format", "tone", "bullet",
]


def needs_research_for_refinement(text: str) -> bool:
    q = (text or "").lower()
    return any(trigger.lower() in q for trigger in RESEARCH_REQUIRED_TRIGGERS)


def looks_like_refine(q: str) -> bool:
    text = (q or "").strip().lower()
    return any(trigger.lower() in text for trigger in STYLE_EDIT_TRIGGERS)


def router_node(state: AgentState) -> dict:
    """Router node: determine intent, write it back to state for downstream routing."""
    query = state["query"]
    incoming_intent = str(state.get("intent") or "").strip().lower()
    has_report = bool(state.get("final_report", "").strip())

    print(f"--- [Router] Routing intent={incoming_intent!r} query={query!r} has_report={has_report} ---")

    # 前端按钮明确指定 intent → 直接透传
    if incoming_intent in ("new_topic", "edit_report", "augment_report"):
        return {
            "intent": incoming_intent,
        }

    # 没有报告 → 只能新开
    if not has_report:
        return {"intent": "new_topic"}

    # 规则：含编辑类关键词 → edit_report（内部再判断是否需要检索）
    if looks_like_refine(query):
        if needs_research_for_refinement(query):
            return {"intent": "augment_report"}
        return {"intent": "edit_report"}

    # LLM 兜底
    report = state["final_report"][:50]
    memory_context = format_short_memory_for_prompt(state.get("short_memory"))
    prompt = f"""
The system already has a generated research report.
Latest user input: "{query}"
Recent report excerpt: "{report}"

Short-term memory:
{memory_context}

Classify the user intent:
1. "new_topic": the user wants to start a completely new research topic.
2. "augment_report": the user wants to supplement the report with new evidence, details, citations, or data that require additional retrieval.
3. "edit_report": the user wants to edit, polish, restyle, or restructure the existing report without needing new evidence.
Return JSON matching the configured schema.
"""

    llm_structured = router_llm.with_structured_output(RouteResult)
    result = llm_structured.invoke([HumanMessage(content=prompt)])

    print(f"--- [Router] LLM route result: {result.intent.value} ---")

    intent = result.intent.value
    if intent == "edit_report" and needs_research_for_refinement(query):
        return {"intent": "augment_report"}

    return {"intent": intent}


def test():
    state: AgentState = {
        "query": "Make the first section more detailed",
        "final_report": "Transformer development",
    }
    print(route_query(state))


# python -m app.graph.nodes.router
# test()

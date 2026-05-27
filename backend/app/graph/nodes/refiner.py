from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from app.graph.nodes.memory import fallback_change_summary, format_short_memory_for_prompt
from app.graph.state import AgentState
from app.utils.llm import get_llm

llm = get_llm()


class RefineResult(BaseModel):
    final_report: str = Field(description="Complete revised user-facing report in Markdown.")
    memory_event: str = Field(description="One short internal sentence summarizing added, removed, or adjusted content.")


def _format_supplemental_evidence(search_results: list[str]) -> str:
    evidence = "\n\n".join(str(item) for item in (search_results or []) if str(item).strip())
    if evidence:
        return f"""
Supplemental evidence:
{evidence}

Use the supplemental evidence to add or revise factual content. New factual claims must be grounded in this evidence and keep source/section citations.
"""

    return """
No supplemental evidence was retrieved.
Do not add new factual claims. Only perform text-level edits requested by the user.
"""


def build_refine_prompt(state: AgentState) -> str:
    query = state["query"]
    old_report = state.get("final_report", "")
    supplemental_evidence = _format_supplemental_evidence(state.get("search_results", []))
    memory_context = format_short_memory_for_prompt(state.get("short_memory"))

    return f"""
You are a professional academic report editor.

Short-term memory:
{memory_context}

Original report:
{old_report}

User edit instruction:
{query}

{supplemental_evidence}

Rules:
1. Preserve the existing Markdown structure unless the user asks to change it.
2. Modify only the requested parts where possible.
3. If supplemental evidence is provided, integrate it into the existing report instead of rewriting from scratch.
4. If supplemental evidence is provided, preserve source/section citations in the form [source: file, section: SectionName].
5. Output the complete revised report only, with no preface or commentary.
6. Return final_report as the complete user-facing revised report.
7. Return memory_event as one short internal sentence describing what was added, removed, or adjusted.
8. Do not include memory_event inside final_report.
"""


def refine_node(state: AgentState):
    query = state["query"]
    print(f"--- [Refiner] Refining report with instruction: {query} ---")

    prompt = build_refine_prompt(state)
    messages = [HumanMessage(content=prompt)]

    try:
        result = llm.with_structured_output(RefineResult).invoke(messages)
        new_report = result.final_report
        memory_event = result.memory_event or fallback_change_summary(query)
    except Exception:
        response = llm.invoke(messages)
        new_report = response.content
        memory_event = fallback_change_summary(query)

    return {
        "final_report": new_report,
        "review_status": "PASS",
        "memory_event": memory_event,
    }

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from app.graph.nodes.memory import fallback_change_summary, format_short_memory_for_prompt
from app.graph.state import AgentState
from app.utils.llm import get_llm

llm = get_llm()


class RefineResult(BaseModel):
    final_report: str = Field(description="Complete revised user-facing report in Markdown.")
    memory_event: str = Field(description="One short internal sentence summarizing added, removed, or adjusted content.")


def build_edit_prompt(state: AgentState) -> str:
    """Prompt for edit_report: pure expression / structure / style editing.

    No new factual claims are permitted — this is text-level refinement only.
    """
    query = state["query"]
    old_report = state.get("final_report", "")
    memory_context = format_short_memory_for_prompt(state.get("short_memory"))

    return f"""
You are a professional academic report editor. Your task is to refine the expression, structure, or style of an existing report according to the user's instruction.

Short-term memory (previous changes to this report):
{memory_context}

Original report:
{old_report}

User edit instruction:
{query}

Rules:
1. This is a **text-level edit only**. Do NOT add new factual claims, evidence, data, citations, or content that was not already present in the original report.
2. Preserve all existing source/section citations in the form [source: file, section: SectionName] exactly as they appear.
3. Modify only the parts the user asked to change. Leave untouched sections as-is.
4. Preserve the existing Markdown structure unless the user explicitly asks to change it.
5. Output the complete revised report only, with no preface or commentary.
6. Return final_report as the complete user-facing revised report.
7. Return memory_event as one short internal sentence describing what was changed (e.g. "rephrased section 2 for conciseness", "converted bullet list to table").
8. Do not include memory_event inside final_report.
"""


def build_augment_prompt(state: AgentState) -> str:
    """Prompt for augment_report: integrate new evidence / search results into the existing report.

    New factual claims must be grounded in the supplemental evidence.
    """
    query = state["query"]
    old_report = state.get("final_report", "")
    memory_context = format_short_memory_for_prompt(state.get("short_memory"))

    evidence = "\n\n".join(
        str(item) for item in (state.get("search_results") or []) if str(item).strip()
    )
    if not evidence:
        evidence = "(No supplemental evidence was retrieved — treat this as an edit-only request.)"

    return f"""
You are a professional academic report editor. Your task is to integrate new evidence and research findings into an existing report according to the user's instruction.

Short-term memory (previous changes to this report):
{memory_context}

Original report:
{old_report}

User instruction:
{query}

Supplemental evidence (retrieved for this request):
{evidence}

Rules:
1. Integrate the supplemental evidence into the existing report where relevant. Do NOT rewrite the entire report from scratch — add and weave in where appropriate.
2. Every new factual claim must be grounded in the supplemental evidence above.
3. Preserve source/section citations in the form [source: file, section: SectionName]. New evidence must carry its source citation.
4. Preserve the existing Markdown structure unless the user asks to change it.
5. Follow the user's specific instruction for where and how to add the new content.
6. Output the complete revised report only, with no preface or commentary.
7. Return final_report as the complete user-facing revised report.
8. Return memory_event as one short internal sentence describing what was added, removed, or adjusted (e.g. "added ablation study comparison from evidence X", "inserted limitation discussion citing source Y").
9. Do not include memory_event inside final_report.
"""


def refine_node(state: AgentState):
    query = state["query"]
    intent = str(state.get("intent") or "").strip().lower()
    print(f"--- [Refiner] Refining report intent={intent} query={query!r} ---")

    if intent == "augment_report":
        prompt = build_augment_prompt(state)
    else:
        prompt = build_edit_prompt(state)

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

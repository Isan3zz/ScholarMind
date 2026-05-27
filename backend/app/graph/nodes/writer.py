from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from app.graph.nodes.memory import fallback_report_summary
from app.graph.state import AgentState
from app.utils.llm import get_llm

llm = get_llm()


class WriteResult(BaseModel):
    final_report: str = Field(description="Complete user-facing report in Markdown.")
    memory_event: str = Field(description="One short internal sentence summarizing what report was generated.")


WRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional academic paper reading and synthesis assistant.

User question:
{query}

Retrieved paper evidence:
{content}

Reviewer feedback, if any:
{critique_section}

Writing rules:
1. Answer only from the provided paper evidence; do not fabricate information that is not in the evidence.
2. Key conclusions must include source/section citations in the form [source: file, section: SectionName].
3. If the evidence is insufficient, clearly state what information is missing.
4. If the user asks about methods, prioritize Method/Approach/Model evidence.
5. If the user asks about experiments, prioritize Experiments/Results/Table evidence.
6. Use Markdown with a clear structure suitable for paper reading or literature review notes.
7. Return final_report as the complete user-facing report.
8. Return memory_event as one short internal sentence explaining what report was generated and its main contents.
9. Do not include memory_event inside final_report.
"""
)


def write_node(state: AgentState):
    print("--- [Node] Writing report ---")
    query = state["query"]
    content = "\n\n".join(state["search_results"])

    critique = state.get("critique", "")
    critique_section = ""
    if critique:
        critique_section = f"""
        Important: the previous report did not pass review.
        Reviewer feedback: {critique}
        Please fix the issue in this version.
        """

    prompt = WRITE_PROMPT.format(
        query=query,
        content=content,
        critique_section=critique_section,
    )

    try:
        result = llm.with_structured_output(WriteResult).invoke(prompt)
        report = result.final_report
        memory_event = result.memory_event or fallback_report_summary(query, report)
    except Exception:
        response = llm.invoke(prompt)
        report = response.content
        memory_event = fallback_report_summary(query, report)

    return {
        "final_report": report,
        "memory_event": memory_event,
    }

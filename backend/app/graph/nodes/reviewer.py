import re
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import AgentState
from app.utils.llm import get_llm

llm = get_llm(model_type="smart")


class ReviewResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    feedback: str = Field(default="", description="Empty for PASS. For FAIL, one concrete improvement suggestion.")


REVIEW_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict academic report reviewer.

Check whether the following report sufficiently answers the user's question:
{query}

Report:
{report}
"""
)


def has_source_citation(report: str) -> bool:
    return bool(
        re.search(
            r"\[source:\s*[^,\]]+,\s*section:\s*[^\]]+\]",
            report or "",
            re.IGNORECASE,
        )
    )


def review_node(state: AgentState):
    print("--- [Node] Reviewing report quality ---")
    query = state["query"]
    report = state["final_report"]
    num = state.get("revision_number", 0)

    # Rule-based check: citations required if evidence had them
    source_evidence_required = any("[source:" in item for item in state.get("search_results", []))
    if source_evidence_required and not has_source_citation(report):
        return {
            "critique": "Report is missing source/section citations. Add citations in the form [source: file, section: SectionName].",
            "revision_number": num + 1,
            "review_status": "FAIL",
        }

    llm_structured = llm.with_structured_output(ReviewResult)
    result = llm_structured.invoke(REVIEW_PROMPT.format(query=query, report=report))

    return {
        "critique": result.feedback,
        "revision_number": num + 1,
        "review_status": result.status,
    }

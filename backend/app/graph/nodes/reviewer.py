import re
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import AgentState
from app.rag.engine import get_available_papers
from app.utils.llm import get_llm

llm = get_llm(model_type="smart")


class InfoGap(BaseModel):
    """One concrete piece of missing information the report should cover."""

    missing_aspect: str = Field(
        description="What specific information is missing. Be precise: name the metric, baseline, method detail, or comparison that is absent."
    )
    target_sections: list[str] = Field(
        default_factory=list,
        description="Which paper sections most likely contain this information. Use actual section names from the paper catalog (e.g. '5 Experiments', '4 Method').",
    )
    priority: Literal["critical", "supplementary"] = Field(
        description="'critical' if this missing info is central to answering the user's question; 'supplementary' if it's nice-to-have detail."
    )


class ReviewResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    gaps: list[InfoGap] = Field(
        default_factory=list,
        description="When FAIL: concrete list of what's missing and where to find it. MUST be non-empty for FAIL. Empty for PASS.",
    )
    summary: str = Field(
        default="",
        description="One-sentence summary of the review outcome, for logging only.",
    )


REVIEW_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict academic report reviewer.

Your job: check whether the report fully answers the user's question using the available paper evidence.

User question:
{query}

Report to review:
{report}

Available papers (with sections):
{paper_catalog}

CRITICAL RULE — Accountability Constraint:
- If you assign FAIL, you MUST list at least one concrete InfoGap.
- Each gap MUST specify (a) exactly what information is missing, and (b) which section(s) of which paper likely contain it.
- If you cannot name a specific missing piece of information AND point to a likely section, then the report is adequate — assign PASS.
- Vague complaints like "not enough detail" or "could be better" are FORBIDDEN. Every gap must be actionable: a downstream retrieval planner will use your gaps to formulate new search queries.

How to evaluate:
1. Read the user's question carefully. Identify what a complete answer requires: methods? experiments? baselines? comparisons? limitations?
2. Scan the report. For each required element, is it present and grounded in evidence?
3. For any missing element, create a gap with:
   - missing_aspect: a one-sentence description of what's absent (e.g. "SafeDecoding's Toxicity score relative to Llama-Guard and GPT-4 baselines")
   - target_sections: 1–3 sections from the paper catalog where this info likely lives
   - priority: critical if the question hinges on it, supplementary if it's a completeness detail
4. If all required elements are present and well-supported, assign PASS with empty gaps.

Output format:
- PASS → status="PASS", gaps=[], summary="Report fully answers the question."
- FAIL → status="FAIL", gaps=[...concrete gaps...], summary="Report missing X, Y, Z."
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


def _build_paper_catalog_for_reviewer() -> str:
    """Build a compact paper catalog with section names for the Reviewer."""
    papers = get_available_papers()
    if not papers:
        return "(No papers indexed)"

    lines = []
    for i, p in enumerate(papers, 1):
        source = p["source"]
        title = p.get("title", source)
        sections = p.get("sections", [])
        # sections is now list[str] from ES terms aggregation; handle both str and legacy dict
        section_names = [
            (s if isinstance(s, str) else s.get("section_name", s.get("name", "?")))
            for s in sections
        ] if sections else ["(sections not available)"]
        lines.append(f"{i}. {source} — \"{title}\"")
        lines.append(f"   Sections: {', '.join(section_names[:20])}")
    return "\n".join(lines)


def _format_gaps_as_critique(gaps: list[InfoGap]) -> str:
    """Convert structured gaps into a Planner-actionable critique string."""
    if not gaps:
        return ""

    lines = ["The report is missing the following information:"]
    for i, gap in enumerate(gaps, 1):
        priority_label = "🔴 CRITICAL" if gap.priority == "critical" else "🟡 SUPPLEMENTARY"
        lines.append(f"\n{i}. [{priority_label}] {gap.missing_aspect}")
        if gap.target_sections:
            lines.append(f"   → Look in: {', '.join(gap.target_sections)}")
    return "\n".join(lines)


def review_node(state: AgentState):
    print("--- [Node] Reviewing report quality ---")
    query = state["query"]
    report = state["final_report"]
    num = state.get("revision_number", 0)
    paper_catalog = _build_paper_catalog_for_reviewer()

    # Rule-based check: citations required if evidence had them
    source_evidence_required = any("[source:" in item for item in state.get("search_results", []))
    if source_evidence_required and not has_source_citation(report):
        return {
            "critique": "Report is missing source/section citations. Add citations in the form [source: file, section: SectionName].",
            "revision_number": num + 1,
            "review_status": "FAIL",
        }

    llm_structured = llm.with_structured_output(ReviewResult)
    result = llm_structured.invoke(
        REVIEW_PROMPT.format(query=query, report=report, paper_catalog=paper_catalog)
    )

    critique_text = _format_gaps_as_critique(result.gaps)
    gaps_data = [gap.model_dump() for gap in result.gaps]

    return {
        "critique": critique_text,
        "reviewer_gaps": gaps_data,
        "revision_number": num + 1,
        "review_status": result.status,
    }

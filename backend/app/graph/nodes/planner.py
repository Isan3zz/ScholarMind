from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from app.graph.nodes.memory import format_short_memory_for_prompt
from app.graph.state import AgentState, RetrievalAuditEntry
from app.rag.engine import get_available_papers
from app.utils.llm import get_llm

llm = get_llm()


class QueryPlan(BaseModel):
    query: str = Field(description="A single English retrieval query for paper vector database search")
    sources: list[str] = Field(
        default_factory=list,
        description="Paper filenames to search for this query. Empty list [] means search all papers.",
    )
    sections: list[str] = Field(
        default_factory=list,
        description="Target sections likely to contain the answer (e.g. '4 Method', '5 Experiments'). Used to boost retrieval ranking.",
    )


class PlanResult(BaseModel):
    queries: list[QueryPlan] = Field(description="3-5 retrieval query plans, each with its target paper sources")


PLAN_PROMPT = ChatPromptTemplate.from_template(
    """You are a retrieval planner for an academic paper QA system.
Rewrite the user's question into 3-5 English retrieval queries for a paper vector database.
For each query, also specify which papers to search.
Output your answer as a json object.

Available papers in the knowledge base:
{paper_catalog}

User question:
{query}

Existing reviewer critique, if any:
{critique}

Short-term memory:
{short_memory_context}

Retrieval audit log (what previous rounds searched and found — use this to avoid dead ends):
{retrieval_audit}

Rules:
1. Output queries that can find evidence in academic papers, not generic web search phrases.
2. Preserve proper nouns, model names, dataset names, acronyms, paper titles, and technical terms from the user question.
3. If the user asks about methods, models, or frameworks, emphasize Method, Approach, Model, Framework, Architecture.
4. If the user asks about experiments, results, comparisons, or ablations, emphasize Experiment, Evaluation, Results, Benchmark, Ablation, Table.
5. If the user asks which baselines, compared methods, or defense methods are used, make the first retrieval query specific to Experimental Setup, Baseline Setup, baseline names, and compared defense methods.
6. If the user asks about innovation or contributions, cover Introduction, Contribution, Method.
7. If the user asks about limitations or future work, cover Limitation, Discussion, Future Work.
8. If reviewer critique mentions missing evidence, generate follow-up queries for that missing information.
8a. DO NOT repeat queries that returned empty results in a previous round (check the audit log's empty_queries). Instead, try different angles or different section targets.
8b. If a previous round's query hit sources/sections but with low relevance, shift sections rather than rephrasing the same query.
8c. If the audit log shows a section was already hit, consider whether a different section of the same paper could fill the gap.
9. FIRST determine whether this is a single-paper or multi-paper question:
   - SINGLE-PAPER: the user asks about one specific paper. ALL queries should target that paper only. Generate focused queries that drill into different sections of THE SAME paper (e.g., its method detail, experimental setup, limitations). Every query's sources = [that_one_paper.pdf].
   - MULTI-PAPER: the user asks about 2+ papers (comparison, contrast, relationships). Split queries by paper: query1 targets paperA's method, query2 targets paperB's method, query3 targets shared concepts. Each query's sources = [one paper] when drilling into that paper, or [paperA, paperB] when directly comparing. Do NOT put both papers in every source list — per-paper queries produce better retrieval.
10. For each query, set `sources` to the relevant paper filename(s). Use an empty list [] only if the question is genuinely unrelated to ALL papers (not the case here).
11. For each query, set `sections` to 1-3 section names that are most likely to contain the answer. Look at the paper catalog above for actual section names. Examples:
    - Method question → sections: ["4 Method", "3 Approach"]
    - Experiment question → sections: ["5 Experiments", "5.1 Experimental Setup"]
    - Contribution question → sections: ["Abstract", "1 Introduction", "6 Conclusion"]
    - Limitation question → sections: ["7 Limitations", "6 Conclusion"]
    Use the actual section names from the papers, not generic placeholders.

Output format example (single-paper):
{{"queries": [{{"query": "SafeDecoding expert model training phase detail", "sources": ["paper.pdf"], "sections": ["4 Method", "4.2 Training"]}}, {{"query": "SafeDecoding inference-time decoding algorithm", "sources": ["paper.pdf"], "sections": ["4 Method", "4.4 Inference"]}}, {{"query": "SafeDecoding experimental baselines and setup", "sources": ["paper.pdf"], "sections": ["5 Experiments", "5.1 Setup"]}}]}}

Output format example (multi-paper):
{{"queries": [{{"query": "SafeDecoding defense mechanism architecture", "sources": ["paperA.pdf"], "sections": ["4 Method"]}}, {{"query": "Adversarial alignment attack methodology", "sources": ["paperB.pdf"], "sections": ["3 Method", "5 Experiments"]}}, {{"query": "defense vs attack evaluation comparison", "sources": ["paperA.pdf", "paperB.pdf"], "sections": ["5 Experiments"]}}]}}
"""
)


def _build_paper_catalog() -> str:
    """构建论文目录文本，供 Planner prompt 使用。"""
    papers = get_available_papers()
    if not papers:
        return "(No papers currently indexed in the knowledge base)"

    lines = []
    for i, p in enumerate(papers, 1):
        source = p["source"]
        title = p.get("title", source)
        abstract = p.get("abstract", "")
        abstract_short = abstract[:300] + ("..." if len(abstract) > 300 else "")
        sections = p.get("sections", [])
        lines.append(f"{i}. source: {source}")
        lines.append(f"   title: {title}")
        if abstract_short:
            lines.append(f"   abstract: {abstract_short}")
        if sections:
            lines.append(f"   sections: {', '.join(sections[:20])}")
    return "\n".join(lines)


def _format_audit_log_for_prompt(audit_log: list[RetrievalAuditEntry] | None) -> str:
    """Format the retrieval audit log as a compact prompt section."""
    if not audit_log:
        return "(No previous retrieval rounds — this is the first attempt.)"

    lines: list[str] = []
    for entry in audit_log:
        round_num = entry.get("round", 0)
        queries = entry.get("queries", [])
        sources_hit = entry.get("sources_hit", [])
        sections_hit = entry.get("sections_hit", [])
        max_score = entry.get("max_relevance", 0.0)
        empty = entry.get("empty_queries", [])

        lines.append(f"Round {round_num}:")
        lines.append(f"  Queries tried: {', '.join(queries) if queries else '(none)'}")
        if sources_hit:
            lines.append(f"  Papers hit: {', '.join(sources_hit)}")
        if sections_hit:
            lines.append(f"  Sections hit: {', '.join(sections_hit)}")
        lines.append(f"  Max relevance score: {max_score:.2f}")
        if empty:
            lines.append(f"  ⚠️ EMPTY (no results): {', '.join(empty)}")
        lines.append("")

    return "\n".join(lines)


def plan_node(state: AgentState):
    print("--- [Planner] Planning paper retrieval queries ---")
    query = state["query"]
    critique = state.get("critique", "")
    memory_context = format_short_memory_for_prompt(state.get("short_memory"))
    paper_catalog = _build_paper_catalog()
    audit_text = _format_audit_log_for_prompt(state.get("retrieval_audit_log"))

    llm_structured = llm.with_structured_output(PlanResult)
    result = llm_structured.invoke(
        PLAN_PROMPT.format(
            query=query,
            critique=critique,
            short_memory_context=memory_context,
            paper_catalog=paper_catalog,
            retrieval_audit=audit_text,
        )
    )

    plans = []
    plan_sources = []
    plan_sections = []
    for qp in result.queries:
        q = qp.query.strip()
        if q:
            plans.append(q)
            plan_sources.append([s for s in qp.sources if s.strip()])
            plan_sections.append([s for s in (qp.sections or []) if s.strip()])

    print(f"--- [Planner] Generated {len(plans)} queries, sources: {[len(s) for s in plan_sources]}, sections: {[len(s) for s in plan_sections]} ---")
    return {"plan": plans, "plan_sources": plan_sources, "plan_sections": plan_sections}


def test():
    state: AgentState = {
        "query": "Transformer development status",
    }
    print(plan_node(state))


# python -m app.graph.nodes.planner
# test()

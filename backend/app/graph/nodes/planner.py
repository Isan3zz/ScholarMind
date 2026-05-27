from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from app.graph.nodes.memory import format_short_memory_for_prompt
from app.graph.state import AgentState
from app.utils.llm import get_llm

llm = get_llm()


class PlanResult(BaseModel):
    queries: list[str] = Field(description="3-5 English retrieval queries for paper vector database search")


PLAN_PROMPT = ChatPromptTemplate.from_template(
    """You are a retrieval planner for an academic paper QA system.
Rewrite the user's question into 3-5 English retrieval queries for a paper vector database.
Return JSON that matches this schema: {{"queries": ["query 1", "query 2", "query 3"]}}.

User question:
{query}

Existing reviewer critique, if any:
{critique}

Short-term memory:
{short_memory_context}

Rules:
1. Output queries that can find evidence in academic papers, not generic web search phrases.
2. Preserve proper nouns, model names, dataset names, acronyms, paper titles, and technical terms from the user question.
3. If the user asks about methods, models, or frameworks, emphasize Method, Approach, Model, Framework, Architecture.
4. If the user asks about experiments, results, comparisons, or ablations, emphasize Experiment, Evaluation, Results, Benchmark, Ablation, Table.
5. If the user asks which baselines, compared methods, or defense methods are used, make the first retrieval query specific to Experimental Setup, Baseline Setup, baseline names, and compared defense methods.
6. If the user asks about innovation or contributions, cover Introduction, Contribution, Method.
7. If the user asks about limitations or future work, cover Limitation, Discussion, Future Work.
8. If reviewer critique mentions missing evidence, generate follow-up queries for that missing information.
"""
)


def plan_node(state: AgentState):
    print("--- [Planner] Planning paper retrieval queries ---")
    query = state["query"]
    critique = state.get("critique", "")
    memory_context = format_short_memory_for_prompt(state.get("short_memory"))

    llm_structured = llm.with_structured_output(PlanResult)
    result = llm_structured.invoke(
        PLAN_PROMPT.format(
            query=query,
            critique=critique,
            short_memory_context=memory_context,
        )
    )

    plans = [q.strip() for q in result.queries if q.strip()]
    return {"plan": plans}


def test():
    state: AgentState = {
        "query": "Transformer development status",
    }
    print(plan_node(state))


# python -m app.graph.nodes.planner
# test()

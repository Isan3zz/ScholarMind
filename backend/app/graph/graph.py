from langgraph.graph import END, StateGraph

from app.graph.nodes.memory import update_short_memory
from app.graph.state import AgentState


def memory_node(state: AgentState):
    return update_short_memory(state)


def planner_node(state: AgentState):
    from app.graph.nodes.planner import plan_node

    return plan_node(state)


def researcher_node(state: AgentState):
    from app.graph.nodes.researcher import research_node

    return research_node(state)


def writer_node(state: AgentState):
    from app.graph.nodes.writer import write_node

    return write_node(state)


def reviewer_node(state: AgentState):
    from app.graph.nodes.reviewer import review_node

    return review_node(state)


def refiner_node(state: AgentState):
    from app.graph.nodes.refiner import refine_node

    return refine_node(state)


def router_node(state: AgentState):
    from app.graph.nodes.router import router_node as _run

    return _run(state)


def route_after_router(state: AgentState) -> str:
    """根据 router 写入的 intent 决定下一步路径。"""
    intent = str(state.get("intent") or "").strip().lower()

    if intent in ("new_topic", "augment_report"):
        return "planner"

    # edit_report: 纯编辑，不需要检索
    return "refiner"


def route_after_research(state: AgentState):
    """Route researcher output based on stop conditions and refinement intent."""
    if state.get("should_stop", False):
        print("--- [route] should_stop=True -> ending task early ---")
        return END

    if state.get("intent") == "augment_report":
        return "refiner"

    return "writer"


def route_after_refiner(state: AgentState):
    """Route refiner output: augment intent needs reviewer, others skip to memory."""
    if state.get("intent") == "augment_report":
        print("--- [route] augment_report -> reviewer ---")
        return "reviewer"
    print("--- [route] style edit -> memory ---")
    return "memory"


def should_continue(state: AgentState):
    """Route reviewer output to retry, finish, or update short memory."""
    current_revision = state.get("revision_number", 0)
    if current_revision >= 3:
        print("--- [route] max revision reached -> ending task ---")
        return END

    review_status = state.get("review_status", "PASS")
    critique = state.get("critique", "")

    if review_status == "FAIL":
        print(f"--- [route] review failed ({critique}) -> planner ---")
        return "planner"

    print("--- [route] review passed -> memory ---")
    return "memory"


def create_graph(memory=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("memory", memory_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "planner": "planner",
            "refiner": "refiner",
        },
    )
    workflow.add_edge("planner", "researcher")
    workflow.add_conditional_edges(
        "researcher",
        route_after_research,
        {
            "writer": "writer",
            "refiner": "refiner",
            END: END,
        },
    )
    workflow.add_edge("writer", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "planner": "planner",
            "memory": "memory",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "refiner",
        route_after_refiner,
        {
            "reviewer": "reviewer",
            "memory": "memory",
        },
    )
    workflow.add_edge("memory", END)

    app = workflow.compile(checkpointer=memory)
    return app

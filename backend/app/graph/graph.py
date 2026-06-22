from langgraph.graph import END, StateGraph

from app.graph.state import AgentState


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
    """Route refiner output: augment intent needs reviewer, others done."""
    if state.get("intent") == "augment_report":
        print("--- [route] augment_report -> reviewer ---")
        return "reviewer"
    print("--- [route] style edit -> END ---")
    return END


def should_continue(state: AgentState):
    """Route reviewer output to retry or finish.

    Exit conditions (any one triggers END):
    1. Reviewer passed → done.
    2. Hard cap: 5 rounds (safety net, should rarely fire).
    3. Info-gain stall: last two audit rounds show no improvement in max_relevance,
       no new sources hit, and empty_queries didn't shrink → further rounds unlikely to help.
    4. All audit queries returned empty → knowledge base has nothing on this topic.
    """
    review_status = state.get("review_status", "PASS")
    critique = state.get("critique", "")
    current_revision = state.get("revision_number", 0)

    if review_status != "FAIL":
        print("--- [route] review passed -> END ---")
        return END

    # Hard safety cap — 5 rounds
    if current_revision >= 5:
        print("--- [route] max revision (5) reached -> ending task ---")
        return END

    # Info-gain analysis from audit log
    audit_log: list = list(state.get("retrieval_audit_log") or [])
    if len(audit_log) >= 2:
        prev = audit_log[-2]
        curr = audit_log[-1]

        prev_max = float(prev.get("max_relevance", 0.0))
        curr_max = float(curr.get("max_relevance", 0.0))
        prev_empty = len(prev.get("empty_queries", []))
        curr_empty = len(curr.get("empty_queries", []))
        prev_sources = set(prev.get("sources_hit", []))
        curr_sources = set(curr.get("sources_hit", []))

        new_sources = curr_sources - prev_sources
        score_improved = curr_max > prev_max + 0.01
        empty_shrank = curr_empty < prev_empty

        if not score_improved and not new_sources and not empty_shrank:
            print(
                f"--- [route] info-gain stall: "
                f"score {prev_max:.2f}→{curr_max:.2f}, "
                f"new_sources={len(new_sources)}, "
                f"empty {prev_empty}→{curr_empty} "
                f"-> ending task ---"
            )
            return END

    # If the latest round had ALL queries empty → nothing retrievable
    if audit_log:
        last = audit_log[-1]
        if last.get("queries") and len(last.get("empty_queries", [])) == len(last.get("queries", [])):
            print("--- [route] all queries empty in last round -> ending task ---")
            return END

    print(f"--- [route] review failed (round {current_revision}) -> planner ---")
    return "planner"


def create_graph(memory=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("refiner", refiner_node)

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
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "refiner",
        route_after_refiner,
        {
            "reviewer": "reviewer",
            END: END,
        },
    )

    app = workflow.compile(checkpointer=memory)
    return app

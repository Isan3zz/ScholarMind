from typing import List, NotRequired, TypedDict


class ShortMemory(TypedDict):
    """Compact per-thread report memory stored in LangGraph checkpoints."""

    topic: str
    report_summary: str
    change_log: list[str]
    last_intent: str


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    query: str
    plan: List[str]
    plan_sources: NotRequired[List[List[str]]]  # per-query paper source lists, aligned with plan
    plan_sections: NotRequired[List[List[str]]]  # per-query target section lists, aligned with plan
    search_results: List[str]
    final_report: str
    critique: str
    revision_number: int
    review_status: str
    search_mode: str
    should_stop: bool
    intent: NotRequired[str | None]  # new_topic | edit_report | augment_report
    short_memory: NotRequired[ShortMemory]

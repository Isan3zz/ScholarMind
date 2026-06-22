# Refinement Routing Design

## Goal

IRIS should distinguish three user intents after a report exists:

1. `new_topic`: start a completely new analysis.
2. `edit_report`: edit the existing report without retrieving new evidence.
3. `augment_report`: add new conditions, evidence, citations, or perspectives to the existing report.

This avoids sending every report modification through the same `refiner` path, while also avoiding full report regeneration when the user only wants to revise the existing report.

## Target Workflows

### New Topic

Use this when the user asks a completely new question or wants a fresh report.

```text
query
-> planner
-> researcher
-> writer
-> reviewer
-> END or planner retry
```

Examples:

- "Analyze this paper's method innovation."
- "Explain the experimental results."
- "Start over and analyze another paper."

### Edit Existing Report

Use this when the user only asks for wording, structure, format, length, or style changes.

```text
old_report + instruction
-> refiner
-> END
```

Examples:

- "Make it more concise."
- "Rewrite this in bullet points."
- "Make the first paragraph more detailed."
- "Use a more formal tone."

### Augment Existing Report

Use this when the user wants to add new evidence, citations, sections, comparisons, or new analytical conditions to the existing report.

```text
old_report + instruction
-> planner
-> researcher
-> refiner
-> END
```

Examples:

- "Add experimental comparison evidence."
- "Add related work citations."
- "Supplement the limitations section."
- "Analyze it again from the efficiency perspective."
- "Add benchmark results."

## Backend Changes

### `backend/app/graph/nodes/router.py`

Extend intent handling from two broad values to three explicit values:

```python
new_topic
edit_report
augment_report
```

Expected routing:

```python
if intent == "new_topic":
    return "planner"

if intent == "edit_report" and has_report:
    return "refiner"

if intent == "augment_report" and has_report:
    return "planner"
```

For backward compatibility, keep supporting old `intent == "refine"`:

```python
if intent == "refine" and has_report:
    return "planner" if needs_research_for_refinement(query) else "refiner"
```

Add a simple rule-based classifier:

```python
RESEARCH_REQUIRED_TRIGGERS = [
    "证据", "引用", "来源", "页码",
    "补充实验", "实验对比", "消融", "benchmark",
    "相关工作", "局限", "future work",
    "再查", "检索", "找更多",
    "evidence", "citation", "source", "page",
    "experiment", "comparison", "ablation",
    "related work", "limitation", "benchmark",
]

def needs_research_for_refinement(text: str) -> bool:
    q = (text or "").lower()
    return any(trigger in q for trigger in RESEARCH_REQUIRED_TRIGGERS)
```

### `backend/app/graph/graph.py`

Change `route_after_research` so `augment_report` goes to `refiner`, not `writer`.

```python
def route_after_research(state: AgentState):
    if state.get("should_stop", False):
        return END

    if state.get("intent") == "augment_report":
        return "refiner"

    return "writer"
```

Graph target shape:

```text
new_topic:
planner -> researcher -> writer -> reviewer

edit_report:
refiner -> END

augment_report:
planner -> researcher -> refiner -> END
```

### `backend/app/graph/nodes/refiner.py`

Refiner should support optional supplemental evidence.

Current inputs:

```text
old_report
user_instruction
```

New inputs:

```text
old_report
user_instruction
optional search_results
```

Prompt behavior:

```text
If supplemental evidence is provided, use it to add or revise factual content.
New factual claims must be grounded in the supplemental evidence and keep source/page citations.
If no supplemental evidence is provided, only perform text-level edits and do not invent new facts.
```

Implementation sketch:

```python
supplemental_evidence = "\n\n".join(state.get("search_results", []))
```

### `backend/app/graph/state.py`

Document the expanded intent values:

```python
intent: NotRequired[str | None]  # new_topic | edit_report | augment_report | refine
```

### `backend/app/api/routes.py`

No major change needed if `intent` is already accepted and copied into `initial_state`.

Expected request body:

```json
{
  "query": "...",
  "search_mode": "hybrid",
  "thread_id": "...",
  "intent": "new_topic | edit_report | augment_report"
}
```

## Frontend Changes

### `frontend/src/App.vue`

Replace the current two-action model with three explicit actions:

```text
Start New Analysis -> new_topic
Edit Wording       -> edit_report
Add Evidence       -> augment_report
```

Suggested UI:

```text
[ Start New Analysis ]
[ Edit Current Report ] [ Add Evidence / Conditions ]
```

Behavior:

- `new_topic`: clear the displayed report and start a fresh workflow.
- `edit_report`: require an existing report, keep current report visible until replacement is produced.
- `augment_report`: require an existing report, keep current report visible while new evidence is retrieved and merged.

### `frontend/src/services/api.js`

No major change needed if `createChatPayload()` already includes optional `intent`.

## Tests

### Router Tests

Add or extend `backend/tests/test_router_intent.py`:

- `intent == "new_topic"` routes to `planner`.
- `intent == "edit_report"` with an existing report routes to `refiner`.
- `intent == "augment_report"` with an existing report routes to `planner`.
- legacy `intent == "refine"` plus evidence-seeking instruction routes to `planner`.
- legacy `intent == "refine"` plus style-only instruction routes to `refiner`.

### Graph Routing Tests

Add tests for `route_after_research`:

- `intent == "augment_report"` routes to `refiner`.
- `intent == "new_topic"` routes to `writer`.
- `should_stop == True` routes to `END`.

### Refiner Tests

Add tests for evidence-aware refinement:

- When `search_results` exists, the refiner prompt includes supplemental evidence.
- When `search_results` is empty, the refiner prompt instructs the model not to invent new facts.

### Frontend Tests

Add or extend frontend tests:

- `createChatPayload()` supports `new_topic`.
- `createChatPayload()` supports `edit_report`.
- `createChatPayload()` supports `augment_report`.

## Implementation Order

1. Extend backend router intent handling.
2. Add `needs_research_for_refinement()` for backward-compatible `refine`.
3. Change `route_after_research()` so `augment_report` goes to `refiner`.
4. Update `refiner` prompt to accept optional `search_results`.
5. Update frontend buttons and intent mapping.
6. Add/extend tests.
7. Run backend unit tests and frontend build.

## Final Desired Behavior

```text
Completely new question
-> planner -> researcher -> writer -> reviewer

Edit old report
-> refiner

Supplement with new condition/evidence
-> planner -> researcher -> refiner
```

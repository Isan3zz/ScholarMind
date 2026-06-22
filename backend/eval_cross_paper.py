"""Cross-paper evaluation: run full pipeline + 5-dim LLM judge."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from app.graph.graph import create_graph
from app.rag.engine import get_available_papers
from app.utils.llm import get_llm

# ── Pick 2 papers for cross-paper comparison ──
papers = get_available_papers()
if len(papers) < 2:
    print("Need >= 2 papers in ES")
    sys.exit(1)

# Find SafeDecoding and the adversarial alignment paper
paper_a = paper_b = None
for p in papers:
    if "SafeDecoding" in p.get("title", ""):
        paper_a = p
    elif "adversarially" in p.get("title", "").lower():
        paper_b = p

if not paper_a or not paper_b:
    # Fallback: first two papers
    paper_a, paper_b = papers[0], papers[1]

print(f"Paper A: {paper_a['source']} — {paper_a.get('title','')[:80]}")
print(f"Paper B: {paper_b['source']} — {paper_b.get('title','')[:80]}")

# ── Cross-paper query ──
query = f"Compare the defense methods proposed in {paper_a['source']} and {paper_b['source']}. What are the key differences in their approaches?"

print(f"\nQuery: {query}\n")

# ── Run graph ──
from langgraph.checkpoint.memory import InMemorySaver

app = create_graph(memory=InMemorySaver())

initial_state = {
    "query": query,
    "revision_number": 0,
    "search_mode": "document",
    "should_stop": False,
    "intent": "new_topic",
}

print("=== Running graph ===\n")
config = {"configurable": {"thread_id": "eval-cross-paper"}}

final_state = None
for event in app.stream(initial_state, config=config):
    for node_name, state_update in event.items():
        print(f"[{node_name}]")
        if node_name == "planner":
            plans = state_update.get("plan", [])
            sources = state_update.get("plan_sources", [])
            for q, s in zip(plans, sources):
                print(f"  query: {q[:80]}")
                print(f"  sources: {s}")
        elif node_name == "researcher":
            n = len(state_update.get("search_results", []))
            print(f"  search_result blocks: {n}")
        elif node_name == "writer":
            report = state_update.get("final_report", "")[:300]
            print(f"  report preview: {report}...")
        elif node_name == "reviewer":
            status = state_update.get("review_status", "")
            critique = state_update.get("critique", "")
            print(f"  status: {status}")
            if critique:
                print(f"  critique: {critique[:150]}")
        print()

final_state = app.get_state(config)
if final_state:
    report = final_state.values.get("final_report", "")
else:
    # fallback: collect from events
    report = ""

print(f"=== Final report ({len(report)} chars) ===\n{report[:2000]}\n")

# ── LLM judge ──
if report:
    print("=== LLM Judge (5 dimensions) ===\n")
    judge_llm = get_llm(model_type="smart")

    judge_prompt = f"""You are an academic evaluation judge. Evaluate the following report generated for a cross-paper comparison query.

User query: {query}

Generated report:
{report[:4000]}

Rate on these 5 dimensions (0.0 to 1.0):
1. answer_correctness: Does the report correctly answer what was asked?
2. faithfulness: Is every factual claim grounded in the cited sources? (no hallucination)
3. key_point_coverage: Are the main differences between the two papers' methods covered?
4. citation_accuracy: Are citations in [source: file, section: Name] format accurate?
5. evidence_sufficiency: Does the report have enough evidence to support its claims?

Return JSON only:
{{"answer_correctness": 0.XX, "faithfulness": 0.XX, "key_point_coverage": 0.XX, "citation_accuracy": 0.XX, "evidence_sufficiency": 0.XX, "critique": "one sentence summary"}}
"""

    result = judge_llm.invoke(judge_prompt)
    print(result.content)

print("\nDone.")

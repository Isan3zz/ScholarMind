"""Cross-paper RAG evaluation — A/B: unified pool vs per-paper routing."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

from run_paper_rag_eval import (
    load_dataset, compute_section_hit, compute_keyword_recall,
    compute_section_hit_rate, compute_mean_mrr, compute_mean_keyword_recall,
    compute_refusal_rejection_rate, compute_mean_precision_recall_f1_at_k,
    evaluate_answers, evaluate_answers_with_judge, JUDGE_FIELDS,
    safe_console_text,
)
from app.rag.engine import hybrid_search

dataset_path = Path(__file__).with_name("cross_paper_dataset.jsonl")
samples = load_dataset(dataset_path)

top_k = 8

def hits_to_dicts(hits):
    return [{
        "chunk_id": h.chunk_id,
        "content": h.context_text,
        "section": h.section,
        "subsection": h.metadata.get("subsection", ""),
        "chunk_type": h.metadata.get("chunk_type", ""),
        "source": h.source,
        "score": h.score,
        "retriever": h.retriever,
    } for h in hits]

# ==============================
# A: Unified pool (sources=None)
# ==============================
results_a = [hits_to_dicts(hybrid_search(s["question"], top_k=top_k, fetch_k=60)) for s in samples]

# ==============================
# B: Explicit sources from dataset
# ==============================
results_b = []
for s in samples:
    sources = s.get("paper", [])
    if isinstance(sources, str):
        sources = [sources]
    if s.get("should_refuse") or len(sources) < 2:
        hits = hybrid_search(s["question"], top_k=top_k, fetch_k=60)
    else:
        hits = hybrid_search(s["question"], sources=sources, top_k=top_k, fetch_k=60)
    results_b.append(hits_to_dicts(hits))


def summarize(label, results):
    answerable = [(s, r) for s, r in zip(samples, results) if not s.get("should_refuse")]
    m = {
        f"section_hit@{k}": compute_section_hit_rate(samples, results, k=k)
        for k in [1, 3, 5, 8]
    }
    m["mrr"] = compute_mean_mrr(samples, results)
    prf = compute_mean_precision_recall_f1_at_k(samples, results, k=top_k)
    m[f"precision@{top_k}"] = prf["precision"]
    m[f"recall@{top_k}"] = prf["recall"]
    m[f"f1@{top_k}"] = prf["f1"]
    m[f"keyword_recall@{top_k}"] = compute_mean_keyword_recall(samples, results, k=top_k)
    m["refusal_rejection_rate"] = compute_refusal_rejection_rate(samples, results, min_score=0.5)
    diversities = [len({h.get("source", "?") for h in r[:top_k]}) for _, r in answerable]
    m[f"avg_source_diversity@{top_k}"] = sum(diversities) / len(diversities) if diversities else 0.0
    m["both_target_papers_hit"] = sum(1 for d in diversities if d >= 2) / len(diversities) if diversities else 0.0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for name, value in m.items():
        print(f"  {name:30s}: {value:.2f}")

def print_per_case(results_a, results_b, answers_a=None, answers_b=None,
                   judgments_a=None, judgments_b=None):
    print(f"\n{'='*100}")
    print("Per-case detail (A → B):")
    for i, (sample, ha, hb) in enumerate(zip(samples, results_a, results_b)):
        if sample.get("should_refuse"):
            continue
        srcs_a = len({h.get("source", "?") for h in ha[:top_k]})
        srcs_b = len({h.get("source", "?") for h in hb[:top_k]})
        kw_a = compute_keyword_recall(sample, ha, k=top_k)
        kw_b = compute_keyword_recall(sample, hb, k=top_k)
        print(f"  [{sample['id']}] papers: {srcs_a}→{srcs_b}  kw: {kw_a:.2f}→{kw_b:.2f}")
        if answers_a and answers_b and judgments_a and judgments_b:
            aa = answers_a[i]
            ab = answers_b[i]
            ja = judgments_a[i]
            jb = judgments_b[i]
            print(f"    answer A: {safe_console_text(aa[:200].replace(chr(10), ' '))}")
            print(f"    answer B: {safe_console_text(ab[:200].replace(chr(10), ' '))}")
            ja_str = ", ".join(f"{f}={ja.get(f, 0):.2f}" for f in JUDGE_FIELDS)
            jb_str = ", ".join(f"{f}={jb.get(f, 0):.2f}" for f in JUDGE_FIELDS)
            print(f"    judge A: {ja_str}")
            print(f"    judge B: {jb_str}")


def main():
    parser = argparse.ArgumentParser(description="Cross-paper RAG evaluation — A/B: unified pool vs per-paper routing.")
    parser.add_argument("--answer-eval", action="store_true", help="Generate answers and compute answer-level metrics.")
    parser.add_argument("--llm-judge", action="store_true", help="Use DeepSeek to judge answers on 5 dimensions.")
    parser.add_argument("--reject-threshold", type=float, default=0.5, help="Min score to treat hits as valid evidence.")
    args = parser.parse_args()

    # --- retrieval ---
    summarize("A: Unified pool (sources=None)", results_a)
    summarize("B: Explicit sources (sources=[...])", results_b)

    answers_a = answers_b = None
    judgments_a = judgments_b = None

    if args.answer_eval:
        from run_paper_rag_eval import generate_answer, compute_answer_metrics

        print("\n" + "=" * 60)
        print("  Answer Evaluation (A → B)")
        print("=" * 60)

        answers_a = [""] * len(samples)
        answers_b = [""] * len(samples)
        for idx, s in enumerate(samples):
            q = s["question"]
            tag = s["id"]
            if s.get("should_refuse"):
                answers_a[idx] = generate_answer(q, results_a[idx], reject_threshold=args.reject_threshold)
                answers_b[idx] = generate_answer(q, results_b[idx], reject_threshold=args.reject_threshold)
            else:
                print(f"  [{tag}] A...", end=" ", flush=True)
                answers_a[idx] = generate_answer(q, results_a[idx], reject_threshold=args.reject_threshold)
                print(f"B...", end=" ", flush=True)
                answers_b[idx] = generate_answer(q, results_b[idx], reject_threshold=args.reject_threshold)
                print("done")
        answer_metrics_a = compute_answer_metrics(samples, answers_a)
        answer_metrics_b = compute_answer_metrics(samples, answers_b)
        for name in sorted(answer_metrics_a):
            va = answer_metrics_a[name]
            vb = answer_metrics_b[name]
            print(f"  {name:30s}: {va:.2f} → {vb:.2f}")

        if args.llm_judge:
            from run_paper_rag_eval import judge_answer_with_llm, compute_judge_metrics

            print("\n" + "=" * 60)
            print("  DeepSeek Judge (A → B)")
            print("=" * 60)
            judgments_a = []
            judgments_b = []
            for idx, s in enumerate(samples):
                tag = s["id"]
                if s.get("should_refuse"):
                    judgments_a.append(judge_answer_with_llm(s, results_a[idx], answers_a[idx]))
                    judgments_b.append(judge_answer_with_llm(s, results_b[idx], answers_b[idx]))
                else:
                    print(f"  [{tag}] A...", end=" ", flush=True)
                    ja = judge_answer_with_llm(s, results_a[idx], answers_a[idx])
                    judgments_a.append(ja)
                    print(f"B...", end=" ", flush=True)
                    jb = judge_answer_with_llm(s, results_b[idx], answers_b[idx])
                    judgments_b.append(jb)
                    print("done")
            judge_metrics_a = compute_judge_metrics(judgments_a)
            judge_metrics_b = compute_judge_metrics(judgments_b)
            for name in sorted(judge_metrics_a):
                va = judge_metrics_a[name]
                vb = judge_metrics_b[name]
                print(f"  {name:30s}: {va:.2f} → {vb:.2f}")

    print_per_case(results_a, results_b, answers_a, answers_b, judgments_a, judgments_b)


if __name__ == "__main__":
    main()

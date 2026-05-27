import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from run_paper_rag_eval import (
    compute_answer_metrics,
    compute_judge_metrics,
    compute_keyword_recall,
    compute_mean_mrr,
    compute_precision_recall_f1_at_k,
    compute_refusal_rejection_rate,
    compute_section_hit,
    compute_section_hit_rate,
    deepseek_judge_config,
    evaluate_with_search_fn,
    formal_agent_search,
    judge_answer_with_llm,
    lexical_search,
    load_dataset,
    planned_retrieval_queries,
    rerank_eval_hits,
    safe_console_text,
)


class PaperRagMetricsTest(unittest.TestCase):
    def test_compute_section_hit(self):
        sample = {"expected_sections": ["Method"]}
        hits = [{"section": "Introduction"}, {"section": "Method"}]

        self.assertTrue(compute_section_hit(sample, hits, k=2))
        self.assertFalse(compute_section_hit(sample, hits, k=1))

    def test_compute_section_hit_matches_subsection_and_combined_label(self):
        subsection_sample = {"expected_sections": ["Experimental Setup"]}
        combined_sample = {"expected_sections": ["Experiments / Experimental Setup"]}
        hits = [{"section": "Experiments", "subsection": "Experimental Setup"}]

        self.assertTrue(compute_section_hit(subsection_sample, hits, k=1))
        self.assertTrue(compute_section_hit(combined_sample, hits, k=1))

    def test_compute_precision_recall_f1_at_k(self):
        sample = {
            "expected_sections": ["Experiments", "Experimental Setup", "A.1 Attack Setup"]
        }
        hits = [
            {"section": "Experiments", "subsection": "Experimental Setup"},
            {"section": "Introduction", "subsection": ""},
            {"section": "A Detailed Experimental Setups", "subsection": "A.1 Attack Setup"},
        ]

        metrics = compute_precision_recall_f1_at_k(sample, hits, k=3)

        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)

    def test_compute_mrr(self):
        sample = {"expected_sections": ["Method"]}
        hits = [{"section": "Introduction"}, {"section": "Method"}]

        self.assertEqual(compute_mean_mrr([sample], [hits]), 0.5)

    def test_compute_section_hit_rate(self):
        samples = [
            {"expected_sections": ["Method"]},
            {"expected_sections": ["Experiments"]},
        ]
        results = [
            [{"section": "Method"}],
            [{"section": "Introduction"}],
        ]

        self.assertEqual(compute_section_hit_rate(samples, results, k=1), 0.5)

    def test_compute_keyword_recall(self):
        sample = {"answer_keywords": ["SafeDecoding", "jailbreak attacks", "HEx-PHI", "missing"]}
        hits = [
            {
                "content": "SafeDecoding defends against jailbreak attacks on HEx PHI.",
                "section": "Introduction",
            }
        ]

        self.assertAlmostEqual(compute_keyword_recall(sample, hits, k=1), 3 / 4)

    def test_compute_refusal_rejection_rate(self):
        samples = [
            {"should_refuse": True},
            {"should_refuse": True},
            {"should_refuse": False},
        ]
        results = [
            [],
            [{"score": 0.1}],
            [{"score": 1.0}],
        ]

        self.assertEqual(compute_refusal_rejection_rate(samples, results, min_score=0.5), 1.0)

    def test_compute_answer_metrics_checks_citations_keywords_and_refusals(self):
        samples = [
            {
                "answer_keywords": ["SafeDecoding", "jailbreak attacks"],
                "should_refuse": False,
            },
            {
                "answer_keywords": [],
                "should_refuse": True,
            },
        ]
        answers = [
            "SafeDecoding defends against jailbreak attacks [source: paper.pdf, section: Introduction].",
            "The provided paper does not contain evidence about that topic.",
        ]

        metrics = compute_answer_metrics(samples, answers)

        self.assertEqual(metrics["answer_keyword_recall"], 1.0)
        self.assertEqual(metrics["citation_format_rate"], 1.0)
        self.assertEqual(metrics["refusal_answer_rate"], 1.0)

    def test_compute_judge_metrics_averages_scores(self):
        judgments = [
            {
                "answer_correctness": 1,
                "faithfulness": 1,
                "key_point_coverage": 0.5,
                "citation_accuracy": 1,
                "evidence_sufficiency": 1,
            },
            {
                "answer_correctness": 0,
                "faithfulness": 0.5,
                "key_point_coverage": 1,
                "citation_accuracy": 0,
                "evidence_sufficiency": 0,
            },
        ]

        metrics = compute_judge_metrics(judgments)

        self.assertEqual(metrics["judge_answer_correctness"], 0.5)
        self.assertEqual(metrics["judge_faithfulness"], 0.75)
        self.assertEqual(metrics["judge_key_point_coverage"], 0.75)
        self.assertEqual(metrics["judge_citation_accuracy"], 0.5)
        self.assertEqual(metrics["judge_evidence_sufficiency"], 0.5)

    def test_judge_answer_scores_correct_refusal_without_llm(self):
        judgment = judge_answer_with_llm(
            {"should_refuse": True},
            [],
            "The provided paper does not contain sufficient evidence to answer this question.",
        )

        self.assertEqual(judgment["answer_correctness"], 1.0)
        self.assertEqual(judgment["faithfulness"], 1.0)
        self.assertEqual(judgment["evidence_sufficiency"], 1.0)

    @patch("dotenv.load_dotenv")
    @patch.dict(
        "os.environ",
        {
            "DEEPSEEK_API_KEY": "sk-test",
            "DEEPSEEK_API_BASE": "https://api.deepseek.test",
            "DEEPSEEK_JUDGE_MODEL": "deepseek-test",
        },
        clear=False,
    )
    def test_deepseek_judge_config_reads_string_env(self, _load_dotenv):
        config = deepseek_judge_config()

        self.assertEqual(config["api_key"], "sk-test")
        self.assertEqual(config["base_url"], "https://api.deepseek.test")
        self.assertEqual(config["model"], "deepseek-test")

    @patch("dotenv.load_dotenv")
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-openai-compatible",
            "OPENAI_API_BASE": "https://api.compatible.test",
            "SMART_LLM_MODEL": "compatible-chat",
        },
        clear=True,
    )
    def test_deepseek_judge_config_can_fallback_to_openai_compatible_env(self, _load_dotenv):
        config = deepseek_judge_config()

        self.assertEqual(config["api_key"], "sk-openai-compatible")
        self.assertEqual(config["base_url"], "https://api.compatible.test")
        self.assertEqual(config["model"], "compatible-chat")

    def test_lexical_search_ranks_matching_section_first(self):
        docs = [
            {
                "content": "The paper introduces the core method for safety-aware decoding.",
                "section": "Method",
            },
            {
                "content": "The experiments compare SafeDecoding with baseline defenses.",
                "section": "Experiments",
            },
        ]

        hits = lexical_search("What baseline defenses are used in experiments?", docs, top_k=1)

        self.assertEqual(hits[0]["section"], "Experiments")

    def test_rerank_eval_hits_uses_reranker_order(self):
        hits = [
            {
                "chunk_id": "a",
                "content": "weak candidate",
                "source": "paper.pdf",
                "section": "Introduction",
                "score": 0.8,
            },
            {
                "chunk_id": "b",
                "content": "strong candidate",
                "source": "paper.pdf",
                "section": "Method",
                "score": 0.1,
            },
        ]

        def fake_reranker(query, candidates, top_k):
            return [candidates[1], candidates[0]][:top_k]

        reranked = rerank_eval_hits("query", hits, top_k=2, rerank_fn=fake_reranker)

        self.assertEqual([hit["chunk_id"] for hit in reranked], ["b", "a"])

    def test_evaluate_with_search_fn_uses_supplied_formal_search(self):
        samples = [
            {
                "question": "method?",
                "expected_sections": ["Method"],
                "answer_keywords": ["expert"],
            }
        ]

        def fake_search(query, top_k, fetch_k):
            return [
                {
                    "section": "Method",
                    "content": "The expert model is used.",
                    "score": 0.9,
                }
            ]

        results, metrics = evaluate_with_search_fn(
            samples,
            fake_search,
            top_k=1,
            fetch_k=3,
            reject_threshold=0.5,
        )

        self.assertEqual(results[0][0]["section"], "Method")
        self.assertEqual(metrics["section_hit@1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_planned_retrieval_queries_include_original_and_unique_rewrites(self):
        queries = planned_retrieval_queries(
            "SafeDecoding 方法是什么？",
            [
                "SafeDecoding method architecture",
                "SafeDecoding method architecture",
                "experimental results benchmark",
            ],
        )

        self.assertEqual(
            queries,
            [
                "SafeDecoding 方法是什么？",
                "SafeDecoding method architecture",
                "experimental results benchmark",
            ],
        )

    def test_planned_retrieval_queries_can_limit_rewrites(self):
        queries = planned_retrieval_queries(
            "SafeDecoding 方法是什么？",
            [
                "SafeDecoding method architecture",
                "SafeDecoding expert model",
                "SafeDecoding experimental setup",
            ],
            max_rewrites=1,
        )

        self.assertEqual(
            queries,
            [
                "SafeDecoding 方法是什么？",
                "SafeDecoding method architecture",
            ],
        )

    def test_formal_agent_search_retrieves_rewrites_then_globally_reranks(self):
        called_queries = []
        rerank_query = []

        def fake_planner(query):
            self.assertEqual(query, "SafeDecoding 方法是什么？")
            return ["SafeDecoding method architecture", "SafeDecoding expert model"]

        def fake_candidate_search(query, fetch_k):
            called_queries.append(query)
            return [
                {
                    "chunk_id": "shared",
                    "section": "Introduction",
                    "content": "duplicate candidate",
                    "score": 0.8,
                },
                {
                    "chunk_id": query,
                    "section": "Method",
                    "content": query,
                    "score": 0.9,
                },
            ]

        def fake_rerank(query, candidates, top_k):
            rerank_query.append(query)
            return [
                hit
                for hit in candidates
                if hit["chunk_id"] != "shared"
            ][:top_k]

        hits = formal_agent_search(
            "SafeDecoding 方法是什么？",
            top_k=5,
            fetch_k=20,
            planner_fn=fake_planner,
            candidate_search_fn=fake_candidate_search,
            rerank_fn=fake_rerank,
        )

        self.assertEqual(
            called_queries,
            [
                "SafeDecoding 方法是什么？",
                "SafeDecoding method architecture",
                "SafeDecoding expert model",
            ],
        )
        self.assertEqual(rerank_query, ["SafeDecoding 方法是什么？"])
        self.assertEqual(
            [hit["chunk_id"] for hit in hits],
            [
                "SafeDecoding 方法是什么？",
                "SafeDecoding method architecture",
                "SafeDecoding expert model",
            ],
        )

    def test_dataset_contains_safedecoding_eval_cases(self):
        samples = load_dataset()
        ids = {sample["id"] for sample in samples}

        self.assertIn("safedecoding_summary", ids)
        self.assertIn("safedecoding_method_sample_space", ids)
        self.assertIn("safedecoding_experimental_setup", ids)
        self.assertIn("safedecoding_limitations", ids)
        self.assertIn("safedecoding_irrelevant_mars", ids)

    def test_load_dataset_accepts_custom_path(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "custom.jsonl"
            path.write_text('{"id": "custom_case", "question": "q"}\n', encoding="utf-8")

            samples = load_dataset(path)

        self.assertEqual(samples, [{"id": "custom_case", "question": "q"}])

    def test_safe_console_text_escapes_gbk_incompatible_characters(self):
        text = safe_console_text("non‑breaking")

        self.assertIn("\\u2011", text)


if __name__ == "__main__":
    unittest.main()

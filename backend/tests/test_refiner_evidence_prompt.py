import unittest
from unittest.mock import Mock, patch

from app.graph.nodes.refiner import build_edit_prompt, build_augment_prompt, refine_node


class RefinerEvidencePromptTest(unittest.TestCase):
    # ── build_augment_prompt ────────────────────────────────────────────

    def test_augment_prompt_includes_supplemental_evidence(self):
        prompt = build_augment_prompt({
            "query": "Add experiment evidence",
            "final_report": "Old report",
            "search_results": ["[source: paper.pdf, section: Experiments]\nAccuracy improves."],
            "short_memory": {
                "topic": "SafeDecoding",
                "report_summary": "Report summary",
                "change_log": [],
                "last_intent": "new_topic",
            },
        })

        self.assertIn("Supplemental evidence", prompt)
        self.assertIn("Accuracy improves.", prompt)
        self.assertIn("source/section citations", prompt)
        self.assertIn("Old report", prompt)

    def test_augment_prompt_handles_empty_evidence(self):
        prompt = build_augment_prompt({
            "query": "Add something",
            "final_report": "Old report",
            "search_results": [],
        })

        self.assertIn("No supplemental evidence was retrieved", prompt)
        self.assertIn("edit-only", prompt)

    # ── build_edit_prompt ───────────────────────────────────────────────

    def test_edit_prompt_forbids_new_facts(self):
        prompt = build_edit_prompt({
            "query": "Make the wording shorter",
            "final_report": "Old report",
            "short_memory": None,
        })

        self.assertIn("text-level edit only", prompt)
        self.assertIn("Do NOT add new factual claims", prompt)
        self.assertNotIn("Supplemental evidence", prompt)

    def test_edit_prompt_includes_short_memory_context(self):
        prompt = build_edit_prompt({
            "query": "Make the method section shorter",
            "final_report": "Full report",
            "short_memory": {
                "topic": "SafeDecoding",
                "report_summary": "Report summary",
                "change_log": ["Added baseline details."],
                "last_intent": "new_topic",
            },
        })

        self.assertIn("Short-term memory", prompt)
        self.assertIn("Current topic: SafeDecoding", prompt)
        self.assertIn("Current report summary: Report summary", prompt)
        self.assertIn("Recent report changes:", prompt)
        self.assertIn("Original report:", prompt)
        self.assertIn("Full report", prompt)

    # ── refine_node dispatch ────────────────────────────────────────────

    def test_refiner_dispatches_to_augment_prompt(self):
        llm = Mock()
        structured_llm = Mock()
        structured_llm.invoke.return_value.final_report = "Updated report"
        structured_llm.invoke.return_value.memory_event = "Added experiment evidence to the current report."
        llm.with_structured_output.return_value = structured_llm
        state = {
            "query": "Add experiment evidence",
            "final_report": "Old report",
            "intent": "augment_report",
            "search_results": ["[source: paper.pdf, section: Experiments]\nAccuracy improves."],
        }

        with patch("app.graph.nodes.refiner.llm", llm):
            result = refine_node(state)

        prompt = structured_llm.invoke.call_args.args[0][0].content
        self.assertIn("Supplemental evidence", prompt)
        self.assertIn("Accuracy improves.", prompt)
        self.assertEqual(result["final_report"], "Updated report")
        self.assertEqual(result["memory_event"], "Added experiment evidence to the current report.")

    def test_refiner_dispatches_to_edit_prompt_by_default(self):
        llm = Mock()
        structured_llm = Mock()
        structured_llm.invoke.return_value.final_report = "Edited report"
        structured_llm.invoke.return_value.memory_event = "Shortened the wording."
        llm.with_structured_output.return_value = structured_llm
        state = {
            "query": "Make the wording shorter",
            "final_report": "Old report",
            # no intent set → should use edit_prompt
            "search_results": [],
        }

        with patch("app.graph.nodes.refiner.llm", llm):
            refine_node(state)

        prompt = structured_llm.invoke.call_args.args[0][0].content
        self.assertIn("text-level edit only", prompt)
        self.assertIn("Do NOT add new factual claims", prompt)

    def test_refiner_uses_structured_memory_event(self):
        llm = Mock()
        structured_llm = Mock()
        structured_llm.invoke.return_value.final_report = "Shorter revised report"
        structured_llm.invoke.return_value.memory_event = "Deleted repeated explanation from the conclusion."
        llm.with_structured_output.return_value = structured_llm
        state = {
            "query": "Make the conclusion shorter",
            "final_report": "Long report",
            "intent": "edit_report",
            "search_results": [],
            "short_memory": {
                "topic": "SafeDecoding",
                "report_summary": "生成了 SafeDecoding 报告。",
                "change_log": [],
                "last_intent": "new_topic",
            },
        }

        with patch("app.graph.nodes.refiner.llm", llm):
            result = refine_node(state)

        self.assertEqual(result["final_report"], "Shorter revised report")
        self.assertEqual(result["memory_event"], "Deleted repeated explanation from the conclusion.")

    def test_refiner_falls_back_when_structured_output_fails(self):
        llm = Mock()
        llm.with_structured_output.side_effect = RuntimeError("structured output failed")
        llm.invoke.return_value.content = "Shorter revised report"
        state = {
            "query": "Make the conclusion shorter",
            "final_report": "Long report",
            "intent": "edit_report",
            "search_results": [],
        }

        with patch("app.graph.nodes.refiner.llm", llm):
            result = refine_node(state)

        self.assertEqual(result["final_report"], "Shorter revised report")
        self.assertIn("Make the conclusion shorter", result["memory_event"])
        llm.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()

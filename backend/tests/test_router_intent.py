import unittest
from unittest.mock import Mock, patch

from app.graph.nodes.router import needs_research_for_refinement, route_query


class RouterIntentTest(unittest.TestCase):
    def test_new_topic_routes_to_planner_without_llm(self):
        state = {
            "query": "Analyze a new paper",
            "final_report": "Existing report",
            "intent": "new_topic",
        }
        router_llm = Mock()

        with patch("app.graph.nodes.router.router_llm", router_llm):
            route = route_query(state)

        self.assertEqual(route, "planner")
        router_llm.invoke.assert_not_called()

    def test_edit_report_routes_to_refiner_without_llm(self):
        state = {
            "query": "Make the conclusion more concise",
            "final_report": "Existing report",
            "intent": "edit_report",
        }
        router_llm = Mock()

        with patch("app.graph.nodes.router.router_llm", router_llm):
            route = route_query(state)

        self.assertEqual(route, "refiner")
        router_llm.invoke.assert_not_called()

    def test_augment_report_routes_to_planner_without_llm(self):
        state = {
            "query": "Add benchmark evidence and citations",
            "final_report": "Existing report",
            "intent": "augment_report",
        }
        router_llm = Mock()

        with patch("app.graph.nodes.router.router_llm", router_llm):
            route = route_query(state)

        self.assertEqual(route, "planner")
        router_llm.invoke.assert_not_called()

    def test_legacy_refine_routes_to_planner_when_instruction_needs_evidence(self):
        state = {
            "query": "\u8865\u5145\u5b9e\u9a8c\u5bf9\u6bd4\u8bc1\u636e\u548c\u5f15\u7528",
            "final_report": "Existing report",
            "intent": "refine",
        }
        router_llm = Mock()

        with patch("app.graph.nodes.router.router_llm", router_llm):
            route = route_query(state)

        self.assertEqual(route, "planner")
        router_llm.invoke.assert_not_called()

    def test_legacy_refine_routes_to_refiner_for_style_only_edit(self):
        state = {
            "query": "Make the wording more concise",
            "final_report": "Existing report",
            "intent": "refine",
        }
        router_llm = Mock()

        with patch("app.graph.nodes.router.router_llm", router_llm):
            route = route_query(state)

        self.assertEqual(route, "refiner")
        router_llm.invoke.assert_not_called()

    def test_needs_research_for_refinement_detects_evidence_requests(self):
        self.assertTrue(needs_research_for_refinement("add related work citations"))
        self.assertTrue(needs_research_for_refinement("\u8865\u5145\u5b9e\u9a8c\u5bf9\u6bd4\u8bc1\u636e"))
        self.assertFalse(needs_research_for_refinement("make the wording shorter"))

    def test_explicit_new_topic_overrides_short_memory(self):
        state = {
            "query": "Analyze a different paper",
            "intent": "new_topic",
            "final_report": "Existing report",
            "short_memory": {
                "topic": "SafeDecoding",
                "report_summary": "生成了 SafeDecoding 报告。",
                "change_log": ["补充了实验基线。"],
                "last_intent": "edit_report",
            },
        }

        self.assertEqual(route_query(state), "planner")

    def test_memory_topic_allows_followup_refinement_detection(self):
        state = {
            "query": "Make this more concise",
            "final_report": "Existing report",
            "short_memory": {
                "topic": "SafeDecoding",
                "report_summary": "生成了 SafeDecoding 报告。",
                "change_log": ["补充了实验基线。"],
                "last_intent": "new_topic",
            },
        }

        self.assertEqual(route_query(state), "refiner")


if __name__ == "__main__":
    unittest.main()

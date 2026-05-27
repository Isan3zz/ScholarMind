import unittest

from app.graph.nodes.planner import PLAN_PROMPT


class PlannerPromptPolicyTest(unittest.TestCase):
    def test_planner_prompt_targets_paper_vector_retrieval(self):
        prompt = PLAN_PROMPT.format(
            query="What is the method innovation in this paper?",
            critique="Missing experimental comparison evidence",
            short_memory_context="No short-term memory available.",
        )

        self.assertIn("paper vector database", prompt)
        self.assertIn("English", prompt)
        self.assertIn("Preserve proper nouns", prompt)
        self.assertIn("3-5 English retrieval queries", prompt)
        self.assertIn("Method", prompt)
        self.assertIn("Experiment", prompt)
        self.assertNotIn("comma-separated", prompt)
        self.assertNotIn("Google", prompt)

    def test_planner_prompt_makes_baseline_queries_specific_to_experimental_setup(self):
        prompt = PLAN_PROMPT.format(
            query="SafeDecoding compare against which baseline defenses?",
            critique="",
            short_memory_context="No short-term memory available.",
        )

        self.assertIn("baseline", prompt.lower())
        self.assertIn("Experimental Setup", prompt)
        self.assertIn("Baseline Setup", prompt)
        self.assertIn("baseline names", prompt)
        self.assertIn("first retrieval query", prompt)

    def test_planner_prompt_accepts_compact_short_memory(self):
        prompt = PLAN_PROMPT.format(
            query="Add baseline details",
            critique="",
            short_memory_context=(
                "Current topic: SafeDecoding\n"
                "Current report summary: Existing report\n"
                "Recent report changes:\n"
                "- Added baseline details"
            ),
        )

        self.assertIn("Short-term memory", prompt)
        self.assertIn("Current topic: SafeDecoding", prompt)
        self.assertIn("Current report summary: Existing report", prompt)
        self.assertIn("Recent report changes:", prompt)


if __name__ == "__main__":
    unittest.main()

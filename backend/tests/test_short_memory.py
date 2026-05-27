import unittest

from app.graph.nodes.memory import (
    empty_short_memory,
    format_short_memory_for_prompt,
    update_short_memory,
)


class ShortMemoryTest(unittest.TestCase):
    def test_empty_short_memory_has_expected_keys(self):
        memory = empty_short_memory()

        self.assertEqual(memory["topic"], "")
        self.assertEqual(memory["report_summary"], "")
        self.assertEqual(memory["change_log"], [])
        self.assertEqual(memory["last_intent"], "")
        self.assertNotIn("evidence_summary", memory)
        self.assertNotIn("last_instruction", memory)

    def test_new_topic_resets_previous_memory_and_summarizes_creation(self):
        state = {
            "query": "Analyze SafeDecoding",
            "intent": "new_topic",
            "final_report": "This report explains SafeDecoding motivation, method, experiments, and limitations.",
            "memory_event": "生成了 SafeDecoding 论文分析报告，主要包括动机、方法、实验和局限性。",
            "short_memory": {
                "topic": "old topic",
                "report_summary": "old report",
                "change_log": ["old change"],
                "last_intent": "edit_report",
            },
        }

        result = update_short_memory(state)
        memory = result["short_memory"]

        self.assertEqual(memory["topic"], "Analyze SafeDecoding")
        self.assertEqual(memory["report_summary"], "生成了 SafeDecoding 论文分析报告，主要包括动机、方法、实验和局限性。")
        self.assertEqual(memory["change_log"], [])
        self.assertEqual(memory["last_intent"], "new_topic")

    def test_refiner_update_keeps_current_summary_and_appends_change(self):
        state = {
            "query": "Make the conclusion shorter",
            "intent": "edit_report",
            "final_report": "Revised report",
            "memory_event": "删除了结论中的重复解释，并保留核心结论。",
            "short_memory": {
                "topic": "Analyze SafeDecoding",
                "report_summary": "生成了 SafeDecoding 报告，主要说明方法和实验。",
                "change_log": ["补充了实验基线说明。"],
                "last_intent": "new_topic",
            },
        }

        result = update_short_memory(state)
        memory = result["short_memory"]

        self.assertEqual(memory["topic"], "Analyze SafeDecoding")
        self.assertEqual(memory["report_summary"], "生成了 SafeDecoding 报告，主要说明方法和实验。")
        self.assertEqual(memory["change_log"], ["补充了实验基线说明。", "删除了结论中的重复解释，并保留核心结论。"])
        self.assertEqual(memory["last_intent"], "edit_report")

    def test_change_log_is_bounded_to_recent_items(self):
        state = {
            "query": "Polish wording",
            "intent": "edit_report",
            "memory_event": "调整了表达。",
            "short_memory": {
                "topic": "Analyze SafeDecoding",
                "report_summary": "Generated report.",
                "change_log": ["change 1", "change 2", "change 3", "change 4", "change 5"],
                "last_intent": "edit_report",
            },
        }

        result = update_short_memory(state)

        self.assertEqual(result["short_memory"]["change_log"], ["change 2", "change 3", "change 4", "change 5", "调整了表达。"])

    def test_format_short_memory_for_prompt_includes_report_and_changes(self):
        text = format_short_memory_for_prompt({
            "topic": "SafeDecoding",
            "report_summary": "生成了 SafeDecoding 报告，主要包括方法和实验。",
            "change_log": ["补充了基线对比。", "删除了冗长背景。"],
            "last_intent": "edit_report",
        })

        self.assertIn("Current topic: SafeDecoding", text)
        self.assertIn("Current report summary: 生成了 SafeDecoding 报告", text)
        self.assertIn("Recent report changes:", text)
        self.assertIn("- 补充了基线对比。", text)
        self.assertIn("- 删除了冗长背景。", text)


if __name__ == "__main__":
    unittest.main()

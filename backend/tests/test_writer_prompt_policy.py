import unittest
from unittest.mock import Mock, patch

from app.graph.nodes.writer import WRITE_PROMPT, write_node


class WriterPromptPolicyTest(unittest.TestCase):
    def test_writer_prompt_requires_source_section_citations(self):
        prompt = WRITE_PROMPT.format(
            query="Summarize the paper method.",
            content="[source: iris.pdf, section: Method]\nmethod text",
            critique_section="",
        )

        self.assertIn("[source:", prompt)
        self.assertIn("section", prompt)
        self.assertIn("[source: file, section: SectionName]", prompt)
        self.assertNotIn("p. page", prompt)
        self.assertIn("do not fabricate", prompt)

    def test_writer_uses_structured_memory_event(self):
        llm = Mock()
        structured_llm = Mock()
        structured_llm.invoke.return_value.final_report = "Report about SafeDecoding methods and experiments."
        structured_llm.invoke.return_value.memory_event = "Generated a SafeDecoding report covering methods and experiments."
        llm.with_structured_output.return_value = structured_llm
        state = {
            "query": "Analyze SafeDecoding",
            "search_results": ["[source: paper.pdf, section: Method]\nSafeDecoding changes decoding."],
            "critique": "",
        }

        with patch("app.graph.nodes.writer.llm", llm):
            result = write_node(state)

        self.assertEqual(result["final_report"], "Report about SafeDecoding methods and experiments.")
        self.assertEqual(result["memory_event"], "Generated a SafeDecoding report covering methods and experiments.")
        llm.with_structured_output.assert_called_once()
        structured_llm.invoke.assert_called_once()

    def test_writer_falls_back_when_structured_output_fails(self):
        llm = Mock()
        llm.with_structured_output.side_effect = RuntimeError("structured output failed")
        llm.invoke.return_value.content = "Report about SafeDecoding methods and experiments."
        state = {
            "query": "Analyze SafeDecoding",
            "search_results": ["[source: paper.pdf, section: Method]\nSafeDecoding changes decoding."],
            "critique": "",
        }

        with patch("app.graph.nodes.writer.llm", llm):
            result = write_node(state)

        self.assertEqual(result["final_report"], "Report about SafeDecoding methods and experiments.")
        self.assertIn("Analyze SafeDecoding", result["memory_event"])
        llm.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()

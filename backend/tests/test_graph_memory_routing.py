import unittest
from unittest.mock import patch

from app.graph.graph import create_graph


class GraphMemoryRoutingTest(unittest.TestCase):
    def test_graph_registers_memory_update_node(self):
        app = create_graph(memory=None)
        graph = app.get_graph()

        node_names = {node.id for node in graph.nodes.values()}

        self.assertIn("memory", node_names)

    def test_memory_node_uses_update_short_memory(self):
        state = {
            "query": "Make it shorter",
            "intent": "edit_report",
            "final_report": "Revised",
        }

        with patch("app.graph.graph.update_short_memory") as fake_update:
            fake_update.return_value = {"short_memory": {"topic": "x"}}
            from app.graph.graph import memory_node

            result = memory_node(state)

        fake_update.assert_called_once_with(state)
        self.assertEqual(result, {"short_memory": {"topic": "x"}})


if __name__ == "__main__":
    unittest.main()

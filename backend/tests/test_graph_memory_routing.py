import unittest

from app.graph.graph import create_graph


class GraphMemoryRoutingTest(unittest.TestCase):
    def test_memory_is_not_a_graph_node(self):
        """short_memory 是后处理，不在 graph 拓扑中。"""
        app = create_graph(memory=None)
        graph = app.get_graph()

        node_names = {node.id for node in graph.nodes.values()}

        self.assertNotIn("memory", node_names)
        self.assertIn("reviewer", node_names)
        self.assertIn("refiner", node_names)

    def test_reviewer_routes_directly_to_end(self):
        """reviewer 的 should_continue 直接返回 END，不再经过 memory 节点。"""
        from app.graph.graph import should_continue

        route = should_continue({"review_status": "PASS", "revision_number": 0})
        from langgraph.graph import END

        self.assertEqual(route, END)

    def test_refiner_routes_to_end_for_edit_report(self):
        """refiner 编辑路径直接返回 END，不再经过 memory 节点。"""
        from app.graph.graph import route_after_refiner
        from langgraph.graph import END

        route = route_after_refiner({"intent": "edit_report"})
        self.assertEqual(route, END)


if __name__ == "__main__":
    unittest.main()

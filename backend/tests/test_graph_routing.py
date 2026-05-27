import unittest

from langgraph.graph import END

from app.graph.graph import route_after_research


class GraphRoutingTest(unittest.TestCase):
    def test_route_after_research_sends_augment_report_to_refiner(self):
        route = route_after_research({"intent": "augment_report", "should_stop": False})

        self.assertEqual(route, "refiner")

    def test_route_after_research_sends_new_topic_to_writer(self):
        route = route_after_research({"intent": "new_topic", "should_stop": False})

        self.assertEqual(route, "writer")

    def test_route_after_research_honors_should_stop(self):
        route = route_after_research({"intent": "augment_report", "should_stop": True})

        self.assertEqual(route, END)


if __name__ == "__main__":
    unittest.main()

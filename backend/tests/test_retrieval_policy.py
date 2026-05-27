import unittest

from app.rag.retrieval import infer_paper_intent, sections_for_intent


class RetrievalPolicyTest(unittest.TestCase):
    def test_method_question_routes_to_method_sections(self):
        intent = infer_paper_intent("这篇论文的方法是什么？模型结构怎么设计？")
        sections = sections_for_intent(intent)

        self.assertEqual(intent, "method")
        self.assertIn("Method", sections)

    def test_experiment_question_routes_to_experiment_sections(self):
        intent = infer_paper_intent("实验结果怎么样？用了哪些数据集和指标？")
        sections = sections_for_intent(intent)

        self.assertEqual(intent, "experiments")
        self.assertIn("Experiments", sections)
        self.assertIn("Results", sections)


if __name__ == "__main__":
    unittest.main()

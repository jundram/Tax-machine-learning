"""Tests for evidence retrieval and narrative generation."""

import unittest

from research_agent import TaxResearchAgent


class TaxResearchAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = TaxResearchAgent()

    def test_work_expense_driver_retrieves_ato_guidance(self):
        sources = self.agent.retrieve_sources(
            [{"feature": "wre_to_salary_ratio", "value": 0.8}]
        )
        self.assertEqual(sources[0]["title"], "ATO: Work-related expenses")
        self.assertTrue(sources[0]["url"].startswith("https://www.ato.gov.au/"))

    def test_review_explains_flag_without_compliance_conclusion(self):
        review = self.agent.compose_review(
            {
                "segment_name": "S2: Wage-dominant (mid income)",
                "anomaly_score": 0.6242,
                "threshold": 0.5640,
                "flagged_for_review": True,
                "main_drivers": [
                    {
                        "feature": "wre_to_salary_ratio",
                        "value": 0.8,
                        "contribution_towards_anomaly": 1.29,
                    },
                    {
                        "feature": "log_WRE_trvl_amt",
                        "value": 18_000.0,
                        "contribution_towards_anomaly": 1.12,
                    },
                ],
            }
        )
        self.assertIn("flagged for statistical review", review["explanation"])
        self.assertIn("do not establish", review["explanation"])
        self.assertGreater(len(review["review_questions"]), 0)
        self.assertGreater(len(review["sources"]), 0)

    def test_chat_explains_flag_from_model_context(self):
        response = self.agent.answer_question(
            "Why was this taxpayer flagged?",
            {
                "segment_name": "S2: Wage-dominant (mid income)",
                "anomaly_score": 0.6242,
                "threshold": 0.5640,
                "flagged_for_review": True,
                "main_drivers": [
                    {
                        "feature": "wre_to_salary_ratio",
                        "value": 0.8,
                        "contribution_towards_anomaly": 1.29,
                    }
                ],
            },
        )
        self.assertIn("0.6242", response["answer"])
        self.assertIn("Work-related expenses", response["sources"][0]["title"])

    def test_chat_rejects_fraud_conclusion(self):
        response = self.agent.answer_question("Does this mean fraud?")
        self.assertIn("does not establish fraud", response["answer"])

    def test_chat_retrieves_rental_guidance(self):
        response = self.agent.answer_question(
            "What should I check for a large rental deduction?"
        )
        self.assertEqual(response["sources"][0]["title"], "ATO: Rental expenses to claim")


if __name__ == "__main__":
    unittest.main()

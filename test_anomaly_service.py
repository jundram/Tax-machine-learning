"""Smoke tests for the reusable anomaly scoring service."""

import math
import unittest

from anomaly_service import TaxpayerAnomalyService
from record_builder import build_record


TYPICAL_WAGE_EARNER = build_record(
    {
        "salary": 75_000, "work_related_expenses": 3_500, "investment_income": 500,
        "gifts": 200, "tax_affairs": 300, "super_balance": 95_000,
    }
)

LARGE_WRE_CLAIM = build_record(
    {
        "salary": 75_000, "work_related_expenses": 40_000, "investment_income": 500,
        "gifts": 200, "tax_affairs": 300, "super_balance": 95_000,
    }
)

STACKED_DEDUCTIONS = build_record(
    {
        "salary": 75_000, "work_related_expenses": 45_000, "investment_income": 500,
        "gifts": 6_000, "personal_super": 15_000, "tax_affairs": 2_500, "super_balance": 95_000,
    }
)


class TaxpayerAnomalyServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = TaxpayerAnomalyService()

    def test_score_typical_record(self):
        result = self.service.score_record(TYPICAL_WAGE_EARNER)

        self.assertIn(result["segment_id"], self.service.segment_models)
        self.assertTrue(math.isfinite(result["anomaly_score"]))
        self.assertTrue(math.isfinite(result["threshold"]))
        self.assertIsInstance(result["flagged_for_review"], bool)
        self.assertEqual(len(result["main_drivers"]), 5)
        self.assertIn("not evidence", result["interpretation"])
        for factor in result["protective_factors"]:
            self.assertLess(factor["contribution_towards_anomaly"], 0.0)

    def test_peer_percentile_when_profiles_available(self):
        result = self.service.score_record(TYPICAL_WAGE_EARNER)
        if not self.service.segment_profiles:
            self.skipTest("segment_profiles.json not present")
        self.assertIsNotNone(result["peer_percentile"])
        self.assertGreaterEqual(result["peer_percentile"], 0.0)
        self.assertLessEqual(result["peer_percentile"], 1.0)
        self.assertEqual(
            result["segment_profile"],
            self.service.segment_profiles[str(result["segment_id"])],
        )

    def test_rejects_non_numeric_amount(self):
        with self.assertRaisesRegex(ValueError, "Sw_amt must be numeric"):
            self.service.score_record({"Sw_amt": "not a number"})

    def test_rejects_top_n_below_one(self):
        with self.assertRaises(ValueError):
            self.service.score_record(TYPICAL_WAGE_EARNER, top_n=0)

    def test_large_wre_claim_scores_higher_with_wre_driver(self):
        typical = self.service.score_record(TYPICAL_WAGE_EARNER)
        large = self.service.score_record(LARGE_WRE_CLAIM)

        self.assertEqual(large["segment_id"], typical["segment_id"])
        self.assertGreater(large["anomaly_score"], typical["anomaly_score"])
        self.assertGreater(large["peer_percentile"], typical["peer_percentile"])
        top = {d["feature"] for d in large["main_drivers"][:3]}
        self.assertTrue(any("wre" in f.lower() or "deduction" in f for f in top), top)

    def test_stacked_deductions_are_flagged(self):
        stacked = self.service.score_record(STACKED_DEDUCTIONS)

        self.assertTrue(stacked["flagged_for_review"])
        self.assertGreaterEqual(stacked["anomaly_score"], stacked["threshold"])
        driver_names = {driver["feature"] for driver in stacked["main_drivers"]}
        # The stacked items (gifts, personal super, tax affairs) and the overall
        # deduction share should be what the forest reacts to.
        self.assertIn("deduction_to_income_ratio", driver_names)
        self.assertTrue(driver_names & {"gift_to_income_ratio", "personal_super_ratio", "log_Spr_Prsnl_Contr",
                                        "log_Non_emp_spr_amt", "tax_affairs_cost_ratio"}, driver_names)


if __name__ == "__main__":
    unittest.main()

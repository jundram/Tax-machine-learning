"""Tests for the simplified-form to ATO-record translation."""

import unittest

from record_builder import FALLBACK_SPLITS, FRANKING_RATE, build_record


class RecordBuilderTests(unittest.TestCase):
    def test_totals_follow_ato_identities(self):
        record = build_record(
            {
                "salary": 80_000, "work_related_expenses": 4_000,
                "rental_income": 20_000, "rental_deductions": 26_000,
                "business_income": 10_000, "business_expenses": 7_000,
                "investment_income": 1_000, "investment_deductions": 300,
                "gifts": 200, "personal_super": 5_000, "tax_affairs": 400,
                "super_balance": 150_000, "lodged_via_agent": False,
            },
            FALLBACK_SPLITS,
        )
        franking = record["Dividends_franking_cr_amt"]
        self.assertAlmostEqual(record["Net_rent_amt"], -6_000)
        self.assertAlmostEqual(record["Net_NPP_BI_amt"], 3_000)
        self.assertAlmostEqual(record["Tot_IncLoss_amt"], 80_000 - 6_000 + 3_000 + 1_000 + franking)
        self.assertAlmostEqual(record["Tot_ded_amt"], 4_000 + 300 + 200 + 5_000 + 400)
        self.assertAlmostEqual(
            record["Taxable_Income"], record["Tot_IncLoss_amt"] - record["Tot_ded_amt"]
        )
        self.assertEqual(record["Lodgment_method"], "S")

    def test_combined_fields_are_spread_across_itemised_columns(self):
        record = build_record({"work_related_expenses": 10_000, "rental_deductions": 1_000}, FALLBACK_SPLITS)
        wre_columns = list(FALLBACK_SPLITS["work_related_expenses"])
        self.assertAlmostEqual(sum(record[c] for c in wre_columns), 10_000)
        self.assertTrue(all(record[c] > 0 for c in wre_columns))
        # Minority lines (self-education, travel) are left untouched so no rare item is manufactured.
        self.assertNotIn("WRE_self_amt", record)
        rent_columns = list(FALLBACK_SPLITS["rental_deductions"])
        self.assertAlmostEqual(sum(record[c] for c in rent_columns), 1_000)

    def test_franking_credit_attached_to_franked_dividends(self):
        record = build_record({"investment_income": 7_000}, FALLBACK_SPLITS)
        self.assertAlmostEqual(record["Dividends_franking_cr_amt"], record["Frk_Div_amt"] * FRANKING_RATE)

    def test_taxable_income_never_negative_and_missing_fields_zero(self):
        record = build_record({"work_related_expenses": 5_000}, FALLBACK_SPLITS)
        self.assertEqual(record["Taxable_Income"], 0.0)
        self.assertEqual(record["Sw_amt"], 0.0)
        self.assertEqual(record["Lodgment_method"], "A")


if __name__ == "__main__":
    unittest.main()

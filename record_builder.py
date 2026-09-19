"""Translate the simplified dashboard form into an ATO sample-file record.

The model was trained on itemised return labels (five work-related expense
lines, three rental-deduction lines, interest versus dividends, ...). The
dashboard asks for a handful of plain-language totals instead, so each total
is spread across its itemised columns in the proportions observed across the
training population (models/input_splits.json). Total income, total
deductions and taxable income are derived using the same accounting
identities that hold in the ATO file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS_PATH = ROOT / "models" / "input_splits.json"

# Used when input_splits.json has not been exported (e.g. synthetic artifacts).
# Only lines used by most claimants receive a share (see tools/export_segment_profiles.py).
FALLBACK_SPLITS: dict[str, dict[str, float]] = {
    "work_related_expenses": {"WRE_car_amt": 0.42, "WRE_uniform_amt": 0.09, "WRE_other_amt": 0.49},
    "rental_deductions": {"Rent_int_ded_amt": 0.44, "Other_rent_ded_amt": 0.48, "Rent_cap_wks_amt": 0.08},
    "investment_income": {"Grs_int_amt": 0.31, "Frk_Div_amt": 0.69},
    "investment_deductions": {"Intrst_Ded_amt": 0.39, "Div_Ded_amt": 0.61},
}

# Franking credit attached to a fully franked dividend at the 30 % company rate.
FRANKING_RATE = 0.30 / 0.70
# Superannuation guarantee rate for the 2022-23 income year; employers must pay
# this on salary, so a wage earner with no employer contributions looks unusual.
SUPER_GUARANTEE_RATE = 0.105

SIMPLE_FIELDS = [
    "salary", "work_related_expenses",
    "rental_income", "rental_deductions",
    "business_income", "business_expenses",
    "investment_income", "investment_deductions",
    "gifts", "personal_super", "tax_affairs", "super_balance",
]


def load_splits(path: str | Path = DEFAULT_SPLITS_PATH) -> dict[str, dict[str, float]]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return FALLBACK_SPLITS


def _spread(total: float, shares: Mapping[str, float]) -> dict[str, float]:
    return {column: float(total) * float(share) for column, share in shares.items()}


def build_record(
    form: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Return an ATO-style record plus the derived totals the form displays."""
    splits = splits or load_splits()
    values = {name: max(float(form.get(name, 0.0) or 0.0), 0.0) for name in SIMPLE_FIELDS}

    record: dict[str, Any] = {
        "Lodgment_method": "A" if form.get("lodged_via_agent", True) else "S",
        "Sw_amt": values["salary"],
        "Gross_rent_amt": values["rental_income"],
        "Total_NPP_BI_amt": values["business_income"],
        "Total_NPP_BE_amt": values["business_expenses"],
        "Gift_amt": values["gifts"],
        "Non_emp_spr_amt": values["personal_super"],
        "Spr_Prsnl_Contr": values["personal_super"],
        "Cost_tax_affairs_amt": values["tax_affairs"],
        "Spr_Ttl_Acnt_Bal": values["super_balance"],
        "Spr_Emplr_Contr": values["salary"] * SUPER_GUARANTEE_RATE,
    }
    record.update(_spread(values["work_related_expenses"], splits["work_related_expenses"]))
    record.update(_spread(values["rental_deductions"], splits["rental_deductions"]))
    record.update(_spread(values["investment_income"], splits["investment_income"]))
    record.update(_spread(values["investment_deductions"], splits["investment_deductions"]))
    record["Dividends_franking_cr_amt"] = record["Frk_Div_amt"] * FRANKING_RATE

    record["Net_rent_amt"] = values["rental_income"] - values["rental_deductions"]
    record["Net_NPP_BI_amt"] = values["business_income"] - values["business_expenses"]

    total_income = (
        values["salary"]
        + record["Net_rent_amt"]
        + record["Net_NPP_BI_amt"]
        + values["investment_income"]
        + record["Dividends_franking_cr_amt"]
    )
    total_deductions = (
        values["work_related_expenses"]
        + values["investment_deductions"]
        + values["gifts"]
        + values["personal_super"]
        + values["tax_affairs"]
    )
    record["Tot_IncLoss_amt"] = total_income
    record["Tot_ded_amt"] = total_deductions
    record["Taxable_Income"] = max(total_income - total_deductions, 0.0)
    return record

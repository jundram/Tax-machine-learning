"""Explain the ten highest-scoring flagged returns in a publishable form.

Reads the row-level results and the confidential sample file (neither is
committed) and writes results/top_flagged_public.json. Each entry keeps the
peer group, the anomaly score and, for the leading SHAP drivers, how the
reported value compares with the population: a multiple of the average
taxpayer, a multiple of the top-1% threshold, and the share of taxpayers who
report the item at all. Record identifiers and dollar amounts are not
written, and multiples are rounded to two significant figures so no line of
the source file can be reconstructed.

Run:  python tools/export_top_flagged.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anomaly_service import TaxpayerAnomalyService  # noqa: E402
from research_agent import TaxResearchAgent  # noqa: E402

SCORES = ROOT / "results" / "taxpayer_scores.csv"
RAW_DATA = ROOT.parent / "data" / "2023_sample_file_SA4.csv"
OUT = ROOT / "results" / "top_flagged_public.json"
TOP_N = 10
DRIVERS_PER_RECORD = 3


def sig2(value: float) -> float:
    """Round to two significant figures (multiples only; never raw amounts)."""
    if value == 0 or not math.isfinite(value):
        return 0.0
    digits = -int(math.floor(math.log10(abs(value)))) + 1
    return round(value, digits)


def compare(value: float, population: pd.Series, kind: str) -> dict:
    """How a reported value sits against everyone else's."""
    mean = float(population.mean())
    p99 = float(population.quantile(0.99))
    out: dict = {"reported": value > 0}
    if value > 0 and mean > 0:
        out["x_average"] = sig2(value / mean)
    if value > 0 and p99 > 0:
        out["x_top1pct"] = sig2(value / p99)
    if kind == "amount":
        out["share_reporting"] = round(float((population > 0).mean()), 3)
    else:
        out["typical_ratio"] = sig2(float(population.median()))
    return out


def main() -> None:
    if not SCORES.exists() or not RAW_DATA.exists():
        raise FileNotFoundError("results/taxpayer_scores.csv and ../data/2023_sample_file_SA4.csv are both required")

    scores = pd.read_csv(SCORES)
    raw = pd.read_csv(RAW_DATA, thousands=",", encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    service = TaxpayerAnomalyService()
    agent = TaxResearchAgent()

    ratio_names = service.metadata["ratio_features"]
    amount_cols = [c for c in service.metadata["amount_columns"] if c in raw.columns]
    amounts = raw[amount_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pop_median_income = float(scores["Tot_IncLoss_amt"].median())
    pop_median_ded = float(scores["Tot_ded_amt"].median())
    ded_ratio_all = scores["Tot_ded_amt"] / scores["Tot_IncLoss_amt"].clip(lower=1000.0)
    pop_median_ded_ratio = float(ded_ratio_all.median())
    seg_median_income = scores.groupby("segment")["Tot_IncLoss_amt"].median()
    seg_median_ded = scores.groupby("segment")["Tot_ded_amt"].median()

    top = scores[scores["flag_seg_2pct"] == 1].nlargest(TOP_N, "anomaly_score_seg")
    entries = []
    for rank, row in enumerate(top.itertuples(index=False), 1):
        record = raw[raw["Ind"] == row.Ind].iloc[0].to_dict()
        result = service.score_record(record, top_n=DRIVERS_PER_RECORD)
        income, deductions = float(row.Tot_IncLoss_amt), float(row.Tot_ded_amt)
        ded_ratio = deductions / max(income, 1000.0)

        drivers = []
        for driver in result["main_drivers"]:
            feature = driver["feature"]
            if feature.startswith("log_"):
                column = feature.removeprefix("log_")
                comparison = compare(float(driver["value"]), amounts[column], "amount")
            elif feature in ratio_names and feature in scores.columns:
                comparison = compare(float(driver["value"]), scores[feature], "ratio")
            else:
                comparison = {"reported": driver["value"] != 0}
            drivers.append({"variable": agent.friendly_feature(feature), **comparison})

        entries.append({
            "rank": rank,
            "segment_id": int(row.segment),
            "segment_name": str(row.segment_name),
            "anomaly_score": round(float(row.anomaly_score_seg), 3),
            "threshold": round(float(service.metadata["segment_thresholds"][str(int(row.segment))]), 3),
            "income_x_segment_median": sig2(income / seg_median_income[row.segment]) if seg_median_income[row.segment] > 0 else None,
            "income_x_population_median": sig2(income / pop_median_income),
            "deductions_x_segment_median": sig2(deductions / seg_median_ded[row.segment]) if seg_median_ded[row.segment] > 0 else None,
            "deductions_x_population_median": sig2(deductions / pop_median_ded),
            "deduction_ratio": round(min(ded_ratio, 5.0), 2),
            "deduction_ratio_x_population_median": sig2(ded_ratio / pop_median_ded_ratio) if pop_median_ded_ratio > 0 else None,
            "drivers": drivers,
        })

    OUT.write_text(json.dumps({"top_n": TOP_N, "note": "Multiples rounded to two significant figures; no identifiers or amounts.",
                               "population_median_deduction_ratio": round(pop_median_ded_ratio, 4), "records": entries}, indent=2),
                   encoding="utf-8")
    print(f"Wrote {OUT} ({len(entries)} records)")


if __name__ == "__main__":
    main()

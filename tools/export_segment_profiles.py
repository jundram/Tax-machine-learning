"""Export aggregate per-segment statistics for the dashboard.

Reads the row-level results/taxpayer_scores.csv written by taxpayer_framework.py
(which is never committed) and writes:

* models/segment_profiles.json  - record counts, medians and the anomaly-score
  quantile curve for each peer group
* models/input_splits.json      - population-level proportions used to itemise
  the dashboard's combined fields
* results/summary_public.json   - summary.json without the keys that quote
  individual records

No individual record is reproduced in any of these files.

Run:  python tools/export_segment_profiles.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "results" / "taxpayer_scores.csv"
RAW_DATA = ROOT.parent / "data" / "2023_sample_file_SA4.csv"
OUT = ROOT / "models" / "segment_profiles.json"
SPLITS_OUT = ROOT / "models" / "input_splits.json"
SUMMARY = ROOT / "results" / "summary.json"
SUMMARY_PUBLIC = ROOT / "results" / "summary_public.json"
# summary.json keys that quote individual sample-file records
PER_RECORD_SUMMARY_KEYS = {"local_cases"}
QUANTILE_GRID = np.linspace(0.0, 1.0, 101)
MIN_LINE_PREVALENCE = 0.25   # a line must be used by >= 25% of claimants to receive a share

# Combined dashboard fields and the itemised ATO columns they are spread across.
SPLIT_GROUPS = {
    "work_related_expenses": [
        "WRE_car_amt", "WRE_trvl_amt", "WRE_uniform_amt", "WRE_self_amt", "WRE_other_amt",
    ],
    "rental_deductions": ["Rent_int_ded_amt", "Other_rent_ded_amt", "Rent_cap_wks_amt"],
    "investment_income": ["Grs_int_amt", "Frk_Div_amt", "Unfranked_Div_amt"],
    "investment_deductions": ["Intrst_Ded_amt", "Div_Ded_amt"],
}


def export_input_splits() -> None:
    """Population-level proportions used to itemise the simplified form's totals."""
    if not RAW_DATA.exists():
        print(f"{RAW_DATA} not found; skipping input_splits.json")
        return
    columns = sorted({c for cols in SPLIT_GROUPS.values() for c in cols})
    raw = pd.read_csv(RAW_DATA, thousands=",", encoding="utf-8-sig", usecols=columns)
    raw = raw.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    splits = {}
    for group, cols in SPLIT_GROUPS.items():
        claimants = raw[cols].sum(axis=1) > 0
        # Spread a total only across the lines most claimants actually use;
        # putting a slice into a minority line would manufacture a rare item.
        prevalence = (raw.loc[claimants, cols] > 0).mean()
        common = [c for c in cols if prevalence[c] >= MIN_LINE_PREVALENCE] or cols
        totals = raw[common].sum()
        share = totals / totals.sum() if totals.sum() > 0 else totals * 0 + 1 / len(common)
        splits[group] = {c: round(float(share[c]), 4) for c in common}
    SPLITS_OUT.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Wrote {SPLITS_OUT}")


def export_public_summary() -> None:
    """Copy summary.json without the keys that quote individual records."""
    if not SUMMARY.exists():
        print(f"{SUMMARY} not found; skipping summary_public.json")
        return
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    public = {k: v for k, v in summary.items() if k not in PER_RECORD_SUMMARY_KEYS}
    SUMMARY_PUBLIC.write_text(json.dumps(public, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_PUBLIC} (dropped {sorted(PER_RECORD_SUMMARY_KEYS & summary.keys())})")


def main() -> None:
    if not SCORES.exists():
        raise FileNotFoundError(
            f"{SCORES} not found. Run taxpayer_framework.py first."
        )
    scores = pd.read_csv(SCORES)
    total = len(scores)
    profiles: dict[str, dict] = {}
    for segment_id, group in scores.groupby("segment"):
        quantiles = np.quantile(group["anomaly_score_seg"], QUANTILE_GRID)
        profiles[str(int(segment_id))] = {
            "n_records": int(len(group)),
            "share_of_population": round(len(group) / total, 4),
            "median_total_income": float(group["Tot_IncLoss_amt"].median()),
            "median_taxable_income": float(group["Taxable_Income"].median()),
            "median_total_deductions": float(group["Tot_ded_amt"].median()),
            "agent_lodgement_share": round(float(group["lodged_via_agent"].mean()), 4),
            "flag_rate_2pct": round(float(group["flag_seg_2pct"].mean()), 4),
            "score_quantile_grid": [round(float(q), 4) for q in QUANTILE_GRID],
            "score_quantiles": [round(float(q), 6) for q in quantiles],
        }
    OUT.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} for {len(profiles)} segments ({total:,} records summarised)")
    export_input_splits()
    export_public_summary()


if __name__ == "__main__":
    main()

"""Reusable scoring service for the segmented taxpayer Isolation Forests.

The research pipeline trains and saves the model artifacts. This module loads
those artifacts and scores new records without retraining the models. Missing
amount fields are treated as zero and are reported in the response so callers
can distinguish a complete return from a partial example.

An anomaly flag is a statistical peer-group signal. It is not evidence of
fraud or non-compliance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models"


class TaxpayerAnomalyService:
    """Load the fitted pipeline and score individual taxpayer records."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. "
                "Run taxpayer_framework.py first."
            )

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.segmentation_scaler = joblib.load(
            self.model_dir / "segmentation_scaler.joblib"
        )
        self.kmeans = joblib.load(self.model_dir / "kmeans.joblib")
        self.anomaly_scaler = joblib.load(self.model_dir / "anomaly_scaler.joblib")
        self.segment_models = {
            int(segment_id): joblib.load(
                self.model_dir / f"isolation_forest_segment_{segment_id}.joblib"
            )
            for segment_id in self.metadata["segment_names"]
        }
        self._explainers: dict[int, shap.TreeExplainer] = {}

        # Optional aggregate per-segment statistics (counts, medians, score
        # quantiles). Written by tools/export_segment_profiles.py; absent for
        # artifacts produced before it existed.
        profiles_path = self.model_dir / "segment_profiles.json"
        self.segment_profiles: dict[str, dict[str, Any]] = (
            json.loads(profiles_path.read_text(encoding="utf-8"))
            if profiles_path.exists()
            else {}
        )

    def _peer_percentile(self, segment_id: int, score: float) -> float | None:
        """Share of the segment's training records scoring below ``score``."""
        profile = self.segment_profiles.get(str(segment_id))
        if not profile:
            return None
        quantiles = np.asarray(profile["score_quantiles"], dtype=float)
        grid = np.asarray(profile["score_quantile_grid"], dtype=float)
        return float(np.clip(np.interp(score, quantiles, grid), 0.0, 1.0))

    @staticmethod
    def _number(record: Mapping[str, Any], name: str) -> float:
        value = record.get(name, 0.0)
        if value is None or value == "":
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric; received {value!r}") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite; received {value!r}")
        return number

    @staticmethod
    def _signed_log(value: float) -> float:
        return float(np.sign(value) * np.log1p(abs(value)))

    def _safe_ratio(
        self, numerator: float, denominator: float, cap: float
    ) -> float:
        floor = float(self.metadata["income_floor"])
        return float(min(numerator / max(denominator, floor), cap))

    def _engineer_features(
        self, record: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]:
        amount_columns = self.metadata.get("amount_columns")
        if amount_columns is None:  # compatibility with artifacts saved before format v1
            non_amount_columns = {
                "Ind",
                "Gender",
                "age_range",
                "Occ_code",
                "Partner_status",
                "SA4",
                "Lodgment_method",
                "PHI_Ind",
            }
            amount_columns = [
                name
                for name in self.metadata["required_input_columns"]
                if name not in non_amount_columns
            ]
        values = {name: self._number(record, name) for name in amount_columns}
        missing = [name for name in amount_columns if name not in record]

        wage = sum(values[name] for name in ["Sw_amt", "Alow_ben_amt", "ETP_txbl_amt"])
        business = sum(
            max(values[name], 0.0)
            for name in ["Total_PP_BI_amt", "Total_NPP_BI_amt"]
        )
        property_income = max(values["Gross_rent_amt"], 0.0)
        investment = sum(
            max(values[name], 0.0)
            for name in [
                "Grs_int_amt",
                "Unfranked_Div_amt",
                "Frk_Div_amt",
                "Dividends_franking_cr_amt",
                "Tot_CY_CG_amt",
            ]
        )
        pension = sum(
            values[name]
            for name in [
                "Aust_govt_pnsn_allw_amt",
                "Taxed_othr_pnsn_amt",
                "Untaxed_othr_pnsn_amt",
            ]
        )
        trust = sum(
            abs(values[name]) for name in ["Net_PT_PP_dsn", "Net_PT_NPP_dsn"]
        )
        other = sum(
            abs(values[name])
            for name in [
                "Other_inc_amt",
                "Other_foreign_inc_amt",
                "Net_farm_management_amt",
            ]
        )
        components = {
            "wage": wage,
            "business": business,
            "property": property_income,
            "investment": investment,
            "pension": pension,
            "trust": trust,
            "other": other,
        }
        gross_activity = sum(components.values())
        segment_values = {
            f"share_{name}": value / gross_activity if gross_activity > 0 else 0.0
            for name, value in components.items()
        }
        segment_values["log_gross_activity"] = float(np.log1p(gross_activity))
        segment_values["log_super_balance"] = float(
            np.log1p(max(values["Spr_Ttl_Acnt_Bal"], 0.0))
        )
        segment_frame = pd.DataFrame(
            [[segment_values[name] for name in self.metadata["segment_features"]]],
            columns=self.metadata["segment_features"],
        )

        income = values["Tot_IncLoss_amt"]
        wre = sum(
            values[name]
            for name in [
                "WRE_car_amt",
                "WRE_trvl_amt",
                "WRE_uniform_amt",
                "WRE_self_amt",
                "WRE_other_amt",
            ]
        )
        rent_deductions = sum(
            values[name]
            for name in [
                "Other_rent_ded_amt",
                "Rent_int_ded_amt",
                "Rent_cap_wks_amt",
            ]
        )
        investment_income = sum(
            values[name]
            for name in ["Grs_int_amt", "Unfranked_Div_amt", "Frk_Div_amt"]
        )
        losses = values["PP_loss_claimed"] + values["NPP_loss_claimed"]
        implied_taxable_income = income - values["Tot_ded_amt"] - losses
        if implied_taxable_income > 0:
            taxable_ratio = values["Taxable_Income"] / max(implied_taxable_income, 1.0)
        elif values["Taxable_Income"] == 0:
            taxable_ratio = 1.0
        else:
            taxable_ratio = 0.0

        lodged_via_agent = float(record.get("Lodgment_method") == "A")
        ratios = {
            "deduction_to_income_ratio": self._safe_ratio(
                values["Tot_ded_amt"], income, 5.0
            ),
            "wre_to_salary_ratio": self._safe_ratio(wre, values["Sw_amt"], 5.0),
            "business_expense_ratio": self._safe_ratio(
                values["Total_NPP_BE_amt"], values["Total_NPP_BI_amt"], 10.0
            ),
            "rental_deduction_ratio": self._safe_ratio(
                rent_deductions, values["Gross_rent_amt"], 10.0
            ),
            "taxable_income_ratio": float(np.clip(taxable_ratio, 0.0, 2.0)),
            "investment_deduction_ratio": self._safe_ratio(
                values["Div_Ded_amt"] + values["Intrst_Ded_amt"],
                investment_income,
                10.0,
            ),
            "gift_to_income_ratio": self._safe_ratio(values["Gift_amt"], income, 2.0),
            "tax_affairs_cost_ratio": self._safe_ratio(
                values["Cost_tax_affairs_amt"], income, 1.0
            ),
            "personal_super_ratio": self._safe_ratio(
                values["Non_emp_spr_amt"] + values["Spr_Prsnl_Contr"],
                income,
                5.0,
            ),
            "payg_instalment_ratio": self._safe_ratio(
                values["Cr_PAYG_ITI_amt"], income, 2.0
            ),
            "items_reported": float(sum(value != 0 for value in values.values())),
            "lodged_via_agent": lodged_via_agent,
        }
        anomaly_values = {
            f"log_{name}": self._signed_log(values[name])
            for name in self.metadata["kept_amount_features"]
        }
        anomaly_values.update(ratios)
        anomaly_frame = pd.DataFrame(
            [[anomaly_values[name] for name in self.metadata["anomaly_features"]]],
            columns=self.metadata["anomaly_features"],
        )

        display_values = {
            name: (
                values[name.removeprefix("log_")]
                if name.startswith("log_")
                else anomaly_values[name]
            )
            for name in self.metadata["anomaly_features"]
        }
        return segment_frame, anomaly_frame, display_values, missing

    def score_record(
        self, record: Mapping[str, Any], top_n: int = 5
    ) -> dict[str, Any]:
        """Return the peer group, anomaly result, and local SHAP drivers."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        segment_frame, anomaly_frame, display_values, missing = (
            self._engineer_features(record)
        )
        segment_scaled = self.segmentation_scaler.transform(segment_frame.values)
        segment_id = int(self.kmeans.predict(segment_scaled)[0])

        anomaly_scaled = self.anomaly_scaler.transform(anomaly_frame.values)
        model = self.segment_models[segment_id]
        anomaly_score = -float(model.score_samples(anomaly_scaled)[0])
        threshold = float(self.metadata["segment_thresholds"][str(segment_id)])

        if segment_id not in self._explainers:
            self._explainers[segment_id] = shap.TreeExplainer(model)
        shap_values = self._explainers[segment_id].shap_values(
            anomaly_scaled, check_additivity=False
        )
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        contributions = -np.asarray(shap_values).reshape(-1)
        feature_names = self.metadata["anomaly_features"]

        def describe(index: int) -> dict[str, Any]:
            return {
                "feature": feature_names[index],
                "value": float(display_values[feature_names[index]]),
                "contribution_towards_anomaly": float(contributions[index]),
            }

        limit = min(top_n, len(feature_names))
        # Positive contributions push the record towards "anomalous";
        # negative ones pull it back towards its peer group.
        drivers = [describe(i) for i in np.argsort(-contributions)[:limit]]
        protective = [
            describe(i)
            for i in np.argsort(contributions)[:limit]
            if contributions[i] < 0
        ]

        return {
            "segment_id": segment_id,
            "segment_name": self.metadata["segment_names"][str(segment_id)],
            "segment_profile": self.segment_profiles.get(str(segment_id)),
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "peer_percentile": self._peer_percentile(segment_id, anomaly_score),
            "flagged_for_review": anomaly_score >= threshold,
            "main_drivers": drivers,
            "protective_factors": protective,
            "defaulted_amount_columns": missing,
            "interpretation": (
                "Statistical peer-group signal only; not evidence of "
                "non-compliance or fraud."
            ),
        }


if __name__ == "__main__":
    example = {
        "Ind": 1,
        "Lodgment_method": "A",
        "Sw_amt": 75_000,
        "Tot_IncLoss_amt": 78_000,
        "Tot_ded_amt": 26_000,
        "Taxable_Income": 52_000,
        "WRE_car_amt": 10_000,
        "WRE_trvl_amt": 6_000,
        "WRE_uniform_amt": 1_000,
        "Cost_tax_affairs_amt": 900,
        "Spr_Ttl_Acnt_Bal": 95_000,
    }
    result = TaxpayerAnomalyService().score_record(example)
    print(json.dumps(result, indent=2))

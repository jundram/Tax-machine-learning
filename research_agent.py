"""Evidence-linked review narratives for taxpayer anomaly results.

This module performs deterministic retrieval from a curated set of official
ATO public guidance. It does not make legal conclusions and does not call an
external language model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_BASE = ROOT / "ato_knowledge_base.json"


FEATURE_TOPICS = {
    "WRE_": "work_related_expenses",
    "wre_to_salary_ratio": "work_related_expenses",
    "rental_deduction_ratio": "rental_expenses",
    "Gross_rent_amt": "rental_expenses",
    "Net_rent_amt": "rental_expenses",
    "rent_": "rental_expenses",
    "business_expense_ratio": "business_expenses",
    "Total_NPP_B": "business_expenses",
    "Total_PP_B": "business_expenses",
    "investment_deduction_ratio": "investment_deductions",
    "Div_Ded_amt": "investment_deductions",
    "Intrst_Ded_amt": "investment_deductions",
    "Gift_amt": "gifts_and_donations",
    "gift_to_income_ratio": "gifts_and_donations",
    "Cost_tax_affairs_amt": "tax_affairs",
    "tax_affairs_cost_ratio": "tax_affairs",
    "Non_emp_spr_amt": "personal_super",
    "Spr_Prsnl_Contr": "personal_super",
    "personal_super_ratio": "personal_super",
}

QUESTION_TOPIC_KEYWORDS = {
    "work_related_expenses": [
        "work expense",
        "work-related",
        "car expense",
        "travel expense",
        "uniform",
        "self education",
        "salary deduction",
    ],
    "rental_expenses": [
        "rent",
        "rental",
        "property expense",
        "capital works",
        "negative gearing",
    ],
    "business_expenses": ["business expense", "business deduction", "business income"],
    "investment_deductions": [
        "investment",
        "interest deduction",
        "dividend deduction",
        "shares",
    ],
    "gifts_and_donations": ["gift", "donation", "deductible gift recipient", "dgr"],
    "tax_affairs": ["tax affairs", "accountant fee", "tax agent fee"],
    "personal_super": [
        "personal super",
        "super contribution",
        "notice of intent",
    ],
    "general_records": ["record", "receipt", "evidence", "document", "substantiation"],
}


FRIENDLY_FEATURE_NAMES = {
    # Amount fields (ATO sample-file column names)
    "Sw_amt": "salary and wages",
    "Grs_int_amt": "gross interest",
    "Aust_govt_pnsn_allw_amt": "government pensions and allowances",
    "Frk_Div_amt": "franked dividends",
    "Net_rent_amt": "net rent",
    "Gross_rent_amt": "gross rent",
    "Net_NPP_BI_amt": "net business income",
    "Total_NPP_BI_amt": "total business income",
    "Net_PT_NPP_dsn": "partnership and trust distributions",
    "Other_inc_amt": "other income",
    "Tot_IncLoss_amt": "total income or loss",
    "WRE_car_amt": "work-related car expenses",
    "WRE_trvl_amt": "work-related travel expenses",
    "WRE_uniform_amt": "work-related uniform expenses",
    "WRE_self_amt": "work-related self-education expenses",
    "WRE_other_amt": "other work-related expenses",
    "Gift_amt": "gifts and donations",
    "Non_emp_spr_amt": "personal super contributions deduction",
    "Cost_tax_affairs_amt": "cost of managing tax affairs",
    "Other_Ded_amt": "other deductions",
    "Tot_ded_amt": "total deductions",
    "Taxable_Income": "taxable income",
    "Spr_Ttl_Acnt_Bal": "total super balance",
    "Alow_ben_amt": "allowances and benefits",
    "ETP_txbl_amt": "employment termination payments",
    "Unfranked_Div_amt": "unfranked dividends",
    "Dividends_franking_cr_amt": "franking credits",
    "Rent_int_ded_amt": "rental interest deductions",
    "Rent_cap_wks_amt": "rental capital-works deductions",
    "Other_rent_ded_amt": "other rental deductions",
    "Total_NPP_BE_amt": "total business expenses",
    "Net_CG_amt": "net capital gain",
    "Tot_CY_CG_amt": "current-year capital gains",
    "Net_PT_PP_dsn": "primary-production partnership/trust distributions",
    "Taxed_othr_pnsn_amt": "taxed pensions and annuities",
    "Untaxed_othr_pnsn_amt": "untaxed pensions and annuities",
    "Other_foreign_inc_amt": "other foreign income",
    "Net_farm_management_amt": "farm management deposits",
    "Div_Ded_amt": "dividend deductions",
    "Intrst_Ded_amt": "interest deductions",
    "Rep_frng_ben_amt": "reportable fringe benefits",
    "Asbl_forgn_source_incm_amt": "assessable foreign income",
    "Rptbl_Empr_spr_cont_amt": "reportable employer super contributions",
    "Cr_PAYG_ITI_amt": "PAYG instalments credited",
    "TFN_amts_wheld_gr_intst_amt": "TFN amounts withheld from interest",
    "Help_debt": "HELP (study loan) debt",
    "Spr_Emplr_Contr": "employer super contributions",
    "Spr_Prsnl_Contr": "personal super contributions",
    "Spr_Othr_Contr": "other super contributions",
    "PP_loss_claimed": "primary-production losses claimed",
    "NPP_loss_claimed": "non-primary-production losses claimed",
    # Behavioural ratios
    "deduction_to_income_ratio": "total deductions relative to income",
    "wre_to_salary_ratio": "work-related expenses relative to salary",
    "business_expense_ratio": "business expenses relative to business income",
    "rental_deduction_ratio": "rental deductions relative to gross rent",
    "taxable_income_ratio": "reported taxable income relative to the implied amount",
    "investment_deduction_ratio": "investment deductions relative to investment income",
    "gift_to_income_ratio": "gifts relative to income",
    "tax_affairs_cost_ratio": "tax-affairs costs relative to income",
    "personal_super_ratio": "personal super contributions relative to income",
    "payg_instalment_ratio": "PAYG instalments relative to income",
    "items_reported": "number of reported amount fields",
    "lodged_via_agent": "tax-agent lodgment indicator",
}


class TaxResearchAgent:
    """Link anomaly drivers to relevant ATO guidance and review questions."""

    def __init__(self, knowledge_base: str | Path = DEFAULT_KNOWLEDGE_BASE):
        path = Path(knowledge_base)
        self.knowledge = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def friendly_feature(feature: str) -> str:
        clean = feature.removeprefix("log_")
        if clean in FRIENDLY_FEATURE_NAMES:
            return FRIENDLY_FEATURE_NAMES[clean]
        return clean.replace("_amt", "").replace("_", " ").strip().lower()

    @staticmethod
    def _topic_for_feature(feature: str) -> str:
        clean = feature.removeprefix("log_")
        for pattern, topic in FEATURE_TOPICS.items():
            if pattern in clean or pattern in feature:
                return topic
        return "general_records"

    def retrieve_sources(
        self, drivers: Iterable[dict[str, Any]], max_sources: int = 3
    ) -> list[dict[str, Any]]:
        topics: list[str] = []
        for driver in drivers:
            topic = self._topic_for_feature(driver["feature"])
            if topic not in topics:
                topics.append(topic)
        if "general_records" not in topics:
            topics.append("general_records")
        return [self.knowledge[topic] for topic in topics[:max_sources]]

    def compose_review(self, result: dict[str, Any]) -> dict[str, Any]:
        drivers = result["main_drivers"]
        status = "above" if result["flagged_for_review"] else "below"
        decision = (
            "The profile is flagged for statistical review"
            if result["flagged_for_review"]
            else "The profile is not flagged at the selected review threshold"
        )

        driver_phrases = []
        for driver in drivers[:3]:
            name = self.friendly_feature(driver["feature"])
            value = driver["value"]
            if "ratio" in driver["feature"]:
                value_text = f"{value:.1%}"
            elif driver["feature"] == "items_reported":
                value_text = f"{value:.0f} items"
            else:
                # Escape the dollar sign so Streamlit markdown does not treat it as LaTeX
                value_text = f"\\${value:,.0f}"
            driver_phrases.append(f"{name} ({value_text})")

        explanation = (
            f"{decision}. It belongs to **{result['segment_name']}**. "
            f"Its anomaly score is **{result['anomaly_score']:.4f}**, which is "
            f"{status} the peer-group threshold of **{result['threshold']:.4f}**. "
            f"The strongest model contributions are: {', '.join(driver_phrases)}. "
            "These factors explain why the statistical model produced this score; "
            "they do not establish that any claim is incorrect."
        )

        sources = self.retrieve_sources(drivers)
        questions: list[str] = []
        for source in sources:
            for question in source["review_questions"]:
                if question not in questions:
                    questions.append(question)

        return {
            "explanation": explanation,
            "review_questions": questions[:8],
            "sources": [
                {
                    "title": source["title"],
                    "url": source["url"],
                    "relevance": source["summary"],
                }
                for source in sources
            ],
            "limitation": (
                "Guidance retrieval is informational only. Apply the law and ATO "
                "guidance to the taxpayer's actual facts and tax year, with human review."
            ),
        }

    def _topics_for_question(self, question: str) -> list[str]:
        normalised = question.lower().replace("_", " ")
        scored: list[tuple[int, str]] = []
        for topic, keywords in QUESTION_TOPIC_KEYWORDS.items():
            score = sum(keyword in normalised for keyword in keywords)
            if score:
                scored.append((score, topic))
        scored.sort(reverse=True)
        return [topic for _, topic in scored]

    @staticmethod
    def _source_view(source: dict[str, Any]) -> dict[str, str]:
        return {
            "title": source["title"],
            "url": source["url"],
            "relevance": source["summary"],
        }

    def answer_question(
        self, question: str, result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Answer a review question using model context and curated ATO sources."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        lower = question.lower()

        if any(term in lower for term in ["fraud", "non-compliance", "guilty"]):
            return {
                "answer": (
                    "No. An Isolation Forest flag only means the record is unusual "
                    "relative to its statistical peer group. It does not establish "
                    "fraud, non-compliance, intent, or an incorrect deduction. The "
                    "underlying facts and supporting records require human review."
                ),
                "sources": [self._source_view(self.knowledge["general_records"])],
            }

        if any(term in lower for term in ["why", "flag", "anomaly", "score"]):
            if result is None:
                return {
                    "answer": (
                        "Analyse a taxpayer record first. I can then explain its peer "
                        "group, anomaly score, threshold and strongest SHAP drivers."
                    ),
                    "sources": [],
                }
            review = self.compose_review(result)
            return {"answer": review["explanation"], "sources": review["sources"]}

        if "threshold" in lower or "2%" in lower or "two percent" in lower:
            context = ""
            if result is not None:
                context = (
                    f" For this record, the score is {result['anomaly_score']:.4f} "
                    f"and the peer-group threshold is {result['threshold']:.4f}."
                )
            return {
                "answer": (
                    "The review threshold is an operating cut-off based on the "
                    "highest-scoring portion of each peer group. It represents a "
                    "review budget, not an estimate of the true rate of tax "
                    f"non-compliance.{context}"
                ),
                "sources": [],
            }

        if any(term in lower for term in ["record", "receipt", "evidence", "document"]):
            if result is not None:
                review = self.compose_review(result)
                questions = review["review_questions"]
                sources = review["sources"]
            else:
                source = self.knowledge["general_records"]
                questions = source["review_questions"]
                sources = [self._source_view(source)]
            bullets = "\n".join(f"- {item}" for item in questions)
            return {
                "answer": f"Suggested evidence checks:\n\n{bullets}",
                "sources": sources,
            }

        topics = self._topics_for_question(question)
        if not topics and result is not None:
            topics = [
                self._topic_for_feature(driver["feature"])
                for driver in result["main_drivers"]
            ]
        if not topics:
            topics = ["general_records"]

        unique_topics: list[str] = []
        for topic in topics:
            if topic not in unique_topics:
                unique_topics.append(topic)
        selected = [self.knowledge[topic] for topic in unique_topics[:3]]
        answer_parts = [source["summary"] for source in selected]
        answer = "\n\n".join(answer_parts)
        answer += (
            "\n\nThis is general guidance only. The correct treatment depends on "
            "the taxpayer's facts and the applicable income year."
        )
        return {
            "answer": answer,
            "sources": [self._source_view(source) for source in selected],
        }

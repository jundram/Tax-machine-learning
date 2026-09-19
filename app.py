"""Tax Insight: Streamlit interface for the peer-group taxpayer anomaly service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from anomaly_service import TaxpayerAnomalyService
from record_builder import build_record, load_splits
from research_agent import TaxResearchAgent


# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tax Insight · Check your tax risk score",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAVY = "#0B2540"
NAVY_2 = "#12365B"
TEAL = "#0E7C86"
TEAL_SOFT = "#E3F4F5"
AMBER = "#F2B33D"
AMBER_SOFT = "#FFF1CC"
GREEN_SOFT = "#E4F5EC"
INK = "#14263A"
MUTED = "#5F6F81"
LINE = "#E2E8F0"
CANVAS = "#F4F7FB"

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{background: {CANVAS};}}
    [data-testid="stAppViewContainer"] > .main .block-container {{
        max-width: 100%; padding: 1.1rem 2rem 2.5rem 2rem;
    }}
    [data-testid="stHeader"] {{background: transparent;}}
    h1, h2, h3, h4 {{color: {INK}; letter-spacing: -0.01em;}}
    [data-testid="stHeaderActionElements"], h3 > a, h4 > a {{display: none !important;}}

    section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {{display: none !important;}}

    /* ---------- top bar ---------- */
    .brand {{display: inline-flex; align-items: center; gap: 0.6rem; padding-top: 0.2rem;}}
    .brand .mark {{width: 40px; height: 40px; border-radius: 12px; background: {NAVY}; color: #fff; display: inline-flex;
                   align-items: center; justify-content: center; font-weight: 800; font-size: 1rem; letter-spacing: 0.05em;}}
    .brand .name {{font-weight: 800; letter-spacing: 0.22em; font-size: 0.95rem; color: {NAVY}; line-height: 1.1;}}
    .brand .tag {{color: {MUTED}; font-size: 0.78rem; letter-spacing: 0;}}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{
        border-radius: 10px !important; font-weight: 600; color: {INK}; background: #fff;
    }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {{
        background: {NAVY} !important; color: #fff !important; border-color: {NAVY} !important;
    }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] * {{color: #fff !important;}}
    .status-pill {{
        display: inline-flex; align-items: center; gap: 0.45rem; background: {TEAL_SOFT}; color: {NAVY};
        padding: 0.45rem 0.95rem; border-radius: 999px; font-size: 0.9rem; font-weight: 600; margin-top: 0.15rem;
    }}
    .status-pill i {{width: 9px; height: 9px; border-radius: 50%; background: {TEAL}; display: inline-block;}}
    div[data-testid="stDownloadButton"] button {{
        background: {NAVY}; color: #fff; border: none; border-radius: 10px; padding: 0.5rem 1rem; font-weight: 600;
    }}
    div[data-testid="stDownloadButton"] button:hover {{background: {NAVY_2}; color: #fff;}}

    /* ---------- page header ---------- */
    .title-row {{display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; margin-top: 0.4rem;}}
    .title-row h1 {{margin: 0; font-size: 2rem; font-weight: 800; color: {INK};}}
    .badge {{display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.35rem 0.85rem; border-radius: 8px; font-weight: 700; font-size: 0.95rem;}}
    .badge.warn {{background: {AMBER_SOFT}; color: #6B4A00;}}
    .badge.ok {{background: {GREEN_SOFT}; color: #155E3C;}}
    .subtitle {{color: {MUTED}; font-size: 1rem; margin: 0.2rem 0 1rem 0;}}

    /* ---------- cards ---------- */
    .card {{background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.15rem; margin-bottom: 1rem;}}
    .card h3 {{font-size: 1.15rem; margin: 0 0 0.6rem 0; display: flex; align-items: center; gap: 0.6rem;}}
    .card p {{color: {INK}; font-size: 0.95rem;}}
    .icon {{width: 38px; height: 38px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 1.1rem; flex: none;}}
    .icon.teal {{background: {TEAL_SOFT}; color: {TEAL};}}
    .icon.navy {{background: #E3EAF3; color: {NAVY};}}
    .kpi {{background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 0.9rem 1.1rem; display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1rem;}}
    .kpi .label {{color: {INK}; font-size: 0.95rem;}}
    .kpi .value {{color: {NAVY}; font-size: 1.85rem; font-weight: 800; line-height: 1.15;}}
    .kpi .side {{margin-left: auto; padding-left: 0.9rem; border-left: 1px solid {LINE}; color: {MUTED}; font-size: 0.8rem; text-align: center; min-width: 62px;}}

    /* ---------- score card ---------- */
    .score {{background: linear-gradient(160deg, {NAVY} 0%, {NAVY_2} 100%); color: #fff; border-radius: 16px; padding: 1.2rem 1.4rem 1.1rem 1.4rem; margin-bottom: 1rem;}}
    .score .head {{display: flex; justify-content: space-between; align-items: center; font-size: 1.15rem; font-weight: 600;}}
    .score .chip {{background: rgba(255,255,255,0.14); border-radius: 999px; padding: 0.3rem 0.85rem; font-size: 0.85rem; font-weight: 500;}}
    .score .big {{font-size: 4.6rem; font-weight: 800; line-height: 1; margin: 0.4rem 0 0.6rem 0; letter-spacing: -0.03em;}}
    .score .flag {{display: inline-block; padding: 0.35rem 0.9rem; border-radius: 8px; font-weight: 700; font-size: 0.95rem;}}
    .score .flag.warn {{background: {AMBER}; color: #3B2A00;}}
    .score .flag.ok {{background: #4CC38A; color: #06301C;}}
    .score .sub {{color: #C9D6E4; font-size: 0.95rem; margin: 0.7rem 0 1rem 0;}}
    .track {{position: relative; height: 10px; border-radius: 999px; background: linear-gradient(90deg, {TEAL} 0%, #5FC3B0 45%, {AMBER} 100%); margin: 1.6rem 0 0.4rem 0;}}
    .track .thr {{position: absolute; top: -8px; width: 3px; height: 26px; background: #fff; border-radius: 2px; transform: translateX(-50%);}}
    .track .dot {{position: absolute; top: -7px; width: 24px; height: 24px; border-radius: 50%; background: {AMBER}; border: 3px solid #fff; transform: translateX(-50%); box-shadow: 0 2px 6px rgba(0,0,0,0.3);}}
    .track .dot.ok {{background: #4CC38A;}}
    .track .lbl {{position: absolute; top: 22px; transform: translateX(-50%); font-size: 0.8rem; color: #C9D6E4; white-space: nowrap; text-align: center;}}
    .track .lbl b {{display: block; color: #fff; font-size: 0.9rem;}}
    .track .end {{position: absolute; top: 22px; font-size: 0.8rem; color: #C9D6E4;}}
    .score .foot {{color: #9FB3C8; font-size: 0.85rem; margin-top: 5.6rem;}}

    /* ---------- context / driver tables ---------- */
    .ctx .row {{display: flex; justify-content: space-between; gap: 1rem; padding: 0.55rem 0; border-bottom: 1px solid {LINE}; font-size: 0.95rem;}}
    .ctx .row:last-of-type {{border-bottom: none;}}
    .ctx .row b {{color: {INK}; text-align: right;}}
    .ctx .lead {{color: {MUTED}; font-size: 0.92rem; margin: -0.2rem 0 0.6rem 0;}}
    .infobox {{background: {CANVAS}; border-radius: 10px; padding: 0.6rem 0.8rem; color: {MUTED}; font-size: 0.88rem; margin-top: 0.6rem;}}
    .tip {{background: {GREEN_SOFT}; border-radius: 10px; padding: 0.6rem 0.9rem; color: #155E3C; font-size: 0.9rem; margin-top: 0.7rem;}}
    table.drivers {{width: 100%; border-collapse: collapse; font-size: 0.95rem;}}
    table.drivers th {{text-align: left; color: {MUTED}; font-weight: 500; padding: 0.3rem 0.5rem; border-bottom: 1px solid {LINE};}}
    table.drivers td {{padding: 0.55rem 0.5rem; border-bottom: 1px solid {LINE}; vertical-align: middle;}}
    table.drivers tr:last-child td {{border-bottom: none;}}
    .rank {{display: inline-flex; width: 28px; height: 28px; border-radius: 50%; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;}}
    .rank.r1 {{background: {AMBER_SOFT}; color: #6B4A00;}}
    .rank.r2 {{background: #E3EAF3; color: {NAVY};}}
    .rank.r3 {{background: {TEAL_SOFT}; color: {TEAL};}}
    .rank.down {{background: {GREEN_SOFT}; color: #155E3C;}}
    .link-row {{display: block; background: {CANVAS}; border-radius: 10px; padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; color: {TEAL} !important; font-weight: 600; text-decoration: none;}}
    .link-row:hover {{background: {TEAL_SOFT};}}
    .small {{color: {MUTED}; font-size: 0.85rem;}}

    /* ---------- assistant column ---------- */
    .assistant {{background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 1.1rem 1.15rem 0.6rem 1.15rem;}}
    .assistant .head {{display: flex; gap: 0.8rem; align-items: center; margin-bottom: 0.4rem;}}
    .assistant .head .icon {{width: 46px; height: 46px; background: {TEAL}; color: #fff; font-size: 1.3rem;}}
    .assistant .head h3 {{margin: 0; font-size: 1.25rem;}}
    .assistant .head p {{margin: 0; color: {MUTED}; font-size: 0.9rem;}}
    .next {{border: 1px solid {LINE}; border-radius: 14px; padding: 0.9rem 1rem; margin: 0.6rem 0; background: #fff;}}
    .next h4 {{margin: 0 0 0.35rem 0; font-size: 1.05rem;}}
    .next p {{margin: 0 0 0.6rem 0; color: {INK}; font-size: 0.95rem;}}
    .next a {{color: {TEAL} !important; font-weight: 600; text-decoration: none;}}
    .check-pill {{display: inline-flex; gap: 0.5rem; align-items: center; background: {GREEN_SOFT}; color: #155E3C; border-radius: 10px; padding: 0.5rem 0.8rem; font-size: 0.88rem; width: 100%; box-sizing: border-box;}}
    [data-testid="stChatMessage"] {{background: {CANVAS}; border-radius: 14px; padding: 0.6rem 0.8rem;}}
    .stButton > button {{border-radius: 10px; border: 1px solid {LINE}; background: #fff; color: {INK}; font-weight: 600;}}
    .stButton > button:hover {{border-color: {TEAL}; color: {TEAL};}}
    .stButton > button[kind="primary"] {{background: {TEAL}; color: #fff; border: none;}}
    .stButton > button[kind="primary"]:hover {{background: #0A6670; color: #fff;}}
    .disclaimer {{margin-top: 1.4rem; background: #fff; border: 1px solid {LINE}; border-left: 4px solid {AMBER}; border-radius: 12px; padding: 0.8rem 1rem; color: {MUTED}; font-size: 0.86rem; line-height: 1.5;}}
    .disclaimer b {{color: {INK};}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_service() -> TaxpayerAnomalyService:
    return TaxpayerAnomalyService()


service = load_service()
research_agent = TaxResearchAgent()
metadata = service.metadata
SPLITS = load_splits()
REAL_DATA = "ATO" in metadata.get("data_source", "")


# ----------------------------------------------------------------------------
# Form definition, scenarios and helpers
# ----------------------------------------------------------------------------
# (session key, label, help text, default, step)
FIELD_GROUPS: dict[str, list[tuple[str, str, str, float, float]]] = {
    "Employment": [
        ("salary", "Salary and wages", "Gross salary and wages from all employers.", 75_000.0, 1_000.0),
        ("work_related_expenses", "Work-related expenses",
         "Total of car, travel, uniform, self-education and other work-related deductions.", 3_500.0, 250.0),
    ],
    "Rental property": [
        ("rental_income", "Rental income", "Gross rent received before any expenses.", 0.0, 1_000.0),
        ("rental_deductions", "Rental deductions",
         "Loan interest, repairs, agent fees, capital works and other rental expenses.", 0.0, 1_000.0),
    ],
    "Business (sole trader)": [
        ("business_income", "Business income", "Total business income before expenses.", 0.0, 1_000.0),
        ("business_expenses", "Business expenses", "Total expenses claimed against business income.", 0.0, 1_000.0),
    ],
    "Investments": [
        ("investment_income", "Investment income", "Bank interest plus dividends received.", 500.0, 100.0),
        ("investment_deductions", "Investment deductions",
         "Interest on investment loans and other costs of earning investment income.", 0.0, 100.0),
    ],
    "Other": [
        ("gifts", "Gifts and donations", "Deductible gifts to registered charities.", 200.0, 100.0),
        ("personal_super", "Personal super contributions", "Deductible personal contributions to super.", 0.0, 500.0),
        ("tax_affairs", "Cost of managing tax affairs", "Tax agent and related fees.", 300.0, 100.0),
        ("super_balance", "Total super balance", "Balance across all super accounts at year end.", 95_000.0, 5_000.0),
    ],
}
ALL_FIELDS = [field for group in FIELD_GROUPS.values() for field in group]
DEFAULTS = {key: default for key, _, _, default, _ in ALL_FIELDS}

SCENARIOS: dict[str, dict[str, object]] = {
    "Typical wage earner": {},
    "One large work-related claim": {"work_related_expenses": 40_000.0},
    "Stacked deductions on a wage": {
        "work_related_expenses": 45_000.0, "gifts": 6_000.0, "personal_super": 15_000.0,
        "tax_affairs": 2_500.0,
    },
    "Negatively geared rental investor": {
        "salary": 110_000.0, "rental_income": 28_000.0, "rental_deductions": 49_000.0,
        "super_balance": 260_000.0,
    },
    "Sole trader with thin margins": {
        "salary": 0.0, "work_related_expenses": 0.0, "business_income": 160_000.0,
        "business_expenses": 152_000.0, "gifts": 0.0, "tax_affairs": 1_800.0,
        "super_balance": 40_000.0, "lodged_via_agent": False,
    },
    "Self-funded retiree": {
        "salary": 0.0, "work_related_expenses": 0.0, "investment_income": 51_000.0,
        "gifts": 500.0, "tax_affairs": 0.0, "super_balance": 900_000.0,
    },
}

WHY_IT_MATTERS = {
    "wre_to_salary_ratio": "High work expenses relative to salary",
    "deduction_to_income_ratio": "Large deduction share of income",
    "business_expense_ratio": "Expenses high relative to business income",
    "rental_deduction_ratio": "Rental deductions high relative to rent",
    "taxable_income_ratio": "Taxable income differs from the implied amount",
    "investment_deduction_ratio": "Deductions high relative to investment income",
    "gift_to_income_ratio": "Donations large relative to income",
    "tax_affairs_cost_ratio": "Tax-affairs cost high relative to income",
    "personal_super_ratio": "Personal super large relative to income",
    "payg_instalment_ratio": "PAYG instalments unusual for this income",
    "items_reported": "Unusual number of items reported",
    "lodged_via_agent": "Lodgment channel unusual for this profile",
}
PAGES = ["Overview", "Record entry", "Guidance", "About"]
NAV_ICONS = {"Overview": "🏠", "Record entry": "📝", "Guidance": "📖", "About": "ℹ️"}


def money(value: float) -> str:
    return f"${value:,.0f}"


def scenario_values(name: str) -> dict[str, object]:
    overrides = SCENARIOS[name]
    values: dict[str, object] = {key: overrides.get(key, default) for key, default in DEFAULTS.items()}
    values["lodged_via_agent"] = bool(overrides.get("lodged_via_agent", True))
    return values


def set_form(values: dict[str, object]) -> None:
    """Store the form durably and give the entry widgets fresh keys so they re-read it."""
    st.session_state["form"] = values
    st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1


def widget_key(name: str) -> str:
    return f"w_{name}_{st.session_state['form_version']}"


def apply_scenario(name: str) -> None:
    set_form(scenario_values(name))
    st.session_state["record_label"] = name
    st.session_state.pop("analysis_result", None)


def analyse(label: str | None = None) -> None:
    """Score the current form. Used as a button callback so it may switch the page."""
    record = build_record(st.session_state["form"], SPLITS)
    st.session_state["analysis_result"] = service.score_record(record, top_n=8)
    st.session_state["analysis_record"] = record
    if label is not None:
        st.session_state["record_label"] = label
    st.session_state["nav"] = "Overview"


def load_and_analyse(name: str) -> None:
    apply_scenario(name)
    analyse()


def ask(question: str) -> None:
    response = research_agent.answer_question(question, result=st.session_state.get("analysis_result"))
    st.session_state["chat_messages"].append({"role": "user", "content": question, "sources": []})
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": response["answer"], "sources": response["sources"]}
    )


def value_text(feature: str, value: float) -> str:
    if "ratio" in feature:
        return f"{value:.1%}"
    if feature == "items_reported":
        return f"{value:.0f}"
    if feature == "lodged_via_agent":
        return "Yes" if value else "No"
    return money(value)


def why_text(driver: dict) -> str:
    feature = driver["feature"].removeprefix("log_")
    if feature in WHY_IT_MATTERS:
        return WHY_IT_MATTERS[feature]
    if driver["contribution_towards_anomaly"] < 0:
        return "Typical for this peer group"
    if driver["value"] == 0:
        return "Absent where peers usually report it"
    return "Amount unusual for this peer group"


def build_report(result: dict, review: dict) -> str:
    record = st.session_state.get("analysis_record", {})
    lines = [
        "# Tax Insight · Peer-group anomaly review",
        f"Generated {datetime.now():%d %B %Y %H:%M}",
        "",
        f"**Record:** {st.session_state.get('record_label', 'Manual entry')}  ",
        f"**Peer group:** {result['segment_name']}  ",
        f"**Anomaly score:** {result['anomaly_score']:.4f} (threshold {result['threshold']:.4f})  ",
        f"**Flagged for review:** {'yes' if result['flagged_for_review'] else 'no'}  ",
    ]
    if result.get("peer_percentile") is not None:
        lines.append(f"**Peer percentile:** {result['peer_percentile']:.0%}")
    lines += ["", "## Return summary", "",
              f"- Total income: {money(record.get('Tot_IncLoss_amt', 0))}",
              f"- Total deductions: {money(record.get('Tot_ded_amt', 0))}",
              f"- Taxable income: {money(record.get('Taxable_Income', 0))}",
              "", "## Driving factors", ""]
    for i, driver in enumerate(result["main_drivers"], 1):
        lines.append(f"{i}. {research_agent.friendly_feature(driver['feature']).capitalize()} — "
                     f"{value_text(driver['feature'], driver['value'])} — {why_text(driver)} "
                     f"(SHAP {driver['contribution_towards_anomaly']:+.3f})")
    lines += ["", "## Suggested evidence checks", ""] + [f"- [ ] {q}" for q in review["review_questions"]]
    lines += ["", "## ATO guidance", ""] + [f"- [{s['title']}]({s['url']})" for s in review["sources"]]
    lines += ["", "---", "An anomaly flag is a statistical signal relative to a peer group. It is not evidence of "
              "error, non-compliance or fraud, and this report is not for decision-making."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Session initialisation (supports ?scenario=<name> links)
# ----------------------------------------------------------------------------
if "form" not in st.session_state:
    requested = st.query_params.get("scenario", "")
    if requested in SCENARIOS:
        apply_scenario(requested)
        analyse()
    else:
        apply_scenario("Typical wage earner")
        st.session_state["record_label"] = "Manual entry"
        st.session_state["nav"] = "Record entry"
st.session_state.setdefault("chat_messages", [])


# ----------------------------------------------------------------------------
# Top bar: brand, navigation, model status, export
# ----------------------------------------------------------------------------
result = st.session_state.get("analysis_result")
review = research_agent.compose_review(result) if result else None

brand_col, nav_col, pill_col, export_col = st.columns([1.6, 3.2, 1.5, 1.3], vertical_alignment="center")
brand_col.markdown(
    '<div class="brand"><span class="mark">TI</span><span><div class="name">TAX INSIGHT</div>'
    '<div class="tag">Peer-group anomaly review</div></span></div>',
    unsafe_allow_html=True,
)
with nav_col:
    page = st.segmented_control(
        "Navigate", PAGES, key="nav", format_func=lambda p: f"{NAV_ICONS[p]}  {p}",
        label_visibility="collapsed", width="stretch",
    ) or "Overview"
pill_col.markdown(
    f'<span class="status-pill"><i></i>{"ATO sample-file model" if REAL_DATA else "Synthetic demo"}</span>',
    unsafe_allow_html=True,
)
if result and review:
    export_col.download_button(
        "⬇ Export report", build_report(result, review),
        file_name="tax_insight_report.md", mime="text/markdown", width="stretch",
    )


# ----------------------------------------------------------------------------
# Overview components
# ----------------------------------------------------------------------------
def render_score_card(result: dict) -> None:
    score, threshold = result["anomaly_score"], result["threshold"]
    profile = result.get("segment_profile") or {}
    quantiles = profile.get("score_quantiles")
    lo = min(quantiles[0], score, threshold) if quantiles else min(0.3, score)
    hi = max(quantiles[-1], score, threshold) if quantiles else max(0.8, score)
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad

    def pos(x: float) -> str:
        return f"{(x - lo) / (hi - lo) * 100:.1f}%"

    flagged = result["flagged_for_review"]
    st.markdown(
        f"""
        <div class="score">
          <div class="head"><span>Anomaly score</span><span class="chip">Peer-group analysis</span></div>
          <div class="big">{score:.2f}</div>
          <span class="flag {'warn' if flagged else 'ok'}">{'Above review threshold' if flagged else 'Below review threshold'}</span>
          <div class="sub">Threshold {threshold:.2f} · Difference {score - threshold:+.2f}</div>
          <div class="track">
            <div class="thr" style="left:{pos(threshold)}"></div>
            <div class="dot {'' if flagged else 'ok'}" style="left:{pos(score)}"></div>
            <span class="end" style="left:0">{lo:.2f}</span>
            <span class="lbl" style="left:{pos(threshold)}"><b>{threshold:.2f}</b>Threshold</span>
            <span class="lbl" style="left:{pos(score)}; top:52px"><b>{score:.2f}</b>Score</span>
            <span class="end" style="right:0">{hi:.2f}</span>
          </div>
          <div class="foot">Statistical unusualness within the peer group, not a fraud probability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context_card(result: dict) -> None:
    profile = result.get("segment_profile") or {}
    short_name = result["segment_name"].split(": ", 1)[-1]
    rows: list[tuple[str, str]] = []
    if result.get("peer_percentile") is not None:
        rows.append(("Peer percentile", f"{result['peer_percentile']:.0%} of peers score lower"))
    if profile:
        rows += [("Peer records", f"{profile['n_records']:,} ({profile['share_of_population']:.1%})"),
                 ("Median total income", money(profile["median_total_income"])),
                 ("Median deductions", money(profile["median_total_deductions"])),
                 ("Lodged via agent", f"{profile['agent_lodgement_share']:.0%}")]
    rows += [("Review budget", f"Top {metadata['contamination']:.0%} per group"),
             ("Detection model", "Isolation Forest"), ("Explanation method", "SHAP")]
    rows_html = "".join(f'<div class="row"><span>{k}</span><b>{v}</b></div>' for k, v in rows)
    st.markdown(
        f"""
        <div class="card ctx">
          <h3><span class="icon teal">👥</span>Peer-group context</h3>
          <div style="font-weight:700;color:{INK};margin-top:-0.3rem">{short_name}</div>
          <div class="lead">Compared with returns sharing a similar income mix.</div>
          {rows_html}
          <div class="infobox">ⓘ A review budget is not an estimated non-compliance rate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_driver_table(result: dict) -> None:
    body = ""
    for i, driver in enumerate(result["main_drivers"][:5], 1):
        body += (f'<tr><td><span class="rank r{min(i, 3)}">{i}</span></td>'
                 f'<td><b>{research_agent.friendly_feature(driver["feature"]).capitalize()}</b></td>'
                 f'<td>{value_text(driver["feature"], driver["value"])}</td>'
                 f'<td>{why_text(driver)}</td>'
                 f'<td class="small">{driver["contribution_towards_anomaly"]:+.3f}</td></tr>')
    for driver in result.get("protective_factors", [])[:2]:
        body += (f'<tr><td><span class="rank down">↓</span></td>'
                 f'<td>{research_agent.friendly_feature(driver["feature"]).capitalize()}</td>'
                 f'<td>{value_text(driver["feature"], driver["value"])}</td>'
                 f'<td>{why_text(driver)}</td>'
                 f'<td class="small">{driver["contribution_towards_anomaly"]:+.3f}</td></tr>')
    st.markdown(
        f"""
        <div class="card">
          <h3><span class="icon teal">📈</span>What is driving the result?
            <span class="small" style="margin-left:auto;font-weight:400">Local SHAP · amounts and ratios describe this record</span></h3>
          <table class="drivers">
            <thead><tr><th>#</th><th>Factor</th><th>Record value</th><th>Why it matters</th><th>SHAP</th></tr></thead>
            <tbody>{body}</tbody>
          </table>
          <div class="tip">💡 The model flags unusual combinations. Supporting evidence determines the next step.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant(review: dict | None) -> None:
    st.markdown(
        '<div class="assistant"><div class="head"><span class="icon">✦</span>'
        "<div><h3>Review assistant</h3><p>Model context + curated guidance</p></div></div></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.button("💡 Explain this score", on_click=ask, args=("Why was this record flagged?",), width="stretch")
    c2.button("💬 What should I check?", on_click=ask, args=("What records should be reviewed?",), width="stretch")

    for message in st.session_state["chat_messages"][-8:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Sources:** " + " · ".join(f"[{s['title']}]({s['url']})" for s in message["sources"]))

    if review:
        source = review["sources"][0]
        st.markdown(
            f"""
            <div class="next">
              <h4>🚩 Recommended next step</h4>
              <p>{review['review_questions'][0]}</p>
              <a href="{source['url']}" target="_blank">📄 {source['title']} ↗</a>
            </div>
            <div class="check-pill">✅ Local guidance · Human review required</div>
            """,
            unsafe_allow_html=True,
        )
    question = st.chat_input("Ask about this result…")
    if question:
        ask(question)
        st.rerun()
    st.markdown(
        '<div class="small" style="margin-top:0.6rem">ⓘ Answers come from the curated local knowledge base. '
        "An anomaly does not establish an incorrect claim.</div>",
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    if not result or not review:
        main, side = st.columns([2.35, 1], gap="medium")
        with main:
            st.markdown(
                '<div class="title-row"><h1>Check your tax risk score</h1></div>'
                '<div class="subtitle">No record has been analysed yet.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card"><h3><span class="icon teal">📝</span>Start with a record</h3>'
                "<p>Enter a return on the <b>Record entry</b> page, or load an example scenario to see how the "
                "peer-group model responds to different taxpayer profiles.</p></div>",
                unsafe_allow_html=True,
            )
            scenario = st.selectbox("Example scenario", list(SCENARIOS))
            st.button("Load and analyse", type="primary", on_click=load_and_analyse, args=(scenario,))
        with side:
            render_assistant(None)
        return

    record = st.session_state["analysis_record"]
    label = st.session_state.get("record_label", "Manual entry")
    short_name = result["segment_name"].split(": ", 1)[-1]
    badge = ('<span class="badge warn">⚠ Review suggested</span>' if result["flagged_for_review"]
             else '<span class="badge ok">✓ No review flag</span>')
    main, side = st.columns([2.35, 1], gap="medium")
    with main:
        st.markdown(
            f'<div class="title-row"><h1>Taxpayer record</h1>{badge}</div>'
            f'<div class="subtitle">Individual return &nbsp;·&nbsp; {short_name} &nbsp;·&nbsp; {label}</div>',
            unsafe_allow_html=True,
        )
        deduction_share = record["Tot_ded_amt"] / max(record["Tot_IncLoss_amt"], 1.0)
        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="kpi"><span class="icon teal">$</span><div><div class="label">Total income</div>'
                    f'<div class="value">{money(record["Tot_IncLoss_amt"])}</div></div>'
                    f'<div class="side">Individual<br>return</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><span class="icon teal">📄</span><div><div class="label">Total deductions</div>'
                    f'<div class="value">{money(record["Tot_ded_amt"])}</div></div>'
                    f'<div class="side">{deduction_share:.1%}<br>of income</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><span class="icon teal">🧮</span><div><div class="label">Taxable income</div>'
                    f'<div class="value">{money(record["Taxable_Income"])}</div></div>'
                    f'<div class="side">{short_name.split(" (")[0]}</div></div>', unsafe_allow_html=True)

        s_col, c_col = st.columns([1.45, 1], gap="medium")
        with s_col:
            render_score_card(result)
        with c_col:
            render_context_card(result)

        render_driver_table(result)

        e_col, g_col = st.columns([1.3, 1], gap="medium")
        with e_col:
            with st.container(border=True):
                st.markdown('<h3 style="font-size:1.15rem;margin:0 0 0.4rem 0">📋 &nbsp;Suggested evidence checks</h3>',
                            unsafe_allow_html=True)
                for i, question in enumerate(review["review_questions"][:6]):
                    st.checkbox(question, key=f"chk_{i}_{abs(hash(question)) % 100000}")
        with g_col:
            links = "".join(f'<a class="link-row" href="{s["url"]}" target="_blank">{s["title"].removeprefix("ATO: ")} ↗</a>'
                            for s in review["sources"])
            st.markdown(
                f'<div class="card"><h3><span class="icon teal">📖</span>ATO guidance</h3>{links}'
                f'<div class="small">Curated references · confirm the applicable tax year.</div></div>',
                unsafe_allow_html=True,
            )
    with side:
        render_assistant(review)


# ----------------------------------------------------------------------------
# Record entry, Guidance and About pages
# ----------------------------------------------------------------------------
def render_record_entry() -> None:
    st.markdown(
        '<div class="title-row"><h1>Record entry</h1></div>'
        '<div class="subtitle">Whole-dollar amounts for one income year. Leave anything that does not apply at zero. '
        "Combined fields are itemised across the return labels the model expects using population-typical proportions.</div>",
        unsafe_allow_html=True,
    )
    top1, top2, top3 = st.columns([2, 1, 1.4])
    scenario = top1.selectbox("Load an example scenario", list(SCENARIOS))
    top2.write("")
    top2.button("Load scenario", on_click=apply_scenario, args=(scenario,), width="stretch")
    form = st.session_state["form"]
    lodgment = top3.selectbox(
        "Lodgment method", ["Tax agent", "Self-lodged"],
        index=0 if form.get("lodged_via_agent", True) else 1, key=widget_key("lodgment"),
    )

    values: dict[str, object] = {}
    columns = st.columns(3, gap="medium")
    for i, (group_name, fields) in enumerate(FIELD_GROUPS.items()):
        with columns[i % 3], st.container(border=True):
            st.markdown(f"**{group_name}**")
            for key, label, help_text, _, step in fields:
                values[key] = float(st.number_input(
                    label, value=float(form[key]), key=widget_key(key), step=step,
                    min_value=0.0, format="%.0f", help=help_text,
                ))
    values["lodged_via_agent"] = lodgment == "Tax agent"
    # Keep the durable form state in step with the widgets.
    st.session_state["form"] = values
    preview = build_record(values, SPLITS)

    st.write("")
    p1, p2, p3, p4 = st.columns([1, 1, 1, 1.2])
    for col, label, key in ((p1, "Total income", "Tot_IncLoss_amt"), (p2, "Total deductions", "Tot_ded_amt"),
                            (p3, "Taxable income", "Taxable_Income")):
        col.markdown(f'<div class="kpi"><div><div class="label">{label}</div><div class="value">{money(preview[key])}</div></div></div>',
                     unsafe_allow_html=True)
    with p4:
        st.write("")
        label = scenario if values == scenario_values(scenario) else "Manual entry"
        st.button("Analyse taxpayer", type="primary", width="stretch", on_click=analyse, args=(label,))


def render_guidance() -> None:
    st.markdown('<div class="title-row"><h1>Guidance</h1></div>'
                '<div class="subtitle">The curated ATO references the review assistant draws on.</div>',
                unsafe_allow_html=True)
    columns = st.columns(2, gap="medium")
    for i, source in enumerate(research_agent.knowledge.values()):
        questions = "".join(f"<li>{q}</li>" for q in source["review_questions"])
        columns[i % 2].markdown(
            f'<div class="card"><h3><span class="icon teal">📖</span>{source["title"].removeprefix("ATO: ")}</h3>'
            f'<p class="small">{source["summary"]}</p><ul class="small">{questions}</ul>'
            f'<a class="link-row" href="{source["url"]}" target="_blank">Open ATO guidance ↗</a></div>',
            unsafe_allow_html=True,
        )


def render_about() -> None:
    n_records = sum(p.get("n_records", 0) for p in service.segment_profiles.values())
    built = datetime.fromtimestamp((Path(service.model_dir) / "model_metadata.json").stat().st_mtime)
    st.markdown('<div class="title-row"><h1>About the model</h1></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1], gap="medium")
    c1.markdown(
        f"""
        <div class="card ctx">
          <h3><span class="icon navy">⚙</span>Fitted pipeline</h3>
          <div class="row"><span>Training data</span><b>{metadata.get('data_source', 'unknown')}</b></div>
          <div class="row"><span>Records</span><b>{n_records:,}</b></div>
          <div class="row"><span>Peer groups</span><b>{len(metadata['segment_names'])} (K-Means on income composition)</b></div>
          <div class="row"><span>Detector</span><b>Isolation Forest per peer group, 200 trees</b></div>
          <div class="row"><span>Features</span><b>{len(metadata['anomaly_features'])} log-amounts and behavioural ratios</b></div>
          <div class="row"><span>Review budget</span><b>{metadata['contamination']:.0%} per peer group</b></div>
          <div class="row"><span>Explanations</span><b>SHAP TreeExplainer</b></div>
          <div class="row"><span>Artifacts built</span><b>{built:%d %B %Y}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    profile_rows = []
    for segment_id, name in metadata["segment_names"].items():
        profile = service.segment_profiles.get(segment_id, {})
        profile_rows.append({"Peer group": name, "Records": profile.get("n_records"),
                             "Share": profile.get("share_of_population"),
                             "Median income": profile.get("median_total_income"),
                             "Threshold": metadata["segment_thresholds"][segment_id]})
    with c2:
        st.markdown("#### Peer groups")
        st.dataframe(pd.DataFrame(profile_rows), hide_index=True, width="stretch",
                     column_config={"Records": st.column_config.NumberColumn(format="%d"),
                                    "Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                    "Median income": st.column_config.NumberColumn(format="$%d"),
                                    "Threshold": st.column_config.NumberColumn(format="%.4f")})


if page == "Overview":
    render_overview()
elif page == "Record entry":
    render_record_entry()
elif page == "Guidance":
    render_guidance()
else:
    render_about()

st.markdown(
    """
    <div class="disclaimer"><b>Not for decision-making.</b> Tax Insight is a research prototype from a Master of Data
    Science capstone. A score or flag is a statistical statement that a return is unusual relative to its peer group;
    it is not evidence of error, non-compliance or fraud, and it is not tax, legal or financial advice. Do not make,
    defer or justify any decision about a real taxpayer, return or audit on the basis of this tool. It is provided as
    is, without warranty of any kind, and the authors accept no responsibility or liability for any loss, action or
    outcome arising from its use. Nothing entered here leaves this machine.</div>
    """,
    unsafe_allow_html=True,
)

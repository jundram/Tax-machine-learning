"""Streamlit interface for the peer-group taxpayer anomaly scoring service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from anomaly_service import TaxpayerAnomalyService
from record_builder import build_record, load_splits
from research_agent import TaxResearchAgent


# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Check Your Tax Risk Score",
    page_icon="🔎",
    layout="wide",
)

NAVY = "#1B2A35"
TEAL = "#0F6E6E"
AMBER = "#B7791F"
RED = "#B42318"
GREEN = "#0E7A4E"
MUTED = "#5C6B7A"

st.markdown(
    f"""
    <style>
    .block-container {{max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}}
    h1, h2, h3 {{color: {NAVY}; letter-spacing: -0.01em;}}
    [data-testid="stMetric"] {{
        background: #ffffff;
        border: 1px solid #E3E9EF;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(27, 42, 53, 0.04);
    }}
    [data-testid="stMetricLabel"] p {{color: {MUTED}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;}}
    .hero {{
        background: linear-gradient(135deg, {NAVY} 0%, #24455A 60%, {TEAL} 100%);
        color: #fff; border-radius: 18px; padding: 1.6rem 1.9rem; margin-bottom: 1.2rem;
    }}
    .hero h1 {{color: #fff; margin: 0 0 0.3rem 0; font-size: 2.1rem;}}
    .hero .kicker {{text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.74rem; opacity: 0.8; margin-bottom: 0.35rem;}}
    .footer {{
        margin-top: 2.5rem; padding: 1.3rem 1.5rem; border-radius: 14px;
        background: #F3F6F9; border: 1px solid #E3E9EF; color: {MUTED}; font-size: 0.86rem; line-height: 1.55;
    }}
    .footer h4 {{margin: 0 0 0.4rem 0; color: {NAVY}; font-size: 0.95rem;}}
    .footer .disclaimer {{
        margin-top: 0.8rem; padding: 0.8rem 1rem; border-radius: 10px;
        background: #FFF4ED; border: 1px solid #F5C6A5; color: #7A2E0E;
    }}
    .hero p {{margin: 0; opacity: 0.88; font-size: 1rem;}}
    .pill {{
        display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; margin-top: 0.7rem;
        background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.28);
    }}
    .kpis {{display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.8rem; margin-bottom: 1.2rem;}}
    .kpi {{background: #fff; border: 1px solid #E3E9EF; border-radius: 14px; padding: 0.9rem 1.1rem; box-shadow: 0 1px 2px rgba(27,42,53,0.04);}}
    .kpi .label {{color: {MUTED}; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em;}}
    .kpi .value {{color: {NAVY}; font-size: 1.8rem; font-weight: 700; line-height: 1.2; margin: 0.15rem 0;}}
    .kpi .sub {{color: {MUTED}; font-size: 0.8rem;}}
    .kpi .sub.up {{color: {RED}; font-weight: 600;}}
    .kpi .sub.down {{color: {GREEN}; font-weight: 600;}}
    .verdict {{
        border-radius: 16px; padding: 1.2rem 1.5rem; margin: 0.4rem 0 1.1rem 0;
        border: 1px solid; display: flex; gap: 1rem; align-items: center;
    }}
    .verdict .icon {{font-size: 2.1rem; line-height: 1;}}
    .verdict h3 {{margin: 0 0 0.25rem 0; font-size: 1.15rem;}}
    .verdict p {{margin: 0; font-size: 0.95rem;}}
    .verdict.flagged {{background: #FFF4ED; border-color: #F5C6A5; color: #7A2E0E;}}
    .verdict.flagged h3 {{color: #9A3412;}}
    .verdict.clear {{background: #EEF8F2; border-color: #B8E0C7; color: #14532D;}}
    .verdict.clear h3 {{color: {GREEN};}}
    .card {{
        background: #fff; border: 1px solid #E3E9EF; border-radius: 14px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
        box-shadow: 0 1px 2px rgba(27, 42, 53, 0.04);
    }}
    .card h4 {{margin: 0 0 0.35rem 0; font-size: 0.98rem; color: {NAVY};}}
    .card p {{margin: 0; color: {MUTED}; font-size: 0.9rem; line-height: 1.45;}}
    .card a {{color: {TEAL}; text-decoration: none; font-weight: 600;}}
    .profile {{
        background: #F3F6F9; border-radius: 14px; padding: 1rem 1.2rem; height: 100%;
    }}
    .profile .label {{color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;}}
    .profile .value {{color: {NAVY}; font-size: 1.15rem; font-weight: 700;}}
    .profile .row {{display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px dashed #D9E1E8;}}
    .profile .row:last-child {{border-bottom: none;}}
    .note {{
        background: #EEF7F5; border-left: 4px solid {TEAL}; border-radius: 8px;
        padding: 0.7rem 1rem; color: #17483F; font-size: 0.9rem;
    }}
    .question {{
        background: #fff; border: 1px solid #E3E9EF; border-left: 4px solid {AMBER};
        border-radius: 10px; padding: 0.55rem 0.9rem; margin-bottom: 0.45rem; font-size: 0.92rem;
    }}
    section[data-testid="stSidebar"] {{background: #F3F6F9;}}
    .summary {{background: #fff; border: 1px solid #E3E9EF; border-radius: 12px; padding: 0.7rem 0.9rem; margin: 0.4rem 0 0.6rem 0;}}
    .summary .title {{font-weight: 700; color: {NAVY}; font-size: 0.88rem; margin-bottom: 0.3rem;}}
    .summary .row {{display: flex; justify-content: space-between; font-size: 0.88rem; padding: 0.22rem 0; color: {MUTED};}}
    .summary .row b {{color: {NAVY};}}
    .summary .row.total {{border-top: 1px solid #E3E9EF; margin-top: 0.2rem; padding-top: 0.4rem; font-size: 0.95rem;}}
    section[data-testid="stSidebar"] [data-testid="stExpander"] {{
        background: #fff; border-radius: 12px; border: 1px solid #E3E9EF;
    }}
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


# ----------------------------------------------------------------------------
# Input form definition
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
SPLITS = load_splits()

# Example scenarios: values override the defaults; everything else resets.
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


def apply_scenario(name: str) -> None:
    """Reset every input to its default, then overlay the scenario values."""
    overrides = SCENARIOS[name]
    for key, default in DEFAULTS.items():
        st.session_state[key] = overrides.get(key, default)
    st.session_state["lodgment"] = "Tax agent" if overrides.get("lodged_via_agent", True) else "Self-lodged"
    st.session_state.pop("analysis_result", None)


def money(value: float) -> str:
    return f"${value:,.0f}"


# ----------------------------------------------------------------------------
# Sidebar: record entry
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Taxpayer record")
    st.caption("Whole-dollar amounts for one income year. Leave anything that does not apply at zero.")

    requested = st.query_params.get("scenario", "")
    scenario = st.selectbox(
        "Load an example scenario", list(SCENARIOS),
        index=list(SCENARIOS).index(requested) if requested in SCENARIOS else 0,
    )
    st.button("Load scenario", on_click=apply_scenario, args=(scenario,), width="stretch")

    if "lodgment" not in st.session_state:
        # A ?scenario=<name> link pre-loads and analyses that example.
        apply_scenario(requested if requested in SCENARIOS else "Typical wage earner")
        st.session_state["auto_analyse"] = requested in SCENARIOS

    st.selectbox("Lodgment method", ["Tax agent", "Self-lodged"], key="lodgment")
    for group_index, (group_name, fields) in enumerate(FIELD_GROUPS.items()):
        with st.expander(group_name, expanded=group_index == 0):
            for key, label, help_text, _, step in fields:
                st.number_input(label, key=key, step=step, min_value=0.0, format="%.0f", help=help_text)

    form = {key: float(st.session_state[key]) for key in DEFAULTS}
    form["lodged_via_agent"] = st.session_state["lodgment"] == "Tax agent"
    record = build_record(form, SPLITS)

    st.markdown(
        f"""
        <div class="summary">
          <div class="title">Derived return summary</div>
          <div class="row"><span>Total income</span><b>{money(record["Tot_IncLoss_amt"])}</b></div>
          <div class="row"><span>Total deductions</span><b>{money(record["Tot_ded_amt"])}</b></div>
          <div class="row total"><span>Taxable income</span><b>{money(record["Taxable_Income"])}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Combined fields are itemised across the return labels the model expects using "
        "population-typical proportions."
    )

    analyse = st.button("Analyse taxpayer", type="primary", width="stretch")

if analyse or st.session_state.pop("auto_analyse", False):
    try:
        st.session_state["analysis_result"] = service.score_record(record, top_n=8)
    except ValueError as exc:
        st.error(str(exc))


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
n_segments = len(metadata["segment_names"])
n_features = len(metadata["anomaly_features"])
data_pill = (
    "Trained on the ATO 2022–23 individual sample file"
    if "ATO" in metadata.get("data_source", "")
    else "Trained on synthetic data"
)
st.markdown(
    f"""
    <div class="hero">
      <div class="kicker">TaxLens · peer-group anomaly review</div>
      <h1>Check your tax risk score</h1>
      <p>Enter a tax return and see how unusual it looks next to taxpayers with a similar income
      mix. The score comes from a peer-group Isolation Forest and every result is explained
      item by item.</p>
      <span class="pill">{data_pill}</span>
      <span class="pill">{n_segments} peer groups</span>
      <span class="pill">{n_features} behavioural features</span>
      <span class="pill">{metadata['contamination']:.0%} review budget</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="note">Runs entirely on this machine. Nothing entered here is sent to the ATO, '
    "to any external service, or to a language model. A flag is a statistical signal that a "
    "return is unusual for its peer group; it is not evidence of non-compliance.</div>",
    unsafe_allow_html=True,
)
st.write("")


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def score_gauge(result: dict) -> alt.Chart:
    """Horizontal gauge: where the score sits relative to the peer-group threshold."""
    score, threshold = result["anomaly_score"], result["threshold"]
    profile = result.get("segment_profile") or {}
    quantiles = profile.get("score_quantiles")
    lo = min(quantiles[0], score) if quantiles else min(0.3, score)
    hi = max(quantiles[-1], score) if quantiles else max(0.8, score)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    bands = pd.DataFrame(
        [
            {"start": lo, "end": threshold, "zone": "Within peer-group norm"},
            {"start": threshold, "end": hi, "zone": "Review zone"},
        ]
    )
    band_layer = (
        alt.Chart(bands)
        .mark_bar(height=26, cornerRadius=6)
        .encode(
            x=alt.X("start:Q", scale=alt.Scale(domain=[lo, hi]), title=None, axis=alt.Axis(grid=False)),
            x2="end:Q",
            color=alt.Color(
                "zone:N",
                scale=alt.Scale(domain=["Within peer-group norm", "Review zone"], range=["#CFE7DC", "#F8D7C4"]),
                legend=None,
            ),
        )
    )
    layers = [band_layer]
    if quantiles:
        grid = profile["score_quantile_grid"]
        ticks = pd.DataFrame(
            {"x": [quantiles[50], quantiles[90], quantiles[99]], "label": ["median", "p90", "p99"]}
        )
        layers.append(
            alt.Chart(ticks).mark_tick(color="#8A98A6", thickness=1.5, size=34).encode(x="x:Q")
        )
        layers.append(
            alt.Chart(ticks).mark_text(dy=-22, color="#8A98A6", fontSize=11).encode(x="x:Q", text="label:N")
        )
    marker = pd.DataFrame({"x": [score], "label": [f"this return  {score:.3f}"]})
    layers.append(
        alt.Chart(marker).mark_point(shape="triangle-down", size=260, filled=True, color=NAVY).encode(x="x:Q")
    )
    layers.append(
        alt.Chart(marker).mark_text(dy=40, fontWeight="bold", color=NAVY, fontSize=12).encode(x="x:Q", text="label:N")
    )
    thr = pd.DataFrame({"x": [threshold], "label": [f"threshold {threshold:.3f}"]})
    layers.append(alt.Chart(thr).mark_rule(color=RED, strokeWidth=2).encode(x="x:Q"))
    layers.append(
        alt.Chart(thr).mark_text(dy=-40, color=RED, fontSize=11, fontWeight="bold").encode(x="x:Q", text="label:N")
    )
    return alt.layer(*layers).properties(height=140).configure_view(strokeWidth=0)


def shap_chart(result: dict) -> alt.Chart:
    """Diverging bar chart of the strongest local SHAP contributions."""
    rows = []
    for driver in result["main_drivers"]:
        rows.append({**driver, "direction": "Raises the score"})
    for driver in result.get("protective_factors", []):
        rows.append({**driver, "direction": "Lowers the score"})
    frame = pd.DataFrame(rows).drop_duplicates("feature")
    frame["label"] = frame["feature"].map(research_agent.friendly_feature).str.capitalize()
    frame["shown_value"] = [
        (f"{v:.1%}" if "ratio" in f else f"{v:.0f}" if f == "items_reported" else money(v))
        for f, v in zip(frame["feature"], frame["value"])
    ]
    frame = frame.sort_values("contribution_towards_anomaly", ascending=False)
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("contribution_towards_anomaly:Q", title="SHAP contribution to anomaly score"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises the score", "Lowers the score"],
                    range=["#D9772F", TEAL],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("shown_value:N", title="Reported value"),
                alt.Tooltip("contribution_towards_anomaly:Q", title="SHAP", format="+.4f"),
            ],
        )
        .properties(height=30 * len(frame) + 40)
        .configure_view(strokeWidth=0)
    )


result = st.session_state.get("analysis_result")

if result is None:
    st.markdown("### How to use this prototype")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        '<div class="card"><h4>1 · Enter or load a record</h4><p>Use the sidebar to type a '
        "return's amounts, or load one of the example scenarios to see how the model responds "
        "to different taxpayer profiles.</p></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        '<div class="card"><h4>2 · Compare with peers</h4><p>The record is assigned to one of '
        f"{n_segments} income-composition peer groups and scored by that group's own "
        "Isolation Forest, so a rental investor is judged against rental investors.</p></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        '<div class="card"><h4>3 · Read the explanation</h4><p>SHAP values show which items '
        "drove the score, matched to the official ATO guidance a reviewer would consult "
        "and the questions they would ask.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Peer groups in the fitted model")
    profile_rows = []
    for segment_id, name in metadata["segment_names"].items():
        profile = service.segment_profiles.get(segment_id, {})
        profile_rows.append(
            {
                "Peer group": name,
                "Records": profile.get("n_records"),
                "Share": profile.get("share_of_population"),
                "Median total income": profile.get("median_total_income"),
                "Median deductions": profile.get("median_total_deductions"),
                "Lodged via agent": profile.get("agent_lodgement_share"),
                "Review threshold": metadata["segment_thresholds"][segment_id],
            }
        )
    st.dataframe(
        pd.DataFrame(profile_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Records": st.column_config.NumberColumn(format="%d"),
            "Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Median total income": st.column_config.NumberColumn(format="$%d"),
            "Median deductions": st.column_config.NumberColumn(format="$%d"),
            "Lodged via agent": st.column_config.NumberColumn(format="percent"),
            "Review threshold": st.column_config.NumberColumn(format="%.4f"),
        },
    )
else:
    percentile = result.get("peer_percentile")
    if result["flagged_for_review"]:
        st.markdown(
            '<div class="verdict flagged"><div class="icon">⚠️</div><div>'
            "<h3>Flagged for statistical review</h3>"
            f"<p>This return is more isolated than its peers in <b>{result['segment_name']}</b>. "
            "A reviewer should examine the items listed below; the flag itself does not establish "
            "that anything is incorrect.</p></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict clear"><div class="icon">✅</div><div>'
            f"<h3>Not flagged at the {metadata['contamination']:.0%} review threshold</h3>"
            f"<p>This return sits within the normal range for <b>{result['segment_name']}</b>. "
            "Its profile is not among the most isolated records in that peer group.</p></div></div>",
            unsafe_allow_html=True,
        )

    margin = result["anomaly_score"] - result["threshold"]
    margin_class = "up" if margin >= 0 else "down"
    margin_arrow = "▲" if margin >= 0 else "▼"
    percentile_text = f"{percentile:.0%}" if percentile is not None else "n/a"
    st.markdown(
        f"""
        <div class="kpis">
          <div class="kpi"><div class="label">Peer group</div><div class="value">Segment {result['segment_id']}</div>
            <div class="sub">{result['segment_name'].split(': ', 1)[-1]}</div></div>
          <div class="kpi"><div class="label">Anomaly score</div><div class="value">{result['anomaly_score']:.4f}</div>
            <div class="sub {margin_class}">{margin_arrow} {abs(margin):.4f} {'above' if margin >= 0 else 'below'} threshold</div></div>
          <div class="kpi"><div class="label">Review threshold</div><div class="value">{result['threshold']:.4f}</div>
            <div class="sub">top {metadata['contamination']:.0%} of this peer group</div></div>
          <div class="kpi"><div class="label">Peer percentile</div><div class="value">{percentile_text}</div>
            <div class="sub">of training records score lower</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("#### Where this return sits in its peer group")
        st.altair_chart(score_gauge(result), width="stretch")
        if percentile is not None:
            st.caption(
                f"The score is higher than {percentile:.0%} of the {result['segment_profile']['n_records']:,} "
                "training records in this peer group (Isolation Forest anomaly score). Green is the normal range, orange the review zone; "
                "grey ticks mark the group's median, 90th and 99th percentiles."
            )
    with right:
        profile = result.get("segment_profile")
        st.markdown("#### Peer group profile")
        if profile:
            st.markdown(
                f"""
                <div class="profile">
                  <div class="label">Peer group</div>
                  <div class="value">{result['segment_name']}</div>
                  <div style="height:0.5rem"></div>
                  <div class="row"><span>Records</span><b>{profile['n_records']:,} ({profile['share_of_population']:.1%})</b></div>
                  <div class="row"><span>Median total income</span><b>{money(profile['median_total_income'])}</b></div>
                  <div class="row"><span>Median deductions</span><b>{money(profile['median_total_deductions'])}</b></div>
                  <div class="row"><span>Lodged via tax agent</span><b>{profile['agent_lodgement_share']:.0%}</b></div>
                  <div class="row"><span>Records above threshold</span><b>{profile['flag_rate_2pct']:.1%}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="profile"><div class="value">{result["segment_name"]}</div></div>', unsafe_allow_html=True)

    st.markdown("#### What drove the score")
    st.altair_chart(shap_chart(result), width="stretch")
    st.caption(
        "Local SHAP contributions from the peer group's Isolation Forest. Orange bars raise the "
        "anomaly score; teal bars lower it. Hover a bar for the reported value."
    )

    review = research_agent.compose_review(result)
    st.markdown("#### Review narrative")
    st.markdown(review["explanation"])

    q_col, s_col = st.columns([1.1, 1])
    with q_col:
        st.markdown("#### Questions for human review")
        for question in review["review_questions"]:
            st.markdown(f'<div class="question">{question}</div>', unsafe_allow_html=True)
    with s_col:
        st.markdown("#### Relevant ATO guidance")
        for source in review["sources"]:
            st.markdown(
                f'<div class="card"><h4><a href="{source["url"]}" target="_blank">{source["title"]} ↗</a></h4>'
                f'<p>{source["relevance"]}</p></div>',
                unsafe_allow_html=True,
            )
    st.caption(review["limitation"])

    with st.expander("Model details and limitations"):
        st.write(result["interpretation"])
        st.write(
            f"Data source for the fitted artifacts: {metadata.get('data_source', 'unknown')}. "
            f"The simplified form leaves {len(result['defaulted_amount_columns'])} return labels "
            "at zero and itemises combined amounts (for example work-related expenses) across "
            "the underlying labels in population-typical proportions, so a real return with an "
            "unusual mix of items could score differently."
        )
        st.write(
            "A production assessment would require approved models, complete input fields, "
            "data-governance controls and human review before any action is taken."
        )


# ----------------------------------------------------------------------------
# Assistant
# ----------------------------------------------------------------------------
st.divider()
st.markdown("### Ask the review assistant")
st.caption(
    "Questions about the current result, thresholds, supporting records or the linked ATO "
    "guidance are answered from the curated local knowledge base. No external model is called."
)

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown(
                "**Sources:** "
                + " · ".join(f"[{s['title']}]({s['url']})" for s in message["sources"])
            )

question = st.chat_input("Try: Why was this record flagged?  ·  What records should be reviewed?")
if question:
    st.session_state["chat_messages"].append({"role": "user", "content": question, "sources": []})
    response = research_agent.answer_question(question, result=result)
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": response["answer"], "sources": response["sources"]}
    )
    st.rerun()


# ----------------------------------------------------------------------------
# Footer: provenance and disclaimer
# ----------------------------------------------------------------------------
_metadata_path = Path(service.model_dir) / "model_metadata.json"
trained_on = datetime.fromtimestamp(_metadata_path.stat().st_mtime).strftime("%d %B %Y")
n_training_records = sum(p.get("n_records", 0) for p in service.segment_profiles.values())
records_text = f"{n_training_records:,} individual returns" if n_training_records else "the training sample"

st.markdown(
    f"""
    <div class="footer">
      <h4>About the model</h4>
      Trained on {metadata.get('data_source', 'unknown data source')} — {records_text},
      grouped into {n_segments} income-composition peer groups, each scored by its own Isolation
      Forest (200 trees, {n_features} features, {metadata['contamination']:.0%} review budget).
      Explanations use SHAP TreeExplainer. Artifacts built on {trained_on}; scikit-learn artifacts
      are version-pinned in <code>requirements.txt</code>. No taxpayer data is stored in this
      repository, and nothing entered on this page leaves this machine.
      <div class="disclaimer">
        <b>Not for decision-making.</b> This is a research prototype developed for a Master of
        Data Science capstone. A score or flag is a statistical statement that a return is
        unusual relative to a peer group — it is not evidence of error, non-compliance or fraud,
        and it is not tax, legal or financial advice. Do not make, defer or justify any decision
        about a real taxpayer, return or audit on the basis of this tool. It is provided as is,
        without warranty of any kind, and the authors accept no responsibility or liability for
        any loss, action or outcome arising from its use.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

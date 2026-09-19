"""TaxLens: Streamlit interface for the peer-group taxpayer anomaly service."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from anomaly_service import TaxpayerAnomalyService
from record_builder import build_record, load_splits
from research_agent import TaxResearchAgent


# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TaxLens · Check your tax risk score",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Design tokens ("Ledger": cool off-white ground, bordered cards, one dark
# score card so the primary number is the only heavy object on the page).
# ----------------------------------------------------------------------------
T: dict[str, str] = dict(
    canvas="#F4F7FB", surface="#FFFFFF", surface_2="#F7F9FC", line="#DFE5EC", line_2="#EDF1F5",
    ink="#14263A", muted="#5B6B7C", navy="#0B2540", navy_2="#12365B",
    accent="#0E7C86", accent_ink="#0A5F67", accent_soft="#E1F1F2",
    amber="#E6A72F", amber_soft="#FFF3D6", amber_ink="#6B4A00",
    green="#1E8E5A", green_soft="#E3F4EA", green_ink="#155E3C", red="#C43D2F",
    radius="14px", radius_sm="10px", card_border="1px solid #DFE5EC", card_shadow="none",
    density="1", display_weight="800",
)

NAVY, NAVY_2, TEAL, TEAL_SOFT = T["navy"], T["navy_2"], T["accent"], T["accent_soft"]
AMBER, AMBER_SOFT, GREEN_SOFT = T["amber"], T["amber_soft"], T["green_soft"]
INK, MUTED, LINE, CANVAS = T["ink"], T["muted"], T["line"], T["canvas"]

# Lucide icons (ISC licence), one stroke weight throughout.
ICON_PATHS = {
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "edit": '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "clipboard": '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>',
    "sparkles": '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" x2="4" y1="22" y2="15"/>',
    "dollar": '<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "calculator": '<rect width="16" height="20" x="4" y="2" rx="2"/><line x1="8" x2="16" y1="6" y2="6"/><line x1="16" x2="16" y1="14" y2="18"/><path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/><path d="M12 14h.01"/><path d="M8 14h.01"/><path d="M12 18h.01"/><path d="M8 18h.01"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "settings": '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
}


def icon(name: str, size: int = 18, cls: str = "") -> str:
    """Inline Lucide icon; colour comes from CSS `currentColor`."""
    return (f'<svg class="ic {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{ICON_PATHS[name]}</svg>')


def tile(name: str, tone: str = "teal") -> str:
    return f'<span class="icon {tone}">{icon(name, 19)}</span>'


st.markdown(
    f"""
    <style>
    :root {{
        --canvas: {T['canvas']}; --surface: {T['surface']}; --surface-2: {T['surface_2']};
        --line: {T['line']}; --line-2: {T['line_2']}; --ink: {T['ink']}; --muted: {T['muted']};
        --navy: {T['navy']}; --navy-2: {T['navy_2']};
        --accent: {T['accent']}; --accent-ink: {T['accent_ink']}; --accent-soft: {T['accent_soft']};
        --amber: {T['amber']}; --amber-soft: {T['amber_soft']}; --amber-ink: {T['amber_ink']};
        --green: {T['green']}; --green-soft: {T['green_soft']}; --green-ink: {T['green_ink']}; --red: {T['red']};
        --radius: {T['radius']}; --radius-sm: {T['radius_sm']};
        --card-border: {T['card_border']}; --card-shadow: {T['card_shadow']};
        --d: {T['density']};
        --ease-out: cubic-bezier(0.23, 1, 0.32, 1); --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
    }}
    html, body, [data-testid="stAppViewContainer"] {{background: var(--canvas); color: var(--ink);}}
    [data-testid="stAppViewContainer"] > .main .block-container {{max-width: 100%; padding: 1rem 2rem 2.5rem 2rem;}}
    [data-testid="stHeader"] {{background: transparent;}}
    section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] {{display: none !important;}}
    [data-testid="stHeaderActionElements"], h3 > a, h4 > a {{display: none !important;}}
    h1, h2, h3, h4 {{color: var(--ink); letter-spacing: -0.02em; text-wrap: balance;}}
    p, li {{color: var(--ink);}}
    ::selection {{background: var(--accent-soft); color: var(--ink);}}
    * {{scrollbar-width: thin; scrollbar-color: var(--line) transparent;}}
    *:focus-visible {{outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px;}}
    .ic {{flex: none; vertical-align: -3px;}}
    .num, td, .value, .stat .value {{font-variant-numeric: tabular-nums;}}

    /* ---------- top bar ---------- */
    .brand {{display: inline-flex; align-items: center; gap: 0.6rem; padding-top: 0.2rem;}}
    .brand .mark {{width: 40px; height: 40px; border-radius: var(--radius-sm); background: var(--navy); color: #fff; display: inline-flex;
                   align-items: center; justify-content: center; font-weight: 800; font-size: 0.95rem; letter-spacing: 0.06em;}}
    .brand .name {{font-weight: 800; letter-spacing: -0.01em; font-size: 1.25rem; color: var(--ink); line-height: 1.1;}}
    .brand .tag {{color: var(--muted); font-size: 0.78rem; letter-spacing: 0;}}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{
        border-radius: var(--radius-sm) !important; font-weight: 600; color: var(--ink); background: var(--surface);
        border-color: var(--line); transition: background-color 120ms ease, color 120ms ease, transform 160ms var(--ease-out);
    }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"]:active {{transform: scale(0.97);}}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {{
        background: var(--navy) !important; color: #fff !important; border-color: var(--navy) !important;
    }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] * {{color: #fff !important;}}
    .status-pill {{
        display: inline-flex; align-items: center; gap: 0.45rem; background: var(--accent-soft); color: var(--accent-ink);
        padding: 0.45rem 0.9rem; border-radius: 999px; font-size: 0.86rem; font-weight: 600; margin-top: 0.15rem;
    }}
    .status-pill i {{width: 8px; height: 8px; border-radius: 50%; background: var(--accent); display: inline-block;}}
    div[data-testid="stDownloadButton"] button, div[data-testid="stDownloadButton"] button * {{color: #fff !important;}}
    div[data-testid="stDownloadButton"] button {{
        background: var(--navy); border: none; border-radius: var(--radius-sm); padding: 0.5rem 1rem; font-weight: 600;
        transition: background-color 120ms ease, transform 160ms var(--ease-out);
    }}
    div[data-testid="stDownloadButton"] button:active {{transform: scale(0.97);}}
    @media (hover: hover) and (pointer: fine) {{ div[data-testid="stDownloadButton"] button:hover {{background: var(--navy-2); color: #fff;}} }}

    /* ---------- page header ---------- */
    .title-row {{display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; margin-top: calc(0.5rem * var(--d));}}
    .title-row h1 {{margin: 0; font-size: 2rem; font-weight: {T['display_weight']}; color: var(--ink);}}
    .badge {{display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.35rem 0.8rem; border-radius: var(--radius-sm); font-weight: 700; font-size: 0.92rem;}}
    .badge.warn {{background: var(--amber-soft); color: var(--amber-ink);}}
    .badge.ok {{background: var(--green-soft); color: var(--green-ink);}}
    .badge.soft {{background: var(--amber-soft); color: var(--amber-ink);}}
    .subtitle {{color: var(--muted); font-size: 1rem; margin: 0.2rem 0 calc(1.1rem * var(--d)) 0; line-height: 1.5;}}

    /* ---------- cards ---------- */
    .card {{background: var(--surface); border: var(--card-border); box-shadow: var(--card-shadow); border-radius: var(--radius); padding: calc(1rem * var(--d)) calc(1.15rem * var(--d)); margin-bottom: calc(1rem * var(--d));}}
    .card h3 {{font-size: 1.12rem; margin: 0 0 0.6rem 0; display: flex; align-items: center; gap: 0.6rem;}}
    .card p {{color: var(--ink); font-size: 0.95rem;}}
    [data-testid="stVerticalBlockBorderWrapper"] {{border-color: var(--line) !important; border-radius: var(--radius) !important; background: var(--surface);}}
    .icon {{width: 38px; height: 38px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; flex: none;}}
    .icon.teal {{background: var(--accent-soft); color: var(--accent-ink);}}
    .icon.navy {{background: var(--surface-2); color: var(--ink);}}
    .icon.amber {{background: var(--amber-soft); color: var(--amber-ink);}}
    .kpi, .stat {{background: var(--surface); border: var(--card-border); box-shadow: var(--card-shadow); border-radius: var(--radius); padding: calc(0.9rem * var(--d)) calc(1.1rem * var(--d)); display: flex; align-items: center; gap: 0.9rem; margin-bottom: calc(1rem * var(--d));}}
    .kpi .label, .stat .label {{color: var(--muted); font-size: 0.9rem;}}
    .kpi .value, .stat .value {{color: var(--ink); font-size: 1.85rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em;}}
    .stat .value {{font-size: 2rem;}}
    .stat .icon {{width: 44px; height: 44px;}}
    .kpi .side {{margin-left: auto; padding-left: 0.9rem; border-left: 1px solid var(--line); color: var(--muted); font-size: 0.8rem; text-align: center; min-width: 62px;}}

    /* ---------- score card ---------- */
    .score {{background: linear-gradient(160deg, var(--navy) 0%, var(--navy-2) 100%); color: #fff; border-radius: var(--radius); padding: 1.2rem 1.4rem 1.1rem 1.4rem; margin-bottom: calc(1rem * var(--d));}}
    .score .head {{display: flex; justify-content: space-between; align-items: center; font-size: 1.1rem; font-weight: 600;}}
    .score .chip {{background: rgba(255,255,255,0.14); border-radius: 999px; padding: 0.3rem 0.85rem; font-size: 0.82rem; font-weight: 500;}}
    .score .big {{font-size: 4.6rem; font-weight: 800; line-height: 1; margin: 0.4rem 0 0.6rem 0; letter-spacing: -0.03em; font-variant-numeric: tabular-nums;}}
    .score .flag {{display: inline-block; padding: 0.35rem 0.9rem; border-radius: var(--radius-sm); font-weight: 700; font-size: 0.92rem;}}
    .score .flag.warn {{background: var(--amber); color: #2B1F00;}}
    .score .flag.ok {{background: #4CC38A; color: #06301C;}}
    .score .sub {{color: #C9D6E4; font-size: 0.95rem; margin: 0.7rem 0 1rem 0;}}
    .track {{position: relative; height: 10px; border-radius: 999px; background: linear-gradient(90deg, #0E7C86 0%, #5FC3B0 45%, #F2B33D 100%); margin: 1.6rem 0 0.4rem 0;}}
    .track .thr {{position: absolute; top: -8px; width: 3px; height: 26px; background: #fff; border-radius: 2px; transform: translateX(-50%);}}
    .track .dot {{position: absolute; top: -7px; width: 24px; height: 24px; border-radius: 50%; background: #F2B33D; border: 3px solid #fff; transform: translateX(-50%);
                  box-shadow: 0 2px 6px rgba(0,0,0,0.3); animation: settle 520ms var(--ease-out) both;}}
    .track .dot.ok {{background: #4CC38A;}}
    @keyframes settle {{from {{transform: translateX(-50%) scale(0.6); opacity: 0.6;}} to {{transform: translateX(-50%) scale(1); opacity: 1;}}}}
    @media (prefers-reduced-motion: reduce) {{ .track .dot {{animation: none;}} }}
    .track .lbl {{position: absolute; top: 22px; transform: translateX(-50%); font-size: 0.8rem; color: #C9D6E4; white-space: nowrap; text-align: center;}}
    .track .lbl b {{display: block; color: #fff; font-size: 0.9rem;}}
    .track .end {{position: absolute; top: 22px; font-size: 0.8rem; color: #C9D6E4;}}
    .score .foot {{color: #9FB3C8; font-size: 0.85rem; margin-top: 6.4rem;}}

    /* ---------- context / driver tables ---------- */
    .ctx .row {{display: flex; justify-content: space-between; gap: 1rem; padding: 0.55rem 0; border-bottom: 1px solid var(--line-2); font-size: 0.95rem; color: var(--muted);}}
    .ctx .row:last-of-type {{border-bottom: none;}}
    .ctx .row b {{color: var(--ink); text-align: right;}}
    .ctx .lead {{color: var(--muted); font-size: 0.92rem; margin: -0.2rem 0 0.6rem 0;}}
    .infobox {{background: var(--surface-2); border-radius: var(--radius-sm); padding: 0.6rem 0.8rem; color: var(--muted); font-size: 0.88rem; margin-top: 0.6rem; display: flex; gap: 0.5rem; align-items: flex-start;}}
    .tip {{background: var(--green-soft); border-radius: var(--radius-sm); padding: 0.6rem 0.9rem; color: var(--green-ink); font-size: 0.9rem; margin-top: 0.7rem; display: flex; gap: 0.5rem; align-items: center;}}
    table.drivers {{width: 100%; border-collapse: collapse; font-size: 0.95rem;}}
    table.drivers th {{text-align: left; color: var(--muted); font-weight: 500; padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--line);}}
    table.drivers td {{padding: 0.55rem 0.5rem; border: none; border-bottom: 1px solid var(--line-2); vertical-align: middle; color: var(--ink);}}
    table.drivers th {{border-left: none; border-right: none; border-top: none;}}
    table.drivers tr:last-child td {{border-bottom: none;}}
    .rank {{display: inline-flex; width: 28px; height: 28px; border-radius: 50%; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;}}
    .rank.r1 {{background: var(--amber-soft); color: var(--amber-ink);}}
    .rank.r2 {{background: var(--surface-2); color: var(--ink);}}
    .rank.r3 {{background: var(--accent-soft); color: var(--accent-ink);}}
    .rank.down {{background: var(--green-soft); color: var(--green-ink);}}
    .link-row, .link-row:hover, .link-row span {{text-decoration: none !important;}}
    .link-row {{display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; background: var(--surface-2); border-radius: var(--radius-sm); padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; color: var(--accent-ink) !important; font-weight: 600; text-decoration: none; transition: background-color 120ms ease;}}
    @media (hover: hover) and (pointer: fine) {{ .link-row:hover {{background: var(--accent-soft);}} }}
    .small {{color: var(--muted); font-size: 0.85rem;}}

    /* ---------- assistant column ---------- */
    .assistant {{background: var(--surface); border: var(--card-border); box-shadow: var(--card-shadow); border-radius: var(--radius); padding: 1.1rem 1.15rem 0.6rem 1.15rem;}}
    .assistant .head {{display: flex; gap: 0.8rem; align-items: center; margin-bottom: 0.4rem;}}
    .assistant .head .icon {{width: 46px; height: 46px; background: var(--accent); color: #fff;}}
    .assistant .head h3 {{margin: 0; font-size: 1.2rem;}}
    .assistant .head p {{margin: 0; color: var(--muted); font-size: 0.9rem;}}
    .next {{border: 1px solid var(--line); border-radius: var(--radius); padding: 0.9rem 1rem; margin: 0.6rem 0; background: var(--surface);}}
    .next h4 {{margin: 0 0 0.35rem 0; font-size: 1.02rem; display: flex; align-items: center; gap: 0.45rem;}}
    .next p {{margin: 0 0 0.6rem 0; color: var(--ink); font-size: 0.95rem;}}
    .next a {{color: var(--accent-ink) !important; font-weight: 600; text-decoration: none; display: inline-flex; align-items: center; gap: 0.4rem;}}
    .check-pill {{display: inline-flex; gap: 0.5rem; align-items: center; background: var(--green-soft); color: var(--green-ink); border-radius: var(--radius-sm); padding: 0.5rem 0.8rem; font-size: 0.88rem; width: 100%; box-sizing: border-box;}}
    [data-testid="stChatMessage"] {{background: var(--surface-2); border-radius: var(--radius); padding: 0.6rem 0.8rem;}}
    .stButton > button {{border-radius: var(--radius-sm); border: 1px solid var(--line); background: var(--surface); color: var(--ink); font-weight: 600; transition: border-color 120ms ease, color 120ms ease, background-color 120ms ease, transform 160ms var(--ease-out);}}
    .stButton > button:active {{transform: scale(0.97);}}
    @media (hover: hover) and (pointer: fine) {{ .stButton > button:hover {{border-color: var(--accent); color: var(--accent-ink);}} }}
    .stButton > button[kind="primary"] {{background: var(--accent); color: #fff; border: none;}}
    @media (hover: hover) and (pointer: fine) {{ .stButton > button[kind="primary"]:hover {{background: var(--accent-ink); color: #fff;}} }}
    .disclaimer {{margin-top: calc(1.4rem * var(--d)); background: var(--amber-soft); border-radius: var(--radius); padding: 0.85rem 1rem; color: var(--amber-ink); font-size: 0.86rem; line-height: 1.5;}}
    .disclaimer b {{color: var(--amber-ink);}}
    .chart-title {{font-weight: 700; font-size: 1.12rem; color: var(--ink); margin: 0 0 0.2rem 0;}}
    .chart-note {{color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem;}}
    .finding-card {{display: flex; gap: 0.9rem; align-items: flex-start; padding: 0.4rem 0.2rem;}}
    .finding-card .no {{background: var(--accent-soft); color: var(--accent-ink); border-radius: var(--radius-sm); width: 48px; height: 48px; display: inline-flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.05rem; flex: none;}}
    .finding-card h4 {{margin: 0 0 0.2rem 0; font-size: 1.02rem;}}
    .finding-card p {{margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.45;}}
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
NAV_ICONS = {"Overview": "Overview", "Record entry": "Record entry", "Guidance": "Guidance", "About": "About"}


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

brand_col, nav_col, pill_col = st.columns([1.6, 3.4, 1.6], vertical_alignment="center")
brand_col.markdown(
    '<div class="brand"><span class="mark">TL</span><span><div class="name">TaxLens</div>'
    '<div class="tag">Explainable tax anomaly insights</div></span></div>',
    unsafe_allow_html=True,
)
with nav_col:
    page = st.segmented_control(
        "Navigate", PAGES, key="nav", format_func=lambda p: NAV_ICONS[p],
        label_visibility="collapsed", width="stretch",
    ) or "Overview"
pill_col.markdown(
    f'<span class="status-pill"><i></i>{"ATO sample-file model" if REAL_DATA else "Synthetic demo"}</span>',
    unsafe_allow_html=True,
)
SUMMARY_PATH = Path(service.model_dir).parent / "results" / "summary_public.json"


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
            <span class="lbl" style="left:{pos(score)}; top:62px"><b>{score:.2f}</b>Score</span>
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
          <h3>{tile("users")}Peer-group context</h3>
          <div style="font-weight:700;color:{INK};margin-top:-0.3rem">{short_name}</div>
          <div class="lead">Compared with returns sharing a similar income mix.</div>
          {rows_html}
          <div class="infobox">{icon("info", 16)}<span>A review budget is not an estimated non-compliance rate.</span></div>
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
          <h3>{tile("trend")}What is driving the result?
            <span class="small" style="margin-left:auto;font-weight:400">Local SHAP · amounts and ratios describe this record</span></h3>
          <table class="drivers">
            <thead><tr><th>#</th><th>Factor</th><th>Record value</th><th>Why it matters</th><th>SHAP</th></tr></thead>
            <tbody>{body}</tbody>
          </table>
          <div class="tip">{icon("bulb", 16)}<span>The model flags unusual combinations. Supporting evidence determines the next step.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant(review: dict | None) -> None:
    st.markdown(
        f'<div class="assistant"><div class="head"><span class="icon">{icon("sparkles", 20)}</span>'
        "<div><h3>Review assistant</h3><p>Model context + curated guidance</p></div></div></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.button("Explain this score", on_click=ask, args=("Why was this record flagged?",), width="stretch")
    c2.button("What should I check?", on_click=ask, args=("What records should be reviewed?",), width="stretch")

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
              <h4>{icon("flag", 16)}Recommended next step</h4>
              <p>{review['review_questions'][0]}</p>
              <a href="{source['url']}" target="_blank">{icon("file", 15)}{source['title']}{icon("external", 13)}</a>
            </div>
            <div class="check-pill">{icon("check", 15)}Local guidance · Human review required</div>
            """,
            unsafe_allow_html=True,
        )
    question = st.chat_input("Ask about this result…")
    if question:
        ask(question)
        st.rerun()
    st.markdown(
        '<div class="small" style="margin-top:0.6rem">Answers come from the curated local knowledge base. '
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
                f'<div class="card"><h3>{tile("edit")}Start with a record</h3>'
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
    badge = (f'<span class="badge warn">{icon("alert", 16)}Review suggested</span>' if result["flagged_for_review"]
             else f'<span class="badge ok">{icon("check", 16)}No review flag</span>')
    main, side = st.columns([2.35, 1], gap="medium")
    with main:
        st.markdown(
            f'<div class="title-row"><h1>Taxpayer record</h1>{badge}</div>'
            f'<div class="subtitle">Individual return &nbsp;·&nbsp; {short_name} &nbsp;·&nbsp; {label}</div>',
            unsafe_allow_html=True,
        )
        deduction_share = record["Tot_ded_amt"] / max(record["Tot_IncLoss_amt"], 1.0)
        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="kpi">{tile("dollar")}<div><div class="label">Total income</div>'
                    f'<div class="value">{money(record["Tot_IncLoss_amt"])}</div></div>'
                    f'<div class="side">Individual<br>return</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi">{tile("file")}<div><div class="label">Total deductions</div>'
                    f'<div class="value">{money(record["Tot_ded_amt"])}</div></div>'
                    f'<div class="side">{deduction_share:.1%}<br>of income</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi">{tile("calculator")}<div><div class="label">Taxable income</div>'
                    f'<div class="value">{money(record["Taxable_Income"])}</div></div>'
                    f'<div class="side">{short_name.split(" (")[0]}</div></div>', unsafe_allow_html=True)

        s_col, c_col = st.columns([1.45, 1], gap="medium")
        with s_col:
            render_score_card(result)
        with c_col:
            render_context_card(result)

        render_driver_table(result)

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
    form = st.session_state["form"]
    top1, _ = st.columns([1.4, 3])
    lodgment = top1.selectbox(
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
        label = next((name for name in SCENARIOS if values == scenario_values(name)), "Manual entry")
        st.button("Analyse taxpayer", type="primary", width="stretch", on_click=analyse, args=(label,))


def render_guidance() -> None:
    st.markdown('<div class="title-row"><h1>Guidance</h1></div>'
                '<div class="subtitle">The curated ATO references the review assistant draws on.</div>',
                unsafe_allow_html=True)
    columns = st.columns(2, gap="medium")
    for i, source in enumerate(research_agent.knowledge.values()):
        questions = "".join(f"<li>{q}</li>" for q in source["review_questions"])
        columns[i % 2].markdown(
            f'<div class="card"><h3>{tile("book")}{source["title"].removeprefix("ATO: ")}</h3>'
            f'<p class="small">{source["summary"]}</p><ul class="small">{questions}</ul>'
            f'<a class="link-row" href="{source["url"]}" target="_blank"><span>Open ATO guidance</span>{icon("external", 14)}</a></div>',
            unsafe_allow_html=True,
        )



# ----------------------------------------------------------------------------
# About page: the model and its label-free evaluation on the real-data run
# ----------------------------------------------------------------------------
@st.cache_data
def load_summary() -> dict | None:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else None


def score_distribution_chart(profile: dict, threshold: float, bins: int = 40) -> alt.Chart:
    """Histogram of the segment's anomaly scores, derived from its quantile curve (CDF)."""
    q = np.asarray(profile["score_quantiles"], dtype=float)
    grid = np.asarray(profile["score_quantile_grid"], dtype=float)
    edges = np.linspace(q[0], q[-1], bins + 1)
    cdf = np.interp(edges, q, grid)
    share = np.diff(cdf)
    mids = (edges[:-1] + edges[1:]) / 2
    frame = pd.DataFrame({"x": mids, "share": share})
    base = alt.Chart(frame).encode(
        x=alt.X("x:Q", title="Isolation Forest anomaly score", scale=alt.Scale(domain=[float(edges[0]), float(edges[-1])]),
                axis=alt.Axis(grid=False, format=".2f")),
        y=alt.Y("share:Q", title="Share of returns", axis=alt.Axis(format=".0%", grid=True)),
        tooltip=[alt.Tooltip("x:Q", title="Score", format=".3f"), alt.Tooltip("share:Q", title="Share of returns", format=".1%")],
    )
    norm = base.mark_area(interpolate="monotone", color="#5FA8A0", opacity=0.85, line={"color": "#2F8F86"})
    review = base.transform_filter(alt.datum.x >= threshold).mark_area(interpolate="monotone", color="#E0A32A", opacity=0.95)
    bars = alt.layer(norm, review)
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="#C43D2F", strokeWidth=2, strokeDash=[5, 3]).encode(x="t:Q")
    label = alt.Chart(pd.DataFrame({"t": [threshold], "l": [f"Review cut-off {threshold:.3f}"]})).mark_text(
        align="right", dx=-6, color="#C43D2F", fontWeight="bold", fontSize=11, baseline="top"
    ).encode(x="t:Q", y=alt.value(6), text="l:N")
    return alt.layer(bars, rule, label).properties(height=230).configure_view(strokeWidth=0)


def threshold_sensitivity_chart(sweep: list[dict], selected: float) -> alt.Chart:
    frame = pd.DataFrame([{"budget": f"{int(round(r['contamination'] * 100))}%", "flagged": r["n_flagged_segmented"],
                           "selected": abs(r["contamination"] - selected) < 1e-9, "jaccard": r.get("jaccard")} for r in sweep])
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, size=54).encode(
        x=alt.X("budget:N", title="Review threshold (top %)", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("flagged:Q", title="Records flagged", axis=alt.Axis(format=",.0f", grid=True)),
        color=alt.Color("selected:N", scale=alt.Scale(domain=[True, False], range=[TEAL, "#9ED3D6"]), legend=None),
        tooltip=[alt.Tooltip("budget:N", title="Budget"), alt.Tooltip("flagged:Q", title="Flagged", format=","),
                 alt.Tooltip("jaccard:Q", title="Agreement with population-wide", format=".2f")],
    )
    text = alt.Chart(frame).mark_text(dy=-8, fontSize=12, fontWeight="bold", color=INK).encode(
        x=alt.X("budget:N", sort=None), y="flagged:Q", text=alt.Text("flagged:Q", format=","))
    return alt.layer(bars, text).properties(height=230).configure_view(strokeWidth=0)


def shap_contribution_chart(importance: dict[str, float], top: int = 8) -> alt.Chart:
    items = list(importance.items())[:top]
    frame = pd.DataFrame({"feature": [research_agent.friendly_feature(f).capitalize() for f, _ in items],
                          "value": [v for _, v in items]})
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color="#2F8F86").encode(
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("value:Q", title="Mean absolute SHAP contribution", axis=alt.Axis(grid=True)),
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("value:Q", title="Mean |SHAP|", format=".3f")],
    ).properties(height=26 * len(frame) + 40).configure_view(strokeWidth=0)


def _rule(summary: dict, name: str) -> dict:
    return next((r for r in summary.get("rule_sanity_checks", []) if r["rule"] == name), {})


def render_about() -> None:
    summary = load_summary()
    if summary is None:
        st.warning("No results summary found. Run `python taxpayer_framework.py` and "
                   "`python tools/export_segment_profiles.py` to produce results/summary_public.json.")
        return

    n_records = summary["n_records"]
    n_ratios = len(summary.get("ratio_features", []))
    n_features = summary.get("n_ad_features", len(metadata["anomaly_features"]))
    n_flagged = summary["n_flagged_seg"]
    budget = metadata["contamination"]
    label_free = summary.get("n_injected", 0) == 0

    t1, t2 = st.columns([4, 1.3], vertical_alignment="center")
    t1.markdown(
        '<div class="title-row"><h1>About the model</h1></div>'
        f'<div class="subtitle">Isolation Forest &nbsp;·&nbsp; {summary["data_source"]}<br>'
        "Exploring unusual financial-ratio profiles within taxpayer peer groups.</div>",
        unsafe_allow_html=True,
    )
    t2.markdown(
        f'<span class="badge soft">{icon("alert", 16)}{"Label-free evaluation" if label_free else "Includes injected test anomalies"}</span>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    for col, icon_name, label, value in (
        (k1, "file", "Records analysed", f"{n_records:,}"),
        (k2, "chart", "Behavioural features", f"{n_features} <span style='font-size:1rem;color:{MUTED};font-weight:600'>incl. {n_ratios} ratios</span>"),
        (k3, "pie", "Review threshold", f"Top {budget:.0%}"),
        (k4, "alert", "Flagged for review", f"{n_flagged:,}"),
    ):
        col.markdown(f'<div class="stat">{tile(icon_name)}<div><div class="label">{label}</div>'
                     f'<div class="value">{value}</div></div></div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1, st.container(border=True):
        st.markdown('<div class="chart-title">Anomaly score distribution</div>', unsafe_allow_html=True)
        names = metadata["segment_names"]
        chosen = st.selectbox("Peer group", list(names), format_func=lambda k: names[k], key="research_segment",
                              label_visibility="collapsed")
        profile = service.segment_profiles.get(chosen)
        if profile:
            st.altair_chart(score_distribution_chart(profile, metadata["segment_thresholds"][chosen]), width="stretch")
            st.markdown(f'<div class="chart-note">Density reconstructed from the score quantiles of {profile["n_records"]:,} returns; '
                        f'the top {budget:.0%} of each peer group lies beyond its own cut-off.</div>', unsafe_allow_html=True)
        else:
            st.info("Segment profiles not exported yet.")
    with c2, st.container(border=True):
        st.markdown('<div class="chart-title">Review threshold sensitivity</div>', unsafe_allow_html=True)
        st.altair_chart(threshold_sensitivity_chart(summary["contamination_sweep"], budget), width="stretch")
        agree = {round(r["contamination"], 3): r.get("jaccard") or 0 for r in summary["contamination_sweep"]}
        st.markdown(
            f'<div class="chart-note">Review budget, not an estimate of non-compliance. Agreement with a single '
            f'population-wide forest rises from {agree.get(0.01, 0):.0%} of flags at 1% to {agree.get(0.1, 0):.0%} at 10%.</div>',
            unsafe_allow_html=True,
        )

    c3, c4 = st.columns([1, 1], gap="medium")
    with c3, st.container(border=True):
        st.markdown('<div class="chart-title">Peer-group results</div>', unsafe_allow_html=True)
        rows = [{"Group": r["segment_name"].split(": ", 1)[-1], "Analysed": r["n"], "Flagged": r["flagged_segmented"],
                 "Flag rate": r["rate_segmented_pct"] / 100, "One forest for all": r["rate_population_wide_pct"] / 100}
                for r in summary["flag_rate_by_segment"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Analysed": st.column_config.NumberColumn(format="localized"),
                                    "Flagged": st.column_config.NumberColumn(format="localized"),
                                    "Flag rate": st.column_config.NumberColumn(format="percent"),
                                    "One forest for all": st.column_config.NumberColumn(format="percent", help="Share of the group a single population-wide forest would flag with the same total budget")})
        st.markdown('<div class="chart-note">Same review threshold applied within each group; the last column shows how unevenly one population-wide forest spends the same budget.</div>', unsafe_allow_html=True)
    with c4, st.container(border=True):
        st.markdown('<div class="chart-title">Feature contributions</div>', unsafe_allow_html=True)
        st.altair_chart(shap_contribution_chart(summary["shap_if_top10"]), width="stretch")
        st.markdown(f'<div class="chart-note">Global SHAP importance across the peer-group forests; the top ten features carry '
                    f'{summary.get("shap_if_top10_share", 0):.0%} of all attribution.</div>', unsafe_allow_html=True)

    rare = _rule(summary, "claims_item_rare_in_segment")
    rare3 = _rule(summary, "claims_3plus_rare_items")
    hgb = next((r for r in summary.get("surrogate_agreement", []) if r["model"].startswith("Hist") and "segmented" in r["reference_flags"]), {})
    income = next((r for r in summary.get("income_confounding", []) if r["design"] == "segmented"), {})
    with st.container(border=True):
        st.markdown('<div class="chart-title">Findings to report</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3, gap="medium")
        f1.markdown(
            '<div class="finding-card"><span class="no">01</span><div><h4>Unusual profiles</h4>'
            f'<p>{rare.get("pct_flagged_segmented", 0):.0f}% of flagged returns claim an item rare in their peer group, against '
            f'{rare.get("pct_unflagged_segmented", 0):.0f}% of unflagged returns; three or more rare items are '
            f'{rare3.get("lift_segmented", 0):.0f}× more common among flags. Flagged returns have a median income '
            f'{income.get("income_multiple_of_population_median", 0):.1f}× the population median.</p></div></div>',
            unsafe_allow_html=True)
        f2.markdown(
            '<div class="finding-card"><span class="no">02</span><div><h4>Threshold stability</h4>'
            f'<p>Re-fitting the forests with five seeds keeps score rankings at Spearman {summary.get("seed_stability_mean_spearman", 0):.2f}; '
            f'{summary.get("pct_primary_flags_in_all_seeds", 0):.0f}% of flags appear under every seed. Peer-group and '
            f'population-wide designs share {summary.get("flag_jaccard", 0):.0%} of their {budget:.0%} lists — they are different review sets.</p></div></div>',
            unsafe_allow_html=True)
        f3.markdown(
            '<div class="finding-card"><span class="no">03</span><div><h4>Model explanations</h4>'
            "<p>Local SHAP sums reproduce each forest's own ranking (ρ = 1.00 in every group). A gradient-boosting surrogate "
            f'recovers the flag rule with ROC-AUC {hgb.get("roc_auc_vs_reference", 0):.3f} and matches '
            f'{hgb.get("jaccard", 0):.0%} of the flagged set at the same budget, so the rule is consistent and learnable.</p></div></div>',
            unsafe_allow_html=True)

    sil = next((r["silhouette"] for r in summary["cluster_selection"] if r["k"] == summary["k_selected"]), 0)
    built = datetime.fromtimestamp((Path(service.model_dir) / "model_metadata.json").stat().st_mtime)
    with st.expander("Methodology, model details & limitations"):
        m1, m2 = st.columns([1, 1.3], gap="medium")
        m1.markdown(
            f"""
            <div class="ctx">
              <div class="row"><span>Training data</span><b>{summary['data_source']}</b></div>
              <div class="row"><span>Records</span><b>{n_records:,}</b></div>
              <div class="row"><span>Peer groups</span><b>{summary['k_selected']} (K-Means, silhouette {sil:.2f})</b></div>
              <div class="row"><span>Detector</span><b>Isolation Forest per group, 200 trees</b></div>
              <div class="row"><span>Features</span><b>{n_features} log-amounts and ratios</b></div>
              <div class="row"><span>Review budget</span><b>{budget:.0%} per peer group</b></div>
              <div class="row"><span>Explanations</span><b>SHAP TreeExplainer</b></div>
              <div class="row"><span>Artifacts built</span><b>{built:%d %B %Y}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m2.markdown(
            f"""
            **Pipeline.** {n_records:,} returns from the {summary['data_source']}. {summary.get('n_amount_features_kept', '')} amount
            variables survive a sparsity screen (non-zero for at least 1% of taxpayers) and a redundancy screen (|r| < 0.95) and are
            signed-log transformed; {n_ratios} behavioural ratios describe the shape of the return. K-Means on nine
            income-composition features gives {summary['k_selected']} peer groups (silhouette {sil:.2f}). One Isolation Forest
            (200 trees) is fitted per group and the top {budget:.0%} of each group is flagged. SHAP TreeExplainer provides
            global and per-return explanations; a Random Forest and histogram gradient boosting are trained only as
            surrogates of the flag rule.

            **Label-free evaluation.** No record was modified and no synthetic anomaly labels were used. Evaluation relies on
            agreement between designs at each budget, flag rates by peer group and income decile, a ratios-only ablation,
            score stability across five seeds and rule-based sanity checks on flagged rows.

            **Limitations.** The file carries no audit outcomes, so nothing here measures non-compliance. A review budget
            flags the top {budget:.0%} of every group even if all returns are correct. Scores still correlate with income
            (ρ = {income.get('spearman_score_vs_income', 0):.2f}) because the amount features carry scale. Results reflect one
            income year and a 2% sample.
            """
        )
    st.markdown('<div class="chart-note">Anomalies support review; they do not establish non-compliance.</div>', unsafe_allow_html=True)


if page == "Overview":
    render_overview()
elif page == "Record entry":
    render_record_entry()
elif page == "About":
    render_about()
else:
    render_guidance()

st.markdown(
    """
    <div class="disclaimer"><b>Not for decision-making.</b> TaxLens is a research prototype from a Master of Data
    Science capstone. A score or flag is a statistical statement that a return is unusual relative to its peer group;
    it is not evidence of error, non-compliance or fraud, and it is not tax, legal or financial advice. Do not make,
    defer or justify any decision about a real taxpayer, return or audit on the basis of this tool. It is provided as
    is, without warranty of any kind, and the authors accept no responsibility or liability for any loss, action or
    outcome arising from its use. Nothing entered here leaves this machine.</div>
    """,
    unsafe_allow_html=True,
)

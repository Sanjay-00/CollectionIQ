"""All CSS and JS for CollectionIQ  -  injected once at app startup."""

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 3rem 2.5rem !important; max-width: 100% !important; }
.stApp { background: #f2f2f2; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f0f0; }
::-webkit-scrollbar-thumb { background: #FFC000; border-radius: 3px; }

/* ── Sidebar collapse button (inside sidebar) ── */
[data-testid="stSidebarCollapseButton"] button {
    background: #FFC000 !important; border: 2px solid #000 !important;
    border-radius: 6px !important; color: #000 !important;
}
[data-testid="stSidebarCollapseButton"] button:hover { background: #e6ac00 !important; }
[data-testid="stSidebarCollapseButton"] svg { fill: #000 !important; }

/* ── Sidebar collapsed state (expand button) ── */
[data-testid="stSidebarCollapsed"] {
    background: #FFC000 !important; border-radius: 0 10px 10px 0 !important;
    width: 36px !important; min-height: 80px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    z-index: 99999 !important; box-shadow: 4px 0 16px rgba(0,0,0,0.35) !important;
    cursor: pointer !important;
    border-top: 2px solid #000 !important; border-right: 2px solid #000 !important;
    border-bottom: 2px solid #000 !important; border-left: none !important;
    visibility: visible !important; opacity: 1 !important;
}
[data-testid="stSidebarCollapsed"] button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    transform: none !important; width: 100% !important; height: 100% !important; min-height: 80px !important;
}
[data-testid="stSidebarCollapsed"] svg { fill: #000000 !important; width: 20px !important; height: 20px !important; }
[data-testid="stSidebarCollapsed"]:hover { background: #e6ac00 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d0d !important; border-right: 3px solid #FFC000 !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.3) !important;
}
[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
[data-testid="stSidebar"] .stSelectbox > label {
    color: #FFC000 !important; font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.8px;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #1a1a1a !important; border: 1px solid #2d2d2d !important;
    color: #fff !important; border-radius: 8px !important;
}
/* ── Sidebar segment expander (checkbox dropdown) ── */
[data-testid="stSidebar"] .stExpander {
    background: #1a1a1a !important; border: 1px solid #2d2d2d !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stExpander summary {
    color: #FFC000 !important; font-weight: 600 !important; font-size: 11px !important;
    text-transform: uppercase !important; letter-spacing: 0.8px !important;
    padding: 10px 12px !important;
}
[data-testid="stSidebar"] .stExpander summary svg { fill: #FFC000 !important; }
[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] {
    background: #111 !important; padding: 6px 12px 10px !important;
    border-top: 1px solid #2d2d2d !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    color: #c8c8c8 !important; font-size: 12px !important;
    text-transform: none !important; letter-spacing: normal !important;
}
[data-testid="stSidebar"] .stCheckbox input:checked + label { color: #FFC000 !important; font-weight: 600 !important; }

[data-testid="stSidebar"] hr { border-color: #222 !important; }
[data-testid="stSidebar"] [data-testid="stButton"][key="clear_cache_btn"] > button {
    background: transparent !important; color: #555 !important;
    border: 1.5px solid #000000 !important; border-radius: 0px !important;
    font-size: 10px !important; font-weight: 600 !important; padding: 4px 10px !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important;
    transition: all 0.2s !important; box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"][key="clear_cache_btn"] > button:hover {
    background: #000000 !important; color: #FFC000 !important;
    border-color: #000000 !important; box-shadow: none !important;
}

/* ── Header ── */
.top-banner {
    background: linear-gradient(90deg, #FFC000 0%, #FFD740 50%, #FFC000 100%);
    height: 4px; margin: 0 -2.5rem; background-size: 200% 100%;
    animation: shimmer 3s infinite linear;
}
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
.dash-header {
    background: #ffffff; display: flex; align-items: center; gap: 20px;
    padding: 14px 28px; border-bottom: 3px solid #FFC000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin: 0 -2.5rem 24px -2.5rem;
}
.dash-logo-box {
    background: #111; border-radius: 10px; padding: 10px 16px;
    display: flex; flex-direction: column; align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.dash-logo-main { font-size: 18px; font-weight: 900; color: #FFC000; letter-spacing: 2px; line-height: 1; }
.dash-logo-sub  { font-size: 10px; font-weight: 500; color: #888; letter-spacing: 1px; margin-top: 2px; }
.dash-title     { font-size: 18px; font-weight: 700; color: #111; letter-spacing: -0.3px; }
.dash-subtitle  { font-size: 12px; color: #999; margin-top: 2px; font-weight: 400; }
.dash-badge {
    margin-left: auto; background: #FFC000; color: #000;
    font-size: 10px; font-weight: 800; padding: 4px 10px;
    border-radius: 20px; letter-spacing: 1px; text-transform: uppercase;
}

/* ── Section labels ── */
.section-label {
    display: flex; align-items: center; gap: 10px;
    font-size: 15px; font-weight: 700; color: #111827;
    margin-bottom: 14px; margin-top: 10px;
}
.section-label::before {
    content: ''; width: 4px; height: 18px;
    background: #FFC000; border-radius: 2px; display: block;
}

/* ── Upload cards ── */
.upload-card {
    background: #fff; border-radius: 12px; border: 1px solid #e5e7eb;
    border-bottom: 3px solid #FFC000; padding: 20px;
    transition: all 0.25s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.upload-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.10); transform: translateY(-1px); }
.upload-card-title { font-size: 13px; font-weight: 700; color: #1f2937; margin-bottom: 3px; }
.upload-card-sub   { font-size: 11px; color: #9ca3af; margin-bottom: 12px; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.kpi-card {
    background: #fff; border-radius: 12px; border-bottom: 3px solid #FFC000;
    padding: 18px 20px; min-width: 140px; flex: 1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.10); }
.kpi-label { font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; }
.kpi-value { font-size: 26px; font-weight: 800; color: #111827; line-height: 1; letter-spacing: -0.5px; white-space: nowrap; }
.kpi-mom   { font-size: 11px; margin-top: 8px; color: #9ca3af; font-weight: 500; }
.kpi-mom-up      { color: #059669; font-weight: 700; }
.kpi-mom-down    { color: #dc2626; font-weight: 700; }
.kpi-mom-neutral { color: #9ca3af; font-weight: 700; }

/* ── Chart containers ── */
.chart-card {
    background: #fff; border-radius: 12px; border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); padding: 4px; overflow: hidden;
    transition: box-shadow 0.2s;
}
.chart-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.10); }

/* ── All buttons base ── */
.stButton > button,
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"] {
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 13px !important; transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important; cursor: pointer !important;
}

/* ── Primary button ── */
[data-testid="baseButton-primary"] {
    background: #FFC000 !important; color: #000000 !important;
    border: 2px solid #000000 !important; box-shadow: 0 2px 8px rgba(255,192,0,0.35) !important;
}
[data-testid="baseButton-primary"]:hover {
    background: #e6ac00 !important; color: #000000 !important;
    border: 2px solid #000000 !important; box-shadow: 0 6px 20px rgba(255,192,0,0.5) !important;
    transform: translateY(-1px) !important;
}
[data-testid="baseButton-primary"]:active {
    background: #cc9900 !important; color: #000000 !important;
    border: 2px solid #000000 !important; transform: translateY(0) !important; box-shadow: none !important;
}

/* ── Secondary button ── */
[data-testid="baseButton-secondary"] {
    background: #ffffff !important; color: #374151 !important; border: 1px solid #d1d5db !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: #FFC000 !important; color: #000000 !important;
    border: 1px solid #000000 !important; box-shadow: 0 4px 12px rgba(255,192,0,0.3) !important;
}

/* ── File uploader browse button ── */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzoneButton"],
[data-testid="stFileUploader"] button {
    background: #ffffff !important; color: #374151 !important;
    border: 1px solid #d1d5db !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzoneButton"]:hover,
[data-testid="stFileUploader"] button:hover {
    background: #FFC000 !important; color: #000000 !important;
    border: 1px solid #000000 !important; box-shadow: 0 4px 12px rgba(255,192,0,0.3) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #1f2937 !important; color: #FFC000 !important;
    border: 1px solid #374151 !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 10px 20px !important; transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #111827 !important; color: #FFD740 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important; transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #e5e7eb !important; margin: 28px 0 !important; }

/* ══ DATE PICKER ══════════════════════════════════════════════════════════ */
[data-testid="stDateInput"] label {
    font-size: 10px !important; font-weight: 700 !important;
    color: #6b7280 !important; text-transform: uppercase !important; letter-spacing: 1.8px !important;
}
[data-testid="stDateInput"] > div {
    background: #0d1117 !important; border: 1.5px solid #21262d !important;
    border-radius: 10px !important; transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stDateInput"] > div:focus-within {
    border-color: #FFC000 !important; box-shadow: 0 0 0 3px rgba(255,192,0,0.18) !important;
}
.stDateInput input {
    color: #FFC000 !important; background: transparent !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 10px 14px !important; letter-spacing: 0.8px !important; caret-color: #FFC000 !important;
}

/* ══ CALENDAR POPUP ═══════════════════════════════════════════════════════ */
[data-baseweb="calendar"] {
    background: #0d1117 !important; border: 1px solid #21262d !important;
    border-radius: 16px !important;
    box-shadow: 0 0 0 1px rgba(255,192,0,0.10), 0 20px 60px rgba(0,0,0,0.75) !important;
    padding: 12px 10px 14px !important; overflow: hidden !important; min-width: 288px !important;
}
[data-baseweb="calendar"] * { color: #8b949e !important; box-sizing: border-box !important; }
[data-baseweb="calendar"] [data-baseweb="select"] > div {
    background: transparent !important; border: none !important;
    padding: 2px 4px !important; border-radius: 6px !important;
}
[data-baseweb="calendar"] [data-baseweb="select"] > div:hover { background: rgba(255,192,0,0.08) !important; }
[data-baseweb="calendar"] [data-baseweb="select"] span {
    color: #FFC000 !important; font-weight: 800 !important;
    font-size: 15px !important; letter-spacing: 0.2px !important;
}
[data-baseweb="calendar"] [data-baseweb="select"] svg { fill: #FFC000 !important; opacity: 0.6 !important; }
[data-baseweb="calendar"] [data-baseweb="button"] {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 8px !important; width: 30px !important; height: 30px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: background 0.15s, border-color 0.15s !important; padding: 0 !important;
}
[data-baseweb="calendar"] [data-baseweb="button"]:hover { background: #FFC000 !important; border-color: #FFC000 !important; }
[data-baseweb="calendar"] [data-baseweb="button"] svg { fill: #8b949e !important; }
[data-baseweb="calendar"] [data-baseweb="button"]:hover svg { fill: #000 !important; }
[data-baseweb="calendar"] abbr {
    color: #30363d !important; font-size: 10px !important; font-weight: 800 !important;
    text-decoration: none !important; text-transform: uppercase !important; letter-spacing: 1.4px !important;
}
[data-baseweb="calendar"] [role="row"]:first-of-type {
    border-bottom: 1px solid #21262d !important; padding-bottom: 6px !important; margin-bottom: 4px !important;
}
[data-baseweb="calendar"] [role="gridcell"] button {
    border-radius: 8px !important; font-size: 13px !important;
    font-weight: 500 !important; transition: background 0.12s !important;
}
[data-baseweb="calendar"] [role="gridcell"]:empty,
[data-baseweb="calendar"] button[disabled],
[data-baseweb="calendar"] [aria-disabled="true"] {
    background: transparent !important; opacity: 0 !important; pointer-events: none !important;
}
[data-baseweb="calendar"] button[disabled] > div,
[data-baseweb="calendar"] [aria-disabled="true"] > div { background: transparent !important; }
[data-baseweb="calendar"] [role="gridcell"] button:not([aria-selected="true"]):hover > div {
    background: rgba(255,192,0,0.10) !important; border-radius: 8px !important;
}
[data-baseweb="calendar"] [role="gridcell"] button:not([aria-selected="true"]):hover * { color: #FFC000 !important; }
[data-baseweb="calendar"] [aria-selected="true"] > div {
    background: #FFC000 !important; border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(255,192,0,0.50) !important;
}
[data-baseweb="calendar"] [aria-selected="true"] * { color: #000 !important; font-weight: 800 !important; }
[data-baseweb="calendar"] [data-today="true"]:not([aria-selected="true"]) > div {
    border-bottom: 2.5px solid #FFC000 !important; border-radius: 0 !important;
}
[data-baseweb="calendar"] [data-today="true"]:not([aria-selected="true"]) * { color: #FFC000 !important; font-weight: 700 !important; }

/* ══ MONTH / YEAR DROPDOWN LIST ══════════════════════════════════════════ */
[data-baseweb="menu"] {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 12px !important; box-shadow: 0 12px 40px rgba(0,0,0,0.65) !important; padding: 4px !important;
}
[data-baseweb="menu"] li {
    color: #8b949e !important; font-size: 13px !important;
    border-radius: 8px !important; margin: 1px 0 !important;
}
[data-baseweb="menu"] li:hover { background: rgba(255,192,0,0.10) !important; color: #FFC000 !important; }
[data-baseweb="menu"] [aria-selected="true"] { background: rgba(255,192,0,0.16) !important; color: #FFC000 !important; font-weight: 700 !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: #111827 !important; color: #FFC000 !important;
    border: 1px solid #374151 !important; border-radius: 10px !important;
    font-size: 14px !important; caret-color: #FFC000 !important; line-height: 1.6 !important;
}
.stTextArea textarea::placeholder { color: #4b5563 !important; }
.stTextArea textarea:focus { border-color: #FFC000 !important; box-shadow: 0 0 0 3px rgba(255,192,0,0.15) !important; }

/* ── Selectbox ── */
[data-baseweb="select"] > div { border-radius: 8px !important; }

/* ── AI panel ── */
.ai-panel {
    background: #0d1117; border-radius: 16px; padding: 28px 32px;
    border: 1px solid #21262d;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-top: 8px;
}
.ai-title    { font-size: 20px; font-weight: 800; color: #FFC000; letter-spacing: -0.3px; }
.ai-subtitle { font-size: 13px; color: #6b7280; margin-bottom: 20px; line-height: 1.7; }

/* ── Result KPI cards ── */
.result-kpi {
    background: #161b22; border-radius: 10px; border-top: 2px solid #FFC000;
    padding: 14px; text-align: center; flex: 1; min-width: 100px;
}
.result-kpi-label { font-size: 10px; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.result-kpi-value { font-size: 22px; font-weight: 800; color: #FFC000; letter-spacing: -0.5px; }

/* ── Ranking cards ── */
.rank-card {
    background: #161b22; border-radius: 12px; border: 1px solid #21262d;
    padding: 16px 18px; transition: box-shadow 0.2s, border-color 0.2s;
}
.rank-card:hover { border-color: rgba(255,192,0,0.35); box-shadow: 0 4px 16px rgba(255,192,0,0.10); }
.rank-title { font-size: 11px; font-weight: 700; color: #FFC000; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
.rank-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #21262d; transition: background 0.15s;
}
.rank-row:hover { background: rgba(255,192,0,0.04); border-radius: 4px; }
.rank-row:last-child { border-bottom: none; }
.rank-name  { font-size: 13px; color: #8b949e; }
.rank-value { font-size: 13px; font-weight: 700; color: #e6edf3; }

/* ── AI observations ── */
.obs-card {
    background: #0d1117; border-radius: 12px; border: 1px solid #21262d;
    border-left: 3px solid #FFC000; padding: 20px 24px; margin-top: 16px;
    transition: border-left-color 0.2s, box-shadow 0.2s;
}
.obs-card:hover { border-left-color: #FFD740; box-shadow: 0 4px 16px rgba(255,192,0,0.10); }
.obs-title { font-size: 11px; font-weight: 700; color: #FFC000; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 1.5px; }
.obs-line  { font-size: 14px; color: #8b949e; line-height: 1.8; padding: 4px 0; }

/* ── Sidebar filter header ── */
.filter-header {
    background: linear-gradient(135deg, #FFC000, #FFD740); color: #000 !important;
    font-weight: 800; font-size: 12px; padding: 10px 0; border-radius: 8px;
    text-align: center; letter-spacing: 2px; margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(255,192,0,0.3);
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #fff !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important; border: 1px solid #e5e7eb !important;
}
.streamlit-expanderContent {
    border: 1px solid #e5e7eb !important; border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; border: none !important; }
div[data-testid="stAlert"][kind="info"]    { background: rgba(255,192,0,0.08) !important; border-left: 3px solid #FFC000 !important; }
div[data-testid="stAlert"][kind="success"] { background: rgba(5,150,105,0.08) !important; border-left: 3px solid #059669 !important; }
div[data-testid="stAlert"][kind="warning"] { background: rgba(217,119,6,0.08) !important; border-left: 3px solid #d97706 !important; }
div[data-testid="stAlert"][kind="error"]   { background: rgba(220,38,38,0.08) !important; border-left: 3px solid #dc2626 !important; }

/* ── Text input ── */
[data-testid="stTextInput"] label {
    font-size: 10px !important; font-weight: 700 !important;
    color: #6b7280 !important; text-transform: uppercase !important; letter-spacing: 1.5px !important;
}
[data-testid="stTextInput"] input {
    background: #111827 !important; color: #e6edf3 !important;
    border: 1px solid #374151 !important; border-radius: 8px !important;
    font-size: 13px !important; padding: 10px 14px !important;
    caret-color: #FFC000 !important; transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] input::placeholder { color: #4b5563 !important; }
[data-testid="stTextInput"] input:focus {
    border-color: #FFC000 !important;
    box-shadow: 0 0 0 3px rgba(255,192,0,0.15) !important; outline: none !important;
}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label { font-size: 13px !important; font-weight: 500 !important; color: #374151 !important; }
[data-testid="stCheckbox"] input[type="checkbox"] + span {
    border: 2px solid #d1d5db !important; border-radius: 4px !important;
    transition: border-color 0.15s, background 0.15s !important;
}
[data-testid="stCheckbox"] input[type="checkbox"]:checked + span {
    background: #FFC000 !important; border-color: #FFC000 !important;
}
[data-testid="stCheckbox"] input[type="checkbox"]:checked + span svg { fill: #000 !important; }

/* ── st.metric ── */
[data-testid="stMetric"] {
    background: #fff; border-radius: 12px; border-bottom: 3px solid #FFC000;
    padding: 14px 16px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.10); }
[data-testid="stMetricLabel"] { font-size: 10px !important; font-weight: 700 !important; color: #9ca3af !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #111827 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 600 !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] { color: #9ca3af !important; font-size: 11px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #FFC000 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }

/* ══ TAB NAVIGATION ══════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #ffffff; padding: 5px 6px; border-radius: 12px;
    border: 1px solid #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px; overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    height: 36px; padding: 0 18px; border-radius: 8px;
    font-size: 13px; font-weight: 600; color: #6b7280;
    background: transparent; border: none;
    transition: background 0.15s, color 0.15s; white-space: nowrap;
}
.stTabs [data-baseweb="tab"]:hover { background: #f3f4f6; color: #374151; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #FFC000 !important; color: #000000 !important;
    font-weight: 700 !important; box-shadow: 0 2px 8px rgba(255,192,0,0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ══ ACTIVE FILTER BAR ═══════════════════════════════════════════════════ */
.filter-bar {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    background: #fffbea; border: 1px solid #fde68a; border-radius: 8px;
    padding: 8px 14px; font-size: 12px; color: #92400e;
    font-weight: 500; margin-bottom: 16px;
}
.filter-chip {
    background: #fef3c7; color: #92400e; border: 1px solid #fcd34d;
    padding: 2px 10px; border-radius: 12px; font-weight: 700; font-size: 11px;
}

/* ══ EMPTY STATE ═════════════════════════════════════════════════════════ */
.empty-state {
    text-align: center; padding: 60px 20px; color: #9ca3af;
}
.empty-state-icon  { font-size: 48px; margin-bottom: 12px; }
.empty-state-title { font-size: 18px; font-weight: 700; color: #374151; margin-bottom: 8px; }
.empty-state-sub   { font-size: 13px; color: #9ca3af; line-height: 1.6; }
</style>

<script>
function fixSidebarToggle() {
    const el = document.querySelector('[data-testid="stSidebarCollapsed"]');
    if (el) {
        el.style.setProperty("background", "#FFC000", "important");
        el.style.setProperty("visibility", "visible", "important");
        el.style.setProperty("opacity", "1", "important");
        el.style.setProperty("display", "flex", "important");
        el.style.setProperty("min-width", "36px", "important");
        el.style.setProperty("min-height", "80px", "important");
        el.style.setProperty("z-index", "99999", "important");
        el.style.setProperty("border-radius", "0 10px 10px 0", "important");
        el.style.setProperty("cursor", "pointer", "important");
        el.querySelectorAll("svg").forEach(s => s.style.setProperty("fill", "#000", "important"));
        el.querySelectorAll("button").forEach(b => {
            b.style.setProperty("background", "transparent", "important");
            b.style.setProperty("border", "none", "important");
            b.style.setProperty("min-height", "80px", "important");
        });
    }
}
const _obs = new MutationObserver(fixSidebarToggle);
_obs.observe(document.body, { childList: true, subtree: true, attributes: true });
fixSidebarToggle();
setInterval(fixSidebarToggle, 500);
</script>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

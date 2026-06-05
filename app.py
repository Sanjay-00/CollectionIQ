import os
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from utils import (
    load_and_validate,
    apply_filters,
    compute_metrics,
    fmt_value,
    build_status_bar_chart,
    build_branch_bar_chart,
    build_closing_pc_chart,
    build_html_export,
    _kpi_card_html,
)
from graph import run_query
from smart_alerts import run_all_alerts


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type object columns to string before st.dataframe display.
    Prevents PyArrow serialization errors when Excel columns contain mixed int/str values."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
    return df


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _dl_btn(df: pd.DataFrame, filename: str, key: str) -> None:
    """Render a compact right-aligned Excel download button."""
    _, col = st.columns([5, 1])
    with col:
        st.download_button(
            "⬇ Excel", data=_excel_bytes(df), file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key, use_container_width=True,
        )


def _load_and_concat(files) -> tuple[pd.DataFrame | None, list[str]]:
    """Load one or more Excel files and concatenate into a single DataFrame.
    Works with a single file or a list — so single-file uploads are unchanged."""
    if not files:
        return None, ["No file uploaded"]
    if not isinstance(files, list):
        files = [files]

    dfs, errors = [], []
    for f in files:
        df, errs = load_and_validate(f)
        if errs:
            errors.append(f"{getattr(f, 'name', 'file')}: {errs[0]}")
        else:
            dfs.append(df)

    if not dfs:
        return None, errors

    if len(dfs) == 1:
        combined = dfs[0]
    else:
        # Normalize datetime columns to ms precision before concat.
        # Different files can have datetime64[ns] vs datetime64[us] which causes
        # OutOfBoundsDatetime when pandas tries to upcast them during concat.
        def _normalize_dt(df):
            for col in df.select_dtypes(include="datetime64").columns:
                try:
                    df[col] = df[col].astype("datetime64[ms]")
                except Exception:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        dfs = [_normalize_dt(df) for df in dfs]
        combined = pd.concat(dfs, ignore_index=True)
        # Drop duplicate Loan No — same loan shouldn't appear in two regional files
        if "Loan No" in combined.columns:
            combined = combined.drop_duplicates(subset=["Loan No"], keep="first")

    return combined, errors  # errors here are warnings (some files failed but ≥1 succeeded)


def _send_report_email(html_report: str, email_to: str, curr_month: str) -> tuple[bool, str]:
    """Send the HTML report via SMTP without regenerating. Returns (success, error_msg)."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders as _enc
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Shriram Finance - Portfolio Intelligence Report {curr_month}"
        msg["From"]    = smtp_user
        msg["To"]      = email_to
        msg.attach(MIMEText(html_report, "html", "utf-8"))
        att = MIMEBase("application", "octet-stream")
        att.set_payload(html_report.encode("utf-8"))
        _enc.encode_base64(att)
        att.add_header("Content-Disposition", f'attachment; filename="portfolio_report_{curr_month}.html"')
        msg.attach(att)
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as srv:
            srv.ehlo(); srv.starttls(); srv.login(smtp_user, smtp_pass)
            srv.sendmail(smtp_user, [r.strip() for r in email_to.split(",") if r.strip()], msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)


def _send_feedback(run_id: str, score: float) -> None:
    """Post thumbs-up (1.0) or thumbs-down (0.0) feedback to LangSmith."""
    try:
        from langsmith import Client as _LSClient
        _LSClient().create_feedback(run_id=run_id, key="result_quality", score=score)
    except Exception:
        pass

st.set_page_config(
    page_title="Shriram Finance Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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
    background: #FFC000 !important;
    border: 2px solid #000 !important;
    border-radius: 6px !important;
    color: #000 !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: #e6ac00 !important;
}
[data-testid="stSidebarCollapseButton"] svg { fill: #000 !important; }

/* ── Sidebar collapsed state (expand button) ── */
[data-testid="stSidebarCollapsed"] {
    background: #FFC000 !important;
    border-radius: 0 10px 10px 0 !important;
    width: 36px !important;
    min-height: 80px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 99999 !important;
    box-shadow: 4px 0 16px rgba(0,0,0,0.35) !important;
    cursor: pointer !important;
    border-top: 2px solid #000 !important;
    border-right: 2px solid #000 !important;
    border-bottom: 2px solid #000 !important;
    border-left: none !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsed"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 80px !important;
}
[data-testid="stSidebarCollapsed"] svg {
    fill: #000000 !important;
    width: 20px !important;
    height: 20px !important;
}
[data-testid="stSidebarCollapsed"]:hover { background: #e6ac00 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 3px solid #FFC000 !important;
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
[data-testid="stSidebar"] hr { border-color: #222 !important; }
[data-testid="stSidebar"] [data-testid="stButton"][key="clear_cache_btn"] > button {
    background: transparent !important;
    color: #555 !important;
    border: 1.5px solid #000000 !important;
    border-radius: 0px !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"][key="clear_cache_btn"] > button:hover {
    background: #000000 !important;
    color: #FFC000 !important;
    border-color: #000000 !important;
    box-shadow: none !important;
}

/* ── Header ── */
.top-banner {
    background: linear-gradient(90deg, #FFC000 0%, #FFD740 50%, #FFC000 100%);
    height: 4px; margin: 0 -2.5rem; background-size: 200% 100%;
    animation: shimmer 3s infinite linear;
}
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
.dash-header {
    background: #ffffff;
    display: flex; align-items: center; gap: 20px;
    padding: 14px 28px;
    border-bottom: 3px solid #FFC000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 0 -2.5rem 24px -2.5rem;
}
.dash-logo-box {
    background: #111; border-radius: 10px;
    padding: 10px 16px; display: flex; flex-direction: column; align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.dash-logo-main { font-size: 18px; font-weight: 900; color: #FFC000; letter-spacing: 2px; line-height: 1; }
.dash-logo-sub  { font-size: 10px; font-weight: 500; color: #888; letter-spacing: 1px; margin-top: 2px; }
.dash-title { font-size: 18px; font-weight: 700; color: #111; letter-spacing: -0.3px; }
.dash-subtitle { font-size: 12px; color: #999; margin-top: 2px; font-weight: 400; }
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
    background: #fff; border-radius: 12px;
    border: 1px solid #e5e7eb;
    border-bottom: 3px solid #FFC000;
    padding: 20px;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.upload-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.10); transform: translateY(-1px); }
.upload-card-title { font-size: 13px; font-weight: 700; color: #1f2937; margin-bottom: 3px; }
.upload-card-sub   { font-size: 11px; color: #9ca3af; margin-bottom: 12px; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.kpi-card {
    background: #fff; border-radius: 12px;
    border-bottom: 3px solid #FFC000;
    padding: 18px 20px; min-width: 140px; flex: 1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.10); }
.kpi-label { font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; }
.kpi-value { font-size: 26px; font-weight: 800; color: #111827; line-height: 1; letter-spacing: -0.5px; white-space: nowrap; }
.kpi-mom   { font-size: 11px; margin-top: 8px; color: #9ca3af; font-weight: 500; }
.kpi-mom-up   { color: #059669; font-weight: 700; }
.kpi-mom-down { color: #dc2626; font-weight: 700; }

/* ── Chart containers ── */
.chart-card {
    background: #fff; border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    padding: 4px; overflow: hidden;
    transition: box-shadow 0.2s;
}
.chart-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.10); }

/* ── All buttons base ── */
.stButton > button,
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
}

/* ── Primary button (Generate Dashboard, Run Query) ── */
[data-testid="baseButton-primary"] {
    background: #FFC000 !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    box-shadow: 0 2px 8px rgba(255,192,0,0.35) !important;
}
[data-testid="baseButton-primary"]:hover {
    background: #e6ac00 !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    box-shadow: 0 6px 20px rgba(255,192,0,0.5) !important;
    transform: translateY(-1px) !important;
}
[data-testid="baseButton-primary"]:active {
    background: #cc9900 !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Secondary button ── */
[data-testid="baseButton-secondary"] {
    background: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: #FFC000 !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    box-shadow: 0 4px 12px rgba(255,192,0,0.3) !important;
}

/* ── File uploader browse button ── */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzoneButton"],
[data-testid="stFileUploader"] button {
    background: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzoneButton"]:hover,
[data-testid="stFileUploader"] button:hover {
    background: #FFC000 !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    box-shadow: 0 4px 12px rgba(255,192,0,0.3) !important;
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
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #e5e7eb !important; margin: 28px 0 !important; }

/* ══ DATE PICKER ══════════════════════════════════════════════════════════ */

/* Label */
[data-testid="stDateInput"] label {
    font-size: 10px !important; font-weight: 700 !important;
    color: #6b7280 !important; text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
}

/* Input wrapper */
[data-testid="stDateInput"] > div {
    background: #0d1117 !important;
    border: 1.5px solid #21262d !important;
    border-radius: 10px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stDateInput"] > div:focus-within {
    border-color: #FFC000 !important;
    box-shadow: 0 0 0 3px rgba(255,192,0,0.18) !important;
}

/* Text field */
.stDateInput input {
    color: #FFC000 !important; background: transparent !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 10px 14px !important; letter-spacing: 0.8px !important;
    caret-color: #FFC000 !important;
}

/* ══ CALENDAR POPUP ═══════════════════════════════════════════════════════ */

/* Outer shell */
[data-baseweb="calendar"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 16px !important;
    box-shadow: 0 0 0 1px rgba(255,192,0,0.10),
                0 20px 60px rgba(0,0,0,0.75) !important;
    padding: 12px 10px 14px !important;
    overflow: hidden !important;
    min-width: 288px !important;
}

/* Reset all text to muted grey — specific rules override below */
[data-baseweb="calendar"] * {
    color: #8b949e !important;
    box-sizing: border-box !important;
}

/* ── Header: month + year selects ── */
[data-baseweb="calendar"] [data-baseweb="select"] > div {
    background: transparent !important;
    border: none !important;
    padding: 2px 4px !important;
    border-radius: 6px !important;
}
[data-baseweb="calendar"] [data-baseweb="select"] > div:hover {
    background: rgba(255,192,0,0.08) !important;
}
[data-baseweb="calendar"] [data-baseweb="select"] span {
    color: #FFC000 !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    letter-spacing: 0.2px !important;
}
[data-baseweb="calendar"] [data-baseweb="select"] svg {
    fill: #FFC000 !important;
    opacity: 0.6 !important;
}

/* ── Nav arrows (prev / next) ── */
[data-baseweb="calendar"] [data-baseweb="button"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    width: 30px !important; height: 30px !important;
    display: flex !important; align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s, border-color 0.15s !important;
    padding: 0 !important;
}
[data-baseweb="calendar"] [data-baseweb="button"]:hover {
    background: #FFC000 !important;
    border-color: #FFC000 !important;
}
[data-baseweb="calendar"] [data-baseweb="button"] svg { fill: #8b949e !important; }
[data-baseweb="calendar"] [data-baseweb="button"]:hover svg { fill: #000 !important; }

/* ── Day-of-week header row ── */
[data-baseweb="calendar"] abbr {
    color: #30363d !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    text-decoration: none !important;
    text-transform: uppercase !important;
    letter-spacing: 1.4px !important;
}

/* ── Separator line below day-of-week headers ── */
[data-baseweb="calendar"] [role="row"]:first-of-type {
    border-bottom: 1px solid #21262d !important;
    padding-bottom: 6px !important;
    margin-bottom: 4px !important;
}

/* ── Regular day numbers ── */
[data-baseweb="calendar"] [role="gridcell"] button {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: background 0.12s !important;
}

/* ── Empty / out-of-month cells — fully hidden ── */
[data-baseweb="calendar"] [role="gridcell"]:empty,
[data-baseweb="calendar"] button[disabled],
[data-baseweb="calendar"] [aria-disabled="true"] {
    background: transparent !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
[data-baseweb="calendar"] button[disabled] > div,
[data-baseweb="calendar"] [aria-disabled="true"] > div {
    background: transparent !important;
}

/* ── Hover (non-selected days) ── */
[data-baseweb="calendar"] [role="gridcell"] button:not([aria-selected="true"]):hover > div {
    background: rgba(255,192,0,0.10) !important;
    border-radius: 8px !important;
}
[data-baseweb="calendar"] [role="gridcell"] button:not([aria-selected="true"]):hover * {
    color: #FFC000 !important;
}

/* ── Selected day — solid gold pill ── */
[data-baseweb="calendar"] [aria-selected="true"] > div {
    background: #FFC000 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(255,192,0,0.50) !important;
}
[data-baseweb="calendar"] [aria-selected="true"] * {
    color: #000 !important;
    font-weight: 800 !important;
}

/* ── Today's date (unselected) — gold dot underline ── */
[data-baseweb="calendar"] [data-today="true"]:not([aria-selected="true"]) > div {
    border-bottom: 2.5px solid #FFC000 !important;
    border-radius: 0 !important;
}
[data-baseweb="calendar"] [data-today="true"]:not([aria-selected="true"]) * {
    color: #FFC000 !important;
    font-weight: 700 !important;
}

/* ══ MONTH / YEAR DROPDOWN LIST ══════════════════════════════════════════ */
[data-baseweb="menu"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.65) !important;
    padding: 4px !important;
}
[data-baseweb="menu"] li {
    color: #8b949e !important; font-size: 13px !important;
    border-radius: 8px !important; margin: 1px 0 !important;
}
[data-baseweb="menu"] li:hover {
    background: rgba(255,192,0,0.10) !important;
    color: #FFC000 !important;
}
[data-baseweb="menu"] [aria-selected="true"] {
    background: rgba(255,192,0,0.16) !important;
    color: #FFC000 !important; font-weight: 700 !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #111827 !important; color: #FFC000 !important;
    border: 1px solid #374151 !important; border-radius: 10px !important;
    font-size: 14px !important; caret-color: #FFC000 !important;
    line-height: 1.6 !important;
}
.stTextArea textarea::placeholder { color: #4b5563 !important; }
.stTextArea textarea:focus { border-color: #FFC000 !important; box-shadow: 0 0 0 3px rgba(255,192,0,0.15) !important; }

/* ── Selectbox ── */
[data-baseweb="select"] > div { border-radius: 8px !important; }

/* ── AI panel ── */
.ai-panel {
    background: #0d1117;
    border-radius: 16px; padding: 28px 32px;
    border: 1px solid #21262d;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-top: 8px;
}
.ai-title { font-size: 20px; font-weight: 800; color: #FFC000; letter-spacing: -0.3px; }
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
    background: #161b22; border-radius: 12px;
    border: 1px solid #21262d; padding: 16px 18px;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.rank-card:hover { border-color: rgba(255,192,0,0.35); box-shadow: 0 4px 16px rgba(255,192,0,0.10); }
.rank-title { font-size: 11px; font-weight: 700; color: #FFC000; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
.rank-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #21262d; transition: background 0.15s; }
.rank-row:hover { background: rgba(255,192,0,0.04); border-radius: 4px; }
.rank-row:last-child { border-bottom: none; }
.rank-name  { font-size: 13px; color: #8b949e; }
.rank-value { font-size: 13px; font-weight: 700; color: #e6edf3; }

/* ── AI observations ── */
.obs-card {
    background: #0d1117; border-radius: 12px;
    border: 1px solid #21262d; border-left: 3px solid #FFC000;
    padding: 20px 24px; margin-top: 16px;
    transition: border-left-color 0.2s, box-shadow 0.2s;
}
.obs-card:hover { border-left-color: #FFD740; box-shadow: 0 4px 16px rgba(255,192,0,0.10); }
.obs-title { font-size: 11px; font-weight: 700; color: #FFC000; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 1.5px; }
.obs-line  { font-size: 14px; color: #8b949e; line-height: 1.8; padding: 4px 0; }

/* ── Sidebar filter header ── */
.filter-header {
    background: linear-gradient(135deg, #FFC000, #FFD740);
    color: #000 !important; font-weight: 800;
    font-size: 12px; padding: 10px 0; border-radius: 8px;
    text-align: center; letter-spacing: 2px; margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(255,192,0,0.3);
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #fff !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important;
    border: 1px solid #e5e7eb !important;
}
.streamlit-expanderContent { border: 1px solid #e5e7eb !important; border-top: none !important; border-radius: 0 0 8px 8px !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; border: none !important; }
div[data-testid="stAlert"][kind="info"]    { background: rgba(255,192,0,0.08) !important; border-left: 3px solid #FFC000 !important; }
div[data-testid="stAlert"][kind="success"] { background: rgba(5,150,105,0.08) !important; border-left: 3px solid #059669 !important; }
div[data-testid="stAlert"][kind="warning"] { background: rgba(217,119,6,0.08) !important; border-left: 3px solid #d97706 !important; }
div[data-testid="stAlert"][kind="error"]   { background: rgba(220,38,38,0.08)  !important; border-left: 3px solid #dc2626 !important; }

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
    padding: 14px 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.10); }
[data-testid="stMetricLabel"]  { font-size: 10px !important; font-weight: 700 !important; color: #9ca3af !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
[data-testid="stMetricValue"]  { font-size: 26px !important; font-weight: 800 !important; color: #111827 !important; }
[data-testid="stMetricDelta"]  { font-size: 12px !important; font-weight: 600 !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] { color: #9ca3af !important; font-size: 11px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #FFC000 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
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
const obs = new MutationObserver(fixSidebarToggle);
obs.observe(document.body, { childList: true, subtree: true, attributes: true });
fixSidebarToggle();
setInterval(fixSidebarToggle, 500);
</script>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
import base64 as _b64, pathlib as _pl
_logo_path = _pl.Path(__file__).parent / "assets" / "shriram_logo.jpg"
_logo_b64  = _b64.b64encode(_logo_path.read_bytes()).decode() if _logo_path.exists() else ""
_logo_html = (
    f'<img src="data:image/jpeg;base64,{_logo_b64}" '
    f'style="height:52px;width:auto;object-fit:contain;display:block;" alt="Shriram Finance">'
    if _logo_b64 else
    '<div class="dash-logo-box"><div class="dash-logo-main">SHRIRAM</div>'
    '<div class="dash-logo-sub">FINANCE</div></div>'
)

st.markdown('<div class="top-banner"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="dash-header">
  <div style="flex-shrink:0;">{_logo_html}</div>
  <div>
    <div class="dash-title">Regional Collection Dashboard</div>
    <div class="dash-subtitle">Credit &amp; Collection Risk Monitoring System</div>
  </div>
  <div class="dash-badge">CollectionIQ</div>
</div>
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
import datetime

# Pre-populate date pickers for sample data — must run before widgets render
if st.session_state.pop("_set_sample_dates", False):
    st.session_state["curr_month_pick"] = datetime.date(2026, 3, 1)
    st.session_state["prev_month_pick"] = datetime.date(2026, 2, 1)

st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown('<div class="upload-card"><div class="upload-card-title">📂 Current Month</div><div class="upload-card-sub">Required - upload one or multiple regional files</div>', unsafe_allow_html=True)
    curr_file = st.file_uploader("Current Month", type=["xlsx", "xls", "xlsb"], key="curr", label_visibility="collapsed", accept_multiple_files=True)
    curr_month_input = st.date_input(
        "Reporting Month",
        value=datetime.date.today().replace(day=1),
        key="curr_month_pick",
        help="Select any date in the reporting month - only Month & Year are used",
        format="DD/MM/YYYY",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_up2:
    st.markdown('<div class="upload-card"><div class="upload-card-title">📂 Previous Month</div><div class="upload-card-sub">Optional - upload one or multiple regional files</div>', unsafe_allow_html=True)
    prev_file = st.file_uploader("Previous Month", type=["xlsx", "xls", "xlsb"], key="prev", label_visibility="collapsed", accept_multiple_files=True)
    prev_month_input = st.date_input(
        "Reporting Month",
        value=(datetime.date.today().replace(day=1) - datetime.timedelta(days=1)).replace(day=1),
        key="prev_month_pick",
        help="Select any date in the previous month - only Month & Year are used",
        format="DD/MM/YYYY",
        disabled=(not prev_file and not st.session_state.get("_sample_loaded")),
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Derive period labels from user-selected dates
curr_month = curr_month_input.strftime("%Y-%m")
prev_month = prev_month_input.strftime("%Y-%m") if (prev_file or st.session_state.get("_sample_loaded")) else None

if not curr_file and not st.session_state.get("_sample_loaded"):
    st.markdown(
        '<div style="margin-top:32px;text-align:center;margin-bottom:40px;">'
        '<div style="font-size:12px;font-weight:800;letter-spacing:4px;color:#FFC000;'
        'text-transform:uppercase;margin-bottom:12px;">📚 COLLECTIONIQ INTELLIGENCE PLATFORM</div>'
        '<div style="font-size:26px;font-weight:800;color:#111;margin-bottom:10px;'
        'letter-spacing:-0.5px;line-height:1.2;">Upload your LCC extract to activate the dashboard</div>'
        '<div style="font-size:13px;color:#999;font-weight:400;letter-spacing:0.2px;">'
        'Supports .xlsx, .xls and .xlsb &nbsp;&#183;&nbsp; Up to 200 MB &nbsp;&#183;&nbsp; Data never leaves your machine'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _FEATURES = [
        ("#FFBF00", "KPI DASHBOARD",       "Collection %, POS, demand, strike rate and NPA with month-on-month movement."),
        ("#111111", "EXECUTIVE SCORECARD", "Every field executive ranked by collection efficiency, strike rate and roll rates."),
        ("#FFBF00", "SMART ALERTS",        "Flagged non-starters, co-lending risk, easy settlements and insurance-driven arrears."),
        ("#111111", "AI QUERY ENGINE",     "Ask any question in plain English. Query priority loans or filter SMA-2 accounts by region."),
        ("#FFBF00", "MONTHLY REPORT",      "Board-ready HTML report with AI narrative, branch league table and a five-point action plan."),
    ]
    for col, (color, title, desc) in zip(st.columns(5), _FEATURES):
        with col:
            st.markdown(
                f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;'
                f'padding:20px 16px;border-top:3px solid {color};">'
                f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:{color};'
                f'text-transform:uppercase;margin-bottom:8px;">{title}</div>'
                f'<div style="font-size:12px;color:#555;line-height:1.6;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="border-top:1px solid #e5e7eb;margin-top:36px;margin-bottom:16px;"></div>',
        unsafe_allow_html=True,
    )

    # ── Fill Sample Data ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;font-size:11px;font-weight:700;color:#9ca3af;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">No file? Try the built-in sample</div>',
        unsafe_allow_html=True,
    )

    @st.cache_data(show_spinner=False)
    def _fetch_sample_from_github():
        import urllib.request, io
        from utils import assign_buckets, COL_ALIASES
        BASE  = "https://raw.githubusercontent.com/Sanjay-00/CollectionIQ/main/sample_data/"
        FILES = {"curr": "Current_Month_Demo.xlsx", "prev": "Previous_Month_Demo.xlsx"}

        def _load(fname):
            with urllib.request.urlopen(BASE + fname, timeout=20) as resp:
                buf = io.BytesIO(resp.read())
            df = pd.read_excel(buf, engine="openpyxl")
            df.rename(columns={k: v for k, v in COL_ALIASES.items() if k in df.columns}, inplace=True)
            for col in ["Ag_Date", "Last Receipt Date", "ParentLDueDate"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            df["Due Dt"] = pd.to_numeric(df["Due Dt"], errors="coerce")
            for col in ["Month Receipt Amount", "Month Collection (Excluding Reserve Collection)", "NET COLLECTION", "Cum Coll (Inst+Exp)", "Total Cum Collection"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            for col in ["NET Collection Demand Inst+Exp", "Net Collection Demand Inst+Exp+BC", "POS", "Arrears / EMI"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["LCC%"]   = pd.to_numeric(df["LCC%"], errors="coerce")
            df["Strike"] = df["Strike"].astype(str).str.strip().str.upper()
            if "Unit" in df.columns:
                df["Unit"] = df["Unit"].astype(str).str.strip().str.upper()
            return assign_buckets(df)

        return _load(FILES["curr"]), _load(FILES["prev"])

    _, col_s, _ = st.columns([2, 1, 2])
    with col_s:
        if st.button("Fill Sample Data", type="primary", use_container_width=True):
            try:
                with st.spinner("Fetching sample data from GitHub..."):
                    _dc, _dp = _fetch_sample_from_github()
                st.session_state["df_curr_raw"]       = _dc
                st.session_state["df_prev_raw"]       = _dp
                st.session_state["_sample_loaded"]    = True
                st.session_state["_set_sample_dates"] = True
                st.rerun()
            except Exception as _e:
                st.error(f"Could not fetch sample data: {_e}")

    st.session_state.pop("df_curr_raw", None)
    st.session_state.pop("df_prev_raw", None)
    st.stop()

col_btn, _ = st.columns([1, 3])
with col_btn:
    generate = st.button("⚡  Generate Dashboard", type="primary", use_container_width=True)

# ── Load data — persisted in session_state so reruns (e.g. Run Query) don't reset ──
if generate and curr_file:
    for _k in ["df_curr_raw", "df_prev_raw", "ai_result", "report_result", "_last_filter_key", "_sample_loaded"]:
        st.session_state.pop(_k, None)

    n_curr = len(curr_file) if isinstance(curr_file, list) else 1
    with st.spinner(f"Loading {n_curr} current month file(s)..."):
        df_curr_raw, err_curr = _load_and_concat(curr_file)
    if df_curr_raw is None:
        st.error(f"Current month: {err_curr[0]}")
        st.stop()
    for _e in err_curr:  # partial failures — show as warnings, not errors
        st.warning(f"Skipped: {_e}")
    if prev_file:
        n_prev = len(prev_file) if isinstance(prev_file, list) else 1
        with st.spinner(f"Loading {n_prev} previous month file(s)..."):
            df_prev_raw, err_prev = _load_and_concat(prev_file)
        if df_prev_raw is None:
            st.error(f"Previous month: {err_prev[0]}")
            st.stop()
        for _e in err_prev:
            st.warning(f"Skipped: {_e}")
    else:
        df_prev_raw = df_curr_raw.iloc[0:0].copy()
    st.session_state["df_curr_raw"] = df_curr_raw
    st.session_state["df_prev_raw"] = df_prev_raw
    st.rerun()

if "df_curr_raw" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:20px 0;color:#aaa;font-size:13px;">
        File ready - click <strong>Generate Dashboard</strong> to build the report.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_curr_raw = st.session_state["df_curr_raw"]
df_prev_raw = st.session_state["df_prev_raw"]

# If the user uploaded a prev file without re-clicking Generate, auto-load it now.
# load_and_validate is @st.cache_data so this is a cache hit — no performance cost.
if prev_file and len(df_prev_raw) == 0:
    _prev_tmp, _prev_err = _load_and_concat(prev_file)
    if _prev_tmp is not None:
        df_prev_raw = _prev_tmp

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="filter-header">⚙ FILTERS</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#2a2a2a;border-radius:8px;padding:10px 14px;margin-bottom:16px;">
      <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Reporting Month</div>
      <div style="font-size:18px;font-weight:800;color:#FFC000;margin-top:2px;">{curr_month}</div>
    </div>
    """, unsafe_allow_html=True)

    regions = ["All"] + sorted(df_curr_raw["RegionName"].dropna().unique().tolist())
    sel_region = st.selectbox("Region", regions)

    # Reset branch when region changes so stale branch never causes silent zero results
    if st.session_state.get("_prev_region") != sel_region:
        st.session_state["_sel_branch"] = "All"
        st.session_state["_prev_region"] = sel_region

    df_for_branch = df_curr_raw if sel_region == "All" else df_curr_raw[df_curr_raw["RegionName"] == sel_region]
    branches = ["All"] + sorted(df_for_branch["Unit"].dropna().unique().tolist())
    default_branch = st.session_state.get("_sel_branch", "All")
    branch_idx = branches.index(default_branch) if default_branch in branches else 0
    sel_branch = st.selectbox("Branch", branches, index=branch_idx)
    st.session_state["_sel_branch"] = sel_branch

    statuses = ["All"] + sorted(df_curr_raw["Loan Status"].dropna().unique().tolist())
    sel_status = st.selectbox("Loan Status", statuses)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px;color:#555;text-align:center;">
        {len(df_curr_raw):,} total records loaded
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    if st.button("Clear cache & relaod", use_container_width=True, key="clear_cache_btn"):
        st.cache_data.clear()
        for _k in ["df_curr_raw", "df_prev_raw", "ai_result", "report_result",
                   "_last_filter_key", "_sample_loaded", "_sel_branch", "_prev_region"]:
            st.session_state.pop(_k, None)
        st.rerun()


# ── Apply filters ──────────────────────────────────────────────────────────────
df_curr = apply_filters(df_curr_raw.copy(), sel_region, sel_branch, sel_status)
df_prev = apply_filters(df_prev_raw.copy(), sel_region, sel_branch, sel_status)

# ── Attach prev_bucket to df_curr so agents can query roll forward / backward ──
# Use df_prev_raw (unfiltered) for the lookup — sidebar filters must not prevent
# bucket migration from being tracked for loans in the current filtered view.
if len(df_prev_raw) > 0 and "Loan No" in df_curr.columns and "curr_bucket" in df_prev_raw.columns:
    _prev_slim = df_prev_raw[["Loan No", "curr_bucket"]].rename(columns={"curr_bucket": "prev_bucket"})
    df_curr = df_curr.merge(_prev_slim, on="Loan No", how="left")

# Clear AI result and report when filters change so stale results are never shown
_filter_key = f"{sel_region}|{sel_branch}|{sel_status}"
if st.session_state.get("_last_filter_key") != _filter_key:
    st.session_state.pop("ai_result", None)
    st.session_state.pop("report_result", None)
    st.session_state["_last_filter_key"] = _filter_key

if len(df_curr) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Metrics ────────────────────────────────────────────────────────────────────
metrics = compute_metrics(df_curr, df_prev)

KIND = {
    "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
    "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct",
    "Count": "count", "SOH": "money", "LCC%": "pct", "CMD %": "pct",
}

KPI_TOP = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %"]
KPI_BOT      = ["Count", "CMD %"]
KPI_EXPOSURE = ["SOH"]


_INVERSE_MOM_LABELS = {"NPA %", "Hard Bucket %"}


def _kpi_card_styled(label, value, mom):
    arrow = "▲" if mom >= 0 else "▼"
    # For bad metrics (NPA, Hard Bucket): up = red, down = green — inverted from normal
    if label in _INVERSE_MOM_LABELS:
        cls = "kpi-mom-down" if mom >= 0 else "kpi-mom-up"
    else:
        cls = "kpi-mom-up" if mom >= 0 else "kpi-mom-down"
    return (
        f'<div class="kpi-card">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-mom">MoM <span class="{cls}">{arrow} {abs(mom):.2f}%</span></div>'
        f'</div>'
    )


def _render_kpi_row(keys):
    html = "".join(
        _kpi_card_styled(k, fmt_value(metrics[k][0], KIND[k]), metrics[k][1])
        for k in keys if k in metrics and k in KIND
    )
    st.markdown(f'<div class="kpi-row">{html}</div>', unsafe_allow_html=True)


# ── KPI top row ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Performance Indicators</div>', unsafe_allow_html=True)
_render_kpi_row(KPI_TOP)

# ── Charts ─────────────────────────────────────────────────────────────────────
fig_status  = build_status_bar_chart(df_curr)
fig_branch  = build_branch_bar_chart(df_curr)
fig_closing = build_closing_pc_chart(df_curr)

st.markdown('<div class="section-label">Portfolio Analysis</div>', unsafe_allow_html=True)

col_bar, col_hbar, col_lcc = st.columns([2, 2, 1])
with col_bar:
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
with col_hbar:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_branch, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_lcc:
    lcc_val  = fmt_value(metrics["LCC%"][0], "pct")
    lcc_mom  = metrics["LCC%"][1]
    lcc_arrow = "▲" if lcc_mom >= 0 else "▼"
    lcc_cls   = "kpi-mom-up" if lcc_mom >= 0 else "kpi-mom-down"
    st.markdown(f"""
    <div class="kpi-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;margin-top:0;">
      <div class="kpi-label">LCC %</div>
      <div style="font-size:36px;font-weight:800;color:#111;line-height:1.1;">{lcc_val}</div>
      <div class="kpi-mom">MoM <span class="{lcc_cls}">{lcc_arrow} {abs(lcc_mom):.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

col_trend, col_bot = st.columns([3, 2])
with col_trend:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_closing, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_bot:
    st.markdown("<br>", unsafe_allow_html=True)
    _render_kpi_row(KPI_BOT)
    _render_kpi_row(KPI_EXPOSURE)

# ── Field Executive Performance Scorecard ─────────────────────────────────────
if "MNT NAME" in df_curr.columns:
    st.markdown("---")
    st.markdown('<div class="section-label">Field Executive Performance Scorecard</div>', unsafe_allow_html=True)

    from analysis.executive_scorecard import compute_executive_scorecard, build_scorecard_table_html

    scorecard_df = compute_executive_scorecard(df_curr)
    if len(scorecard_df) > 0:
        top_count = (scorecard_df["Tier"] == "top").sum()
        bot_count = (scorecard_df["Tier"] == "bottom").sum()

        avg_coll   = scorecard_df["Collection %"].mean()
        avg_strike = scorecard_df["Strike Rate %"].mean()
        avg_npa    = scorecard_df["NPA %"].mean() if "NPA %" in scorecard_df.columns else 0.0

        sc_col1, sc_col2, sc_col3, sc_col4, sc_col5 = st.columns(5)
        with sc_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:#16a34a;">
              <div class="kpi-label">Top Performers</div>
              <div class="kpi-value" style="color:#16a34a;">{top_count}</div>
              <div class="kpi-mom">Top 25% by collection %</div>
            </div>""", unsafe_allow_html=True)
        with sc_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:#dc2626;">
              <div class="kpi-label">Need Attention</div>
              <div class="kpi-value" style="color:#dc2626;">{bot_count}</div>
              <div class="kpi-mom">Bottom 25% by collection %</div>
            </div>""", unsafe_allow_html=True)
        with sc_col3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Avg Collection %</div>
              <div class="kpi-value">{avg_coll:.1f}%</div>
              <div class="kpi-mom">{len(scorecard_df)} executives ranked</div>
            </div>""", unsafe_allow_html=True)
        with sc_col4:
            strike_color = "#16a34a" if avg_strike >= 70 else "#d97706" if avg_strike >= 50 else "#dc2626"
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{strike_color};">
              <div class="kpi-label">Avg Strike Rate %</div>
              <div class="kpi-value" style="color:{strike_color};">{avg_strike:.1f}%</div>
              <div class="kpi-mom">Full payment received %</div>
            </div>""", unsafe_allow_html=True)
        with sc_col5:
            npa_color = "#16a34a" if avg_npa < 5 else "#d97706" if avg_npa < 10 else "#dc2626"
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{npa_color};">
              <div class="kpi-label">Avg NPA %</div>
              <div class="kpi-value" style="color:{npa_color};">{avg_npa:.1f}%</div>
              <div class="kpi-mom">Across all executives</div>
            </div>""", unsafe_allow_html=True)

        with st.expander(f"View Executive Performance Table ({len(scorecard_df)} executives)", expanded=False):
            st.markdown(build_scorecard_table_html(scorecard_df), unsafe_allow_html=True)
            _dl_btn(scorecard_df.drop(columns=["Tier"], errors="ignore"),
                    "executive_scorecard.xlsx", "dl_scorecard")
    else:
        st.info("Not enough data per executive to build scorecard (minimum 5 accounts required).")

# ── Smart Alerts ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Smart Alerts</div>', unsafe_allow_html=True)

SEVERITY_STYLE = {
    "critical": ("#dc2626", "#fff5f5"),
    "high":     ("#f97316", "#fff7ed"),
    "medium":   ("#d97706", "#fffbea"),
}
CLEAR_STYLE = ("#16a34a", "#f0fdf4")   # green when count == 0

alerts = run_all_alerts(df_curr)

for i in range(0, len(alerts), 2):
    row_alerts = alerts[i:i+2]
    cols = st.columns(len(row_alerts))
    for col, alert in zip(cols, row_alerts):
        is_clear = alert["count"] == 0
        if is_clear:
            title_color, bg = CLEAR_STYLE
        else:
            title_color, bg = SEVERITY_STYLE.get(alert["severity"], ("#d97706", "#fffbea"))

        border = f"border-left:4px solid {title_color};"
        count_color = "#16a34a" if is_clear else title_color
        pos_fmt = fmt_value(alert["pos"], "money")
        pc_fmt  = fmt_value(alert["closing_arrears"], "money")

        with col:
            st.markdown(f"""
            <div style="background:{bg};border-radius:12px;{border}
                        padding:16px 18px;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:4px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="font-size:20px;">{"✅" if is_clear else alert["icon"]}</span>
                <span style="font-size:14px;font-weight:700;color:{title_color};">{alert["title"]}</span>
              </div>
              <div style="font-size:12px;color:#666;margin-bottom:10px;">{alert["subtitle"]}</div>
              <div style="display:flex;gap:16px;margin-bottom:10px;">
                <div>
                  <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Accounts</div>
                  <div style="font-size:26px;font-weight:800;color:{count_color};">{alert["count"]}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">POS</div>
                  <div style="font-size:22px;font-weight:800;color:#111;">{pos_fmt if not is_clear else "—"}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Closing Arrears</div>
                  <div style="font-size:22px;font-weight:800;color:{title_color};">{pc_fmt if not is_clear else "—"}</div>
                </div>
              </div>
              <div style="font-size:11px;color:#555;font-style:italic;border-top:1px solid rgba(0,0,0,0.08);
                          padding-top:8px;">
                {"✓ All clear - no accounts flagged" if is_clear else f"💬 {alert['action']}"}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if not is_clear:
                with st.expander(f"View {alert['count']} accounts"):
                    display_df = alert["df"]
                    display_df = display_df.loc[:, ~display_df.columns.duplicated()]
                    st.dataframe(
                        _safe_df(display_df.reset_index(drop=True)),
                        use_container_width=True,
                        hide_index=True,
                        height=min(300, 40 + alert["count"] * 35),
                    )
                    _dl_btn(display_df.reset_index(drop=True),
                            f"alert_{alert['title'].replace(' ', '_')}.xlsx",
                            f"dl_alert_{alert['title']}")

# ── Bucket Migration / Roll-Rate Analysis ─────────────────────────────────────
if len(df_prev_raw) > 0:
    st.markdown("---")
    st.markdown('<div class="section-label">Bucket Migration / Roll-Rate Analysis</div>', unsafe_allow_html=True)

    from analysis.roll_rate import compute_roll_rate_matrix, compute_roll_rate_kpis, build_roll_rate_heatmap

    rr_matrix, rr_meta = compute_roll_rate_matrix(df_curr, df_prev)
    fig_rr = build_roll_rate_heatmap(rr_matrix)

    rr_col1, rr_col2, rr_col3, rr_col4 = st.columns(4)
    rr_kpis = [
        ("Roll-Forward Rate", rr_meta["roll_forward_rate"], "%", "#dc2626", "Accounts that worsened bucket"),
        ("Roll-Backward Rate", rr_meta["roll_backward_rate"], "%", "#16a34a", "Delinquent accounts returned to STD"),
        ("NPA Formation",     rr_meta["npa_formation_rate"],"%", "#991b1b", "Non-NPA accounts that became NPA"),
        ("Matched Accounts",  rr_meta["matched_count"],     "",  "#111827", "Accounts in both months"),
    ]
    for col, (label, val, unit, color, tip) in zip([rr_col1, rr_col2, rr_col3, rr_col4], rr_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{color};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};font-size:24px;">{val:,.1f}{unit}</div>
              <div class="kpi-mom" style="color:#9ca3af;">{tip}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="chart-card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_rr, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(
        f"{rr_meta['matched_count']:,} matched accounts | "
        f"{rr_meta['new_entries']:,} new this month | "
        f"{rr_meta['exits']:,} closed/exited"
    )

# ── HTML export ────────────────────────────────────────────────────────────────
st.markdown("---")
col_dl, _ = st.columns([1, 3])
with col_dl:
    filters_applied = {"Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status, "Year Month": str(curr_month)}
    html_content = build_html_export(
        df_curr, df_prev, metrics, fig_status, fig_branch, fig_closing, filters_applied,
        curr_month=curr_month,
        alerts=alerts,
        scorecard_df=scorecard_df if "MNT NAME" in df_curr.columns else None,
        roll_rate_meta=rr_meta if len(df_prev_raw) > 0 else None,
    )
    st.download_button(
        label="⬇  Download as HTML",
        data=html_content.encode("utf-8"),
        file_name=f"shriram_dashboard_{curr_month}.html",
        mime="text/html",
        use_container_width=True,
    )

# ── Portfolio Intelligence Report ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">AI Portfolio Intelligence Report</div>', unsafe_allow_html=True)

with st.expander("Configure & Generate Monthly Report", expanded=False):
    st.markdown("""
    <div class="ai-panel">
      <div class="ai-title">Monthly Portfolio Intelligence Report</div>
      <div class="ai-subtitle">
        Generates a board-ready HTML report with AI executive narrative, branch rankings,
        field executive scorecard, bucket migration analysis, and 5 prioritized action items.
        Download as HTML or send via email.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top:14px;'><strong style='font-size:13px;'>Report Sections:</strong></div>", unsafe_allow_html=True)
    rpt_c1, rpt_c2, rpt_c3 = st.columns(3)
    with rpt_c1:
        inc_health  = st.checkbox("Portfolio Health", value=True, key="rpt_health")
        inc_flags   = st.checkbox("Risk Flags",       value=True, key="rpt_flags")
    with rpt_c2:
        inc_migrate = st.checkbox(
            "Bucket Migration", value=(len(df_prev_raw) > 0), key="rpt_migrate",
            disabled=(len(df_prev_raw) == 0), help="Upload previous month file to enable",
        )
        inc_branch  = st.checkbox("Branch Performance", value=True, key="rpt_branch")
    with rpt_c3:
        inc_exec    = st.checkbox("Executive Rankings", value=True, key="rpt_exec")

    smtp_ok = bool(os.environ.get("SMTP_HOST", ""))
    if smtp_ok:
        rpt_email_to = st.text_input(
            "Send report to (email address)",
            placeholder="manager@shriram.com, head@shriram.com",
            key="rpt_email_to",
            help="Separate multiple addresses with a comma",
        )
    else:
        rpt_email_to = ""
        st.caption("Email not configured — add SMTP_HOST / SMTP_USER / SMTP_PASS to .env to enable.")

    _gc, _sc, _ = st.columns([2, 1, 3])
    with _gc:
        rpt_btn = st.button("Generate Monthly Report", type="primary", key="rpt_generate", use_container_width=True)
    with _sc:
        rpt_send_btn = st.button("📧 Send", key="rpt_send_only", use_container_width=True,
                                  disabled=not (smtp_ok and st.session_state.get("report_result", {}).get("html_report")))

if rpt_send_btn:
    _cached = st.session_state.get("report_result", {})
    if not rpt_email_to.strip():
        st.warning("Enter an email address in the field above.")
    else:
        with st.spinner("Sending report..."):
            _ok, _err = _send_report_email(_cached["html_report"], rpt_email_to.strip(), curr_month)
        if _ok:
            st.success(f"Report sent to: {rpt_email_to}")
        else:
            st.error(f"Failed: {_err}")

if rpt_btn:
    enabled_sections = []
    if st.session_state.get("rpt_health"):   enabled_sections.append("portfolio_health")
    if st.session_state.get("rpt_flags"):    enabled_sections.append("risk_flags")
    if st.session_state.get("rpt_migrate"):  enabled_sections.append("bucket_migration")
    if st.session_state.get("rpt_branch"):   enabled_sections.append("branch_performance")
    if st.session_state.get("rpt_exec"):     enabled_sections.append("executive_rankings")

    from report_agent.graph import run_report

    with st.spinner("Running Portfolio Intelligence Agent (30–60 seconds)..."):
        _rpt_result = run_report(
            df_curr=df_curr,
            df_prev=df_prev,
            curr_month=curr_month,
            prev_month=prev_month,
            enabled_sections=enabled_sections,
            filters_applied={"Region": sel_region, "Branch": sel_branch, "Loan Status": sel_status},
            email_to=rpt_email_to,
        )
    st.session_state["report_result"] = _rpt_result

_rpt = st.session_state.get("report_result")
if _rpt:
    if _rpt.get("error"):
        st.error(f"Report generation failed: {_rpt['error']}")
    elif _rpt.get("html_report"):
        st.success("Report generated successfully.")

        if _rpt.get("email_sent"):
            st.info(f"Report emailed to: {_rpt.get('email_to', '')}")
        elif _rpt.get("email_error") and os.environ.get("SMTP_HOST", "") and _rpt.get("email_to", ""):
            st.warning(f"Email failed: {_rpt['email_error']}")

        rpt_dl_col, _ = st.columns([1, 3])
        with rpt_dl_col:
            st.download_button(
                label="⬇  Download Intelligence Report (HTML)",
                data=_rpt["html_report"].encode("utf-8"),
                file_name=f"portfolio_intelligence_{curr_month}.html",
                mime="text/html",
                use_container_width=True,
            )


        if _rpt.get("executive_narrative"):
            bullets = [l.strip().lstrip("- ").strip() for l in _rpt["executive_narrative"].split("\n") if l.strip()]
            narrative_html = "".join(
                f'<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #1e293b;">'
                f'<span style="color:#FFC000;font-size:14px;font-weight:900;flex-shrink:0;">&#8226;</span>'
                f'<span style="color:#c9d1d9;font-size:13px;line-height:1.6;">{b}</span>'
                f'</div>'
                for b in bullets
            )
            st.markdown(f"""
            <div class="obs-card">
              <div class="obs-title">AI Executive Narrative</div>
              <div style="margin-top:8px;">{narrative_html}</div>
            </div>""", unsafe_allow_html=True)

        if _rpt.get("action_plan"):
            lines = [l.strip() for l in _rpt["action_plan"].split("\n") if l.strip()][:5]
            action_html = "".join(
                f'<div style="padding:8px 0;border-bottom:1px solid #21262d;font-size:13px;color:#8b949e;">{l}</div>'
                for l in lines
            )
            st.markdown(f"""
            <div class="obs-card" style="margin-top:12px;">
              <div class="obs-title">Prioritized Action Plan</div>
              {action_html}
            </div>""", unsafe_allow_html=True)

# ── AI Query Assistant ─────────────────────────────────────────────────────────
st.components.v1.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #f2f2f2; font-family: 'Inter', sans-serif; padding: 2px 0 0 0; }
.panel {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 24px;
}
.hdr { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.icon { font-size: 22px; line-height: 1; }
.title { font-size: 20px; font-weight: 800; color: #FFC000; letter-spacing: -0.3px; }
.sub { font-size: 13px; color: #6b7280; line-height: 1.7; margin-bottom: 14px; }
.chip {
    display: inline-block;
    background: #161b22;
    color: #8b949e;
    border: 1px solid #2d333b;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    font-style: italic;
    cursor: pointer;
    margin: 0 6px 0 0;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    line-height: 1.4;
    outline: none;
}
.chip:hover { background: #2a2a2a; color: #d0d0d0; border-color: #555; }
</style>
<div class="panel">
  <div class="hdr">
    <span class="icon">🤖</span>
    <span class="title">AI Query Assistant</span>
  </div>
  <div class="sub">Ask any question about your loan portfolio in plain English. Click an example to try:</div>
  <button class="chip" onclick="fill('Show customers who haven\\'t paid for last 3 months')">Show customers who haven't paid for last 3 months</button>
  <button class="chip" onclick="fill('List NPA accounts in MAHAD with POS above 1 lakh')">List NPA accounts in MAHAD with POS above 1 lakh</button>
<button class="chip" onclick="fill('Show all accounts with arrears greater than 2 EMI from November 2025 advances')">Show all accounts &gt;2 bucket from Nov 2025 advances</button>
  <button class="chip" onclick="fill('Show those accounts that need immediate action')">Show accounts that need immediate action</button>
</div>
<script>
function fill(text) {
    var doc  = window.parent.document;
    var ta   = doc.querySelector('[data-testid="stTextArea"] textarea');
    if (!ta) return;
    var set  = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
    set.call(ta, text);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
}
</script>
""", height=200, scrolling=False)

ai_query = st.text_area(
    "Query",
    key="ai_query_input",
    placeholder="Type your question here...",
    height=90,
    label_visibility="collapsed",
)

col_run, col_hint = st.columns([1, 4])
with col_run:
    run_btn = st.button("🔍  Run Query", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        "<div style='padding-top:10px;font-size:12px;color:#aaa;'>Powered by Gemini Flash 2.5 · LangGraph multi-agent pipeline</div>",
        unsafe_allow_html=True,
    )

if run_btn:
    if not ai_query.strip():
        st.warning("Please enter a question.")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found in .env file.")
    else:
        with st.spinner("Running multi-agent pipeline..."):
            st.session_state["ai_result"] = run_query(ai_query.strip(), df_curr)

# ── Render AI result (persists across reruns via session_state) ───────────────
result = st.session_state.get("ai_result")
if result:
    if result.get("error"):
        st.error(f"Query failed: {result['error']}")
    else:
        filtered_df = result["result_df"]
        kpis_q      = result["result_kpis"]
        rankings    = result["result_rankings"]
        insights    = result["insights"]
        plain       = result["parsed_filters"].get("plain_english", "")

        # Domain Expert interpretation card
        is_priority    = result.get("priority_mode", False)
        is_aggregation = result.get("aggregation_mode", False)
        result_type    = result.get("result_type", "loan_table")
        category   = result.get("query_category", "general").replace("_", " ").title()
        query_title = result.get("query_title", "")
        enriched   = result.get("enriched_query", "")
        risk_flag  = result.get("risk_flag", "medium")
        risk_color = {"high": "#dc2626", "medium": "#d97706", "low": "#16a34a"}.get(risk_flag, "#d97706")
        risk_label = {"high": "🔴 High Risk", "medium": "🟡 Medium Risk", "low": "🟢 Low Risk"}.get(risk_flag, "🟡 Medium Risk")

        # Domain Expert card
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #2a2a3e;border-radius:12px;padding:16px 20px;margin:16px 0;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="font-size:11px;font-weight:700;background:#1e293b;color:#94a3b8;
                           padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:1px;">{category}</span>
              <span style="font-size:14px;font-weight:700;color:#f1f5f9;">{query_title}</span>
            </div>
            <span style="font-size:11px;font-weight:700;color:{risk_color};">{risk_label}</span>
          </div>
          <div style="font-size:12px;color:#94a3b8;font-style:italic;line-height:1.6;border-top:1px solid #1e293b;padding-top:10px;">
            <span style="color:#FFC000;font-weight:600;font-style:normal;">🧠 Domain Expert:</span> &nbsp;{enriched}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Priority mode: account count selector + grouped display
        if is_priority:
            from agents.domain_expert import PRIORITY_RULES
            from agents.data_executor import distribute_priority_accounts

            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #FFC000;border-radius:12px;
                        padding:16px 20px;margin:0 0 16px 0;">
              <div style="font-size:13px;font-weight:800;color:#FFC000;margin-bottom:8px;letter-spacing:1px;">
                🎯 PRIORITY ACTION MODE
              </div>
              <div style="font-size:11px;color:#94a3b8;">
                Accounts ranked across 7 priority rules. Each account appears only once under its highest priority.
                Total available: <strong style="color:#fff">{kpis_q.get('Count', 0)} accounts</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
<style>
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stSelectbox"] [data-baseweb="select"] div,
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #FFC000 !important;
}
</style>""", unsafe_allow_html=True)
            sel_col, _ = st.columns([1, 3])
            with sel_col:
                n_accounts = st.selectbox(
                    "How many accounts to review?",
                    options=[20, 30, 40, 50],
                    index=1,
                    key="priority_n",
                )

            distributed = distribute_priority_accounts(filtered_df, n_accounts)

            # Show each priority group separately
            for priority_label, grp in distributed.groupby("Priority", sort=False):
                p_num = priority_label.split(":")[0].strip()
                p_name = priority_label.split(":")[-1].strip()
                p_color = "#dc2626" if p_num in ("P1", "P5") else "#f97316" if p_num in ("P2", "P3", "P6", "P7") else "#d97706"

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin:16px 0 6px 0;">
                  <span style="background:{p_color};color:#fff;font-size:11px;font-weight:800;
                               padding:3px 10px;border-radius:12px;">{p_num}</span>
                  <span style="font-size:14px;font-weight:700;color:#1a1a1a;">{p_name}</span>
                  <span style="font-size:12px;color:#888;">{len(grp)} accounts</span>
                </div>
                """, unsafe_allow_html=True)

                display_grp = grp.drop(columns=["Priority", "Why", "_rank"], errors="ignore")
                display_grp = display_grp.loc[:, ~display_grp.columns.duplicated()]
                _disp_grp = display_grp.reset_index(drop=True)
                st.dataframe(_safe_df(_disp_grp.head(1000)), use_container_width=True,
                             height=min(280, 45 + min(len(grp), 1000) * 36), hide_index=True)
                if len(_disp_grp) > 1000:
                    st.caption(f"Showing 1,000 of {len(_disp_grp):,} rows — download Excel for full list.")
                _dl_btn(_disp_grp, f"priority_{p_num}.xlsx", f"dl_priority_{p_num}")

        elif result_type == "single_stat" and not is_aggregation:
            # ── Single stat — simple count / sum over filtered rows ─────────────
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:14px;
                        padding:32px 36px;margin:0 0 20px 0;text-align:center;">
              <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                          letter-spacing:2px;margin-bottom:16px;">{query_title}</div>
              <div style="display:flex;gap:24px;justify-content:center;flex-wrap:wrap;">
            """ + "".join(
                f'<div style="min-width:140px;">'
                f'<div style="font-size:11px;color:#6b7280;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px;">{k}</div>'
                f'<div style="font-size:42px;font-weight:900;color:#FFC000;line-height:1;letter-spacing:-1px;">'
                f'{fmt_value(v, {"Count":"count","Total POS":"money","Avg Arrears/EMI":"pct","Total Demand":"money","Total Collection":"money","Collection %":"pct"}.get(k,"count"))}'
                f'</div></div>'
                for k, v in kpis_q.items() if v not in (0, 0.0, "")
            ) + f"""
              </div>
              <div style="font-size:13px;color:#4b5563;margin-top:20px;font-style:italic;">{plain}</div>
            </div>
            """, unsafe_allow_html=True)

        elif result_type == "single_stat" and is_aggregation:
            # ── Single stat from aggregation — show top answer + compact table ──
            agg_spec     = result.get("aggregation_spec", {})
            metric_label = agg_spec.get("metric_label", "Metric")
            _gb          = agg_spec.get("group_by", "Group")
            group_col    = f"{_gb[0]} ({_gb[1]})" if isinstance(_gb, list) else str(_gb)
            sort_asc     = agg_spec.get("sort_asc", True)
            if len(filtered_df) > 0 and metric_label in filtered_df.columns:
                top_row   = filtered_df.iloc[0]
                top_name  = top_row.get(group_col, "—")
                top_val   = top_row.get(metric_label, 0)
                direction = "lowest" if sort_asc else "highest"
                st.markdown(f"""
                <div style="background:#0d1117;border:1px solid #21262d;border-radius:14px;
                            padding:28px 36px;margin:0 0 20px 0;text-align:center;">
                  <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                              letter-spacing:2px;margin-bottom:12px;">{direction} {metric_label}</div>
                  <div style="font-size:36px;font-weight:900;color:#FFC000;letter-spacing:-0.5px;">{top_name}</div>
                  <div style="font-size:22px;font-weight:700;color:#e6edf3;margin-top:6px;">{int(top_val) if isinstance(top_val, (int, float)) and top_val == int(top_val) else round(top_val, 4)}</div>
                  <div style="font-size:12px;color:#4b5563;margin-top:12px;font-style:italic;">{plain}</div>
                </div>
                """, unsafe_allow_html=True)

            # Also show compact full ranking below
            if len(filtered_df) > 1:
                st.markdown("<div style='font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Full Ranking</div>", unsafe_allow_html=True)
                st.dataframe(_safe_df(filtered_df.head(1000)), use_container_width=True,
                             height=min(400, 50 + min(len(filtered_df), 1000) * 36), hide_index=True)
                if len(filtered_df) > 1000:
                    st.caption(f"Showing 1,000 of {len(filtered_df):,} rows — download Excel for full list.")
                _dl_btn(filtered_df, "ranking_result.xlsx", "dl_ranking")

        elif is_aggregation:
            # ── Aggregation result — ranked executive/branch/region table ──────
            agg_spec     = result.get("aggregation_spec", {})
            metric_label = agg_spec.get("metric_label", "Metric")
            _gb          = agg_spec.get("group_by", "Group")
            group_col    = f"{_gb[0]} ({_gb[1]})" if isinstance(_gb, list) else str(_gb)

            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #FFC000;border-radius:12px;
                        padding:16px 20px;margin:0 0 16px 0;">
              <div style="font-size:13px;font-weight:800;color:#FFC000;margin-bottom:6px;letter-spacing:1px;">
                📊 AGGREGATION RESULT — {metric_label.upper()}
              </div>
              <div style="font-size:12px;color:#94a3b8;">
                {plain}&nbsp; &nbsp;
                <strong style="color:#fff">{len(filtered_df)} {group_col}s</strong> ranked
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Build styled HTML table
            if len(filtered_df) > 0:
                header_cols = list(filtered_df.columns)
                th_cells = "".join(
                    f'<th style="padding:10px 14px;text-align:{"right" if c not in ("Rank", group_col) else "left"};'
                    f'font-size:10px;font-weight:800;color:#6b7280;text-transform:uppercase;'
                    f'letter-spacing:1.2px;border-bottom:1px solid #21262d;">{c}</th>'
                    for c in header_cols
                )
                rows_html = ""
                for i, row in filtered_df.iterrows():
                    rank_val = int(row.get("Rank", i + 1))
                    # Highlight top 3
                    row_bg = "rgba(255,192,0,0.06)" if rank_val <= 3 else "transparent"
                    cells = ""
                    for c in header_cols:
                        val = row[c]
                        if c == "Rank":
                            cells += f'<td style="padding:10px 14px;font-weight:800;color:#FFC000;">#{rank_val}</td>'
                        elif c == group_col:
                            cells += f'<td style="padding:10px 14px;font-weight:600;color:#e6edf3;font-size:13px;">{val}</td>'
                        elif c == metric_label:
                            _disp = int(val) if isinstance(val, (int, float)) and val == int(val) else round(val, 4)
                            cells += f'<td style="padding:10px 14px;text-align:right;font-weight:800;color:#FFC000;font-size:14px;">{_disp}</td>'
                        else:
                            cells += f'<td style="padding:10px 14px;text-align:right;color:#8b949e;font-size:13px;">{int(val) if isinstance(val, (int, float)) and val == int(val) else val}</td>'
                    rows_html += f'<tr style="background:{row_bg};border-bottom:1px solid #0d1117;">{cells}</tr>'

                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;overflow:hidden;margin-top:8px;">
                  <table style="width:100%;border-collapse:collapse;">
                    <thead><tr style="background:#0d1117;">{th_cells}</tr></thead>
                    <tbody>{rows_html}</tbody>
                  </table>
                </div>
                """, unsafe_allow_html=True)
                _dl_btn(filtered_df, "aggregation_result.xlsx", "dl_aggregation")
            else:
                st.warning("No data returned for this aggregation.")

        else:
            # ── Normal row-level filter result ─────────────────────────────────
            st.markdown(f"""
            <div style="background:#1a2e1a;border-left:4px solid #16a34a;border-radius:8px;
                        padding:12px 16px;margin:0 0 16px 0;color:#86efac;font-weight:600;font-size:14px;">
                ✓ Found <strong style="color:#fff">{kpis_q.get('Count',0)} accounts</strong>
                &nbsp; {plain}
            </div>
            """, unsafe_allow_html=True)

            # Result KPIs
            QKIND = {"Count":"count","Total POS":"money","Avg Arrears/EMI":"pct",
                     "Total Demand":"money","Total Collection":"money","Collection %":"pct"}
            kpi_html = "".join(
                f'<div class="result-kpi"><div class="result-kpi-label">{k}</div>'
                f'<div class="result-kpi-value">{fmt_value(v, QKIND[k])}</div></div>'
                for k, v in kpis_q.items()
            )
            st.markdown(
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">{kpi_html}</div>',
                unsafe_allow_html=True,
            )

            # Rankings — 2 columns: left (regions + bucket dist + MNT NAME) | right (branches)
            r1, r2 = st.columns(2)

            def _rank_rows(items, val_fmt="count"):
                rows = ""
                for name, val in list(items)[:5]:
                    display = fmt_value(val, val_fmt) if val_fmt == "money" else f"{int(val)}"
                    rows += f'<div class="rank-row"><span class="rank-name">{name}</span><span class="rank-value">{display}</span></div>'
                return rows

            with r1:
                left_html = ""
                if rankings.get("region_counts"):
                    left_html += f'<div class="rank-card" style="margin-bottom:12px;"><div class="rank-title">🗺 Top Regions by Account Count</div>{_rank_rows(rankings["region_counts"].items())}</div>'
                if rankings.get("bucket_dist"):
                    bucket_rows = "".join(
                        f'<div class="rank-row"><span class="rank-name">{b}</span><span class="rank-value">{p}%</span></div>'
                        for b, p in rankings["bucket_dist"].items()
                    )
                    left_html += f'<div class="rank-card" style="margin-bottom:12px;"><div class="rank-title">📊 Bucket Distribution</div>{bucket_rows}</div>'
                if rankings.get("mnt_details"):
                    mnt_rows = ""
                    for e in rankings["mnt_details"]:
                        mnt_rows += (
                            f'<div class="rank-row" style="gap:6px;">'
                            f'<span class="rank-name" style="flex:1.4;font-weight:600;">{e["name"]}</span>'
                            f'<span class="rank-name" style="flex:0.9;color:#9ca3af;font-size:11px;">{e["branch"]}</span>'
                            f'<span class="rank-value" style="min-width:36px;text-align:right;">{e["count"]}</span>'
                            f'<span class="rank-value" style="min-width:52px;text-align:right;color:#FFC000;">{fmt_value(e["pos"], "money")}</span>'
                            f'</div>'
                        )
                    left_html += f'<div class="rank-card"><div class="rank-title">👤 Top Executives by Account Count</div>{mnt_rows}</div>'
                st.markdown(left_html, unsafe_allow_html=True)

            with r2:
                right_html = ""
                if rankings.get("branch_counts"):
                    right_html += f'<div class="rank-card" style="margin-bottom:12px;"><div class="rank-title">🏢 Top Branches by Account Count</div>{_rank_rows(rankings["branch_counts"].items())}</div>'
                if rankings.get("branch_pos"):
                    right_html += f'<div class="rank-card"><div class="rank-title">💰 Top Branches by POS</div>{_rank_rows(rankings["branch_pos"].items(), "money")}</div>'
                st.markdown(right_html, unsafe_allow_html=True)

            # Customer table
            st.markdown("<div style='margin-top:20px;font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Matching Customer Records</div>", unsafe_allow_html=True)
            display_filtered = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
            st.dataframe(_safe_df(display_filtered.head(1000)), use_container_width=True, height=320, hide_index=True)
            if len(display_filtered) > 1000:
                st.caption(f"Showing 1,000 of {len(display_filtered):,} rows — download Excel for full list.")
            _dl_btn(display_filtered, "filtered_accounts.xlsx", "dl_filter_table")

        # AI observations (always shown)
        obs_lines = "".join(
            f'<div class="obs-line">{line}</div>'
            for line in insights.split("\n") if line.strip()
        )
        st.markdown(f"""
        <div class="obs-card">
          <div class="obs-title">💡 AI Observations</div>
          {obs_lines}
        </div>
        """, unsafe_allow_html=True)

        # User feedback — only shown when LangSmith is configured
        _ls_key = (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY", "")).strip()
        _run_id = result.get("run_id", "")
        if _ls_key and _run_id:
            _fb_cols = st.columns([2.5, 0.5, 0.5, 5])
            with _fb_cols[0]:
                st.markdown(
                    "<div style='font-size:12px;color:#888;padding-top:8px;'>Was this result helpful?</div>",
                    unsafe_allow_html=True,
                )
            with _fb_cols[1]:
                if st.button("👍", key="fb_up", help="Helpful result"):
                    _send_feedback(_run_id, score=1.0)
                    st.session_state["_fb_sent"] = True
            with _fb_cols[2]:
                if st.button("👎", key="fb_down", help="Result needs improvement"):
                    _send_feedback(_run_id, score=0.0)
                    st.session_state["_fb_sent"] = True
            if st.session_state.pop("_fb_sent", False):
                st.toast("Feedback saved to LangSmith", icon="✅")

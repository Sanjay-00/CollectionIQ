"""Shared UI helpers used across multiple tab modules."""

from io import BytesIO

import pandas as pd
import streamlit as st

from utils import load_and_validate


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type object columns to string for safe st.dataframe display."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
    return df


def _chart_card(fig) -> None:
    """Render a Plotly figure inside the standard bordered .chart-card wrapper."""
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)


def _divider(margin: str = "24px 0") -> None:
    """Standard horizontal section divider."""
    st.markdown(f'<div style="border-top:1px solid #e5e7eb;margin:{margin};"></div>', unsafe_allow_html=True)


def _empty_state(icon: str, title: str, sub: str) -> None:
    """Standard 'nothing to show yet' placeholder block."""
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-state-icon">{icon}</div>'
        f'<div class="empty-state-title">{title}</div>'
        f'<div class="empty-state-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _dl_btn(df: pd.DataFrame, filename: str, key: str) -> None:
    """Right-aligned compact Excel download button."""
    _, col = st.columns([5, 1])
    with col:
        st.download_button(
            "⬇ Excel", data=_excel_bytes(df), file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key, width='stretch',
        )


def _kpi_card_html(
    label: str, value: str, delta, *, unit: str = "%", inverse: bool = False,
    count_delta: int | None = None, zero_delta_bad: bool = False,
) -> str:
    """Shared KPI card markup: label + big value + MoM delta arrow.

    `inverse=True` means a falling value is the good direction (e.g. NPA %)
    so the arrow color logic flips. `delta=None` renders "no prev data"
    instead of an arrow (used when no previous-month file was uploaded).

    When the displayed delta rounds to 0.00%, two independent tabs in this
    app have always disagreed on the color (Dashboard: red: Portfolio
    Intelligence: green) - `zero_delta_bad` lets each caller keep its own
    pre-existing convention instead of silently picking one for both.

    If `count_delta` (the underlying raw account-count movement) is also
    passed and the displayed delta rounds to 0.00%, it takes priority over
    `zero_delta_bad` and breaks the tie using the real count instead: a
    +/-1 account move that got rounded away by the percentage still shows
    as worsened/improved, and a true zero-count move shows neutral/grey.
    """
    if delta is None:
        mom_html = '<div class="kpi-mom" style="color:#9ca3af;">no prev data</div>'
    else:
        is_tied = round(delta, 2) == 0.0
        if is_tied and count_delta:
            arrow = "▲" if count_delta > 0 else "▼"
            good  = (count_delta < 0) if inverse else (count_delta > 0)
            cls   = "kpi-mom-up" if good else "kpi-mom-down"
        elif is_tied and count_delta == 0:
            arrow, cls = "–", "kpi-mom-neutral"
        elif is_tied and zero_delta_bad:
            arrow = "▲" if delta >= 0 else "▼"
            cls   = "kpi-mom-down" if inverse else "kpi-mom-up"
        else:
            arrow = "▲" if delta >= 0 else "▼"
            good  = (delta <= 0) if inverse else (delta >= 0)
            cls   = "kpi-mom-up" if good else "kpi-mom-down"
        mom_html = f'<div class="kpi-mom">MoM <span class="{cls}">{arrow} {abs(delta):.2f}{unit}</span></div>'
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{mom_html}</div>'
    )


def _static_kpi_card_html(
    label: str, value, caption: str = "", *,
    color: str = "", value_style: str = "", card_style: str = "",
) -> str:
    """A plain (no MoM-delta) KPI card: colored top border, label, big value,
    optional caption line. Used for one-off summary numbers (e.g. "Fleet
    Operators: 12") where there's no month-over-month comparison to show."""
    border    = f"border-top-color:{color};" if color else ""
    card_attr = f' style="{border}{card_style}"' if (border or card_style) else ""
    val_attr  = f' style="color:{color};{value_style}"' if (color or value_style) else ""
    caption_html = f'<div class="kpi-mom">{caption}</div>' if caption else ""
    return (
        f'<div class="kpi-card"{card_attr}>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value"{val_attr}>{value}</div>'
        f'{caption_html}</div>'
    )


def _npa_pct_color(val: float, hi: float = 10.0, mid: float = 5.0) -> str:
    """Red/orange/green threshold color for an NPA%-style risk percentage."""
    v = val or 0
    return "#dc2626" if v > hi else ("#d97706" if v > mid else "#16a34a")


def _sma2_pct_color(val: float, hi: float = 10.0, mid: float = 5.0) -> str:
    """Red/orange/grey threshold color for a SMA-2%-style risk percentage."""
    v = val or 0
    return "#ef4444" if v > hi else ("#d97706" if v > mid else "#374151")


def _send_feedback(run_id: str, score: float) -> None:
    """Post thumbs-up (1.0) or thumbs-down (0.0) feedback to LangSmith."""
    try:
        from langsmith import Client as _LSClient
        _LSClient().create_feedback(run_id=run_id, key="result_quality", score=score)
    except Exception:
        pass


def _send_report_email(html_report: str, email_to: str, curr_month: str) -> tuple[bool, str]:
    """Send the HTML report via SMTP. Returns (success, error_msg).

    Thin wrapper around report_agent's send_report_email - the same function
    the auto-send-on-generate LangGraph node uses, so this tab's standalone
    "resend" button can't silently drift out of sync with it.
    """
    from report_agent.nodes.email_dispatcher import send_report_email
    return send_report_email(html_report, email_to, curr_month)


def _load_and_concat(files) -> tuple[pd.DataFrame | None, list[str]]:
    """Load one or more Excel files and concatenate into a single DataFrame."""
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
        return dfs[0], errors

    def _normalize_dt(df):
        for col in df.select_dtypes(include="datetime64").columns:
            try:
                df[col] = df[col].astype("datetime64[ms]")
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    dfs = [_normalize_dt(d) for d in dfs]
    combined = pd.concat(dfs, ignore_index=True)
    if "Loan No" in combined.columns:
        combined = combined.drop_duplicates(subset=["Loan No"], keep="first")
    return combined, errors

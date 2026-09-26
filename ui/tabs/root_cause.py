"""
Root Cause tab  -  WHY collection%/NPA%/SMA-2% are moving, region by region and
branch by branch. Zero AI calls. Built on top of Portfolio Intelligence's own
region scorecard (region_df) rather than recomputing it.

Tables here match the rest of the app's own convention (see
ui/tabs/portfolio_intelligence.py): a raw HTML <table>, black header with
amber (#FFC000) text, red/orange/green severity coloring on the columns
where "higher is worse" is unambiguous -- never a plain st.dataframe grid.
"""
import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _divider, append_total_row, _esc
from analysis.root_cause import (
    build_root_cause_workbook, load_daily_missed_feed,
    compute_recent_advances_daily_match, compute_recent_advances_status_by_grain,
)

_STATUS_COLOR = {"Improving": "#16a34a", "Worsening": "#dc2626", "Stable": "#d97706", "-": "#9ca3af"}
_STATUS_ICON  = {"Improving": "🟢", "Worsening": "🔴", "Stable": "🟡", "-": "⚪"}

# The exact red/orange/green hexes ui/components.py's _npa_pct_color/_sma2_pct_color
# and every Portfolio Intelligence table already use -- reused here so severity
# means the same three colors everywhere in the app, not a second palette.
_RED, _ORANGE, _GREEN, _NEUTRAL = "#dc2626", "#d97706", "#16a34a", "#374151"

# Per-bucket severity for the Recent Advances bucket tables (STD..NPA) -- a
# different coloring shape from the threshold-based severity below: here the
# BUCKET ITSELF carries the severity, not a numeric cutoff on the cell's value.
_BUCKET_COLOR = {"STD": _GREEN, "0-1": _NEUTRAL, "1-2": _ORANGE, "2-3": "#ea580c", "NPA": _RED}


def _section(title: str, margin_top: str = "0px") -> None:
    st.markdown(
        f'<div class="section-label" style="margin-top:{margin_top};">{title}</div>',
        unsafe_allow_html=True,
    )


def _badge(status: str) -> str:
    c = _STATUS_COLOR.get(status, "#9ca3af")
    i = _STATUS_ICON.get(status, "⚪")
    return (
        f'<span style="background:{c}18;color:{c};font-size:11px;font-weight:700;'
        f'padding:2px 9px;border-radius:12px;white-space:nowrap;">{i} {status}</span>'
    )


# ── Shared value formatters (used as `fmt` in the _html_table column specs) ──

def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or v == ""


def _fmt_text(v, _row=None) -> str:
    return "-" if _is_missing(v) else _esc(v)


def _fmt_int(v, _row=None) -> str:
    return "-" if _is_missing(v) else f"{int(v):,}"


def _fmt_pct(decimals: int = 1):
    def _f(v, _row=None) -> str:
        return "-" if _is_missing(v) else f"{float(v):.{decimals}f}%"
    return _f


def _fmt_cr(v, _row=None) -> str:
    return "-" if _is_missing(v) else f"₹{float(v):,.2f}Cr"


def _severity_color(hi: float, mid: float):
    """Red above `hi`, orange above `mid`, green otherwise -- same threshold
    shape as ui/components.py::_npa_pct_color, tuned per-metric below."""
    def _c(v, _row=None):
        if _is_missing(v):
            return None
        v = float(v)
        return _RED if v > hi else (_ORANGE if v > mid else _GREEN)
    return _c


def _bucket_row_color(_v, row):
    return _BUCKET_COLOR.get(row.get("Bucket"))


def _bucket_col_color(bucket_name: str):
    def _c(_v, _row=None):
        return _BUCKET_COLOR.get(bucket_name)
    return _c


# ── Shared HTML table builder (matches ui/tabs/portfolio_intelligence.py) ────

def _html_table(df: pd.DataFrame, cols: list[dict], last_row_is_total: bool = False) -> str:
    """cols: list of {"key", "label"?, "align"? ("left"/"right", default
    right), "fmt"?: (val, row) -> str, "color"?: (val, row) -> str|None,
    "bold"?: bool}. When last_row_is_total, that row (the caller's own
    Grand Total / Total row) renders bold with an amber top border and never
    gets severity-colored -- an aggregate isn't a peer to color-rank."""
    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:7px 12px;font-size:11px;'
        f'text-align:{c.get("align", "right")};white-space:nowrap;">{_esc(c.get("label", c["key"]))}</th>'
        for c in cols
    )
    n = len(df)
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        is_total = last_row_is_total and i == n - 1
        cells = ""
        for c in cols:
            val = row.get(c["key"])
            align = c.get("align", "right")
            fmt = c.get("fmt", _fmt_text)
            display = fmt(val, row)
            color = None if is_total else (c["color"](val, row) if c.get("color") else None)
            style = f"padding:7px 12px;font-size:12px;text-align:{align};"
            if color:
                style += f"color:{color};font-weight:600;"
            elif c.get("bold"):
                style += "font-weight:700;"
            if is_total:
                style += "font-weight:800;border-top:2px solid #FFC000;"
            cells += f'<td style="{style}">{display}</td>'
        row_bg = "#fffbea" if is_total else "#fff"
        border = "" if is_total else "border-bottom:1px solid #f0f0f0;"
        rows_html += f'<tr style="background:{row_bg};{border}">{cells}</tr>'
    return (
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


def _render_contamination_notice(contamination: dict) -> None:
    if not contamination:
        return
    items = ", ".join(f"{col} ({n} rows)" for col, n in contamination.items())
    st.markdown(
        f'<div style="background:#fef3c7;border:1px solid #fbbf24;border-radius:8px;'
        f'padding:10px 14px;font-size:12px;color:#92400e;margin-bottom:16px;">'
        f'⚠️ Data quality: unrecognized values found and excluded from the flags below '
        f'(not silently trusted as Yes/No) &mdash; {_esc(items)}.'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_why_table(why_df: pd.DataFrame) -> None:
    _section("Region Diagnosis: Why Is It Moving?")
    st.caption(
        "One row per region: current NPA%/Collection% and MoM movement, plus the single "
        "driver claiming the largest share of that region's delinquent book. "
        "Worsening regions are shown first."
    )
    if why_df.empty:
        st.info("Not enough data to compute region diagnostics.")
        return

    display_cols = ["Region", "Status", "NPA%", "Δ NPA%", "Collection%", "Dominant Driver", "Driver Share %", "Delinquent Accounts"]
    display_cols = [c for c in display_cols if c in why_df.columns]
    view = why_df[display_cols].copy()

    for _, row in view.iterrows():
        badge = _badge(row.get("Status", "-"))
        delta = row.get("Δ NPA%")
        delta_txt = f"{delta:+.2f}pp" if delta is not None and pd.notna(delta) else "—"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:14px;padding:10px 14px;'
            f'border:1px solid #e5e7eb;border-radius:8px;margin-bottom:8px;">'
            f'<div style="min-width:120px;font-weight:700;font-size:13px;">{_esc(row["Region"])}</div>'
            f'{badge}'
            f'<div style="font-size:12px;color:#374151;">NPA% <b>{row.get("NPA%", 0):.1f}%</b> ({delta_txt})</div>'
            f'<div style="font-size:12px;color:#374151;">Collection% <b>{row.get("Collection%", 0):.1f}%</b></div>'
            f'<div style="flex:1;text-align:right;font-size:12px;">'
            f'<span style="color:#6b7280;">Dominant driver:</span> '
            f'<b style="color:#111827;">{_esc(row.get("Dominant Driver", "-"))}</b> '
            f'<span style="color:#6b7280;">({row.get("Driver Share %", 0):.0f}% of delinquents)</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("View as table"):
        cols = [
            {"key": "Region", "align": "left", "bold": True},
            {"key": "Status", "align": "left", "fmt": lambda v, _r=None: _badge(str(v))},
            {"key": "NPA%", "fmt": _fmt_pct(1), "color": _severity_color(hi=10, mid=5)},
            {"key": "Δ NPA%", "fmt": lambda v, _r=None: "-" if _is_missing(v) else f"{v:+.2f}pp"},
            {"key": "Collection%", "fmt": _fmt_pct(1)},
            {"key": "Dominant Driver", "align": "left"},
            {"key": "Driver Share %", "fmt": _fmt_pct(0)},
            {"key": "Delinquent Accounts", "fmt": _fmt_int},
        ]
        cols = [c for c in cols if c["key"] in view.columns]
        st.markdown(_html_table(view, cols), unsafe_allow_html=True)
    _dl_btn(why_df, "region_why_diagnosis.xlsx", "dl_root_cause_why")


def _render_insurance_split(ins_df: pd.DataFrame) -> None:
    _section("Insurance-vs-Installment Split, by Branch", margin_top="24px")
    st.caption(
        "Among delinquent accounts: Insurance-Only means the customer is current on the "
        "loan itself (installment arrears ≤ 0) but an unpaid insurance/expense charge is "
        "creating the delinquency — fixable with a cash/WCL adjustment, not a credit "
        "problem. Both = genuine shortfall on both legs (colored by severity). "
        "Sorted by highest Insurance-Only share first."
    )
    if ins_df.empty:
        st.info("Not enough data to compute the insurance split.")
        return
    show_cols = ["Unit", "RegionName", "Delinquent Accounts", "Insurance-Only", "Insurance-Only %",
                 "Installment-Only", "Installment-Only %", "Both", "Both %"]
    show_cols = [c for c in show_cols if c in ins_df.columns]
    ratio_cols = {
        "Insurance-Only %": ("Insurance-Only", "Delinquent Accounts"),
        "Installment-Only %": ("Installment-Only", "Delinquent Accounts"),
        "Both %": ("Both", "Delinquent Accounts"),
    }
    view = append_total_row(ins_df[show_cols], ratio_cols=ratio_cols)
    cols = [
        {"key": "Unit", "align": "left", "bold": True},
        {"key": "RegionName", "align": "left", "label": "Region"},
        {"key": "Delinquent Accounts", "fmt": _fmt_int},
        {"key": "Insurance-Only", "fmt": _fmt_int},
        {"key": "Insurance-Only %", "fmt": _fmt_pct(1)},
        {"key": "Installment-Only", "fmt": _fmt_int},
        {"key": "Installment-Only %", "fmt": _fmt_pct(1)},
        {"key": "Both", "fmt": _fmt_int},
        # "Both" = genuine shortfall on both legs -- the most severe of the three, colored accordingly.
        {"key": "Both %", "fmt": _fmt_pct(1), "color": _severity_color(hi=50, mid=25)},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    st.markdown(_html_table(view, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(ins_df, "insurance_vs_installment_split.xlsx", "dl_root_cause_insurance")


def _render_chronic_shock_split(cs_df: pd.DataFrame) -> None:
    _section("Chronic vs. Shock Split, by Branch", margin_top="24px")
    st.caption(
        "Chronic = no collection for 3+ months and arrears already past 6 EMIs at some "
        "point (behavioral history). Hard = currently ≥6 EMIs overdue (today's snapshot). "
        "The two can diverge: Chronic+Hard are write-off/legal candidates, Chronic+Not Hard "
        "are worth a call before they relapse, Shock+Hard are sudden deterioration worth a "
        "restructuring conversation."
    )
    if cs_df.empty:
        st.info("Not enough data to compute the chronic/shock split.")
        return
    show_cols = ["Unit", "RegionName", "Delinquent Accounts",
                 "Chronic + Hard", "Chronic + Hard %", "Chronic + Not Hard", "Chronic + Not Hard %",
                 "Shock + Hard", "Shock + Hard %", "Neither", "Neither %"]
    show_cols = [c for c in show_cols if c in cs_df.columns]
    ratio_cols = {
        "Chronic + Hard %": ("Chronic + Hard", "Delinquent Accounts"),
        "Chronic + Not Hard %": ("Chronic + Not Hard", "Delinquent Accounts"),
        "Shock + Hard %": ("Shock + Hard", "Delinquent Accounts"),
        "Neither %": ("Neither", "Delinquent Accounts"),
    }
    view = append_total_row(cs_df[show_cols], ratio_cols=ratio_cols)
    cols = [
        {"key": "Unit", "align": "left", "bold": True},
        {"key": "RegionName", "align": "left", "label": "Region"},
        {"key": "Delinquent Accounts", "fmt": _fmt_int},
        {"key": "Chronic + Hard", "fmt": _fmt_int},
        {"key": "Chronic + Hard %", "fmt": _fmt_pct(1), "color": _severity_color(hi=15, mid=7)},
        {"key": "Chronic + Not Hard", "fmt": _fmt_int},
        {"key": "Chronic + Not Hard %", "fmt": _fmt_pct(1)},
        {"key": "Shock + Hard", "fmt": _fmt_int},
        {"key": "Shock + Hard %", "fmt": _fmt_pct(1), "color": _severity_color(hi=15, mid=7)},
        {"key": "Neither", "fmt": _fmt_int},
        {"key": "Neither %", "fmt": _fmt_pct(1)},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    st.markdown(_html_table(view, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(cs_df, "chronic_vs_shock_split.xlsx", "dl_root_cause_chronic_shock")


def _render_recent_advances_summary(summary: dict) -> None:
    if not summary:
        st.info("Not enough data to compute the Recent Advances cohort (needs an Ag_Date column).")
        return
    cards = [
        ("Portfolio Loans", f"{summary['portfolio_count']:,}"),
        ("Portfolio SOH", f"₹{summary['portfolio_soh_cr']:,.2f} Cr"),
        ("Cohort Loans", f"{summary['cohort_count']:,}", f"{summary['cohort_count_pct']:.1f}% of portfolio"),
        ("Cohort SOH", f"₹{summary['cohort_soh_cr']:,.2f} Cr", f"{summary['cohort_soh_pct']:.1f}% of portfolio"),
    ]
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        label, value = card[0], card[1]
        sub = card[2] if len(card) > 2 else ""
        with col:
            st.markdown(
                f'<div style="border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;">'
                f'<div style="font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;">{_esc(label)}</div>'
                f'<div style="font-size:20px;font-weight:800;color:#111827;margin-top:2px;">{_esc(value)}</div>'
                f'<div style="font-size:11px;color:#9ca3af;margin-top:2px;">{_esc(sub)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_recent_advances_bucket(bucket_df: pd.DataFrame) -> None:
    if bucket_df.empty:
        st.info("Not enough data to compute the bucket split.")
        return
    st.caption(
        "Bucket-wise split of the cohort itself, colored by severity (green=STD ... red=NPA). "
        "% is of the cohort's own graded total."
    )
    cols = [
        {"key": "Bucket", "align": "left", "bold": True, "color": _bucket_row_color},
        {"key": "Count", "fmt": _fmt_int},
        {"key": "Count %", "fmt": _fmt_pct(1), "color": _bucket_row_color},
        {"key": "SOH (Cr)", "fmt": _fmt_cr},
        {"key": "SOH %", "fmt": _fmt_pct(1), "color": _bucket_row_color},
    ]
    cols = [c for c in cols if c["key"] in bucket_df.columns]
    st.markdown(_html_table(bucket_df, cols), unsafe_allow_html=True)
    _dl_btn(bucket_df, "recent_advances_bucket_split.xlsx", "dl_root_cause_recent_bucket")


def _render_recent_advances_by_group(count_df: pd.DataFrame, soh_df: pd.DataFrame, label: str, group_col: str) -> None:
    if count_df.empty and soh_df.empty:
        return
    group_key = group_col
    tab_count, tab_soh = st.tabs([f"By {label}  —  Count %", f"By {label}  —  SOH %"])
    with tab_count:
        if count_df.empty:
            st.info("No data.")
        else:
            cols = [{"key": group_key, "align": "left", "bold": True, "label": label}]
            if "RegionName" in count_df.columns and group_key != "RegionName":
                cols.append({"key": "RegionName", "align": "left", "label": "Region"})
            for bucket in ("STD", "0-1", "1-2", "2-3", "NPA"):
                if bucket in count_df.columns:
                    cols.append({"key": bucket, "fmt": _fmt_int})
                if f"{bucket} %" in count_df.columns:
                    cols.append({"key": f"{bucket} %", "fmt": _fmt_pct(1), "color": _bucket_col_color(bucket)})
            if "Total Count" in count_df.columns:
                cols.append({"key": "Total Count", "fmt": _fmt_int, "bold": True})
            st.markdown(_html_table(count_df, cols), unsafe_allow_html=True)
            _dl_btn(count_df, f"recent_advances_by_{label.lower()}_count.xlsx", f"dl_root_cause_recent_{label.lower()}_count")
    with tab_soh:
        if soh_df.empty:
            st.info("No data.")
        else:
            cols = [{"key": group_key, "align": "left", "bold": True, "label": label}]
            if "RegionName" in soh_df.columns and group_key != "RegionName":
                cols.append({"key": "RegionName", "align": "left", "label": "Region"})
            for bucket in ("STD", "0-1", "1-2", "2-3", "NPA"):
                if f"{bucket} (Cr)" in soh_df.columns:
                    cols.append({"key": f"{bucket} (Cr)", "fmt": _fmt_cr})
                if f"{bucket} %" in soh_df.columns:
                    cols.append({"key": f"{bucket} %", "fmt": _fmt_pct(1), "color": _bucket_col_color(bucket)})
            if "Total SOH (Cr)" in soh_df.columns:
                cols.append({"key": "Total SOH (Cr)", "fmt": _fmt_cr, "bold": True})
            st.markdown(_html_table(soh_df, cols), unsafe_allow_html=True)
            _dl_btn(soh_df, f"recent_advances_by_{label.lower()}_soh.xlsx", f"dl_root_cause_recent_{label.lower()}_soh")


# Thresholds are tuned to the Nov'25+ cohort's own observed scale (verified on
# real data), NOT the portfolio-wide NPA%/SMA-2% thresholds elsewhere in the
# app: Delinquent % (any missed due date) legitimately runs high (10-50%+) on
# a young book, while PNPA%/NPA% within that same cohort are naturally much
# smaller (any NPA at all this early is already a bad sign) -- so each metric
# gets its own hi/mid cutoffs rather than sharing one scale.
_STATUS_SEVERITY = {
    "Delinquent %": _severity_color(hi=25, mid=12),
    "PNPA %": _severity_color(hi=5, mid=2),
    "NPA %": _severity_color(hi=3, mid=1.5),
}


def _render_status_table(status_df: pd.DataFrame, grain_label: str, required_cols_hint: str) -> None:
    if status_df.empty:
        st.warning(f"Could not build the {grain_label.lower()} table ({required_cols_hint}).")
        return
    total_col = next(c for c in status_df.columns if c.startswith("Total "))
    label_cols = [c for c in ("Region", "Branch", "Executive") if c in status_df.columns]
    primary_label = label_cols[0]
    unmapped = status_df[status_df[total_col].isna() & (status_df[primary_label] != "Grand Total")][primary_label].tolist()
    if unmapped:
        shown = ", ".join(map(str, unmapped[:10])) + (f" (+{len(unmapped) - 10} more)" if len(unmapped) > 10 else "")
        st.warning(
            f"No matching {grain_label.lower()} in the master file for: {shown} "
            "(not in this master, or spelled differently). Their counts are shown but "
            "their % is blank, and the Grand Total % is understated."
        )
    cols = []
    for lc in label_cols:
        cols.append({"key": lc, "align": "left", "bold": True})
    cols.append({"key": total_col, "fmt": _fmt_int})
    cols.append({"key": "Delinquent Cases", "fmt": _fmt_int})
    cols.append({"key": "Delinquent %", "fmt": _fmt_pct(2), "color": _STATUS_SEVERITY["Delinquent %"]})
    cols.append({"key": "PNPA (SMA-2) Cases", "fmt": _fmt_int})
    cols.append({"key": "PNPA %", "fmt": _fmt_pct(2), "color": _STATUS_SEVERITY["PNPA %"]})
    cols.append({"key": "NPA Cases", "fmt": _fmt_int})
    cols.append({"key": "NPA %", "fmt": _fmt_pct(2), "color": _STATUS_SEVERITY["NPA %"]})
    st.markdown(_html_table(status_df, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(status_df, f"cohort_delinquency_by_{grain_label.lower()}.xlsx", f"dl_root_cause_{grain_label.lower()}_status")


def _render_status_by_grain(df_curr: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="region")
    branch_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="branch")
    executive_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="executive")

    tab_r, tab_b, tab_e = st.tabs(["By Region", "By Branch", "By Executive"])
    with tab_r:
        _render_status_table(region_status, "Region", "the feed needs REGIONNAME, ARREARS / EMI and LOAN NO columns")
    with tab_b:
        _render_status_table(branch_status, "Branch", "the feed needs UNIT, ARREARS / EMI and LOAN NO columns")
    with tab_e:
        _render_status_table(executive_status, "Executive", "the feed needs RE NAME, UNIT, ARREARS / EMI and LOAN NO columns")
    return region_status, branch_status, executive_status


def _render_daily_match(df_curr: pd.DataFrame) -> dict | None:
    """Optional: upload today's due-date-missed daily feed (a DIFFERENT file/
    schema than the master LCC -- see analysis.root_cause module docstring) to
    cross-check how many of the cohort are CURRENTLY missing a due date,
    fresher than master's own snapshot. Returns the match dict (also stashed
    in session_state) so the caller can fold it into the Excel export."""
    _section("Cohort Delinquency Status: Today's Due-Date-Missed Feed", margin_top="24px")
    st.caption(
        "Optional. Upload the Excel Automation project's daily due-date-missed extract: "
        "totals come from the master file above, delinquent / PNPA (SMA-2) / NPA counts "
        "from the feed's own Arrears/EMI (today's live status), each as a % of the "
        "group's total cohort. Sorted worst first, colored red (worst) to green (best)."
    )
    uploaded = st.file_uploader("Daily due-date-missed feed (.xlsb/.xlsx/.xls)", type=["xlsb", "xlsx", "xls"], key="rc_daily_feed_upload")
    if uploaded is None:
        return st.session_state.get("rc_daily_match")

    daily_df, err = load_daily_missed_feed(uploaded)
    if err:
        st.error(err)
        return None

    match = compute_recent_advances_daily_match(df_curr, daily_df)
    if not match:
        st.warning("Could not match this file against the cohort (check Ag_Date/Loan No columns).")
        return None

    st.session_state["rc_daily_match"] = match
    region_status, branch_status, executive_status = _render_status_by_grain(df_curr, daily_df)
    st.session_state["rc_region_status"] = region_status
    st.session_state["rc_branch_status"] = branch_status
    st.session_state["rc_executive_status"] = executive_status
    c1, c2, c3 = st.columns(3)
    c1.metric("Cohort on today's missed-due-date list", f"{match['matched_count']:,}", f"{match['pct_of_cohort_on_daily_list']:.1f}% of cohort")
    c2.metric("Daily feed loans", f"{match['daily_feed_count']:,}", f"{match['pct_of_daily_list_in_cohort']:.1f}% found in cohort")
    c3.metric("Not yet in master snapshot", f"{match['unmatched_daily_count']:,}", "likely disbursed after this closing", delta_color="off")
    if match["unmatched_daily_count"]:
        with st.expander(f"View the {match['unmatched_daily_count']} loans not found in this master snapshot"):
            st.markdown(
                _html_table(pd.DataFrame({"Loan No": match["unmatched_daily_loan_nos"]}), [{"key": "Loan No", "align": "left"}]),
                unsafe_allow_html=True,
            )
    return match


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_workbook_bytes(
    why_df: pd.DataFrame, insurance_split_df: pd.DataFrame,
    chronic_shock_df: pd.DataFrame, contamination: dict, generated_for: str,
    recent_summary: dict, recent_bucket_df: pd.DataFrame,
    recent_by_region_count: pd.DataFrame, recent_by_region_soh: pd.DataFrame,
    recent_by_branch_count: pd.DataFrame, recent_by_branch_soh: pd.DataFrame,
    daily_match: dict, region_status: pd.DataFrame,
    branch_status: pd.DataFrame, executive_status: pd.DataFrame,
) -> bytes:
    return build_root_cause_workbook(
        why_df, insurance_split_df, chronic_shock_df, contamination, generated_for,
        recent_summary=recent_summary, recent_bucket_df=recent_bucket_df,
        recent_by_region_count=recent_by_region_count, recent_by_region_soh=recent_by_region_soh,
        recent_by_branch_count=recent_by_branch_count, recent_by_branch_soh=recent_by_branch_soh,
        daily_match=daily_match, region_status=region_status,
        branch_status=branch_status, executive_status=executive_status,
    )


def render_root_cause_tab(
    why_df: pd.DataFrame,
    insurance_split_df: pd.DataFrame,
    chronic_shock_df: pd.DataFrame,
    contamination: dict,
    curr_month: str = "",
    df_curr: pd.DataFrame | None = None,
    recent_summary: dict | None = None,
    recent_bucket_df: pd.DataFrame | None = None,
    recent_by_region_count: pd.DataFrame | None = None,
    recent_by_region_soh: pd.DataFrame | None = None,
    recent_by_branch_count: pd.DataFrame | None = None,
    recent_by_branch_soh: pd.DataFrame | None = None,
) -> None:
    recent_bucket_df = recent_bucket_df if recent_bucket_df is not None else pd.DataFrame()
    recent_by_region_count = recent_by_region_count if recent_by_region_count is not None else pd.DataFrame()
    recent_by_region_soh = recent_by_region_soh if recent_by_region_soh is not None else pd.DataFrame()
    recent_by_branch_count = recent_by_branch_count if recent_by_branch_count is not None else pd.DataFrame()
    recent_by_branch_soh = recent_by_branch_soh if recent_by_branch_soh is not None else pd.DataFrame()

    header_col, dl_col = st.columns([4, 1])
    with header_col:
        st.markdown("""
        <div class="ai-panel">
          <div class="ai-title">Root Cause Diagnostics</div>
          <div class="ai-subtitle">
            Not just "what the numbers are" — why Collection%/NPA%/SMA-2% are moving,
            region by region and branch by branch, so mitigation can follow.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with dl_col:
        if not why_df.empty:
            st.download_button(
                "⬇ Download Report (Excel)",
                data=_cached_workbook_bytes(
                    why_df, insurance_split_df, chronic_shock_df, contamination, curr_month,
                    recent_summary or {}, recent_bucket_df,
                    recent_by_region_count, recent_by_region_soh,
                    recent_by_branch_count, recent_by_branch_soh,
                    st.session_state.get("rc_daily_match") or {},
                    st.session_state.get("rc_region_status", pd.DataFrame()),
                    st.session_state.get("rc_branch_status", pd.DataFrame()),
                    st.session_state.get("rc_executive_status", pd.DataFrame()),
                ),
                file_name=f"root_cause_diagnostics_{curr_month or 'report'}.xlsx".replace(" ", "_"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_root_cause_workbook",
                width='stretch',
            )

    _render_contamination_notice(contamination)
    _render_why_table(why_df)
    _divider()
    _render_insurance_split(insurance_split_df)
    _divider()
    _render_chronic_shock_split(chronic_shock_df)
    _divider()

    _section("Recent Advances Cohort", margin_top="24px")
    st.caption(f"Loans agreed from {(recent_summary or {}).get('cohort_start', 'the configured cutoff')} onward.")
    _render_recent_advances_summary(recent_summary or {})
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    _render_recent_advances_bucket(recent_bucket_df)
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    _render_recent_advances_by_group(recent_by_region_count, recent_by_region_soh, "Region", group_col="RegionName")
    _render_recent_advances_by_group(recent_by_branch_count, recent_by_branch_soh, "Branch", group_col="Unit")

    if df_curr is not None:
        _divider()
        _render_daily_match(df_curr)

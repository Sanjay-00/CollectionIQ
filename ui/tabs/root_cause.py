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
from ui.glossary import DRIVER_HELP, help_for, info_icon
from config import HARD_BUCKET_ARREARS_EMI_MIN
from analysis.root_cause import (
    build_root_cause_workbook, load_daily_missed_feed,
    ARREARS_GROUPS, HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, EARLY_DELINQUENCY,
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
        f'text-align:{c.get("align", "right")};white-space:nowrap;">{_esc(c.get("label", c["key"]))}'
        f'{info_icon(c.get("help", help_for(c["key"])), color="#fde68a")}</th>'
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
        f'⚠️ Some yes/no columns contain values the app doesn\'t recognise: {_esc(items)}. '
        f'Those rows are left out of these tables rather than guessed as Yes or No.'
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
        delta_txt = f"{delta:+.2f}pp" if delta is not None and pd.notna(delta) else "no prev month"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:14px;padding:10px 14px;'
            f'border:1px solid #e5e7eb;border-radius:8px;margin-bottom:8px;">'
            f'<div style="min-width:120px;font-weight:700;font-size:13px;">{_esc(row["Region"])}</div>'
            f'{badge}'
            f'<div style="font-size:12px;color:#374151;">NPA% <b>{row.get("NPA%", 0):.1f}%</b> ({delta_txt})</div>'
            f'<div style="font-size:12px;color:#374151;">Collection% <b>{row.get("Collection%", 0):.1f}%</b></div>'
            f'<div style="flex:1;text-align:right;font-size:12px;">'
            f'<span style="color:#6b7280;">Dominant driver{info_icon(help_for("Dominant Driver"))}:</span> '
            f'<b style="color:#111827;">{_esc(row.get("Dominant Driver", "-"))}</b>'
            f'{info_icon(DRIVER_HELP.get(row.get("Dominant Driver")))} '
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
            {"key": "Dominant Driver", "align": "left",
             "fmt": lambda v, _r=None: _fmt_text(v) + info_icon(DRIVER_HELP.get(v))},
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
        "creating the delinquency: fixable with a cash/WCL adjustment, not a credit "
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
    _section("Deep Arrears: Still Paying vs Not Paying, by Branch", margin_top="24px")
    st.caption(
        f"Every delinquent loan in one of four groups, by how far behind it is today "
        f"({HARD_BUCKET_ARREARS_EMI_MIN}+ EMIs = hard) and whether the customer has paid anything in "
        f"the last 3 months. Hover the ⓘ on each column for what to do with that group."
    )
    if cs_df.empty:
        st.info("Not enough data to compute the chronic/shock split.")
        return
    show_cols = ["Unit", "RegionName", "Delinquent Accounts"] + [x for g in ARREARS_GROUPS for x in (g, f"{g} %")]
    show_cols = [c for c in show_cols if c in cs_df.columns]
    ratio_cols = {f"{g} %": (g, "Delinquent Accounts") for g in ARREARS_GROUPS}
    view = append_total_row(cs_df[show_cols], ratio_cols=ratio_cols)
    cols = [
        {"key": "Unit", "align": "left", "bold": True},
        {"key": "RegionName", "align": "left", "label": "Region"},
        {"key": "Delinquent Accounts", "fmt": _fmt_int},
        {"key": HARD_NOT_PAYING, "fmt": _fmt_int},
        {"key": f"{HARD_NOT_PAYING} %", "fmt": _fmt_pct(1), "color": _severity_color(hi=15, mid=7)},
        {"key": HARD_STILL_PAYING, "fmt": _fmt_int},
        {"key": f"{HARD_STILL_PAYING} %", "fmt": _fmt_pct(1), "color": _severity_color(hi=15, mid=7)},
        {"key": WAS_SILENT, "fmt": _fmt_int},
        {"key": f"{WAS_SILENT} %", "fmt": _fmt_pct(1)},
        {"key": EARLY_DELINQUENCY, "fmt": _fmt_int},
        {"key": f"{EARLY_DELINQUENCY} %", "fmt": _fmt_pct(1)},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    st.markdown(_html_table(view, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(cs_df, "deep_arrears_paying_vs_not_paying.xlsx", "dl_root_cause_chronic_shock")


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
    tab_count, tab_soh = st.tabs([f"By {label}: Count %", f"By {label}: SOH %"])
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
# a young book, while SMA-2%/NPA% within that same cohort are naturally much
# smaller (any NPA at all this early is already a bad sign) -- so each metric
# gets its own hi/mid cutoffs rather than sharing one scale.
_STATUS_SEVERITY = {
    "Delinquent": _severity_color(hi=25, mid=12),
    "SMA-2": _severity_color(hi=5, mid=2),
    "NPA": _severity_color(hi=3, mid=1.5),
}


def _count_with_pct(key: str):
    """'150 (7.5%)' -- the count plus its share of the row's Running Loans."""
    def _f(v, row) -> str:
        if _is_missing(v):
            return "-"
        pct = row.get(f"{key} %")
        pct_html = "" if _is_missing(pct) else f' <span style="color:#6b7280;font-weight:400;">({float(pct):.1f}%)</span>'
        return f"{int(v):,}{pct_html}"
    return _f


def _color_by_pct(key: str):
    severity = _STATUS_SEVERITY.get(key)

    def _c(_v, row):
        return severity(row.get(f"{key} %")) if severity else None
    return _c


def _render_status_table(status_df: pd.DataFrame, grain_label: str) -> None:
    if status_df.empty:
        st.warning(
            f"Could not build the {grain_label.lower()} table: the list needs LOAN NO and "
            "ARREARS / EMI columns, and the LCC needs running loans from the cohort start onward."
        )
        return
    label_cols = [c for c in status_df.columns if c in ("Region", "Branch", "Executive")]
    count_cols = [c for c in status_df.columns if c not in label_cols and c != "Running Loans" and not c.endswith(" %")]
    cols = [{"key": lc, "align": "left", "bold": lc == label_cols[0]} for lc in label_cols]
    cols.append({"key": "Running Loans", "fmt": _fmt_int, "bold": True})
    for c in count_cols:
        cols.append({"key": c, "fmt": _count_with_pct(c), "color": _color_by_pct(c)})
    st.markdown(_html_table(status_df, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(status_df, f"recent_advances_delinquency_by_{grain_label.lower()}.xlsx", f"dl_root_cause_{grain_label.lower()}_status")


def _render_status_by_grain(df_curr: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="region")
    branch_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="branch")
    executive_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="executive")

    tab_r, tab_b, tab_e = st.tabs(["By Region", "By Branch", "By Executive"])
    with tab_r:
        _render_status_table(region_status, "Region")
    with tab_b:
        _render_status_table(branch_status, "Branch")
    with tab_e:
        _render_status_table(executive_status, "Executive")
    return region_status, branch_status, executive_status


def _render_match_notes(match: dict) -> None:
    """Explain every list loan the table does NOT count, in plain words."""
    notes = []
    if match["exec_changed_count"]:
        notes.append(
            f"**{match['exec_changed_count']:,}** delinquent loans have a different executive on the list "
            "than in the LCC. They are shown under their LCC executive, so each executive's % "
            "compares the same loans."
        )
    if match["other_scope_count"]:
        notes.append(f"**{match['other_scope_count']:,}** list loans are from regions/zones not in this LCC: ignored.")
    if match["hidden_by_filter_count"]:
        notes.append(f"**{match['hidden_by_filter_count']:,}** list loans are hidden by the sidebar filters.")
    if notes:
        st.markdown("\n".join(f"- {n}" for n in notes))

    if match["missing_from_lcc_count"]:
        by_branch = ", ".join(f"{_esc(b)}: {n}" for b, n in match["missing_from_lcc_by_branch"].items())
        with st.expander(
            f"⚠ {match['missing_from_lcc_count']:,} list loans are from regions this LCC covers, "
            "but the loans themselves are missing from it"
        ):
            st.caption(
                "Not counted (no LCC row means no total to compare against). Usually a branch missing "
                f"from this LCC extract. By branch on the list: {by_branch}"
            )
            st.markdown(
                _html_table(pd.DataFrame({"Loan No": match["missing_from_lcc_loan_nos"]}), [{"key": "Loan No", "align": "left"}]),
                unsafe_allow_html=True,
            )
    if match["in_view_not_counted_count"]:
        with st.expander(
            f"⚠ {match['in_view_not_counted_count']:,} list loans are in the LCC but not counted "
            "as running recent loans there"
        ):
            st.caption(
                "The LCC gives these an agreement date before the cohort start, or a status other "
                "than RUN. If the list shows a later date, the LCC date may have day and month swapped."
            )
            st.markdown(
                _html_table(pd.DataFrame({"Loan No": match["in_view_not_counted_loan_nos"]}), [{"key": "Loan No", "align": "left"}]),
                unsafe_allow_html=True,
            )


def _render_daily_match(df_curr: pd.DataFrame, df_all: pd.DataFrame | None = None) -> dict | None:
    """Optional: upload today's due-date-missed list to see, per region/branch/
    executive, how many running recent loans are delinquent and in which bucket.
    Returns the match dict (also stashed in session_state) for the Excel export."""
    _section("Recent Advances: Delinquency Status from Today's Due-Date-Missed List", margin_top="24px")
    st.caption(
        "Running Loans = loans agreed from the cohort start onward that are still running in the LCC. "
        "Delinquent = those same loans found on the uploaded list (matched by Loan No), split into "
        "buckets by the list's own Arrears/EMI (today's position). Every % is of Running Loans. "
        "Region, branch and executive come from the LCC. Sorted worst first."
    )
    uploaded = st.file_uploader("Due-date-missed list (.xlsb/.xlsx/.xls)", type=["xlsb", "xlsx", "xls"], key="rc_daily_feed_upload")
    if uploaded is None:
        return st.session_state.get("rc_daily_match")

    daily_df, err = load_daily_missed_feed(uploaded)
    if err:
        st.error(err)
        return None

    match = compute_recent_advances_daily_match(df_curr, daily_df, df_all=df_all)
    if not match:
        st.warning(
            "Could not match this list against the LCC: the list needs LOAN NO and ARREARS / EMI "
            "columns, and the LCC needs running loans from the cohort start onward."
        )
        return None

    st.session_state["rc_daily_match"] = match
    c1, c2, c3 = st.columns(3)
    c1.metric("Running loans", f"{match['cohort_count']:,}", help="Running loans from the cohort start onward, in the LCC")
    c2.metric("Delinquent", f"{match['matched_count']:,} ({match['pct_of_cohort_on_daily_list']:.1f}%)", help="Running loans found on the uploaded list")
    c3.metric("On the list", f"{match['daily_feed_count']:,}", help="All loans in the uploaded file")
    region_status, branch_status, executive_status = _render_status_by_grain(df_curr, daily_df)
    st.session_state["rc_region_status"] = region_status
    st.session_state["rc_branch_status"] = branch_status
    st.session_state["rc_executive_status"] = executive_status
    _render_match_notes(match)
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
    df_all: pd.DataFrame | None = None,
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
            Not just "what the numbers are", but why Collection%/NPA%/SMA-2% are moving,
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
        _render_daily_match(df_curr, df_all)

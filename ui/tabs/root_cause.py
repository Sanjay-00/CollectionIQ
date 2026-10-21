"""
Root Cause tab: WHY delinquency, NPA and collection are moving, region by
region and branch by branch. One question per view (VIEWS), each with a
one-line takeaway and one table in the app's shared format
(ui.components.html_table). Zero AI calls.
"""
import pandas as pd
import streamlit as st

from ui.components import _dl_btn, append_total_row, _esc, html_table, section_label, takeaway
from ui.glossary import DRIVER_HELP, help_for, info_icon
from config import HARD_BUCKET_ARREARS_EMI_MIN
from analysis.root_cause import (
    build_root_cause_workbook, load_daily_missed_feed,
    ARREARS_GROUPS, GROUP_LABELS, HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, EARLY_DELINQUENCY,
    compute_recent_advances_daily_match, story_for_export, bucket_moves_for_export,
    compute_recent_advances_bucket_movement,
    compute_recent_advances_story_by_grain,
    list_report_date, list_month_warning,
)

_STATUS_COLOR = {"Improving": "#16a34a", "Worsening": "#dc2626", "Stable": "#d97706", "-": "#9ca3af"}
_STATUS_ICON  = {"Improving": "🟢", "Worsening": "🔴", "Stable": "🟡", "-": "⚪"}

VIEWS = ["Why, by region", "Insurance vs installment", "Paying vs not paying", "Recent advances"]

# Buckets that are a risk (everything except STD) get heat shading.
_RISK_BUCKETS = ("1-30 DPD", "SMA-1", "SMA-2", "NPA")


def _badge(status: str) -> str:
    c = _STATUS_COLOR.get(status, "#9ca3af")
    i = _STATUS_ICON.get(status, "⚪")
    return (
        f'<span style="background:{c}18;color:{c};font-size:11px;font-weight:700;'
        f'padding:2px 9px;border-radius:12px;white-space:nowrap;">{i} {status}</span>'
    )


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or v == ""


_PREV_DELINQ_HELP = "Delinquency % in last month's file (loans with any EMI or charge overdue)."
_DELTA_DELINQ_HELP = "This month minus last month, in percentage points. Positive means worse."
_DELQ_COLS = ["Delinquency%", "Prev Delinquency%", "Δ Delinquency%"]
# Total row: overall Delinquency% = all delinquent / all accounts (not an average of branches).
_DELQ_RATIO = {"Delinquency%": ("Delinquent Accounts", "Accounts")}


def _summary_line(total_row: pd.Series, parts: list[tuple[str, str]]) -> None:
    """One visible line with the all-branch totals, so the table can stay collapsed."""
    bits = [f"<b>{int(total_row['Delinquent Accounts']):,}</b> delinquent loans"]
    if not _is_missing(total_row.get("Accounts")) and not _is_missing(total_row.get("Delinquency%")):
        bits = [f"<b>{int(total_row['Accounts']):,}</b> accounts",
                f"<b>{int(total_row['Delinquent Accounts']):,}</b> delinquent (<b>{float(total_row['Delinquency%']):.1f}%</b>)"]
    bits += [f"{_esc(label)} <b>{float(total_row[key]):.1f}%</b>" for label, key in parts
             if key in total_row and not _is_missing(total_row[key])]
    takeaway(f"All branches: {', '.join(bits)}.")


def _table(df: pd.DataFrame, cols: list[dict], last_row_is_total: bool = False, max_height: int | None = None) -> str:
    """The app's shared table (ui.components.html_table); each column's ⓘ
    comes from the glossary unless the spec gives one. last_row_is_total:
    the frame's last row is its Total row. max_height: scroll inside the
    table (header kept in view) instead of stretching the page."""
    cols = [{**c, "help": c.get("help") or help_for(c["key"])} for c in cols]
    if last_row_is_total and len(df):
        # append_total_row repeats a text column's own name in the Total row; show it blank.
        total = {k: ("" if isinstance(v, str) and v == k else v) for k, v in df.iloc[-1].to_dict().items()}
        return html_table(df.iloc[:-1], cols, total=total, max_height=max_height)
    return html_table(df, cols, max_height=max_height)


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


def _why_total(df_curr: pd.DataFrame | None) -> dict | None:
    """The whole view as one Total row (sum over sum), for the region table."""
    if df_curr is None or df_curr.empty:
        return None
    from utils import unit_metrics
    t = unit_metrics(df_curr, []).iloc[0]
    return {"Region": "Total", "Status": "", "Accounts": int(t["Accounts"]),
            "Delinquent Accounts": int(t["Delinquent"]), "Delinquency%": float(t["Delinquency%"]),
            "NPA%": float(t["NPA%"]), "Collection%": float(t["Collection%"])}


def _why_takeaway(why_df: pd.DataFrame) -> str:
    """How many regions got worse, and the driver behind most of them."""
    n = len(why_df)
    worse = int((why_df.get("Status") == "Worsening").sum()) if "Status" in why_df.columns else 0
    text = f"<b>{worse}</b> of {n} regions are worsening." if "Status" in why_df.columns else f"{n} regions."
    if "Dominant Driver" in why_df.columns and why_df["Dominant Driver"].notna().any():
        top = why_df["Dominant Driver"].value_counts()
        text += f" The most common driver is <b>{_esc(top.index[0])}</b> ({int(top.iloc[0])} of {n} regions)."
    return text


def _driver_cell(v, row) -> str:
    """'Deep arrears, still paying · 37%' with the driver's own explanation on hover."""
    if not isinstance(v, str) or not v:
        return "-"
    share = row.get("Driver Share %") if hasattr(row, "get") else None
    tail = "" if share is None or pd.isna(share) else \
        f' <span style="color:#6b7280;font-weight:400;">· {float(share):.0f}%</span>'
    return f"{_esc(v)}{tail}{info_icon(DRIVER_HELP.get(v))}"


def _render_why_table(why_df: pd.DataFrame, df_curr: pd.DataFrame | None = None) -> None:
    section_label("Why It's Moving, Region by Region")
    if why_df.empty:
        st.info("Not enough data to compute region diagnostics.")
        return
    takeaway(_why_takeaway(why_df))
    st.caption("Worsening regions first. Main driver: the cause behind the largest share of the region's "
               "delinquent loans (its share after the dot).")
    display_cols = ["Region", "Status", "Accounts", "Delinquent Accounts", "Delinquency%", "Prev Delinquency%",
                    "Δ Delinquency%", "NPA%", "Δ NPA%", "Collection%", "Dominant Driver", "Driver Share %"]
    view = why_df[[c for c in display_cols if c in why_df.columns]].copy()
    cols = [
        {"key": "Region", "align": "left", "bold": True},
        {"key": "Status", "align": "left", "fmt": lambda v, _r=None: _badge(str(v)) if v else ""},
        {"key": "Accounts", "fmt": "int"},
        {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct_count", "count": "Delinquent Accounts", "heat": True},
        {"key": "Prev Delinquency%", "label": "Last month", "fmt": "pct", "help": _PREV_DELINQ_HELP},
        {"key": "Δ Delinquency%", "label": "Change", "fmt": "pp", "help": _DELTA_DELINQ_HELP},
        {"key": "NPA%", "label": "NPA", "fmt": "pct", "heat": True},
        {"key": "Δ NPA%", "label": "NPA change", "fmt": "pp"},
        {"key": "Collection%", "label": "Collection", "fmt": "pct"},
        {"key": "Dominant Driver", "label": "Main driver", "align": "left", "fmt": _driver_cell},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    total = _why_total(df_curr)
    if total:
        view = pd.concat([view, pd.DataFrame([total])], ignore_index=True)
    st.markdown(_table(view, cols, last_row_is_total=bool(total)), unsafe_allow_html=True)
    _dl_btn(why_df, "region_why_diagnosis.xlsx", "dl_root_cause_why")


def _render_insurance_split(ins_df: pd.DataFrame) -> None:
    section_label("Insurance or Instalment: What's Unpaid, by Branch")
    st.caption("Shares of each branch's delinquent loans. Insurance-only: the EMI is paid, only an insurance "
               "or expense charge is due (a cash or WCL adjustment clears it). Highest insurance-only first.")
    if ins_df.empty:
        st.info("Not enough data to compute the insurance split.")
        return
    show_cols = ["Unit", "RegionName", "Accounts", "Delinquent Accounts", *_DELQ_COLS, "Insurance-Only", "Insurance-Only %",
                 "Installment-Only", "Installment-Only %", "Both", "Both %", "Other", "Other %"]
    show_cols = [c for c in show_cols if c in ins_df.columns]
    ratio_cols = {
        **_DELQ_RATIO,
        "Insurance-Only %": ("Insurance-Only", "Delinquent Accounts"),
        "Installment-Only %": ("Installment-Only", "Delinquent Accounts"),
        "Both %": ("Both", "Delinquent Accounts"),
        "Other %": ("Other", "Delinquent Accounts"),
    }
    view = append_total_row(ins_df[show_cols], ratio_cols=ratio_cols)
    cols = [
        {"key": "Unit", "align": "left", "bold": True},
        {"key": "RegionName", "align": "left", "label": "Region"},
        {"key": "Accounts", "fmt": "int"},
        {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct_count", "count": "Delinquent Accounts", "heat": True},
        {"key": "Prev Delinquency%", "fmt": "pct", "help": _PREV_DELINQ_HELP},
        {"key": "Δ Delinquency%", "fmt": "pp", "help": _DELTA_DELINQ_HELP},
        # Shares of the DELINQUENT loans (each with its count).
        {"key": "Insurance-Only %", "label": GROUP_LABELS["Insurance-Only"], "fmt": "pct_count", "count": "Insurance-Only"},
        {"key": "Installment-Only %", "label": GROUP_LABELS["Installment-Only"], "fmt": "pct_count", "count": "Installment-Only"},
        # "Both" = genuine shortfall on both legs -- the most severe of the three, shaded accordingly.
        {"key": "Both %", "label": GROUP_LABELS["Both"], "fmt": "pct_count", "count": "Both", "heat": True},
        {"key": "Other %", "label": GROUP_LABELS["Other"], "fmt": "pct_count", "count": "Other"},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    _summary_line(view.iloc[-1], [(GROUP_LABELS[g], f"{g} %") for g in ("Insurance-Only", "Installment-Only", "Both", "Other")])
    st.markdown(_table(view, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(ins_df, "insurance_vs_installment_split.xlsx", "dl_root_cause_insurance")


def _render_chronic_shock_split(cs_df: pd.DataFrame) -> None:
    section_label("Paying or Not: Delinquent Loans in Four Groups, by Branch")
    st.caption(f"By how far behind the loan is ({HARD_BUCKET_ARREARS_EMI_MIN}+ EMIs = hard) and whether anything "
               "was paid in the last 3 months. Hover the ⓘ on a column for what to do with that group.")
    if cs_df.empty:
        st.info("Not enough data to compute the paying vs not paying split.")
        return
    show_cols = ["Unit", "RegionName", "Accounts", "Delinquent Accounts", *_DELQ_COLS] + [x for g in ARREARS_GROUPS for x in (g, f"{g} %")]
    show_cols = [c for c in show_cols if c in cs_df.columns]
    ratio_cols = {**_DELQ_RATIO, **{f"{g} %": (g, "Delinquent Accounts") for g in ARREARS_GROUPS}}
    view = append_total_row(cs_df[show_cols], ratio_cols=ratio_cols)
    cols = [
        {"key": "Unit", "align": "left", "bold": True},
        {"key": "RegionName", "align": "left", "label": "Region"},
        {"key": "Accounts", "fmt": "int"},
        {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct_count", "count": "Delinquent Accounts", "heat": True},
        {"key": "Prev Delinquency%", "fmt": "pct", "help": _PREV_DELINQ_HELP},
        {"key": "Δ Delinquency%", "fmt": "pp", "help": _DELTA_DELINQ_HELP},
        # Shares of the DELINQUENT loans (each with its count).
        {"key": f"{HARD_NOT_PAYING} %", "label": GROUP_LABELS[HARD_NOT_PAYING], "fmt": "pct_count", "count": HARD_NOT_PAYING, "heat": True},
        {"key": f"{HARD_STILL_PAYING} %", "label": GROUP_LABELS[HARD_STILL_PAYING], "fmt": "pct_count", "count": HARD_STILL_PAYING, "heat": True},
        {"key": f"{WAS_SILENT} %", "label": GROUP_LABELS[WAS_SILENT], "fmt": "pct_count", "count": WAS_SILENT},
        {"key": f"{EARLY_DELINQUENCY} %", "label": GROUP_LABELS[EARLY_DELINQUENCY], "fmt": "pct_count", "count": EARLY_DELINQUENCY},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    _summary_line(view.iloc[-1], [(GROUP_LABELS[g], f"{g} %") for g in ARREARS_GROUPS])
    st.markdown(_table(view, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(cs_df, "deep_arrears_paying_vs_not_paying.xlsx", "dl_root_cause_chronic_shock")


# ── Recent advances: one story, month end vs a due-date list ─────────────────
# Read top to bottom, the page answers a manager's questions in order: how
# many recent loans are behind then and now (cards), where (one table, one
# "View by" control), and the detail behind it (folded away). Every figure
# comes from analysis.root_cause; this module only lays it out.

_GRAIN_LABELS = {"zone": "Zone", "region": "Region", "branch": "Branch", "executive": "Executive"}
_PROFILE_GROUP_COL = {"zone": "Zone", "region": "RegionName", "branch": "Unit"}
_PROFILE_BUCKETS = ("STD",) + _RISK_BUCKETS


def _card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div style="border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;height:100%;">'
        f'<div style="font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;">{_esc(label)}</div>'
        f'<div style="font-size:22px;font-weight:800;color:#111827;margin-top:2px;">{_esc(value)}</div>'
        f'<div style="font-size:12px;color:#4b5563;margin-top:2px;">{_esc(sub)}</div>'
        f'</div>'
    )


def _cards(cards: list[tuple[str, str, str]]) -> None:
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        with col:
            st.markdown(_card(label, value, sub), unsafe_allow_html=True)


def _pct(v) -> str:
    return "-" if _is_missing(v) else f"{float(v):.1f}%"


def _signed(v, _row=None, unit: str = "", decimals: int = 0) -> str:
    """A change as a plain signed figure: no colour or arrow, so no verdict."""
    if _is_missing(v):
        return '<span style="color:#9ca3af;">-</span>'
    v = float(v)
    return "0" if v == 0 else f"{v:+,.{decimals}f}{unit}"


def _render_recent_advances_summary(summary: dict) -> None:
    if not summary:
        st.info("Not enough data for recent loans: the month-end file needs an Ag_Date column.")
        return
    _cards([
        ("All loans", f"{summary['portfolio_count']:,}", f"₹{summary['portfolio_soh_cr']:,.2f} Cr SOH"),
        ("Recent loans", f"{summary['cohort_count']:,}", f"{summary['cohort_count_pct']:.1f}% of all loans"),
        ("Recent loans SOH", f"₹{summary['cohort_soh_cr']:,.2f} Cr", f"{summary['cohort_soh_pct']:.1f}% of all SOH"),
    ])


def _render_profile(recent_summary: dict, bucket_df: pd.DataFrame, by_group: dict, grain: str) -> None:
    """Month-end only: the recent loans by bucket, overall and for one grain
    (zone/region/branch; executive falls back to branch)."""
    _render_recent_advances_summary(recent_summary)
    if bucket_df.empty:
        st.info("Not enough data for the bucket split.")
        return
    st.caption("Bucket shares are of recent loans that have had an EMI due; new loans with no EMI yet are left out.")
    st.markdown(_table(bucket_df, [
        {"key": "Bucket", "align": "left", "bold": True},
        {"key": "Count %", "label": "Loans", "fmt": "pct_count", "count": "Count"},
        {"key": "SOH %", "label": "SOH", "fmt": "pct_cr", "amount": "SOH (Cr)"},
    ]), unsafe_allow_html=True)
    _dl_btn(bucket_df, "recent_loans_bucket_split.xlsx", "dl_root_cause_recent_bucket")

    if grain not in by_group:
        st.caption("The month-end profile is not split by executive; showing branches.")
        grain = "branch"
    count_df, soh_df = by_group.get(grain, (pd.DataFrame(), pd.DataFrame()))
    if count_df.empty:
        return
    label, key = _GRAIN_LABELS[grain], _PROFILE_GROUP_COL[grain]
    head = [{"key": key, "align": "left", "bold": True, "label": label}]
    if "RegionName" in count_df.columns and key != "RegionName":
        head.append({"key": "RegionName", "align": "left", "label": "Region"})
    tab_count, tab_soh = st.tabs(["By loan count", "By SOH"], key="rc_ra_profile_measure")
    with tab_count:
        cols = head + [{"key": f"{b} %", "label": b, "fmt": "pct_count", "count": b, "heat": b in _RISK_BUCKETS}
                       for b in _PROFILE_BUCKETS if f"{b} %" in count_df.columns]
        cols.append({"key": "Total Count", "label": "Loans", "fmt": "int", "bold": True})
        ratio = {f"{b} %": (b, "Total Count") for b in _PROFILE_BUCKETS if b in count_df.columns}
        st.markdown(_table(append_total_row(count_df, ratio_cols=ratio), cols, last_row_is_total=True,
                           max_height=480), unsafe_allow_html=True)
        _dl_btn(count_df, f"recent_loans_by_{label.lower()}_count.xlsx", "dl_root_cause_recent_group_count")
    with tab_soh:
        cols = head + [{"key": f"{b} %", "label": b, "heat": b in _RISK_BUCKETS, "fmt": "pct_cr", "amount": f"{b} (Cr)"}
                       for b in _PROFILE_BUCKETS if f"{b} %" in soh_df.columns]
        cols.append({"key": "Total SOH (Cr)", "label": "SOH", "fmt": "cr", "bold": True})
        ratio = {f"{b} %": (f"{b} (Cr)", "Total SOH (Cr)") for b in _PROFILE_BUCKETS if f"{b} (Cr)" in soh_df.columns}
        st.markdown(_table(append_total_row(soh_df, ratio_cols=ratio), cols, last_row_is_total=True,
                           max_height=480), unsafe_allow_html=True)
        _dl_btn(soh_df, f"recent_loans_by_{label.lower()}_soh.xlsx", "dl_root_cause_recent_group_soh")


def _list_bundle(df_curr: pd.DataFrame, df_all: pd.DataFrame, uploaded, curr_month: str) -> dict:
    """Read the list and compute every Recent advances figure once per (list
    file, filtered loans, month): a click on "View by" or the bucket toggle
    must not recompute ~94k loans. Kept in session_state, newest only."""
    loans = pd.util.hash_pandas_object(df_curr["Loan No"], index=False).sum() if "Loan No" in df_curr.columns else 0
    key = (getattr(uploaded, "file_id", None) or getattr(uploaded, "name", ""), int(loans), len(df_curr), curr_month)
    cached = st.session_state.get("rc_ra_bundle")
    if cached and cached["key"] == key:
        return cached
    daily_df, err = load_daily_missed_feed(uploaded)
    if err:
        return {"key": key, "error": err}
    match = compute_recent_advances_daily_match(df_curr, daily_df, df_all=df_all)
    if not match:
        return {"key": key, "error": "Could not match this list with the month-end file: the list needs LOAN NO and "
                                     "ARREARS / EMI columns, and the month-end file needs running loans from "
                                     "the recent-loans start date onward."}
    list_date = list_report_date(daily_df)
    story = {g: compute_recent_advances_story_by_grain(df_curr, daily_df, grain=g, list_date=list_date)
             for g in _GRAIN_LABELS}
    bundle = {
        "key": key, "error": None, "match": match, "list_date": list_date, "story": story,
        "buckets": compute_recent_advances_bucket_movement(df_curr, daily_df, list_date=list_date),
        "warnings": [w for w in (
            list_month_warning(curr_month, list_date),
            None if "Due Dt" in df_curr.columns else
            "The month-end file has no Due Dt column, so every loan clean at month end counts as due by the list date.",
        ) if w],
    }
    st.session_state["rc_ra_bundle"] = bundle
    return bundle


def _story_cards(total: pd.Series, close_lbl: str, list_lbl: str) -> None:
    run = int(total["Running Loans"])
    newly_sub = f"{int(total['New Slippage']):,} of {int(total['Due by List Date']):,} loans with an EMI due"
    if "First EMI Missed" in total and not _is_missing(total["First EMI Missed"]):
        newly_sub += f", plus {int(total['First EMI Missed']):,} missed their first EMI"
    _cards([
        (f"Behind at {close_lbl}", _pct(total["Close Delinquent %"]), f"{int(total['Close Delinquent']):,} of {run:,} loans"),
        (f"Behind on {list_lbl}", _pct(total["List Delinquent %"]),
         f"{int(total['List Delinquent']):,} loans, {_signed(total['Change in Delinquent % (pts)'], unit=' % points', decimals=2)}"),
        ("Month-end cases cleared", _pct(total["Cleared %"]),
         f"{int(total['Cleared']):,} of {int(total['Pending at Close']):,}, {int(total['Still Pending']):,} still behind"),
        ("Newly behind", _pct(total["New Slippage %"]), newly_sub),
    ])


def _story_table(df: pd.DataFrame, close_lbl: str, list_lbl: str, show_buckets: bool) -> None:
    label_cols = [c for c in df.columns if c in _GRAIN_LABELS.values()]
    cols = [{"key": lc, "align": "left", "bold": lc == label_cols[0]} for lc in label_cols]
    cols.append({"key": "Running Loans", "label": "Loans", "fmt": "int", "bold": True,
                 "help": "Recent loans still running in the month-end file. Every % below is of these loans, "
                         "at both dates, so a change is a change in the same loans."})

    def _side(side: str, lbl: str) -> list[dict]:
        when = f"at {lbl}" if side == "Close" else f"on {lbl}"
        out = [{"key": f"{side} Delinquent %", "label": f"Behind {when}", "fmt": "pct_count",
                "count": f"{side} Delinquent", "heat": True,
                "help": f"Loans behind on EMI (delinquent) {when}: % of Loans, with the count."}]
        if show_buckets:
            out += [{"key": f"{side} {b} %", "label": f"{b} ({lbl})", "fmt": "pct_count", "count": f"{side} {b}"}
                    for b in _RISK_BUCKETS if f"{side} {b} %" in df.columns]
        return out

    cols += _side("Close", close_lbl) + _side("List", list_lbl)
    if show_buckets and "List Other %" in df.columns:
        cols.append({"key": "List Other %", "label": f"Other ({list_lbl})", "fmt": "pct_count", "count": "List Other",
                     "help": "On the list with no EMI overdue (usually only a charge)."})
    cols.append({"key": "Change in Delinquent % (pts)", "label": "Change (% points)",
                 "fmt": lambda v, row: _signed(v, decimals=2),
                 "help": f"Behind on {list_lbl} minus behind at {close_lbl}, in percentage points."})
    if show_buckets:
        cols.append({"key": "Change in Delinquent", "label": "Change (loans)", "fmt": _signed,
                     "help": f"Loans behind on {list_lbl} minus loans behind at {close_lbl}."})
    cols += [
        {"key": "Cleared %", "label": "Month-end cases cleared", "fmt": "pct_count", "count": "Cleared",
         "help": f"Behind at {close_lbl} and not on the list now: % of the loans behind at month end."},
        {"key": "New Slippage %", "label": "Newly behind", "fmt": "pct_count", "count": "New Slippage", "heat": True,
         "help": f"Not behind at {close_lbl}, EMI due by {list_lbl}, and now on the list: "
                 "% of those loans with an EMI due."},
    ]
    if "First EMI Missed" in df.columns:
        cols.append({"key": "First EMI Missed", "label": "Missed first EMI", "fmt": "int",
                     "help": "New loans with no EMI yet at month end that are now on the list."})
    if "On List, Not Yet Due" in df.columns:
        cols.append({"key": "On List, Not Yet Due", "label": "On list before due date", "fmt": "int",
                     "help": "Not behind at month end and on the list, though the EMI date has not come yet "
                             "(often an overdue charge). Not in Newly behind."})
    st.markdown(_table(df, cols, last_row_is_total=True, max_height=560), unsafe_allow_html=True)


def _render_bucket_moves(moves: pd.DataFrame) -> None:
    if moves.empty:
        st.info("Could not build the bucket movement: the month-end file needs an Arrears / EMI column.")
        return
    st.caption("Each row is where loans stood at month end; the columns show where the same loans stand on the "
               "list date. Better, Same and Worse compare the two.")
    view = moves.replace({"Bucket at Close": {"No EMI Yet": "New, no EMI yet"}})
    cols = [{"key": "Bucket at Close", "label": "At month end", "align": "left", "bold": True}]
    cols += [{"key": c, "label": "Not behind now" if c == "Not on List" else c, "fmt": "int"}
             for c in view.columns if c not in ("Bucket at Close", "Better", "Same", "Worse", "Total")]
    cols += [{"key": "Better", "fmt": "int"}, {"key": "Same", "fmt": "int"}, {"key": "Worse", "fmt": "int"},
             {"key": "Total", "label": "Loans", "fmt": "int", "bold": True}]
    st.markdown(_table(view, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(bucket_moves_for_export(moves), "recent_loans_bucket_movement.xlsx", "dl_root_cause_bucket_movement")


def _render_about(match: dict, close_lbl: str, list_lbl: str) -> None:
    """Definitions and every list loan the tables do not count, in plain words
    (no nested expanders: this already sits inside one)."""
    st.markdown(
        "- **Loans**: recent loans (agreed from the start date) still running in the month-end file.\n"
        f"- **Behind**: at least part of an EMI overdue (delinquent). At {close_lbl} from the month-end file, "
        f"on {list_lbl} from the list.\n"
        "- **Month-end cases cleared**: behind at month end, not on the list now.\n"
        "- **Newly behind**: not behind at month end, EMI due by the list date, now on the list.\n"
        "- **Missed first EMI**: new loans with no EMI yet at month end that are now on the list.\n"
        "- The list date can fall mid-month, before customers who missed an EMI have paid, "
        "so figures on it run higher than at a month end."
    )
    st.markdown(f"**Matching the list:** {match['daily_feed_count']:,} loans on the list, "
                f"{match['matched_count']:,} found among the recent loans and counted.")
    notes = []
    if match["exec_changed_count"]:
        notes.append(f"{match['exec_changed_count']:,} counted loans show a different executive on the list. "
                     "They stay under their month-end executive, so each executive's % compares the same loans.")
    if match["hidden_by_filter_count"]:
        notes.append(f"{match['hidden_by_filter_count']:,} list loans are hidden by the sidebar filters.")
    if match["other_scope_count"]:
        notes.append(f"{match['other_scope_count']:,} list loans are from regions or zones not in the month-end file.")
    if match["in_view_not_counted_count"]:
        notes.append(f"{match['in_view_not_counted_count']:,} list loans are in the month-end file but not counted "
                     "as running recent loans there (an earlier agreement date, or a status other than RUN).")
    if match["missing_from_lcc_count"]:
        by_branch = ", ".join(f"{_esc(b)}: {n}" for b, n in match["missing_from_lcc_by_branch"].items())
        notes.append(f"{match['missing_from_lcc_count']:,} list loans are from regions the month-end file covers, "
                     f"but are missing from it, usually a branch missing from the extract. By branch: {by_branch}.")
    if notes:
        st.markdown("\n".join(f"- {n}" for n in notes))
    loan_lists = [("Missing from the month-end file", match["missing_from_lcc_loan_nos"]),
                  ("In the month-end file but not counted", match["in_view_not_counted_loan_nos"])]
    for title, loans in loan_lists:
        if loans:
            st.caption(f"{title}: {len(loans):,} loan numbers")
            st.markdown(_table(pd.DataFrame({"Loan No": loans}), [{"key": "Loan No", "align": "left"}],
                               max_height=240), unsafe_allow_html=True)


def _render_recent_advances(
    df_curr: pd.DataFrame | None, df_all: pd.DataFrame | None, curr_month: str,
    recent_summary: dict, bucket_df: pd.DataFrame, by_group: dict,
) -> None:
    start = recent_summary.get("cohort_start")
    start_txt = f"{pd.Timestamp(start):%d %b %Y}" if start else "the start date"
    close_lbl = f"{pd.Period(curr_month, freq='M').strftime('%b')} month end" if curr_month else "month end"

    uploaded = None
    if df_curr is not None:
        uploaded = st.file_uploader(
            "Compare with a due-date-missed list (.xlsb, .xlsx, .xls)", type=["xlsb", "xlsx", "xls"],
            key="rc_daily_feed_upload",
            help="The daily list of loans that missed a due date. It is matched to the month-end file by loan number.",
        )
    if uploaded is None:
        for k in ("rc_ra_bundle", "rc_daily_match", "rc_movement"):
            st.session_state.pop(k, None)
        if df_curr is not None:
            st.caption("Upload a due-date list to see how the same recent loans moved since month end, "
                       "by zone, region, branch and executive.")
        section_label(f"Recent loans at {close_lbl} (agreed from {start_txt})", margin_top="16px")
        grains = [g for g in ("zone", "region", "branch") if g in by_group]
        grain = st.segmented_control("View by", [_GRAIN_LABELS[g] for g in grains], default="Region",
                                     key="rc_ra_profile_grain") or "Region"
        _render_profile(recent_summary, bucket_df, by_group, grain.lower())
        return

    with st.spinner("Matching the list with the month-end file..."):
        bundle = _list_bundle(df_curr, df_all if df_all is not None else df_curr, uploaded, curr_month)
    if bundle["error"]:
        st.error(bundle["error"])
        return
    list_date = bundle["list_date"]
    list_lbl = f"{list_date:%d %b}" if list_date is not None else "the list date"
    story = bundle["story"]
    st.session_state["rc_daily_match"] = bundle["match"]
    st.session_state["rc_movement"] = {**{g: story[g] for g in ("region", "branch", "executive")},
                                       "buckets": bundle["buckets"]}

    section_label(f"Recent loans: {close_lbl} vs {list_lbl}", margin_top="16px")
    region = story["region"]
    loans_txt = f"{int(region.iloc[-1]['Running Loans']):,} " if not region.empty else ""
    st.caption(f"The same {loans_txt}loans agreed from {start_txt}, at both dates. Mid-month figures run higher "
               "than month end, because some customers pay later in the month.")
    for w in bundle["warnings"]:
        st.warning(w)
    if region.empty:
        st.warning("Could not build the comparison: the month-end file needs an Arrears / EMI column.")
        return
    _story_cards(region.iloc[-1], close_lbl, list_lbl)

    grains = [g for g in _GRAIN_LABELS if not story[g].empty]
    c_view, c_toggle, c_dl = st.columns([3, 1.4, 1], vertical_alignment="bottom")
    with c_view:
        picked = st.segmented_control("View by", [_GRAIN_LABELS[g] for g in grains], default="Region",
                                      key="rc_ra_grain") or "Region"
    with c_toggle:
        show_buckets = st.toggle("Show bucket split", key="rc_ra_buckets")
    grain = picked.lower()
    with c_dl:
        _dl_btn(story_for_export(story[grain]), f"recent_loans_month_end_vs_{list_lbl.replace(' ', '_')}_by_{grain}.xlsx",
                "dl_root_cause_story")
    _story_table(story[grain], close_lbl, list_lbl, show_buckets)

    with st.expander(f"How loans moved between buckets, {close_lbl} to {list_lbl}"):
        _render_bucket_moves(bundle["buckets"])
    with st.expander(f"Month-end profile ({close_lbl}): loans and SOH by bucket"):
        _render_profile(recent_summary, bucket_df, by_group, grain)
    with st.expander("About this data"):
        _render_about(bundle["match"], close_lbl, list_lbl)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_workbook_bytes(
    why_df: pd.DataFrame, insurance_split_df: pd.DataFrame,
    chronic_shock_df: pd.DataFrame, contamination: dict, generated_for: str,
    recent_summary: dict, recent_bucket_df: pd.DataFrame,
    recent_by_region_count: pd.DataFrame, recent_by_region_soh: pd.DataFrame,
    recent_by_branch_count: pd.DataFrame, recent_by_branch_soh: pd.DataFrame,
    daily_match: dict, region_status: pd.DataFrame,
    branch_status: pd.DataFrame, executive_status: pd.DataFrame, movement: dict,
) -> bytes:
    return build_root_cause_workbook(
        why_df, insurance_split_df, chronic_shock_df, contamination, generated_for,
        recent_summary=recent_summary, recent_bucket_df=recent_bucket_df,
        recent_by_region_count=recent_by_region_count, recent_by_region_soh=recent_by_region_soh,
        recent_by_branch_count=recent_by_branch_count, recent_by_branch_soh=recent_by_branch_soh,
        daily_match=daily_match, region_status=region_status,
        branch_status=branch_status, executive_status=executive_status, movement=movement,
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
    recent_by_zone_count: pd.DataFrame | None = None,
    recent_by_zone_soh: pd.DataFrame | None = None,
) -> None:
    recent_bucket_df = recent_bucket_df if recent_bucket_df is not None else pd.DataFrame()
    recent_by_region_count = recent_by_region_count if recent_by_region_count is not None else pd.DataFrame()
    recent_by_region_soh = recent_by_region_soh if recent_by_region_soh is not None else pd.DataFrame()
    recent_by_branch_count = recent_by_branch_count if recent_by_branch_count is not None else pd.DataFrame()
    recent_by_branch_soh = recent_by_branch_soh if recent_by_branch_soh is not None else pd.DataFrame()

    header_col, dl_col = st.columns([4, 1], vertical_alignment="center")
    with header_col:
        st.markdown('<div style="font-size:13.5px;color:#374151;">Why delinquency, NPA and collection are '
                    'moving, region by region and branch by branch, so the fix can follow.</div>',
                    unsafe_allow_html=True)
    _render_contamination_notice(contamination)
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="rc_view", label_visibility="collapsed") or VIEWS[0]
    if view == VIEWS[0]:
        _render_why_table(why_df, df_curr)
    elif view == VIEWS[1]:
        _render_insurance_split(insurance_split_df)
    elif view == VIEWS[2]:
        _render_chronic_shock_split(chronic_shock_df)
    else:
        by_group = {g: pair for g, pair in (
            ("zone", (recent_by_zone_count, recent_by_zone_soh)),
            ("region", (recent_by_region_count, recent_by_region_soh)),
            ("branch", (recent_by_branch_count, recent_by_branch_soh)),
        ) if pair[0] is not None and not pair[0].empty}
        _render_recent_advances(df_curr, df_all, curr_month, recent_summary or {}, recent_bucket_df, by_group)

    # The workbook button sits in the header but is built LAST: the Recent
    # advances view computes the list figures above, and a button built first
    # would hand out a workbook without them (it did, on the first download
    # after an upload).
    with dl_col:
        _daily_match = st.session_state.get("rc_daily_match") or {}
        _movement = st.session_state.get("rc_movement") or {}
        if not why_df.empty:
            st.download_button(
                "⬇ Root Cause workbook",
                data=lambda: _cached_workbook_bytes(
                    why_df, insurance_split_df, chronic_shock_df, contamination, curr_month,
                    recent_summary or {}, recent_bucket_df,
                    recent_by_region_count, recent_by_region_soh,
                    recent_by_branch_count, recent_by_branch_soh,
                    _daily_match, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), _movement,
                ),
                file_name=f"root_cause_diagnostics_{curr_month or 'report'}.xlsx".replace(" ", "_"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_root_cause_workbook",
                width='stretch',
            )

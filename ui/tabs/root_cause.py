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
    ARREARS_GROUPS, HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, EARLY_DELINQUENCY,
    compute_recent_advances_daily_match, compute_recent_advances_status_by_grain,
)

_STATUS_COLOR = {"Improving": "#16a34a", "Worsening": "#dc2626", "Stable": "#d97706", "-": "#9ca3af"}
_STATUS_ICON  = {"Improving": "🟢", "Worsening": "🔴", "Stable": "🟡", "-": "⚪"}

VIEWS = ["Why, by region", "Insurance vs installment", "Paying vs not paying", "Recent advances"]

# Buckets that are a risk (everything except STD) get heat shading.
_RISK_BUCKETS = ("0-1", "1-2", "2-3", "NPA")


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


def _table(df: pd.DataFrame, cols: list[dict], last_row_is_total: bool = False) -> str:
    """The app's shared table (ui.components.html_table); each column's ⓘ
    comes from the glossary unless the spec gives one. last_row_is_total:
    the frame's last row is its Total row."""
    cols = [{**c, "help": c.get("help") or help_for(c["key"])} for c in cols]
    if last_row_is_total and len(df):
        # append_total_row repeats a text column's own name in the Total row; show it blank.
        total = {k: ("" if isinstance(v, str) and v == k else v) for k, v in df.iloc[-1].to_dict().items()}
        return html_table(df.iloc[:-1], cols, total=total)
    return html_table(df, cols)


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
        {"key": "Insurance-Only %", "label": "Insurance-Only", "fmt": "pct_count", "count": "Insurance-Only"},
        {"key": "Installment-Only %", "label": "Installment-Only", "fmt": "pct_count", "count": "Installment-Only"},
        # "Both" = genuine shortfall on both legs -- the most severe of the three, shaded accordingly.
        {"key": "Both %", "label": "Both", "fmt": "pct_count", "count": "Both", "heat": True},
        {"key": "Other %", "label": "Other", "fmt": "pct_count", "count": "Other"},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    _summary_line(view.iloc[-1], [("Insurance-Only", "Insurance-Only %"), ("Installment-Only", "Installment-Only %"), ("Both", "Both %"), ("Other", "Other %")])
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
        {"key": f"{HARD_NOT_PAYING} %", "label": HARD_NOT_PAYING, "fmt": "pct_count", "count": HARD_NOT_PAYING, "heat": True},
        {"key": f"{HARD_STILL_PAYING} %", "label": HARD_STILL_PAYING, "fmt": "pct_count", "count": HARD_STILL_PAYING, "heat": True},
        {"key": f"{WAS_SILENT} %", "label": WAS_SILENT, "fmt": "pct_count", "count": WAS_SILENT},
        {"key": f"{EARLY_DELINQUENCY} %", "label": EARLY_DELINQUENCY, "fmt": "pct_count", "count": EARLY_DELINQUENCY},
    ]
    cols = [c for c in cols if c["key"] in view.columns]
    _summary_line(view.iloc[-1], [(g, f"{g} %") for g in ARREARS_GROUPS])
    st.markdown(_table(view, cols, last_row_is_total=True), unsafe_allow_html=True)
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
        "Bucket-wise split of the cohort itself. % is of the cohort's own graded total."
    )
    cols = [
        {"key": "Bucket", "align": "left", "bold": True},
        {"key": "Count %", "label": "Loans", "fmt": "pct_count", "count": "Count"},
        {"key": "SOH %", "label": "SOH", "fmt": "pct_cr", "amount": "SOH (Cr)"},
    ]
    cols = [c for c in cols if c["key"] in bucket_df.columns]
    st.markdown(_table(bucket_df, cols), unsafe_allow_html=True)
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
                if f"{bucket} %" in count_df.columns:
                    cols.append({"key": f"{bucket} %", "label": bucket, "fmt": "pct_count", "count": bucket,
                                 "heat": bucket in _RISK_BUCKETS})
            if "Total Count" in count_df.columns:
                cols.append({"key": "Total Count", "fmt": "int", "bold": True})
            ratio = {f"{b} %": (b, "Total Count") for b in ("STD", "0-1", "1-2", "2-3", "NPA") if b in count_df.columns}
            st.markdown(_table(append_total_row(count_df, ratio_cols=ratio), cols, last_row_is_total=True),
                        unsafe_allow_html=True)
            _dl_btn(count_df, f"recent_advances_by_{label.lower()}_count.xlsx", f"dl_root_cause_recent_{label.lower()}_count")
    with tab_soh:
        if soh_df.empty:
            st.info("No data.")
        else:
            cols = [{"key": group_key, "align": "left", "bold": True, "label": label}]
            if "RegionName" in soh_df.columns and group_key != "RegionName":
                cols.append({"key": "RegionName", "align": "left", "label": "Region"})
            for bucket in ("STD", "0-1", "1-2", "2-3", "NPA"):
                if f"{bucket} %" in soh_df.columns:   # "34.6% (₹1.21 Cr)": the money in brackets
                    amount = f"{bucket} (Cr)" if f"{bucket} (Cr)" in soh_df.columns else None
                    cols.append({"key": f"{bucket} %", "label": bucket, "heat": bucket in _RISK_BUCKETS,
                                 "fmt": "pct_cr" if amount else "pct", "amount": amount})
            if "Total SOH (Cr)" in soh_df.columns:
                cols.append({"key": "Total SOH (Cr)", "fmt": "cr", "bold": True})
            ratio = {f"{b} %": (f"{b} (Cr)", "Total SOH (Cr)") for b in ("STD", "0-1", "1-2", "2-3", "NPA")
                     if f"{b} (Cr)" in soh_df.columns}
            st.markdown(_table(append_total_row(soh_df, ratio_cols=ratio), cols, last_row_is_total=True),
                        unsafe_allow_html=True)
            _dl_btn(soh_df, f"recent_advances_by_{label.lower()}_soh.xlsx", f"dl_root_cause_recent_{label.lower()}_soh")


def _render_status_table(status_df: pd.DataFrame, grain_label: str) -> None:
    if status_df.empty:
        st.warning(
            f"Could not build the {grain_label.lower()} table: the list needs LOAN NO and "
            "ARREARS / EMI columns, and the LCC needs running loans from the cohort start onward."
        )
        return
    label_cols = [c for c in status_df.columns if c in ("Zone", "Region", "Branch", "Executive")]
    count_cols = [c for c in status_df.columns if c not in label_cols and c != "Running Loans" and not c.endswith(" %")]
    cols = [{"key": lc, "align": "left", "bold": lc == label_cols[0]} for lc in label_cols]
    cols.append({"key": "Running Loans", "fmt": "int", "bold": True})
    for c in count_cols:
        # Every status column is a risk: its share of Running Loans (count), shaded.
        cols.append({"key": f"{c} %", "label": c, "fmt": "pct_count", "count": c, "heat": True})
    st.markdown(_table(status_df, cols, last_row_is_total=True), unsafe_allow_html=True)
    _dl_btn(status_df, f"recent_advances_delinquency_by_{grain_label.lower()}.xlsx", f"dl_root_cause_{grain_label.lower()}_status")


def _render_status_by_grain(df_curr: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zone_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="zone")
    region_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="region")
    branch_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="branch")
    executive_status = compute_recent_advances_status_by_grain(df_curr, daily_df, grain="executive")

    if not zone_status.empty:
        tab_z, tab_r, tab_b, tab_e = st.tabs(["By Zone", "By Region", "By Branch", "By Executive"])
        with tab_z:
            _render_status_table(zone_status, "Zone")
    else:
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
                _table(pd.DataFrame({"Loan No": match["missing_from_lcc_loan_nos"]}), [{"key": "Loan No", "align": "left"}]),
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
                _table(pd.DataFrame({"Loan No": match["in_view_not_counted_loan_nos"]}), [{"key": "Loan No", "align": "left"}]),
                unsafe_allow_html=True,
            )


def _render_daily_match(df_curr: pd.DataFrame, df_all: pd.DataFrame | None = None) -> dict | None:
    """Optional: upload today's due-date-missed list to see, per region/branch/
    executive, how many running recent loans are delinquent and in which bucket.
    Returns the match dict (also stashed in session_state) for the Excel export."""
    section_label("Recent Advances: Delinquency Status from Today's Due-Date-Missed List", margin_top="24px")
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
    with dl_col:
        # Read now (page render), used when the button is clicked.
        _daily_match = st.session_state.get("rc_daily_match") or {}
        _region_status = st.session_state.get("rc_region_status", pd.DataFrame())
        _branch_status = st.session_state.get("rc_branch_status", pd.DataFrame())
        _executive_status = st.session_state.get("rc_executive_status", pd.DataFrame())
        if not why_df.empty:
            st.download_button(
                "⬇ Root Cause workbook",
                data=lambda: _cached_workbook_bytes(
                    why_df, insurance_split_df, chronic_shock_df, contamination, curr_month,
                    recent_summary or {}, recent_bucket_df,
                    recent_by_region_count, recent_by_region_soh,
                    recent_by_branch_count, recent_by_branch_soh,
                    _daily_match, _region_status, _branch_status, _executive_status,
                ),
                file_name=f"root_cause_diagnostics_{curr_month or 'report'}.xlsx".replace(" ", "_"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_root_cause_workbook",
                width='stretch',
            )

    _render_contamination_notice(contamination)
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="rc_view", label_visibility="collapsed") or VIEWS[0]
    if view == VIEWS[0]:
        _render_why_table(why_df, df_curr)
    elif view == VIEWS[1]:
        _render_insurance_split(insurance_split_df)
    elif view == VIEWS[2]:
        _render_chronic_shock_split(chronic_shock_df)
    else:
        section_label("Recent Advances Cohort")
        st.caption(f"Loans agreed from {(recent_summary or {}).get('cohort_start', 'the configured cutoff')} onward.")
        _render_recent_advances_summary(recent_summary or {})
        _render_recent_advances_bucket(recent_bucket_df)
        if recent_by_zone_count is not None and not recent_by_zone_count.empty:
            _render_recent_advances_by_group(recent_by_zone_count, recent_by_zone_soh, "Zone", group_col="Zone")
        _render_recent_advances_by_group(recent_by_region_count, recent_by_region_soh, "Region", group_col="RegionName")
        _render_recent_advances_by_group(recent_by_branch_count, recent_by_branch_soh, "Branch", group_col="Unit")
        if df_curr is not None:
            _render_daily_match(df_curr, df_all)

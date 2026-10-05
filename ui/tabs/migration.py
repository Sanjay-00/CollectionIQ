import pandas as pd
import streamlit as st

from ui.components import (
    _static_kpi_card_html, _chart_card, _empty_state, heat_range, heat_style, _dl_btn, _safe_df, _esc,
)
from ui.glossary import help_for, info_icon
from config import ROLL_STEPS


def _filter_options(df: pd.DataFrame, col: str) -> list[str]:
    """"All" plus every distinct value in `col`, or just ["All"] if the
    column isn't present (e.g. an older LCC extract missing MNT NAME)."""
    if col not in df.columns:
        return ["All"]
    return ["All"] + sorted(df[col].dropna().unique().tolist())


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_filtered_roll_rate(_df_curr_f: pd.DataFrame, _df_prev_f: pd.DataFrame, data_version: int, drilldown_key: str):
    """Same cached-recompute pattern app.py's own `_cached_roll_rate` uses for
    the unfiltered case -- without this, a drill-down filter recomputes the
    full roll-rate matrix (a set-build + merge over every matched account) on
    EVERY Streamlit rerun, including reruns triggered by unrelated widgets
    elsewhere on the page, not just when the filter selection actually changes.

    _df_curr_f/_df_prev_f are underscore-prefixed (never hashed by Streamlit --
    hashing the full sliced DataFrame every rerun was the actual cost this
    cache was meant to avoid); data_version + drilldown_key (which already
    captures both the sidebar's filter selection AND this tab's own local
    Region/Branch/Executive drill-down) are the cheap, explicit stand-ins."""
    from analysis.roll_rate import compute_roll_rate_matrix
    return compute_roll_rate_matrix(_df_curr_f, _df_prev_f)


def _bucket_summary_table_html(df: pd.DataFrame) -> str:
    headers = ["Bucket", "Accounts", "Roll Fwd%", "Stable%", "Roll Bwd%"]
    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:8px 12px;font-size:11px;'
        f'text-align:{"left" if h == "Bucket" else "right"};white-space:nowrap;">{h}</th>'
        for h in headers
    )
    rows_html = ""
    # Roll Fwd% is a risk (no "safe" level): red shading by rank, never green.
    # Roll Bwd% is the good direction, so it keeps its green threshold.
    fwd_rng = heat_range(df["Roll Fwd%"].tolist())
    for _, row in df.iterrows():
        fwd, stable, bwd = row["Roll Fwd%"], row["Stable%"], row["Roll Bwd%"]
        bwd_color = "#16a34a" if bwd >= 10 else ("#d97706" if bwd >= 5 else "#374151")
        rows_html += (
            f'<tr style="border-bottom:1px solid #f0f0f0;">'
            f'<td style="padding:8px 12px;font-size:12px;font-weight:700;">{row["Bucket"]}</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;">{row["Accounts"]:,}</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;font-weight:700;{heat_style(fwd, fwd_rng)}">{fwd:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;color:#6b7280;">{stable:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;text-align:right;font-weight:700;color:{bwd_color};">{bwd:.1f}%</td>'
            f'</tr>'
        )
    return (
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


# ── Roll / flow by count and SOH (analysis/roll_flow.py) ─────────────────────

_MATRIX_MEASURES = {"Accounts %": ("count_pct", "pct"), "Accounts": ("count", "count"),
                    "SOH %": ("soh_pct", "pct"), "SOH (₹ Cr)": ("soh", "soh")}
_GRAIN_TABS = [("By Region", "region"), ("By Branch", "branch"), ("By Executive", "executive")]
# The first two steps are the early slips a manager can still catch cheaply.
_CATCH_EARLY = set(list(ROLL_STEPS)[:2])


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_roll_flow(_df: pd.DataFrame, data_version: int, drilldown_key: str) -> dict:
    """Every roll/flow view for one drill-down selection, computed once per
    selection (same cache-key pattern as _cached_filtered_roll_rate)."""
    from analysis import roll_flow as rf
    return {
        "summary": rf.roll_steps_summary(_df),
        "matrix": {m: rf.migration_matrix(_df, m) for m, _ in _MATRIX_MEASURES.values()},
        "by": {g: rf.roll_steps_by(_df, g) for _, g in _GRAIN_TABS},
        "loans": {label: rf.loans_that_rolled(_df, b) for label, b in ROLL_STEPS.items()},
    }


def _step_card(label: str, d: dict, from_bucket: str) -> str:
    is_cure = from_bucket == ""
    early = label in _CATCH_EARLY
    accent = "#16a34a" if is_cure else ("#dc2626" if early else "#9ca3af")
    tag = ('<span style="background:#fee2e2;color:#991b1b;font-size:10px;font-weight:700;'
           'padding:1px 7px;border-radius:9px;margin-left:6px;">CATCH EARLY</span>') if early else ""
    base_name = "behind" if is_cure else from_bucket
    return (
        f'<div style="border:1px solid #e5e7eb;border-top:4px solid {accent};border-radius:10px;'
        f'padding:12px 14px;background:#fff;height:100%;">'
        f'<div style="font-size:12px;font-weight:700;color:#374151;">{_esc(label)}{tag}'
        f'{info_icon(help_for(label))}</div>'
        f'<div style="font-size:26px;font-weight:800;color:#111827;margin-top:4px;">{d["n"]:,}'
        f'<span style="font-size:13px;font-weight:600;color:#6b7280;"> loans</span></div>'
        f'<div style="font-size:12px;color:#374151;"><b>{d["pct"]:.1f}%</b> of {d["base"]:,} {base_name} loans</div>'
        f'<div style="font-size:12px;color:#374151;margin-top:2px;"><b>₹{d["soh_cr"]:,.2f} Cr</b> SOH '
        f'(<b>{d["soh_pct"]:.1f}%</b> of ₹{d["base_soh_cr"]:,.2f} Cr)</div>'
        f'</div>'
    )


def _render_early_warning(summary: dict) -> None:
    st.markdown('<div class="section-label" style="margin-top:8px;">Early Warning: Who Slipped This Month</div>',
                unsafe_allow_html=True)
    st.caption(
        "Loans in each bucket last month that are in a worse bucket now, by number of loans and by "
        "last month's SOH. New defaulters and 1-30 DPD loans slipping further are the cheapest to bring "
        "back; left alone they become NPA in two or three months."
    )
    labels = [*ROLL_STEPS, "Back to STD"]
    froms = [*ROLL_STEPS.values(), ""]
    for col, label, fb in zip(st.columns(len(labels)), labels, froms):
        with col:
            st.markdown(_step_card(label, summary[label], fb), unsafe_allow_html=True)


def _step_cell(row, label: str, by_soh: bool) -> tuple[str, float]:
    """'11.9%' over '37 of 310' (or '₹1.20 of 10.30 Cr'); returns (html, % for shading)."""
    if by_soh:
        pct, sub = row[f"{label} | SOH %"], f"₹{row[f'{label} | SOH (Cr)']:,.2f} of {row[f'{label} | Base SOH (Cr)']:,.2f} Cr"
    else:
        pct, sub = row[f"{label} | %"], f"{int(row[f'{label} | Accounts']):,} of {int(row[f'{label} | Base']):,}"
    if not row[f"{label} | Base"]:
        return '<span style="color:#9ca3af;">-</span>', 0.0
    return (f'<div style="font-weight:700;">{pct:.1f}%</div>'
            f'<div style="font-size:10.5px;color:#6b7280;white-space:nowrap;">{sub}</div>'), float(pct)


def _with_total(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Append a Total row: sums of counts/SOH, % recomputed from those sums."""
    sums = df.drop(columns=id_cols).sum(numeric_only=True)
    for label in [*ROLL_STEPS, "Back to STD"]:
        base, base_soh = sums[f"{label} | Base"], sums[f"{label} | Base SOH (Cr)"]
        sums[f"{label} | %"] = round(sums[f"{label} | Accounts"] / base * 100, 2) if base else 0.0
        sums[f"{label} | SOH %"] = round(sums[f"{label} | SOH (Cr)"] / base_soh * 100, 2) if base_soh else 0.0
    total = {c: "" for c in id_cols} | sums.to_dict()
    total[id_cols[0]] = "Total"
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _roll_table_html(df: pd.DataFrame, by_soh: bool) -> str:
    id_cols = [c for c in ("Region", "Branch", "Executive") if c in df.columns]
    id_cols = ([c for c in id_cols if c not in ("Region",)] + (["Region"] if "Region" in id_cols else []))
    labels = [*ROLL_STEPS, "Back to STD"]
    view = _with_total(df, id_cols)
    n = len(view)
    ranges = {lb: heat_range([r[f"{lb} | SOH %" if by_soh else f"{lb} | %"] for _, r in df.iterrows()
                              if r[f"{lb} | Base"]]) for lb in ROLL_STEPS}
    size_col = "Matched SOH (Cr)" if by_soh else "Matched Accounts"
    head = "".join(f'<th style="text-align:left;">{c}</th>' for c in id_cols)
    head += f'<th>{"SOH last month" if by_soh else "Accounts"}</th>'
    head += "".join(f'<th>{_esc(lb)}{info_icon(help_for(lb), color="#fde68a")}</th>' for lb in labels)
    rows = ""
    for i, (_, r) in enumerate(view.iterrows()):
        total = i == n - 1
        style = "font-weight:800;border-top:2px solid #FFC000;background:#fffbea;" if total else ""
        cells = "".join(f'<td style="text-align:left;{"font-weight:700;" if j == 0 else ""}">{_esc(r[c])}</td>'
                        for j, c in enumerate(id_cols))
        size = r[size_col]
        cells += f'<td>{f"₹{size:,.2f} Cr" if by_soh else f"{int(size):,}"}</td>'
        for lb in labels:
            html, pct = _step_cell(r, lb, by_soh)
            shade = "" if total or lb not in ranges else heat_style(pct, ranges[lb])
            if lb == "Back to STD":
                shade = "color:#15803d;"
            cells += f'<td style="{shade}">{html}</td>'
        rows += f'<tr style="{style}border-bottom:1px solid #f0f0f0;">{cells}</tr>'
    css = ("<style>.rf-tbl th{background:#111;color:#FFC000;padding:7px 10px;font-size:11px;"
           "text-align:right;white-space:nowrap;position:sticky;top:0;z-index:1;}"
           ".rf-tbl td{padding:6px 10px;font-size:12px;text-align:right;vertical-align:top;}</style>")
    return (f'{css}<div style="overflow:auto;max-height:520px;border-radius:10px;border:1px solid #e5e7eb;">'
            f'<table class="rf-tbl" style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
            f'<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def _render_where(by: dict) -> None:
    st.markdown('<div class="section-label" style="margin-top:20px;">Where It\'s Happening</div>',
                unsafe_allow_html=True)
    by_soh = st.radio("Measure roll rates by", ["Accounts", "SOH"], horizontal=True, key="mig_where_measure") == "SOH"
    st.caption(
        "Each cell: % of the loans (or of last month's SOH) in that bucket last month that are in a worse "
        "bucket now, with the numbers behind it. Sorted by new defaulters first. Units below the minimum "
        "size are left out."
    )
    for tab, (name, grain) in zip(st.tabs([t for t, _ in _GRAIN_TABS]), _GRAIN_TABS):
        with tab:
            df = by[grain]
            if df.empty:
                st.info("Not enough matched loans for this view.")
                continue
            st.markdown(_roll_table_html(df, by_soh), unsafe_allow_html=True)
            _dl_btn(df, f"roll_rates_by_{grain}.xlsx", f"dl_roll_{grain}")


def _render_call_list(loans: dict) -> None:
    st.markdown('<div class="section-label" style="margin-top:20px;">Loans To Call First</div>',
                unsafe_allow_html=True)
    st.caption("The loans behind each step, largest SOH first, with the executive to follow up.")
    labels = list(ROLL_STEPS)
    for tab, label in zip(st.tabs([f"{lb} ({len(loans[lb]):,})" for lb in labels]), labels):
        with tab:
            df = loans[label]
            if df.empty:
                st.caption("No loans made this move.")
                continue
            soh_cr = df["SOH"].sum() / 1e7 if "SOH" in df.columns else 0.0
            st.caption(f"{len(df):,} loans, ₹{soh_cr:,.2f} Cr SOH today.")
            st.dataframe(_safe_df(df), use_container_width=True, hide_index=True, height=360)
            slug = label.split(" ")[0].lower().replace("-", "")
            _dl_btn(df, f"call_list_{slug}.xlsx", f"dl_call_{slug}")


def render_migration_tab(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    rr_matrix,
    rr_meta: dict | None,
    data_version: int = 0,
    sidebar_filter_key: str = "",
) -> None:
    if rr_meta is None:
        _empty_state(
            "📈", "Previous month data not loaded",
            "Upload a previous month LCC file alongside the current month file,<br>"
            "then click <strong>Generate Dashboard</strong> to see bucket migration and roll-rate analysis.",
        )
        return

    from analysis.roll_rate import compute_bucket_roll_summary, build_roll_rate_heatmap

    # ── Drill-down filters (local to this tab -- narrows the migration view
    # without touching the sidebar's global filter used by every other tab) ──
    st.markdown('<div class="section-label">Drill Down</div>', unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        sel_region = st.selectbox("Region", _filter_options(df_curr, "RegionName"), key="mig_region")

    df_for_branch = df_curr if sel_region == "All" else df_curr[df_curr["RegionName"] == sel_region]
    with f_col2:
        sel_branch = st.selectbox("Branch", _filter_options(df_for_branch, "Unit"), key="mig_branch")

    df_for_exec = df_for_branch if sel_branch == "All" else df_for_branch[df_for_branch["Unit"] == sel_branch]
    with f_col3:
        sel_exec = st.selectbox("Executive", _filter_options(df_for_exec, "MNT NAME"), key="mig_exec")

    has_filter = sel_region != "All" or sel_branch != "All" or sel_exec != "All"

    if has_filter:
        def _apply(df: pd.DataFrame) -> pd.DataFrame:
            # One combined boolean mask, one indexing op -- not up to 3
            # sequential boolean-index copies for the up-to-3 active filters.
            mask = pd.Series(True, index=df.index)
            if sel_region != "All" and "RegionName" in df.columns:
                mask &= df["RegionName"] == sel_region
            if sel_branch != "All" and "Unit" in df.columns:
                mask &= df["Unit"] == sel_branch
            if sel_exec != "All" and "MNT NAME" in df.columns:
                mask &= df["MNT NAME"] == sel_exec
            return df[mask]

        df_curr_f = _apply(df_curr)
        df_prev_f = _apply(df_prev) if df_prev is not None else pd.DataFrame()
        if len(df_curr_f) == 0:
            st.warning("No accounts match this Region / Branch / Executive combination.")
            return
        # Combines the sidebar's own filter selection (already baked into
        # df_curr's content, but not otherwise visible to this cache key) with
        # this tab's own local drill-down -- both determine df_curr_f's content.
        drilldown_key = f"{sidebar_filter_key}|{sel_region}|{sel_branch}|{sel_exec}"
        rr_matrix, rr_meta = _cached_filtered_roll_rate(df_curr_f, df_prev_f, data_version, drilldown_key)
    else:
        df_curr_f = df_curr
        drilldown_key = f"{sidebar_filter_key}|All|All|All"

    flow = _cached_roll_flow(df_curr_f, data_version, drilldown_key)
    if flow["summary"]:
        _render_early_warning(flow["summary"])

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:20px;">Overall Roll Rates</div>', unsafe_allow_html=True)
    rr_kpis = [
        ("Roll-Forward Rate",  rr_meta["roll_forward_rate"],  "%", "#dc2626", "Accounts that worsened bucket"),
        ("Roll-Backward Rate", rr_meta["roll_backward_rate"], "%", "#16a34a", "Delinquent accounts returned to STD"),
        ("NPA Formation",      rr_meta["npa_formation_rate"], "%", "#991b1b", "Non-NPA accounts that became NPA"),
        ("Matched Accounts",   rr_meta["matched_count"],      "",  "#111827", "Accounts in both months"),
    ]
    for col, (label, val, unit, color, tip) in zip(st.columns(4), rr_kpis):
        with col:
            st.markdown(
                _static_kpi_card_html(label, f"{val:,.1f}{unit}", tip, color=color, value_style="font-size:24px;"),
                unsafe_allow_html=True,
            )

    # ── Heatmap ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:20px;">Bucket Migration Matrix</div>', unsafe_allow_html=True)
    measure = st.radio("Show matrix as", list(_MATRIX_MEASURES), horizontal=True, key="mig_matrix_measure")
    m_key, kind = _MATRIX_MEASURES[measure]
    _chart_card(build_roll_rate_heatmap(flow["matrix"][m_key], kind) if flow["summary"] else build_roll_rate_heatmap(rr_matrix))
    if measure.endswith("%"):
        st.caption("Each row adds to 100%: of what was in that bucket last month, where it is now.")
    st.caption(
        f"{rr_meta['matched_count']:,} matched accounts | "
        f"{rr_meta['new_entries']:,} new this month | "
        f"{rr_meta['exits']:,} closed/exited"
    )

    # ── Bucket-wise Roll Forward / Stable / Roll Backward % table ─────────────
    st.markdown('<div class="section-label" style="margin-top:20px;">Bucket-wise Roll Rates</div>', unsafe_allow_html=True)
    bucket_summary = compute_bucket_roll_summary(rr_matrix)
    if bucket_summary.empty or bucket_summary["Accounts"].sum() == 0:
        st.caption("No matched accounts for this selection.")
    else:
        st.markdown(_bucket_summary_table_html(bucket_summary), unsafe_allow_html=True)
        st.caption(
            "Per previous-month bucket: % that rolled forward (worsened), stayed stable, "
            "or rolled backward (improved) this month."
        )

    if flow["summary"]:
        _render_where(flow["by"])
        _render_call_list(flow["loans"])

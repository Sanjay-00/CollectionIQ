"""
Portfolio Intelligence tab  -  pre-computed answers to the 5 portfolio questions.
Zero AI calls. Loads instantly.
"""
import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _safe_df

_STATUS_COLOR = {"Improving": "#16a34a", "Worsening": "#dc2626", "Stable": "#d97706", "-": "#9ca3af"}
_STATUS_ICON  = {"Improving": "🟢", "Worsening": "🔴", "Stable": "🟡", "-": "⚪"}
_SEV_COLOR    = {"critical": "#dc2626", "high": "#f97316", "medium": "#d97706", "low": "#16a34a"}


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def _kpi_mini(label: str, value: str, delta, unit: str = "", inverse: bool = False) -> str:
    if delta is None:
        mom_html = '<div class="kpi-mom" style="color:#9ca3af;">no prev data</div>'
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


def _delta_html(val, unit: str = "pp") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '<span style="color:#9ca3af;"> - </span>'
    color = "#dc2626" if val > 0.1 else ("#16a34a" if val < -0.1 else "#d97706")
    arrow = "▲" if val > 0.1 else ("▼" if val < -0.1 else "-")
    return f'<span style="color:{color};font-weight:700;">{arrow} {abs(val):.2f}{unit}</span>'


def _count_delta_html(val) -> str:
    if val is None:
        return '<span style="color:#9ca3af;"> - </span>'
    if val == 0:
        return '<span style="color:#9ca3af;font-weight:700;">-</span>'
    color = "#dc2626" if val > 0 else "#16a34a"
    arrow = "▲" if val > 0 else "▼"
    return f'<span style="color:{color};font-weight:700;">{arrow} {abs(int(val))}</span>'


# ── Section 1: Portfolio Pulse ────────────────────────────────────────────────

def _render_pulse(kpis: list, fig_waterfall, rr_meta: dict | None, has_prev: bool) -> None:
    _section("Section 1  -  Portfolio Pulse: State of the Book in 30 Seconds")

    def _kpi_row(items):
        return "".join(_kpi_mini(k["label"], k["value"], k["delta"], k["unit"], k["inverse"]) for k in items)

    st.markdown(f'<div class="kpi-row">{_kpi_row(kpis[:4])}</div>', unsafe_allow_html=True)
    if len(kpis) > 4:
        st.markdown(f'<div class="kpi-row" style="margin-top:8px;">{_kpi_row(kpis[4:])}</div>', unsafe_allow_html=True)

    col_wf, col_npa = st.columns([3, 1])
    with col_wf:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_waterfall, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_npa:
        if rr_meta and rr_meta.get("matched_count", 0) > 0:
            npa_form = rr_meta["npa_formation_rate"]
            rbwd     = rr_meta["roll_backward_rate"]
            rfwd     = rr_meta["roll_forward_rate"]
            for label, val, color, tip in [
                ("Fresh NPA Formation", f"{npa_form:.1f}%", "#991b1b", "Non-NPA → NPA this month"),
                ("Roll-Backward Rate",  f"{rbwd:.1f}%",     "#16a34a", "Accounts rescued"),
                ("Roll-Forward Rate",   f"{rfwd:.1f}%",     "#dc2626", "Accounts worsened"),
            ]:
                st.markdown(
                    f'<div class="kpi-card" style="border-top-color:{color};margin-bottom:8px;">'
                    f'<div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value" style="color:{color};font-size:22px;">{val}</div>'
                    f'<div class="kpi-mom" style="color:#9ca3af;">{tip}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Upload previous month file to see roll rates and NPA formation rate.")


# ── Section 2: Region Scorecard ───────────────────────────────────────────────

def _render_region_scorecard(df: pd.DataFrame, has_prev: bool) -> None:
    display_cols = [
        "Region", "Accounts",
        "SMA-2", "SMA-2%",
        "NPA% (Curr)",
        *( ["NPA% (Prev)", "Δ NPA%"] if has_prev else []),
        "Collection%", "Hard Bucket%", "SOH (Cr)",
        *( ["Roll Fwd%", "Roll Bwd%"] if "Roll Fwd%" in df.columns and df["Roll Fwd%"].notna().any() else []),
        "Status",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:7px 12px;font-size:11px;'
        f'text-align:{"left" if c in ("Region","Status") else "right"};white-space:nowrap;">{c}</th>'
        for c in display_cols
    )

    rows_html = ""
    for _, row in df.iterrows():
        status = str(row.get("Status", "-"))
        row_bg = "#fff5f5" if status == "Worsening" else ("#f0fdf4" if status == "Improving" else "#fff")
        cells = ""
        for col in display_cols:
            val = row.get(col)
            align = "left" if col in ("Region", "Status") else "right"
            style = f"padding:7px 12px;font-size:12px;text-align:{align};"

            if col == "Region":
                cells += f'<td style="{style}font-weight:700;">{val}</td>'
            elif col == "Status":
                cells += f'<td style="{style}">{_badge(str(val))}</td>'
            elif col == "Δ NPA%":
                cells += f'<td style="{style}">{_delta_html(val)}</td>'
            elif col == "SMA-2%":
                c = "#ef4444" if (val or 0) > 10 else ("#d97706" if (val or 0) > 5 else "#374151")
                display = f"{val:.1f}%" if val is not None else " - "
                cells += f'<td style="{style}color:{c};font-weight:600;">{display}</td>'
            elif col == "SMA-2":
                c = "#ef4444" if (val or 0) > 50 else "#374151"
                display = f"{int(val):,}" if val is not None else " - "
                cells += f'<td style="{style}color:{c};">{display}</td>'
            elif col in ("NPA% (Curr)", "NPA% (Prev)"):
                c = "#dc2626" if (val or 0) > 10 else ("#d97706" if (val or 0) > 5 else "#16a34a")
                display = f"{val:.1f}%" if val is not None else " - "
                cells += f'<td style="{style}color:{c};font-weight:600;">{display}</td>'
            elif col == "Roll Fwd%":
                c = "#dc2626" if (val or 0) > 20 else ("#d97706" if (val or 0) > 10 else "#16a34a")
                cells += f'<td style="{style}color:{c};font-weight:600;">{val:.1f}%</td>' if val is not None else f'<td style="{style}"> - </td>'
            elif col == "Roll Bwd%":
                c = "#16a34a" if (val or 0) > 10 else "#d97706"
                cells += f'<td style="{style}color:{c};font-weight:600;">{val:.1f}%</td>' if val is not None else f'<td style="{style}"> - </td>'
            elif col in ("Collection%", "Hard Bucket%"):
                cells += f'<td style="{style}">{val:.1f}%</td>' if val is not None else f'<td style="{style}"> - </td>'
            elif col == "SOH (Cr)":
                cells += f'<td style="{style}">₹{val:.2f}Cr</td>'
            elif isinstance(val, (int, float)) and not pd.isna(val):
                cells += f'<td style="{style}">{int(val):,}</td>'
            else:
                cells += f'<td style="{style}">{val if val is not None else " - "}</td>'

        rows_html += f'<tr style="background:{row_bg};border-bottom:1px solid #f0f0f0;">{cells}</tr>'

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )
    _dl_btn(df, "region_scorecard.xlsx", "dl_region")


def _render_scorecard_section(region_df, branch_df, fig_quadrant, exec_recovery_df, has_prev, npa_sma2_cmp):
    _section("Section 2  -  Who Needs Attention?  (Region / Branch / Executive)", margin_top="24px")

    sub_tabs = st.tabs(["Region View", "Branch Quadrant", "Executive Recovery", "NPA & SMA-2 Comparison"])

    with sub_tabs[0]:
        if not region_df.empty:
            n_wors = (region_df["Status"] == "Worsening").sum() if "Status" in region_df.columns else 0
            n_impr = (region_df["Status"] == "Improving").sum() if "Status" in region_df.columns else 0
            n_stbl = len(region_df) - n_wors - n_impr
            c1, c2, c3 = st.columns(3)
            for col, label, val, color in [
                (c1, "Worsening Regions", n_wors, "#dc2626"),
                (c2, "Improving Regions", n_impr, "#16a34a"),
                (c3, "Stable Regions", n_stbl, "#d97706"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="kpi-card" style="border-top-color:{color};">'
                        f'<div class="kpi-label">{label}</div>'
                        f'<div class="kpi-value" style="color:{color};">{val}</div></div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("<br>", unsafe_allow_html=True)
            _render_region_scorecard(region_df, has_prev)
        else:
            st.info("No region data (RegionName column not found).")

    with sub_tabs[1]:
        if not branch_df.empty:
            col_chart, col_legend = st.columns([3, 1])
            with col_chart:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_quadrant, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
            with col_legend:
                for label, color, desc in [
                    ("Intervene Now", "#dc2626", "High NPA, Low Collection"),
                    ("Watch",         "#f97316", "High NPA, High Collection"),
                    ("Underperforming", "#d97706", "Low NPA, Low Collection"),
                    ("Healthy",       "#16a34a", "Low NPA, High Collection"),
                ]:
                    st.markdown(
                        f'<div style="border-left:4px solid {color};padding:6px 10px;'
                        f'margin-bottom:8px;background:{color}0D;border-radius:0 6px 6px 0;">'
                        f'<div style="font-size:12px;font-weight:700;color:{color};">{label}</div>'
                        f'<div style="font-size:11px;color:#6b7280;">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )
                st.caption("Bubble size = SOH. Dashed lines = portfolio median.")

            st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin-top:12px;">Branch Rankings (Sortable)</div>', unsafe_allow_html=True)
            with st.expander(f"View all {len(branch_df)} branches", expanded=False):
                st.dataframe(_safe_df(branch_df), use_container_width=True, hide_index=True)
                _dl_btn(branch_df, "branch_quadrant.xlsx", "dl_branch_quad")
        else:
            st.info("No branch data (Unit column not found).")

    with sub_tabs[2]:
        if not exec_recovery_df.empty:
            st.caption("Accounts rescued = moved from NPA/SMA-2/SMA-1 to a better bucket vs last month. Net Recovery = Rescued − Slipped.")
            _render_exec_recovery(exec_recovery_df)
        elif not has_prev:
            st.info("Upload previous month file to see the executive recovery leaderboard.")
        else:
            st.info("No executive-level prev_bucket data available.")

    with sub_tabs[3]:
        _render_npa_sma2_comparison(npa_sma2_cmp or {}, has_prev)


# ── Section 2: NPA & SMA-2 Comparison ────────────────────────────────────────

def _render_npa_sma2_comparison(cmp_data: dict, has_prev: bool) -> None:
    import plotly.graph_objects as go

    if not cmp_data:
        st.info("No comparison data available.")
        return

    dim_tabs_avail = []
    if "region"    in cmp_data: dim_tabs_avail.append(("Region",    "RegionName", "region"))
    if "branch"    in cmp_data: dim_tabs_avail.append(("Branch",    "Unit",       "branch"))
    if "executive" in cmp_data: dim_tabs_avail.append(("Executive", "MNT NAME",   "executive"))

    if not dim_tabs_avail:
        st.info("No dimension data found.")
        return

    dim_sub = st.tabs([t[0] for t in dim_tabs_avail])
    for tab, (label, col, key) in zip(dim_sub, dim_tabs_avail):
        with tab:
            df = cmp_data[key].copy()
            names = df[col].tolist()

            # ── grouped bar chart ────────────────────────────────────────────
            fig = go.Figure()

            def _bar(name, y_vals, color, text_color="#fff"):
                y_clean = [v if v is not None and not (isinstance(v, float) and pd.isna(v)) else 0 for v in y_vals]
                return go.Bar(
                    name=name, x=names, y=y_clean,
                    marker=dict(color=color, line=dict(width=0)),
                    text=[str(int(v)) if v else "" for v in y_clean],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=11, color=text_color, family="Arial Black"),
                    hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y}}<extra></extra>",
                )

            fig.add_trace(_bar("NPA (Curr)", df["NPA (Curr)"].tolist(), "#dc2626"))
            if has_prev and df["NPA (Prev)"].notna().any():
                fig.add_trace(_bar("NPA (Prev)", df["NPA (Prev)"].tolist(), "#fca5a5", "#374151"))
            fig.add_trace(_bar("SMA-2 (Curr)", df["SMA-2 (Curr)"].tolist(), "#ea580c"))
            if has_prev and df["SMA-2 (Prev)"].notna().any():
                fig.add_trace(_bar("SMA-2 (Prev)", df["SMA-2 (Prev)"].tolist(), "#fed7aa", "#374151"))

            fig.update_layout(
                barmode="group",
                bargap=0.20, bargroupgap=0.06,
                title=dict(text=f"NPA & SMA-2 Count  -  Current vs Previous  ({label})", font=dict(size=13, color="#111"), x=0),
                xaxis=dict(tickangle=-30, showgrid=False, tickfont=dict(color="#374151")),
                yaxis=dict(title="Account Count", showgrid=True, gridcolor="#f3f4f6", tickfont=dict(color="#374151")),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
                margin=dict(l=20, r=20, t=60, b=90),
                height=430,
            )
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

            # ── delta table ──────────────────────────────────────────────────
            st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin:12px 0 6px;">Detailed Comparison Table</div>', unsafe_allow_html=True)

            show_cols = [col, "Accounts", "NPA (Curr)", "SMA-2 (Curr)"]
            if has_prev:
                show_cols += ["NPA (Prev)", "NPA Δ", "NPA Δ%", "SMA-2 (Prev)", "SMA-2 Δ", "SMA-2 Δ%"]

            th = "".join(
                f'<th style="background:#111;color:#FFC000;padding:6px 10px;font-size:11px;'
                f'text-align:{"left" if c == col else "center"};white-space:nowrap;">{c}</th>'
                for c in show_cols if c in df.columns
            )
            rows_html = ""
            for _, row in df.iterrows():
                cells = ""
                for c in [c for c in show_cols if c in df.columns]:
                    val = row[c]
                    align = "left" if c == col else "center"
                    style = f"padding:6px 10px;font-size:12px;text-align:{align};"

                    if c == col:
                        cells += f'<td style="{style}font-weight:700;">{val}</td>'
                    elif c in ("NPA Δ", "SMA-2 Δ"):
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            cells += f'<td style="{style}color:#9ca3af;"> - </td>'
                        else:
                            color = "#dc2626" if val > 0 else ("#16a34a" if val < 0 else "#9ca3af")
                            arrow = "▲" if val > 0 else ("▼" if val < 0 else "-")
                            cells += f'<td style="{style}color:{color};font-weight:700;">{arrow} {abs(int(val))}</td>'
                    elif c in ("NPA Δ%", "SMA-2 Δ%"):
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            cells += f'<td style="{style}color:#9ca3af;"> - </td>'
                        else:
                            color = "#dc2626" if val > 0 else ("#16a34a" if val < 0 else "#9ca3af")
                            arrow = "▲" if val > 0 else ("▼" if val < 0 else "-")
                            cells += f'<td style="{style}color:{color};font-weight:700;">{arrow} {abs(val):.1f}%</td>'
                    elif c in ("NPA (Curr)",):
                        color = "#dc2626" if (val or 0) > 20 else "#374151"
                        cells += f'<td style="{style}color:{color};font-weight:700;">{int(val):,}</td>'
                    elif c in ("SMA-2 (Curr)",):
                        color = "#f97316" if (val or 0) > 20 else "#374151"
                        cells += f'<td style="{style}color:{color};font-weight:700;">{int(val):,}</td>'
                    elif isinstance(val, (int, float)) and not pd.isna(val):
                        cells += f'<td style="{style}">{int(val):,}</td>'
                    else:
                        cells += f'<td style="{style}">{val if val is not None else " - "}</td>'
                rows_html += f'<tr style="border-bottom:1px solid #f0f0f0;">{cells}</tr>'

            st.markdown(
                f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #e5e7eb;">'
                f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
                f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
                f'</table></div>',
                unsafe_allow_html=True,
            )
            _dl_btn(df[[c for c in show_cols if c in df.columns]], f"npa_sma2_{key}.xlsx", f"dl_cmp_{key}")

            if not has_prev:
                st.caption("Upload previous month file to see Δ columns.")


# ── Section 3: Good vs Bad ────────────────────────────────────────────────────

def _render_good_bad(good_bad: dict, has_prev: bool) -> None:
    _section("Section 3  -  The Honest Mirror: What Went Right / Concerns", margin_top="24px")
    good = good_bad.get("good", [])
    bad  = good_bad.get("bad", [])

    if not good and not bad:
        st.info("Upload both current and previous month files to see the Good vs Bad summary.")
        return

    col_g, col_b = st.columns(2)
    with col_g:
        items = "".join(
            f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #dcfce7;">{item}</li>'
            for item in good
        ) or '<li style="color:#9ca3af;font-style:italic;">No notable improvements detected.</li>'
        st.markdown(
            f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px;">'
            f'<div style="font-size:13px;font-weight:700;color:#16a34a;margin-bottom:10px;">🟢 What Went Right</div>'
            f'<ul style="list-style:none;padding:0;margin:0;">{items}</ul></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        items = "".join(
            f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #fee2e2;">{item}</li>'
            for item in bad
        ) or '<li style="color:#9ca3af;font-style:italic;">No notable concerns detected.</li>'
        st.markdown(
            f'<div style="background:#fff5f5;border:1px solid #fecaca;border-radius:10px;padding:16px;">'
            f'<div style="font-size:13px;font-weight:700;color:#dc2626;margin-bottom:10px;">🔴 Concerns This Month</div>'
            f'<ul style="list-style:none;padding:0;margin:0;">{items}</ul></div>',
            unsafe_allow_html=True,
        )


# ── Section 4: Risk Flag Deep Dive ────────────────────────────────────────────

def _render_risk_flags(flag_df: pd.DataFrame, alerts_curr: list) -> None:
    _section("Section 4  -  Risk Flag Deep Dive", margin_top="24px")

    if flag_df.empty:
        st.info("No risk flag data available.")
        return

    alerts_by_title = {a["title"]: a for a in (alerts_curr or [])}

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    flag_df = flag_df.copy()
    flag_df["_sev_ord"] = flag_df["Severity"].map(lambda s: sev_order.get(s, 9))
    flag_df = flag_df.sort_values("_sev_ord").drop(columns=["_sev_ord"])

    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:7px 12px;font-size:11px;'
        f'text-align:{"left" if h == "Risk Type" else "center"};white-space:nowrap;">{h}</th>'
        for h in ["Risk Type", "Accounts", "SOH Exposure", "Last Month", "Δ vs Last Month"]
    )

    rows_html = ""
    for _, row in flag_df.iterrows():
        sev   = row.get("Severity", "medium")
        color = _SEV_COLOR.get(sev, "#d97706")
        cnt   = int(row["Accounts"])
        soh   = row["SOH (Cr)"]
        prev  = row["Last Month"]
        delta = row["Δ"]
        rows_html += (
            f'<tr style="border-bottom:1px solid #f0f0f0;">'
            f'<td style="padding:7px 12px;font-size:12px;font-weight:700;border-left:3px solid {color};">{row["Risk Type"]}</td>'
            f'<td style="padding:7px 12px;font-size:14px;font-weight:800;color:{color};text-align:center;">{cnt}</td>'
            f'<td style="padding:7px 12px;font-size:12px;text-align:center;">₹{soh:.2f}Cr</td>'
            f'<td style="padding:7px 12px;font-size:12px;color:#6b7280;text-align:center;">{int(prev) if prev is not None else " - "}</td>'
            f'<td style="padding:7px 12px;text-align:center;">{_count_delta_html(delta)}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin-top:20px;">Account Drilldown</div>', unsafe_allow_html=True)
    st.caption("Accounts pre-filtered per flag  -  instant, no AI call needed.")

    _BG = {"critical": "#fff5f5", "high": "#fff7ed", "medium": "#fffbea", "low": "#f0fdf4"}

    for _, row in flag_df.iterrows():
        title = row["Risk Type"]
        cnt   = int(row["Accounts"])
        if cnt == 0:
            continue
        alert = alerts_by_title.get(title)
        if not alert or "df" not in alert or alert["df"].empty:
            continue
        sev    = row.get("Severity", "medium")
        color  = _SEV_COLOR.get(sev, "#d97706")
        bg     = _BG.get(sev, "#fffbea")
        icon   = alert.get("icon", "⚠️")
        sub    = alert.get("subtitle", "")
        action = alert.get("action", "")
        pos    = f"₹{alert.get('pos', 0) / 1e7:.2f}Cr"
        arrears = f"₹{alert.get('closing_arrears', 0) / 1e7:.2f}Cr"

        st.markdown(f"""
        <div style="background:{bg};border-radius:12px;border-left:4px solid {color};
                    padding:14px 18px;margin-bottom:4px;box-shadow:0 2px 6px rgba(0,0,0,0.05);">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <span style="font-size:20px;">{icon}</span>
            <span style="font-size:14px;font-weight:700;color:{color};">{title}</span>
          </div>
          <div style="font-size:12px;color:#666;margin-bottom:10px;">{sub}</div>
          <div style="display:flex;gap:24px;margin-bottom:10px;">
            <div>
              <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Accounts</div>
              <div style="font-size:24px;font-weight:800;color:{color};">{cnt:,}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">POS</div>
              <div style="font-size:20px;font-weight:800;color:#111;">{pos}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Closing Arrears</div>
              <div style="font-size:20px;font-weight:800;color:{color};">{arrears}</div>
            </div>
          </div>
          <div style="font-size:11px;color:#555;font-style:italic;border-top:1px solid rgba(0,0,0,0.08);padding-top:8px;">
            💬 {action}
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"View {cnt:,} accounts", expanded=False):
            st.dataframe(_safe_df(alert["df"]), use_container_width=True, hide_index=True)
            _dl_btn(alert["df"], f"flag_{title.lower().replace(' ','_')}.xlsx", f"dl_flag_{title[:8]}")


# ── Section 5: Vintage & Sourcing ─────────────────────────────────────────────

def _render_product_table(df: pd.DataFrame, npa_col: str = "NPA%") -> None:
    if df.empty:
        return
    headers = list(df.columns)
    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:6px 10px;font-size:11px;'
        f'text-align:{"left" if i == 0 else "right"};white-space:nowrap;">{h}</th>'
        for i, h in enumerate(headers)
    )
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for i, col in enumerate(headers):
            val = row[col]
            align = "left" if i == 0 else "right"
            style = f"padding:6px 10px;font-size:12px;text-align:{align};"
            if col == npa_col:
                c = "#dc2626" if (val or 0) > 10 else ("#d97706" if (val or 0) > 5 else "#16a34a")
                cells += f'<td style="{style}color:{c};font-weight:700;">{val:.1f}%</td>'
            elif col == "SMA-2%":
                c = "#ef4444" if (val or 0) > 10 else ("#d97706" if (val or 0) > 5 else "#374151")
                cells += f'<td style="{style}color:{c};font-weight:700;">{val:.1f}%</td>'
            elif isinstance(val, float) and "%" in col:
                cells += f'<td style="{style}">{val:.1f}%</td>'
            elif isinstance(val, float) and any(k in col for k in ("SOH", "Loan", "Avg")):
                cells += f'<td style="{style}">₹{val:.2f}</td>'
            elif isinstance(val, float):
                cells += f'<td style="{style}">{val:.2f}</td>'
            elif isinstance(val, int):
                cells += f'<td style="{style}">{val:,}</td>'
            else:
                cells += f'<td style="{style}">{val}</td>'
        rows_html += f'<tr style="border-bottom:1px solid #f0f0f0;">{cells}</tr>'

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def _roll_vintage(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Roll monthly cohorts up to Quarter / Half-Year / Year and recompute NPA% + SMA-2%."""
    if granularity == "Monthly" or df.empty:
        return df

    def _label(month_str: str) -> str:
        try:
            y, m = int(month_str[:4]), int(month_str[5:7])
        except Exception:
            return month_str
        if granularity == "Quarterly":
            q = (m - 1) // 3 + 1
            return f"Q{q}-{y}"
        if granularity == "Half-Yearly":
            h = 1 if m <= 6 else 2
            return f"H{h}-{y}"
        return str(y)  # Yearly

    df = df.copy()
    df["_period"] = df["Disbursement Month"].apply(_label)

    agg = (
        df.groupby("_period", sort=False)
        .agg(
            Accounts=("Accounts", "sum"),
            **{"NPA Count":   ("NPA Count",   "sum")},
            **{"SMA-2 Count": ("SMA-2 Count", "sum")},
            **{"SOH (Cr)":    ("SOH (Cr)",    "sum")},
        )
        .reset_index()
        .rename(columns={"_period": "Disbursement Month"})
    )
    agg["NPA%"]   = (agg["NPA Count"]   / agg["Accounts"] * 100).round(2)
    agg["SMA-2%"] = (agg["SMA-2 Count"] / agg["Accounts"] * 100).round(2)

    # Sort chronologically: extract year + sub-period for stable ordering
    def _sort_key(label):
        try:
            if label.startswith("Q"):
                q, y = label[1:].split("-")
                return int(y) * 10 + int(q)
            if label.startswith("H"):
                h, y = label[1:].split("-")
                return int(y) * 10 + int(h)
            return int(label) * 10
        except Exception:
            return 0

    agg["_sk"] = agg["Disbursement Month"].apply(_sort_key)
    return agg.sort_values("_sk").drop(columns=["_sk"]).reset_index(drop=True)


def _render_vintage_sourcing(product_data: dict) -> None:
    _section("Section 5  -  Vintage & Sourcing Analysis", margin_top="24px")

    if not product_data:
        st.info("No segment, fuel type, vintage, or source channel data found in this file.")
        return

    tabs_avail = []
    if "vintage" in product_data: tabs_avail.append(("Disbursement Vintage", "vintage"))
    if "source"  in product_data: tabs_avail.append(("Sourcing Channel", "source"))
    if "segment" in product_data: tabs_avail.append(("Vehicle Segment", "segment"))
    if "fuel"    in product_data: tabs_avail.append(("Fuel Type", "fuel"))

    if not tabs_avail:
        st.info("Insufficient data for product analysis.")
        return

    sub_tabs = st.tabs([t[0] for t in tabs_avail])
    for sub_tab, (label, key) in zip(sub_tabs, tabs_avail):
        with sub_tab:
            df = product_data[key]
            if key == "vintage":
                st.caption(
                    "Rising NPA% on older cohorts = expected ageing. "
                    "Spike on a specific month = sourcing quality issue that month  -  collections can't fix it, credit can stop repeating it."
                )
                granularity = st.radio(
                    "Group by", ["Monthly", "Quarterly", "Half-Yearly", "Yearly"],
                    horizontal=True, key="vintage_gran", index=1,
                )
                plot_df = _roll_vintage(df, granularity)
                from analysis.portfolio_intelligence import build_vintage_chart
                fig_v = build_vintage_chart(plot_df)
                if fig_v.data:
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig_v, width='stretch')
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                _render_product_table(plot_df.drop(columns=["NPA Count", "SMA-2 Count"], errors="ignore"))
            elif key == "source":
                st.caption(
                    "DSA/sourcing channel NPA%. "
                    "Politically sensitive but extremely valuable  -  bad sources get delisted."
                )
                _render_product_table(df)
            else:
                _render_product_table(df)
            _dl_btn(df, f"product_{key}.xlsx", f"dl_prod_{key}")


# ── Section 6: Concentration & Exposure ───────────────────────────────────────

def _render_exec_recovery(df: pd.DataFrame) -> None:
    if df.empty:
        return
    headers = list(df.columns)
    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:7px 10px;font-size:11px;'
        f'text-align:{"left" if h == "Executive" else "center"};white-space:nowrap;">{h}</th>'
        for h in headers
    )
    rows_html = ""
    for _, row in df.iterrows():
        net = row.get("Net Recovery", 0)
        row_bg = "#f0fdf4" if net > 0 else ("#fff5f5" if net < 0 else "#fff")
        cells = ""
        for col in headers:
            val = row[col]
            align = "left" if col == "Executive" else "center"
            style = f"padding:7px 10px;font-size:12px;text-align:{align};"
            if col == "Net Recovery":
                c = "#16a34a" if val > 0 else ("#dc2626" if val < 0 else "#9ca3af")
                cells += f'<td style="{style}font-weight:800;font-size:14px;color:{c};">{val:+d}</td>'
            elif col == "Rescued":
                cells += f'<td style="{style}color:#16a34a;font-weight:700;">{val}</td>'
            elif col == "Slipped":
                cells += f'<td style="{style}color:#dc2626;font-weight:700;">{val}</td>'
            elif isinstance(val, int):
                cells += f'<td style="{style}">{val:,}</td>'
            else:
                cells += f'<td style="{style}">{val}</td>'
        rows_html += f'<tr style="background:{row_bg};border-bottom:1px solid #f0f0f0;">{cells}</tr>'

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    _dl_btn(df, "executive_recovery.xlsx", "dl_exec_recovery")


def _render_concentration(fig_treemap, fleet: dict, top_accounts: pd.DataFrame) -> None:
    _section("Section 6  -  Concentration & Exposure Map", margin_top="24px")

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_treemap, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Size = SOH (Cr). Color = NPA% (green = low risk → red = high risk). Click a region to drill into its branches.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_fleet, col_top = st.columns(2)

    with col_fleet:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:8px;">Fleet Operator Exposure (3+ Loans per Customer)</div>', unsafe_allow_html=True)
        cnt     = fleet.get("count", 0)
        soh     = fleet.get("total_soh_cr", 0.0)
        npa_ops = fleet.get("npa_operators", 0)
        if cnt == 0:
            st.info("No fleet operators found (no customer with 3+ loans). Uses Cust Mob No as customer identifier.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Fleet Operators</div>'
                    f'<div class="kpi-value">{cnt:,}</div>'
                    f'<div class="kpi-mom">≥3 loans per customer</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="kpi-card" style="border-top-color:#dc2626;"><div class="kpi-label">SOH</div>'
                    f'<div class="kpi-value" style="color:#dc2626;">₹{soh:.2f}Cr</div>'
                    f'<div class="kpi-mom">Fleet total exposure</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                npa_color = "#dc2626" if npa_ops > 0 else "#16a34a"
                st.markdown(
                    f'<div class="kpi-card" style="border-top-color:{npa_color};"><div class="kpi-label">Operators with NPA</div>'
                    f'<div class="kpi-value" style="color:{npa_color};">{npa_ops}</div>'
                    f'<div class="kpi-mom">≥1 NPA loan in fleet</div></div>',
                    unsafe_allow_html=True,
                )
            top_fleet = fleet.get("top_df", pd.DataFrame())
            if not top_fleet.empty:
                st.caption("⚠️ Customer identity uses Cust Mob No  -  same person with different numbers may appear separately.")
                with st.expander("Top 20 Fleet Operators by SOH", expanded=False):
                    st.dataframe(_safe_df(top_fleet), use_container_width=True, hide_index=True)
                    _dl_btn(top_fleet, "fleet_operators.xlsx", "dl_fleet")

    with col_top:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:8px;">Top 20 Accounts by SOH</div>', unsafe_allow_html=True)
        if not top_accounts.empty:
            with st.expander("View Top 20 Accounts", expanded=False):
                st.dataframe(_safe_df(top_accounts), use_container_width=True, hide_index=True)
                _dl_btn(top_accounts, "top_accounts_soh.xlsx", "dl_top_accounts")
        else:
            st.info("SOH column not found.")


# ── Risk Indicators table ─────────────────────────────────────────────────────

def _render_risk_indicators(indicators: list[dict]) -> None:
    _section("Section 5b  -  Is the Risk Profile Changing?", margin_top="24px")
    if not indicators:
        st.info("No risk indicators computed.")
        return

    headers = ["Signal", "This Month", "Last Month", "Δ", "Direction", "Note"]
    th = "".join(
        f'<th style="background:#111;color:#FFC000;padding:7px 12px;font-size:11px;'
        f'text-align:{"left" if h in ("Signal","Note") else "center"};white-space:nowrap;">{h}</th>'
        for h in headers
    )
    rows_html = ""
    _DIR_COLOR = {"Improving": "#16a34a", "Worsening": "#dc2626", "Stable": "#d97706", " - ": "#9ca3af"}
    for ind in indicators:
        d = ind["Direction"]
        dc = _DIR_COLOR.get(d, "#9ca3af")
        rows_html += (
            f'<tr style="border-bottom:1px solid #f0f0f0;">'
            f'<td style="padding:7px 12px;font-size:12px;font-weight:700;">{ind["Signal"]}</td>'
            f'<td style="padding:7px 12px;font-size:13px;font-weight:800;text-align:center;">{ind["This Month"]}</td>'
            f'<td style="padding:7px 12px;font-size:12px;color:#6b7280;text-align:center;">{ind["Last Month"]}</td>'
            f'<td style="padding:7px 12px;font-size:12px;font-weight:700;color:{dc};text-align:center;">{ind["Δ"]}</td>'
            f'<td style="padding:7px 12px;text-align:center;">{_badge(d)}</td>'
            f'<td style="padding:7px 12px;font-size:11px;color:#6b7280;">{ind["Note"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


# ── Shared: Top-5 region/branch breakdown ─────────────────────────────────────

def _top5_breakdown(df: pd.DataFrame, accent: str = "#ef4444") -> None:
    """Render top-5 region and top-5 branch by account count side-by-side."""
    col_r, col_b = st.columns(2)

    def _table_html(title: str, rows: list[tuple[str, int]], total: int) -> str:
        header = (
            f'<div style="font-size:12px;font-weight:700;color:#6b7280;'
            f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">{title}</div>'
        )
        body = ""
        for rank, (name, cnt) in enumerate(rows, 1):
            pct = cnt / total * 100 if total else 0
            bar_w = max(int(pct * 1.8), 2)
            body += (
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="min-width:16px;font-size:11px;color:#9ca3af;font-weight:700;">#{rank}</span>'
                f'<span style="flex:1;font-size:12px;color:#111;white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis;" title="{name}">{name}</span>'
                f'<div style="width:{bar_w}px;height:6px;background:{accent};border-radius:3px;flex-shrink:0;"></div>'
                f'<span style="min-width:28px;font-size:12px;font-weight:700;color:{accent};text-align:right;">{cnt}</span>'
                f'</div>'
            )
        return f'<div style="background:#fafafa;border-radius:8px;padding:12px 14px;">{header}{body}</div>'

    total = len(df)
    with col_r:
        if "RegionName" in df.columns:
            top = df["RegionName"].value_counts().head(5).items()
            st.markdown(_table_html("Top 5 Regions", list(top), total), unsafe_allow_html=True)
        else:
            st.caption("Region data unavailable")
    with col_b:
        if "Unit" in df.columns:
            top = df["Unit"].value_counts().head(5).items()
            st.markdown(_table_html("Top 5 Branches", list(top), total), unsafe_allow_html=True)
        else:
            st.caption("Branch data unavailable")


# ── Repossession Analysis ─────────────────────────────────────────────────────

def _render_repossession(repo_df: pd.DataFrame) -> None:
    _section("Section 7  -  Repossession Priority List", margin_top="24px")
    st.caption(
        "Accounts in SMA-2 or NPA bucket sanctioned within the last 18 months. "
        "These still have collateral value  -  act now before the asset depreciates further."
    )

    if repo_df.empty:
        st.info("No accounts match repossession criteria (SMA-2/NPA + sanctioned ≤ 18 months ago).")
        return

    n_total = len(repo_df)
    soh_col  = "SOH" in repo_df.columns
    total_soh = repo_df["SOH"].sum() / 1e7 if soh_col else 0.0

    c1, c2, c3 = st.columns(3)
    sma2_n = int((repo_df["curr_bucket"] == "SMA-2").sum()) if "curr_bucket" in repo_df.columns else 0
    npa_n  = int((repo_df["curr_bucket"] == "NPA").sum())  if "curr_bucket" in repo_df.columns else 0
    with c1:
        st.markdown(
            f'<div class="kpi-card" style="border-top-color:#ef4444;">'
            f'<div class="kpi-label">Eligible Accounts</div>'
            f'<div class="kpi-value" style="color:#ef4444;">{n_total:,}</div>'
            f'<div class="kpi-mom">SMA-2: {sma2_n:,} | NPA: {npa_n:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card" style="border-top-color:#dc2626;">'
            f'<div class="kpi-label">Total SOH at Risk</div>'
            f'<div class="kpi-value" style="color:#dc2626;">₹{total_soh:.2f}Cr</div>'
            f'<div class="kpi-mom">Asset value to be recovered</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        lcc_col = "LCC%" in repo_df.columns
        avg_lcc = repo_df["LCC%"].dropna().mean() if lcc_col else 0.0
        lcc_color = "#dc2626" if avg_lcc < 50 else "#d97706"
        st.markdown(
            f'<div class="kpi-card" style="border-top-color:{lcc_color};">'
            f'<div class="kpi-label">Avg LCC %</div>'
            f'<div class="kpi-value" style="color:{lcc_color};">{avg_lcc:.1f}%</div>'
            f'<div class="kpi-mom">Lower = worse payer history</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    _top5_breakdown(repo_df, accent="#ef4444")
    st.markdown("<br>", unsafe_allow_html=True)

    sub_tabs = st.tabs(["By Recency", "By LCC% (Worst Payers)", "By SOH (Largest Exposure)"])

    with sub_tabs[0]:
        st.caption("Newest delinquent accounts first  -  recently sanctioned but already deep in arrears. Highest urgency.")
        if "Ag_Date" in repo_df.columns:
            view = repo_df.sort_values("Ag_Date", ascending=False).reset_index(drop=True)
        else:
            view = repo_df.copy()
        st.dataframe(_safe_df(view), use_container_width=True, hide_index=True)
        _dl_btn(view, "repo_by_recency.xlsx", "dl_repo_recency")

    with sub_tabs[1]:
        st.caption("Worst payment history first (LCC% ascending). Chronically non-paying accounts  -  least likely to self-cure.")
        if "LCC%" in repo_df.columns:
            view = repo_df.sort_values("LCC%", ascending=True).reset_index(drop=True)
        else:
            view = repo_df.copy()
        st.dataframe(_safe_df(view), use_container_width=True, hide_index=True)
        _dl_btn(view, "repo_by_lcc.xlsx", "dl_repo_lcc")

    with sub_tabs[2]:
        st.caption("Largest SOH exposure first  -  accounts where repossession recovers the most. Prioritise field resources here.")
        if "SOH" in repo_df.columns:
            view = repo_df.sort_values("SOH", ascending=False).reset_index(drop=True)
        else:
            view = repo_df.copy()
        st.dataframe(_safe_df(view), use_container_width=True, hide_index=True)
        _dl_btn(view, "repo_by_soh.xlsx", "dl_repo_soh")


# ── Main render entry point ───────────────────────────────────────────────────

def render_portfolio_intelligence_tab(
    pulse_kpis: list,
    fig_waterfall,
    region_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    fig_quadrant,
    exec_recovery_df: pd.DataFrame,
    product_data: dict,
    risk_indicators: list,
    good_bad: dict,
    flag_df: pd.DataFrame,
    fig_treemap,
    fleet: dict,
    top_accounts: pd.DataFrame,
    alerts_curr: list,
    has_prev: bool,
    rr_meta: dict | None,
    repo_df: pd.DataFrame | None = None,
    npa_sma2_cmp: dict | None = None,
    good_customers: pd.DataFrame | None = None,
) -> None:
    if not has_prev:
        st.info(
            "**Previous month file not loaded.** Upload it to unlock trend analysis (Δ NPA%, roll rates, recovery leaderboard, Good vs Bad). "
            "All current-month views are available now."
        )

    _render_pulse(pulse_kpis, fig_waterfall, rr_meta, has_prev)
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_scorecard_section(region_df, branch_df, fig_quadrant, exec_recovery_df, has_prev, npa_sma2_cmp or {})
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_good_bad(good_bad, has_prev)
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_risk_flags(flag_df, alerts_curr)
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_vintage_sourcing(product_data)
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_risk_indicators(risk_indicators)
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_concentration(fig_treemap, fleet, top_accounts)
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_repossession(repo_df if repo_df is not None else pd.DataFrame())
    st.markdown('<div style="border-top:1px solid #e5e7eb;margin:24px 0;"></div>', unsafe_allow_html=True)

    _render_good_customers(good_customers if good_customers is not None else pd.DataFrame())


# ── Section 8: Good Customers ─────────────────────────────────────────────────

def _render_good_customers(good_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Section 8 - Good Customers: Refinance &amp; Relationship Candidates</div>', unsafe_allow_html=True)
    st.caption("Criteria: 70%+ tenure completed AND LCC% >= 100%. Flag for refinance offer or relationship management.")

    if good_df.empty:
        st.info("No accounts meet the good customer criteria (70%+ tenure + LCC% >= 100%) in the current selection.")
        return

    n = len(good_df)
    soh_col = "SOH" if "SOH" in good_df.columns else None
    tenure_col = "Tenure Completed %" if "Tenure Completed %" in good_df.columns else None

    total_soh_cr = round(pd.to_numeric(good_df[soh_col], errors="coerce").sum() / 1e7, 2) if soh_col else 0.0
    avg_tenure = round(pd.to_numeric(good_df[tenure_col], errors="coerce").mean(), 1) if tenure_col else 0.0

    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi-card">'
        f'<div class="kpi-label">Good Customers</div>'
        f'<div class="kpi-value">{n:,}</div>'
        f'<div class="kpi-mom">Refinance eligible</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-label">Total SOH (Cr)</div>'
        f'<div class="kpi-value">&#8377;{total_soh_cr:,.2f}</div>'
        f'<div class="kpi-mom">Outstanding principal</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-label">Avg Tenure Completed</div>'
        f'<div class="kpi-value">{avg_tenure}%</div>'
        f'<div class="kpi-mom">Across all good customers</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _top5_breakdown(good_df, accent="#16a34a")
    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(_safe_df(good_df), use_container_width=True, hide_index=True)
    _dl_btn(good_df, "good_customers.xlsx", "dl_good_customers")

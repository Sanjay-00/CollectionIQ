"""The Dashboard tab's summary of the whole portfolio, below the KPI cards:
what changed and where, early warning, league tables, business, alerts, and
what to do. Each section is short and links to the tab with the detail.
Numbers come from analysis/summary.py and analysis/action_center.py; this
file only lays them out."""
import pandas as pd
import streamlit as st

from config import DASHBOARD_TOP_N
from ui.components import _dl_btn, _esc, _safe_df, goto_tab_button, heat_range, heat_style
from ui.glossary import help_for, info_icon

_TONE = {"bad": ("#dc2626", "#fef2f2"), "good": ("#16a34a", "#f0fdf4"), "info": ("#d97706", "#fffbeb")}


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_action(_df_c: pd.DataFrame, _df_p: pd.DataFrame, data_version: int, filter_key: str) -> dict:
    from analysis.action_center import build
    return build(_df_c, _df_p)


def _label(text: str, top: str = "0px") -> None:
    st.markdown(f'<div class="section-label" style="margin-top:{top};">{text}</div>', unsafe_allow_html=True)


def _header(text: str, tab: str | None, key: str, top: str = "22px") -> None:
    """Section title with a "See details" button on the right."""
    left, right = st.columns([4, 1])
    with left:
        _label(text, top)
    if tab:
        with right:
            st.markdown(f'<div style="height:{top};"></div>', unsafe_allow_html=True)
            goto_tab_button(tab, key)


def _tile(title: str, value: str, sub: str = "", accent: str = "#e5e7eb", help_text: str | None = None) -> str:
    return (f'<div style="border:1px solid #e5e7eb;border-top:4px solid {accent};border-radius:10px;'
            f'padding:9px 12px;background:#fff;height:100%;">'
            f'<div style="font-size:11.5px;font-weight:700;color:#374151;">{_esc(title)}{info_icon(help_text)}</div>'
            f'<div style="font-size:21px;font-weight:800;color:#111827;margin-top:2px;">{value}</div>'
            f'<div style="font-size:11.5px;color:#4b5563;">{sub}</div></div>')


def _arrow(change, worse_if_up: bool = True, unit: str = " pts") -> str:
    if change is None or pd.isna(change):
        return '<span style="color:#6b7280;">-</span>'
    if change == 0:
        return '<span style="color:#6b7280;">no change</span>'
    color = "#dc2626" if (change > 0) == worse_if_up else "#16a34a"
    return f'<span style="color:{color};font-weight:700;">{"▲" if change > 0 else "▼"} {abs(change):.2f}{unit}</span>'


def _grid(cards: list[str], cols: int = 2) -> str:
    return (f'<div style="display:grid;grid-template-columns:repeat({cols},minmax(0,1fr));gap:8px;">'
            f'{"".join(cards)}</div>')


def _pulse_html(pulse: list[dict]) -> str:
    chips = ""
    for x in pulse:
        if x["change"] is None:
            move = '<span style="color:#6b7280;">no last month</span>'
        elif x["change"] == 0:
            move = '<span style="color:#6b7280;">unchanged</span>'
        else:
            color = "#dc2626" if x["worse"] else "#16a34a"
            move = (f'<span style="color:{color};font-weight:700;">'
                    f'{"▲" if x["change"] > 0 else "▼"} {abs(x["change"]):.2f} pts</span>')
        chips += (f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:6px 12px;background:#fff;">'
                  f'<span style="font-size:11px;color:#6b7280;font-weight:600;">{_esc(x["label"])}</span> '
                  f'<span style="font-size:15px;font-weight:800;color:#111827;">{x["now"]:.1f}%</span> {move}</div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">{chips}</div>'


def _headline_cards(items: list[dict]) -> list[str]:
    cards = []
    for h in items:
        line, bg = _TONE[h["tone"]]
        cards.append(f'<div style="border-left:4px solid {line};background:{bg};border-radius:6px;padding:8px 12px;">'
                     f'<div style="font-size:13.5px;font-weight:700;color:#111827;">{_esc(h["title"])}</div>'
                     f'<div style="font-size:12px;color:#374151;margin-top:2px;">{_esc(h["detail"])}</div></div>')
    return cards


def _focus_table(focus: list[dict]) -> str:
    th = "".join(f'<th style="text-align:{a};">{h}</th>' for h, a in
                 (("#", "center"), ("Group", "left"), ("Loans", "right"), ("SOH", "right"),
                  ("If ignored, likely to slip next month", "right"), ("What to do", "left")))
    rows = ""
    for f in focus:
        risk = f["risk"]
        risk_txt = (f'<b>{risk["n"]:,}</b> loans · ₹{risk["soh"] / 1e7:,.2f} Cr' if risk and risk["n"] else
                    '<span style="color:#9ca3af;">-</span>')
        rows += (f'<tr><td style="text-align:center;"><span class="ac-rank">{f["rank"]}</span></td>'
                 f'<td style="font-weight:700;">{_esc(f["name"])}</td>'
                 f'<td style="text-align:right;">{f["loans"]:,}</td>'
                 f'<td style="text-align:right;">₹{f["soh"] / 1e7:,.2f} Cr</td>'
                 f'<td style="text-align:right;color:#991b1b;">{risk_txt}</td>'
                 f'<td style="color:#374151;">{_esc(f["action"])}</td></tr>')
    css = ("<style>.ac-tbl th{background:#111;color:#FFC000;padding:7px 10px;font-size:11px;white-space:nowrap;}"
           ".ac-tbl td{padding:7px 10px;font-size:12.5px;border-bottom:1px solid #f0f0f0;background:#fff;}"
           ".ac-rank{display:inline-block;width:22px;height:22px;line-height:22px;border-radius:11px;"
           "background:#111;color:#FFC000;font-weight:800;font-size:11px;}</style>")
    return (f'{css}<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
            f'<table class="ac-tbl" style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
            f'<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>')


def _outlook_card(title: str, x: dict, help_text: str) -> str:
    return (
        f'<div style="border:1px solid #e5e7eb;border-top:4px solid #dc2626;border-radius:10px;padding:10px 14px;background:#fff;">'
        f'<div style="font-size:12px;font-weight:700;color:#374151;">{title}{info_icon(help_text)}</div>'
        f'<div style="font-size:24px;font-weight:800;color:#111827;">about {x["n"]:,}'
        f'<span style="font-size:12px;font-weight:600;color:#6b7280;"> loans (likely {x["low"]:,} to {x["high"]:,})</span></div>'
        f'<div style="font-size:12px;color:#374151;">about ₹{x["soh"] / 1e7:,.2f} Cr SOH</div></div>'
    )


def _attention_cards(units: dict) -> str:
    if not units["worst"]:
        return '<div style="font-size:12.5px;color:#374151;">Nothing stands out against the rest of the portfolio.</div>'
    cards = []
    for u in units["worst"]:
        chips = "".join(f'<span class="ac-chip">{_esc(r)}</span>' for r in u["reasons"])
        region = f' · {_esc(u["region"])}' if u["region"] else ""
        cards.append(f'<div style="border:1px solid #fecaca;border-radius:8px;padding:8px 12px;background:#fff;">'
                     f'<div style="font-size:13px;font-weight:700;color:#111827;">{_esc(u["name"])}'
                     f'<span style="font-weight:500;color:#6b7280;">{region} · {u["accounts"]:,} loans</span></div>'
                     f'<div style="margin-top:5px;">{chips}</div></div>')
    css = ("<style>.ac-chip{display:inline-block;background:#fef2f2;color:#991b1b;border:1px solid #fecaca;"
           "border-radius:10px;padding:1px 8px;margin:2px 4px 2px 0;font-size:11.5px;}</style>")
    html = css + _grid(cards, 2)
    if units["improving"]:
        good = "; ".join(f'<b>{_esc(u["name"])}</b> ({_esc(u["why"])})' for u in units["improving"])
        html += f'<div style="font-size:12px;color:#15803d;margin-top:8px;">Improving: {good}.</div>'
    return html


def _render_call_lists(lists: list[dict], unique: int) -> None:
    with st.expander(f"Call lists ({unique:,} loans; a loan can be in more than one list)", expanded=False):
        if not lists:
            st.caption("No loans need a call.")
            return
        st.caption("Each list shows the columns that prove why the loan is on it (e.g. for insurance-only: "
                   "installment arrears 0, expense arrears above the threshold). Largest SOH first.")
        execs = sorted({str(e) for t in lists if "Executive" in t["loans"].columns for e in t["loans"]["Executive"].dropna()})
        pick = st.selectbox("Executive", ["All", *execs], key="action_call_exec")
        tabs = st.tabs([f'{t["name"]} ({len(t["loans"]):,})' for t in lists])
        for tab, (i, t) in zip(tabs, enumerate(lists, start=1)):
            with tab:
                view = t["loans"] if pick == "All" else t["loans"][t["loans"]["Executive"].astype(str) == pick]
                if view.empty:
                    st.caption("No loans for this executive in this list.")
                    continue
                st.dataframe(_safe_df(view), use_container_width=True, hide_index=True, height=380)
                _dl_btn(view, f"call_list_{i}.xlsx", f"dl_action_calls_{i}")


def _render_moves(moves: dict) -> None:
    """Worse / better: biggest delinquency moves by region and branch."""
    def side(title: str, color: str, key: str) -> str:
        items = ""
        for level in ("Region", "Branch"):
            for u in moves[level][key]:
                region = f' <span style="color:#6b7280;">({_esc(u["region"])})</span>' if u.get("region") else ""
                items += (f'<li style="margin:3px 0;"><b>{_esc(u["name"])}</b>{region} '
                          f'<span style="color:#6b7280;">{level.lower()}</span>: delinquency {u["now"]:.1f}% '
                          f'{_arrow(u["change"])}</li>')
        body = items or '<li style="color:#6b7280;">Nothing moved by much.</li>'
        return (f'<div style="border:1px solid #e5e7eb;border-left:4px solid {color};border-radius:8px;'
                f'padding:8px 12px;background:#fff;"><div style="font-weight:800;font-size:13px;color:{color};">{title}</div>'
                f'<ul style="margin:4px 0 0 16px;padding:0;font-size:12.5px;color:#111827;">{body}</ul></div>')
    st.markdown(_grid([side("Got worse", "#dc2626", "worse"), side("Got better", "#16a34a", "better")], 2),
                unsafe_allow_html=True)


def _render_early_warning(r: dict) -> None:
    from analysis import roll_flow as rf
    s = r.get("roll_summary")
    if not s:
        st.caption("Upload last month's file to see which loans slipped or recovered.")
        return
    tiles = []
    for step in rf.step_labels():
        d = s[step]
        good = step == "Back to STD"
        tiles.append(_tile(step, f"{d['n']:,}",
                           f"{d['pct']:.1f}% of loans · {d['soh_pct']:.1f}% of SOH",
                           "#16a34a" if good else ("#dc2626" if step in list(rf.ROLL_STEPS)[:2] else "#f59e0b"),
                           help_for(step)))
    out = r.get("outlook")
    if out:
        tiles.append(_tile("Next month: new NPAs", f"~{out['new_npa']['n']:,}",
                           f"likely {out['new_npa']['low']:,} to {out['new_npa']['high']:,} · ₹{out['new_npa']['soh'] / 1e7:,.1f} Cr",
                           "#7f1d1d", "Expected if this month's roll rates repeat."))
    st.markdown(_grid(tiles, len(tiles)), unsafe_allow_html=True)


_LEAGUE_FMT = {"Collection %", "Strike %"}


def _render_league(df_curr, df_prev, data_version: int, filter_key: str) -> None:
    from analysis.summary import LEAGUE_METRICS
    c1, c2, c3 = st.columns([2, 1, 1])
    metric = c1.selectbox("Metric", list(LEAGUE_METRICS), key="dash_league_metric")
    grain = c2.selectbox("Level", ["Region", "Branch", "Executive"], index=1, key="dash_league_level")
    side = c3.radio("Show", ["Worst", "Best"], horizontal=True, key="dash_league_side")
    table, ranked = _cached_league(df_curr, df_prev, data_version, filter_key, metric, grain, side == "Worst")
    if table.empty:
        st.caption("Not enough data for this view.")
        return
    worse_if_up = LEAGUE_METRICS[metric][1]
    rows = ""
    rng = heat_range(table["Now"].tolist()) if worse_if_up else None
    for _, r in table.iterrows():
        prev = "-" if r["Last Month"] is None or pd.isna(r["Last Month"]) else f'{r["Last Month"]:.1f}%'
        prank = r["Rank Last Month"]
        rank_move = "" if prank is None or pd.isna(prank) else (
            f' <span style="color:#6b7280;font-size:11px;">(was {int(prank)})</span>')
        region = f'<td>{_esc(r["Region"])}</td>' if "Region" in table.columns else ""
        shade = heat_style(r["Now"], rng) if rng else ""
        delq = "" if metric == "Delinquency %" else f'<td style="text-align:right;color:#374151;">{r["Delinquency %"]:.1f}%</td>'
        rows += (f'<tr><td style="text-align:center;">{int(r["Rank"])}{rank_move}</td>'
                 f'<td style="font-weight:700;">{_esc(r[grain])}</td>{region}'
                 f'<td style="text-align:right;">{int(r["Accounts"]):,}</td>'
                 f'<td style="text-align:right;{shade}">{r["Now"]:.1f}%</td>'
                 f'<td style="text-align:right;color:#4b5563;">{prev}</td>'
                 f'<td style="text-align:right;">{_arrow(r["Change"], worse_if_up)}</td>{delq}</tr>')
    head = "".join(f"<th>{h}</th>" for h in ["Rank", grain, *(["Region"] if "Region" in table.columns else []),
                                             "Accounts", metric, "Last Month", "Change",
                                             *([] if metric == "Delinquency %" else ["Delinquency %"])])
    st.markdown(f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;">'
                f'<table class="ac-tbl" style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
                f'<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    st.caption(f"{side} {len(table)} of {ranked} {grain.lower()}s ranked by {metric}"
               f"{'' if grain == 'Region' else ' (units too small to judge fairly are left out)'}. "
               "Rank 1 is the " + ("worst." if side == "Worst" else "best."))


def _render_business(b: dict) -> None:
    if not b.get("month"):
        st.caption("No agreement dates in this file.")
        return
    mom = (lambda v: "" if v is None else _arrow(v, worse_if_up=False, unit="%"))
    tiles = [
        _tile(f"New loans ({b['month']})", f"{b['accounts']:,}", f"{mom(b['accounts_mom_pct'])} vs last month"),
        _tile("Amount funded", f"₹{b['funded_cr']:,.2f} Cr", f"{mom(b['funded_mom_pct'])} vs last month"),
        _tile("Average ticket", f"₹{b['avg_ticket_l']:,.2f} L"),
    ]
    st.markdown(_grid(tiles, 3), unsafe_allow_html=True)
    lines = []
    if b["top_segments"]:
        lines.append("Top segments: " + ", ".join(f"<b>{_esc(x['Segment'])}</b> ({x['Accounts']:,})" for x in b["top_segments"]))
    if b["top_branches"]:
        lines.append("Top branches: " + ", ".join(f"<b>{_esc(x['Branch'])}</b> ({x['Accounts This Month']:,})" for x in b["top_branches"]))
    if lines:
        st.markdown(f'<div style="font-size:12.5px;color:#374151;margin-top:6px;">{"<br>".join(lines)}</div>',
                    unsafe_allow_html=True)


_SEV = {"critical": "#dc2626", "high": "#f97316", "medium": "#d97706", "low": "#6b7280"}


def _render_alerts(alerts: list[dict]) -> None:
    if not alerts:
        st.caption("No alerts.")
        return
    chips = ""
    for a in alerts:
        delta = "" if a["prev"] is None else f' {_arrow(a["count"] - a["prev"], unit="")}'
        chips += (f'<div style="border:1px solid #e5e7eb;border-left:4px solid {_SEV.get(a["severity"], "#6b7280")};'
                  f'border-radius:8px;padding:6px 10px;background:#fff;">'
                  f'<div style="font-size:11.5px;color:#374151;font-weight:600;">{_esc(a["title"])}</div>'
                  f'<div style="font-size:16px;font-weight:800;color:#111827;">{a["count"]:,}{delta}</div></div>')
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;">{chips}</div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_league(_df_c, _df_p, data_version: int, filter_key: str, metric: str, grain: str, worst: bool):
    from analysis.summary import league_table
    return league_table(_df_c, _df_p, metric, grain, worst, DASHBOARD_TOP_N)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_summary(_df_c, _df_p, data_version: int, filter_key: str, as_of: str) -> dict:
    from analysis import summary as sm
    return {
        "moves": {lvl: sm.biggest_moves(_df_c, _df_p, lvl, 2 if lvl == "Region" else 3) for lvl in ("Region", "Branch")},
        "business": sm.business_snapshot(_df_c, as_of),
    }


def render_summary(df_curr: pd.DataFrame, df_prev: pd.DataFrame, data_version: int, filter_key: str,
                   curr_month: str, alerts: list, alerts_prev: list) -> None:
    """Everything below the KPI cards on the Dashboard tab."""
    from analysis.summary import alert_snapshot
    from ui.components import TAB_LABELS
    tab = {t.split(" ", 1)[1]: t for t in TAB_LABELS}
    r = _cached_action(df_curr, df_prev, data_version, filter_key)
    sm = _cached_summary(df_curr, df_prev, data_version, filter_key, str(curr_month))
    has_prev = r["has_prev"]

    _header("What Changed, and Where", tab["Portfolio Intelligence"], "go_pi_moves")
    if has_prev:
        st.caption("Regions and branches whose delinquency moved most since last month.")
        _render_moves(sm["moves"])
    else:
        st.caption("Upload last month's file to compare.")

    _header("Early Warning: Loans That Slipped", tab["Migration"], "go_migration")
    _render_early_warning(r)

    _header("League Table", tab["Portfolio Intelligence"], "go_pi_league")
    _render_league(df_curr, df_prev, data_version, filter_key)

    _header("New Business", tab["Business"], "go_business")
    _render_business(sm["business"])

    _header("Risk Alerts", tab["Alerts"], "go_alerts")
    _render_alerts(alert_snapshot(alerts, alerts_prev))

    _header("What To Do First", None, "")
    st.caption("Today's loans, cheapest and most urgent first. \"If ignored\" uses this month's roll rate for each bucket.")
    if r["focus"]:
        st.markdown(_focus_table(r["focus"]), unsafe_allow_html=True)
    if r["attention"]["branch"]["worst"] or r["attention"]["executive"]["worst"]:
        with st.expander("Branches and executives needing attention", expanded=False):
            tab_b, tab_e = st.tabs(["Branches", "Executives"])
            with tab_b:
                st.markdown(_attention_cards(r["attention"]["branch"]), unsafe_allow_html=True)
            with tab_e:
                st.markdown(_attention_cards(r["attention"]["executive"]), unsafe_allow_html=True)
    from analysis.action_center import unique_call_count
    _render_call_lists(r["calls"], unique_call_count(r["calls"]))

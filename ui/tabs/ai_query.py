import os

import pandas as pd
import streamlit as st

from utils import fmt_value
from ui.components import _dl_btn, _safe_df, _send_feedback, _kpi_card_html

# result_grain -> (singular, plural) display noun, used wherever the UI used to
# hardcode "accounts"/"Customer Records" regardless of the result's actual row
# grain (a region_scorecard result is 6 regions, not 6 accounts).
_GRAIN_LABELS = {
    "loan":      ("account", "accounts"),
    "customer":  ("customer", "customers"),
    "region":    ("region", "regions"),
    "branch":    ("branch", "branches"),
    "executive": ("executive", "executives"),
    "segment":   ("segment", "segments"),
    "signal":    ("signal", "signals"),
    "matrix":    ("row", "rows"),
    "portfolio": ("KPI", "KPIs"),
}


def _grain_noun(grain: str, plural: bool = True) -> str:
    singular, plural_noun = _GRAIN_LABELS.get(grain, _GRAIN_LABELS["loan"])
    return plural_noun if plural else singular


def render_ai_query_tab(
    df_curr: pd.DataFrame,
    snapshot_dates: dict | None = None,
    df_prev: pd.DataFrame | None = None,
    precomputed_views: dict | None = None,
    alerts_curr: list | None = None,
    alerts_prev: list | None = None,
    rr_meta: dict | None = None,
) -> None:
    from graph import run_query

    # ── Example chips (cross-frame JS fill) ──────────────────────────────────
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
    background: #161b22; color: #8b949e;
    border: 1px solid #2d333b; border-radius: 6px;
    padding: 5px 12px; font-size: 11px; font-style: italic;
    cursor: pointer; margin: 0 6px 0 0;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    line-height: 1.4; outline: none;
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
  <button class="chip" onclick="fill('Give me all branches with previous NPA count and current NPA count, sorted descending with the biggest reduction on top')">Branches with the biggest NPA drop vs last month</button>
  <button class="chip" onclick="fill('Show all accounts with arrears greater than 2 EMI from November 2025 onward advances')">Show all accounts &gt;2 bucket from Nov 2025 onward advances</button>
  <button class="chip" onclick="fill('Show those accounts that need immediate action')">Show accounts that need immediate action</button>
</div>
<script>
function fill(text) {
    var doc = window.parent.document;
    var ta  = doc.querySelector('[data-testid="stTextArea"] textarea');
    if (!ta) return;
    var set = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
    set.call(ta, text);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
}
</script>
""", height=200, scrolling=False)

    ai_query = st.text_area(
        "Query", key="ai_query_input",
        placeholder="Type your question here...",
        height=90, label_visibility="collapsed",
    )

    col_run, col_hint = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🔍  Run Query", type="primary", width='stretch')
    with col_hint:
        st.markdown(
            "<div style='padding-top:10px;font-size:12px;color:#aaa;'>"
            "Powered by Gemini 2.5 Flash · LangGraph multi-agent pipeline</div>",
            unsafe_allow_html=True,
        )

    if run_btn:
        if not ai_query.strip():
            st.warning("Please enter a question.")
        elif not os.environ.get("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY not found in .env file.")
        else:
            with st.status("Running AI pipeline...", expanded=True) as _status:
                def _on_step(label: str) -> None:
                    _status.write(label)
                _ai_result = run_query(ai_query.strip(), df_curr, on_step=_on_step,
                                       snapshot_dates=snapshot_dates, df_prev=df_prev,
                                       precomputed_views=precomputed_views,
                                       alerts_curr=alerts_curr, alerts_prev=alerts_prev,
                                       rr_meta=rr_meta)
                _status.update(label="Query complete", state="complete", expanded=False)
            st.session_state["ai_result"] = _ai_result

    # ── Render result ─────────────────────────────────────────────────────────
    result = st.session_state.get("ai_result")
    if not result:
        return

    if result.get("error"):
        st.error(f"Query failed: {result['error']}")
        return

    # ── Clarification: query was ambiguous  -  ask instead of guessing ───────────
    if result.get("needs_clarification"):
        q_question = result.get("clarification_question") or "Your query could be read a few ways  -  which did you mean?"
        q_options  = result.get("clarification_options") or []
        orig_query = result.get("query") or ""

        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #FFC000;border-radius:12px;
                    padding:16px 20px;margin:16px 0 10px 0;">
          <div style="font-size:13px;font-weight:800;color:#FFC000;margin-bottom:8px;letter-spacing:1px;">
            🤔 NEED A QUICK CLARIFICATION
          </div>
          <div style="font-size:13px;color:#e6edf3;line-height:1.6;">{q_question}</div>
        </div>
        """, unsafe_allow_html=True)

        for i, opt in enumerate(q_options):
            if st.button(opt, key=f"clarify_opt_{i}", width='stretch'):
                augmented = f"{orig_query} (interpretation: {opt})"
                with st.status("Running AI pipeline...", expanded=True) as _status:
                    def _on_step(label: str) -> None:
                        _status.write(label)
                    _res = run_query(augmented, df_curr, on_step=_on_step,
                                     snapshot_dates=snapshot_dates, allow_clarification=False,
                                     df_prev=df_prev, precomputed_views=precomputed_views,
                                     alerts_curr=alerts_curr, alerts_prev=alerts_prev,
                                     rr_meta=rr_meta)
                    _status.update(label="Query complete", state="complete", expanded=False)
                st.session_state["ai_result"] = _res
                st.rerun()

        st.caption("None of these? Rephrase your question above with the detail you meant, and run again.")
        return

    filtered_df = result["result_df"]
    kpis_q      = result["result_kpis"]
    rankings    = result["result_rankings"]
    insights    = result.get("insights") or ""
    plain       = (result.get("parsed_filters") or {}).get("plain_english") or ""

    is_priority    = result.get("priority_mode", False)
    is_aggregation = result.get("aggregation_mode", False)
    result_type    = result.get("result_type") or "loan_table"
    result_grain   = result.get("result_grain") or "loan"
    view_render    = result.get("view_render") or ""
    category       = (result.get("query_category") or "general").replace("_", " ").title()
    query_title    = result.get("query_title") or ""
    enriched       = result.get("enriched_query") or ""
    risk_flag      = result.get("risk_flag") or "medium"
    risk_color     = {"high": "#dc2626", "medium": "#d97706", "low": "#16a34a"}.get(risk_flag, "#d97706")
    risk_label     = {"high": "🔴 High Risk", "medium": "🟡 Medium Risk", "low": "🟢 Low Risk"}.get(risk_flag, "🟡 Medium Risk")

    # ── Domain Expert card ────────────────────────────────────────────────────
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

    # ── Fast-path view: KPI cards (e.g. portfolio pulse) ──────────────────────
    if view_render == "kpi_cards":
        rows = (kpis_q.get("_agg_rows") or []).copy()
        # _agg_rows is capped at 5 by view_node (shared with the insight-generator
        # context); re-derive the FULL card set straight from filtered_df so every
        # KPI is shown, not just the first 5.
        if len(filtered_df) and {"label", "value", "delta"}.issubset(filtered_df.columns):
            rows = filtered_df.to_dict(orient="records")
        cards_html = "".join(
            _kpi_card_html(
                r.get("label", ""), r.get("value", ""), r.get("delta"),
                unit=r.get("unit") or "%", inverse=bool(r.get("inverse")),
            )
            for r in rows
        )
        st.markdown(f'<div class="kpi-row" style="flex-wrap:wrap;">{cards_html}</div>', unsafe_allow_html=True)

        # Comparison table: This Month | Previous Month | Delta | % Change, using
        # the raw (unformatted) curr_raw/prev_raw numbers so Delta/%Change are the
        # literal arithmetic difference, not the sign-flipped-for-card-color delta.
        if rows and any(r.get("prev_raw") is not None for r in rows):
            def _fmt_side(val, unit):
                if val is None:
                    return "-"
                if unit == "Cr":
                    return f"₹{val:.2f}Cr"
                if unit == "%":
                    return f"{val:.2f}%"
                return f"{val:,.0f}"

            table_rows = ""
            for r in rows:
                cv, pv = r.get("curr_raw"), r.get("prev_raw")
                unit, inverse = r.get("unit") or "", bool(r.get("inverse"))
                if pv is None:
                    delta_cell, pct_cell = "-", "-"
                    color = "#9ca3af"
                else:
                    diff = cv - pv
                    pct = (diff / pv * 100) if pv else None
                    good = (diff < 0) if inverse else (diff > 0)
                    color = "#16a34a" if (diff != 0 and good) else ("#dc2626" if diff != 0 else "#9ca3af")
                    arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "-")
                    delta_cell = f"{arrow} {abs(diff):,.2f}" if unit != "" else f"{arrow} {abs(int(diff)):,}"
                    pct_cell = f"{pct:+.1f}%" if pct is not None else "-"
                table_rows += (
                    f'<tr style="background:#161b22;border-bottom:1px solid #0d1117;">'
                    f'<td style="padding:8px 14px;font-size:12px;font-weight:700;color:#e6edf3;">{r.get("label","")}</td>'
                    f'<td style="padding:8px 14px;font-size:13px;font-weight:800;text-align:right;color:#e6edf3;">{_fmt_side(cv, unit)}</td>'
                    f'<td style="padding:8px 14px;font-size:12px;text-align:right;color:#8b949e;">{_fmt_side(pv, unit)}</td>'
                    f'<td style="padding:8px 14px;font-size:12px;font-weight:700;text-align:right;color:{color};">{delta_cell}</td>'
                    f'<td style="padding:8px 14px;font-size:12px;font-weight:700;text-align:right;color:{color};">{pct_cell}</td>'
                    f'</tr>'
                )
            st.markdown(
                "<div style='margin-top:20px;font-size:11px;font-weight:700;color:#888;"
                "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>This Month vs Previous Month</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #21262d;border-radius:10px;overflow-x:auto;">'
                f'<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr>'
                f'<th style="background:#0d1117;color:#FFC000;padding:7px 14px;font-size:11px;text-align:left;">KPI</th>'
                f'<th style="background:#0d1117;color:#FFC000;padding:7px 14px;font-size:11px;text-align:right;">This Month</th>'
                f'<th style="background:#0d1117;color:#FFC000;padding:7px 14px;font-size:11px;text-align:right;">Previous Month</th>'
                f'<th style="background:#0d1117;color:#FFC000;padding:7px 14px;font-size:11px;text-align:right;">Δ</th>'
                f'<th style="background:#0d1117;color:#FFC000;padding:7px 14px;font-size:11px;text-align:right;">% Change</th>'
                f'</tr></thead><tbody>{table_rows}</tbody></table></div>',
                unsafe_allow_html=True,
            )

    # ── Fast-path view: risk indicator signal table ───────────────────────────
    elif view_render == "risk_indicator_table":
        rows = filtered_df.to_dict(orient="records") if len(filtered_df) else []
        _DIR_COLOR = {"Improving": "#16a34a", "Worsening": "#dc2626", "Stable": "#d97706"}
        headers = ["Signal", "This Month", "Last Month", "Δ", "Direction", "Note"]
        th = "".join(
            f'<th style="background:#0d1117;color:#FFC000;padding:7px 12px;font-size:11px;'
            f'text-align:{"left" if h in ("Signal","Note") else "center"};white-space:nowrap;">{h}</th>'
            for h in headers
        )
        rows_html = ""
        for r in rows:
            d = r.get("Direction", "-")
            dc = _DIR_COLOR.get(d, "#9ca3af")
            rows_html += (
                f'<tr style="background:#161b22;border-bottom:1px solid #0d1117;">'
                f'<td style="padding:7px 12px;font-size:12px;font-weight:700;color:#e6edf3;">{r.get("Signal","")}</td>'
                f'<td style="padding:7px 12px;font-size:13px;font-weight:800;text-align:center;color:#e6edf3;">{r.get("This Month","")}</td>'
                f'<td style="padding:7px 12px;font-size:12px;color:#8b949e;text-align:center;">{r.get("Last Month","")}</td>'
                f'<td style="padding:7px 12px;font-size:12px;font-weight:700;color:{dc};text-align:center;">{r.get("Δ","")}</td>'
                f'<td style="padding:7px 12px;text-align:center;color:{dc};font-weight:700;font-size:12px;">{d}</td>'
                f'<td style="padding:7px 12px;font-size:11px;color:#8b949e;">{r.get("Note","")}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #21262d;border-radius:10px;overflow-x:auto;">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    # ── Priority mode ─────────────────────────────────────────────────────────
    elif is_priority:
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
div[data-testid="stSelectbox"] [data-baseweb="select"] span { color: #FFC000 !important; }
</style>""", unsafe_allow_html=True)

        sel_col, _ = st.columns([1, 3])
        with sel_col:
            n_accounts = st.selectbox(
                "How many accounts to review?",
                options=[20, 30, 40, 50], index=1, key="priority_n",
            )

        distributed = distribute_priority_accounts(filtered_df, n_accounts)
        for priority_label, grp in distributed.groupby("Priority", sort=False):
            p_num   = priority_label.split(":")[0].strip()
            p_name  = priority_label.split(":")[-1].strip()
            p_color = "#dc2626" if p_num in ("P1", "P5") else "#f97316" if p_num in ("P2", "P3", "P6", "P7") else "#d97706"

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:16px 0 6px 0;">
              <span style="background:{p_color};color:#fff;font-size:11px;font-weight:800;
                           padding:3px 10px;border-radius:12px;">{p_num}</span>
              <span style="font-size:14px;font-weight:700;color:#1a1a1a;">{p_name}</span>
              <span style="font-size:12px;color:#888;">{len(grp)} accounts</span>
            </div>
            """, unsafe_allow_html=True)

            disp = grp.drop(columns=["Priority", "Why", "_rank"], errors="ignore")
            disp = disp.loc[:, ~disp.columns.duplicated()].reset_index(drop=True)
            st.dataframe(_safe_df(disp.head(1000)), width='stretch',
                         height=min(280, 45 + min(len(grp), 1000) * 36), hide_index=True)
            if len(disp) > 1000:
                st.caption(f"Showing 1,000 of {len(disp):,} rows  -  download Excel for full list.")
            _dl_btn(disp, f"priority_{p_num}.xlsx", f"dl_priority_{p_num}")

    elif result.get("plan_mode"):
        # ── Multi-step plan result ────────────────────────────────────────────
        plan = result.get("plan") or []

        def _step_text(s: dict) -> str:
            op = s.get("op")
            if op == "group_aggregate":
                gb = ", ".join(str(c) for c in (s.get("group_by") if isinstance(s.get("group_by"), list) else [s.get("group_by")]))
                aggs = ", ".join(
                    f"{a.get('func')}({a.get('column') or 'rows'}) as {a.get('alias')}"
                    for a in (s.get("aggregations") or [])
                )
                return f"Group by [{gb}] → {aggs}"
            if op == "filter":
                conds = ", ".join(
                    f"{c.get('column')} {c.get('op')} {c.get('value')}"
                    for c in (s.get("conditions") or [])
                )
                return f"Keep rows where {conds}"
            if op == "derive":
                return f"Add column '{s.get('column')}' = {s.get('expr')}"
            if op == "sort":
                return f"Sort by {s.get('by')} ({'ascending' if s.get('ascending') else 'descending'})"
            if op == "limit":
                return f"Take top {s.get('n')}"
            return str(op)

        steps_html = "".join(
            f'<div style="display:flex;gap:10px;margin:4px 0;font-size:12px;color:#94a3b8;">'
            f'<span style="color:#FFC000;font-weight:800;min-width:18px;">{i}.</span>'
            f'<span>{_step_text(s)}</span></div>'
            for i, s in enumerate(plan, start=1)
        )
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #FFC000;border-radius:12px;
                    padding:16px 20px;margin:0 0 16px 0;">
          <div style="font-size:13px;font-weight:800;color:#FFC000;margin-bottom:8px;letter-spacing:1px;">
            🧩 MULTI-STEP PLAN  -  {len(filtered_df)} rows
          </div>
          <div style="font-size:12px;color:#94a3b8;margin-bottom:8px;">{plain}</div>
          {steps_html}
        </div>
        """, unsafe_allow_html=True)

        if len(filtered_df) > 0:
            display_plan = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
            st.dataframe(_safe_df(display_plan.head(1000)), width='stretch',
                         height=min(420, 50 + min(len(display_plan), 1000) * 36), hide_index=True)
            if len(display_plan) > 1000:
                st.caption(f"Showing 1,000 of {len(display_plan):,} rows  -  download Excel for full list.")
            _dl_btn(display_plan, "plan_result.xlsx", "dl_plan")
        else:
            st.warning("The plan returned no rows.")

    elif result_type == "single_stat" and not is_aggregation:
        # ── Scalar result ─────────────────────────────────────────────────────
        _QKIND = {"Count":"count","Total POS":"money","Avg Arrears/EMI":"pct",
                  "Total Demand":"money","Total Collection":"money","Collection %":"pct"}
        kpi_html = "".join(
            f'<div style="min-width:140px;text-align:center;">'
            f'<div style="font-size:11px;color:#6b7280;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:8px;">{k}</div>'
            f'<div style="font-size:42px;font-weight:900;color:#FFC000;line-height:1;letter-spacing:-1px;">'
            f'{fmt_value(v, _QKIND.get(k, "count"))}</div></div>'
            for k, v in kpis_q.items() if v not in (0, 0.0, "")
        )
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:14px;
                    padding:32px 36px;margin:0 0 20px 0;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                      letter-spacing:2px;margin-bottom:16px;">{query_title}</div>
          <div style="display:flex;gap:24px;justify-content:center;flex-wrap:wrap;">{kpi_html}</div>
          <div style="font-size:13px;color:#4b5563;margin-top:20px;font-style:italic;">{plain}</div>
        </div>
        """, unsafe_allow_html=True)

    elif result_type == "single_stat" and is_aggregation:
        # ── Aggregation single-answer ─────────────────────────────────────────
        agg_spec     = result.get("aggregation_spec") or {}
        _metrics     = agg_spec.get("metrics") or []
        metric_label = _metrics[0]["label"] if _metrics else (agg_spec.get("metric_label") or "Metric")
        _gb          = agg_spec.get("group_by") or "Group"
        group_col    = f"{_gb[0]} ({_gb[1]})" if isinstance(_gb, list) else str(_gb)
        sort_asc     = agg_spec.get("sort_asc")
        if sort_asc is None:
            sort_asc = True
        if len(filtered_df) > 0 and metric_label in filtered_df.columns:
            top_row   = filtered_df.iloc[0]
            top_name  = top_row.get(group_col, " - ")
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
        if len(filtered_df) > 1:
            st.markdown("<div style='font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Full Ranking</div>", unsafe_allow_html=True)
            st.dataframe(_safe_df(filtered_df.head(1000)), width='stretch',
                         height=min(400, 50 + min(len(filtered_df), 1000) * 36), hide_index=True)
            if len(filtered_df) > 1000:
                st.caption(f"Showing 1,000 of {len(filtered_df):,} rows - download Excel for full list.")
            _dl_btn(filtered_df, "ranking_result.xlsx", "dl_ranking")

    elif is_aggregation:
        # ── Ranked aggregation table ──────────────────────────────────────────
        agg_spec      = result.get("aggregation_spec") or {}
        _metrics_list = agg_spec.get("metrics") or []
        metric_labels = [m["label"] for m in _metrics_list if m.get("label")]
        metric_label  = metric_labels[0] if metric_labels else (agg_spec.get("metric_label") or "Metric")
        _gb           = agg_spec.get("group_by") or "Group"
        group_col     = f"{_gb[0]} ({_gb[1]})" if isinstance(_gb, list) else str(_gb)
        header_title  = " · ".join(metric_labels) if metric_labels else metric_label

        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #FFC000;border-radius:12px;
                    padding:16px 20px;margin:0 0 16px 0;">
          <div style="font-size:13px;font-weight:800;color:#FFC000;margin-bottom:6px;letter-spacing:1px;">
            📊 AGGREGATION RESULT - {header_title.upper()}
          </div>
          <div style="font-size:12px;color:#94a3b8;">
            {plain}&nbsp; &nbsp;
            <strong style="color:#fff">{len(filtered_df)} {group_col}s</strong> ranked
          </div>
        </div>
        """, unsafe_allow_html=True)

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
                row_bg   = "rgba(255,192,0,0.06)" if rank_val <= 3 else "transparent"
                cells = ""
                for c in header_cols:
                    val = row[c]
                    if c == "Rank":
                        cells += f'<td style="padding:10px 14px;font-weight:800;color:#FFC000;">#{rank_val}</td>'
                    elif c == group_col:
                        cells += f'<td style="padding:10px 14px;font-weight:600;color:#e6edf3;font-size:13px;">{val}</td>'
                    elif c in metric_labels or c == metric_label:
                        _disp = int(val) if isinstance(val, (int, float)) and val == int(val) else round(val, 2)
                        cells += f'<td style="padding:10px 14px;text-align:right;font-weight:800;color:#FFC000;font-size:14px;">{_disp}</td>'
                    else:
                        cells += f'<td style="padding:10px 14px;text-align:right;color:#8b949e;font-size:13px;">{int(val) if isinstance(val, (int, float)) and val == int(val) else val}</td>'
                rows_html += f'<tr style="background:{row_bg};border-bottom:1px solid #0d1117;">{cells}</tr>'

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;overflow-x:auto;margin-top:8px;">
              <table style="width:100%;border-collapse:collapse;min-width:600px;">
                <thead><tr style="background:#0d1117;">{th_cells}</tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)
            _dl_btn(filtered_df, "aggregation_result.xlsx", "dl_aggregation")
        else:
            st.warning("No data returned for this aggregation.")

    else:
        # ── Row-level filter result ───────────────────────────────────────────
        st.markdown(f"""
        <div style="background:#1a2e1a;border-left:4px solid #16a34a;border-radius:8px;
                    padding:12px 16px;margin:0 0 16px 0;color:#86efac;font-weight:600;font-size:14px;">
            ✓ Found <strong style="color:#fff">{kpis_q.get('Count',0)} {_grain_noun(result_grain)}</strong>
            &nbsp; {plain}
        </div>
        """, unsafe_allow_html=True)

        _QKIND = {"Count":"count","Total POS":"money","Avg Arrears/EMI":"pct",
                  "Total Demand":"money","Total Collection":"money","Collection %":"pct"}
        kpi_html = "".join(
            f'<div class="result-kpi"><div class="result-kpi-label">{k}</div>'
            f'<div class="result-kpi-value">{fmt_value(v, _QKIND.get(k, "count"))}</div></div>'
            # Keys starting with "_" (e.g. "_agg_rows") are internal-only, consumed
            # by the insight generator, never meant to render as a KPI card -- they
            # aren't scalars, so fmt_value would crash on them (e.g. abs() on a list).
            for k, v in kpis_q.items() if not str(k).startswith("_")
        )
        st.markdown(
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">{kpi_html}</div>',
            unsafe_allow_html=True,
        )

        def _rank_rows(items, val_fmt="count"):
            rows = ""
            for name, val in list(items)[:5]:
                display = fmt_value(val, val_fmt) if val_fmt == "money" else f"{int(val)}"
                rows += (
                    f'<div class="rank-row">'
                    f'<span class="rank-name">{name}</span>'
                    f'<span class="rank-value">{display}</span>'
                    f'</div>'
                )
            return rows

        r1, r2 = st.columns(2)
        with r1:
            left_html = ""
            if rankings.get("region_counts"):
                left_html += (
                    f'<div class="rank-card" style="margin-bottom:12px;">'
                    f'<div class="rank-title">🗺 Top Regions by Account Count</div>'
                    f'{_rank_rows(rankings["region_counts"].items())}</div>'
                )
            if rankings.get("bucket_dist"):
                bucket_rows = "".join(
                    f'<div class="rank-row">'
                    f'<span class="rank-name">{b}</span>'
                    f'<span class="rank-value">{p}%</span>'
                    f'</div>'
                    for b, p in rankings["bucket_dist"].items()
                )
                left_html += (
                    f'<div class="rank-card" style="margin-bottom:12px;">'
                    f'<div class="rank-title">📊 Bucket Distribution</div>'
                    f'{bucket_rows}</div>'
                )
            if rankings.get("mnt_details"):
                mnt_rows = "".join(
                    f'<div class="rank-row" style="gap:6px;">'
                    f'<span class="rank-name" style="flex:1.4;font-weight:600;">{e["name"]}</span>'
                    f'<span class="rank-name" style="flex:0.9;color:#9ca3af;font-size:11px;">{e["branch"]}</span>'
                    f'<span class="rank-value" style="min-width:36px;text-align:right;">{e["count"]}</span>'
                    f'<span class="rank-value" style="min-width:52px;text-align:right;color:#FFC000;">{fmt_value(e["pos"], "money")}</span>'
                    f'</div>'
                    for e in rankings["mnt_details"]
                )
                left_html += (
                    f'<div class="rank-card">'
                    f'<div class="rank-title">👤 Top Executives by Account Count</div>'
                    f'{mnt_rows}</div>'
                )
            st.markdown(left_html, unsafe_allow_html=True)

        with r2:
            right_html = ""
            if rankings.get("branch_counts"):
                right_html += (
                    f'<div class="rank-card" style="margin-bottom:12px;">'
                    f'<div class="rank-title">🏢 Top Branches by Account Count</div>'
                    f'{_rank_rows(rankings["branch_counts"].items())}</div>'
                )
            if rankings.get("branch_pos"):
                right_html += (
                    f'<div class="rank-card">'
                    f'<div class="rank-title">💰 Top Branches by POS</div>'
                    f'{_rank_rows(rankings["branch_pos"].items(), "money")}</div>'
                )
            st.markdown(right_html, unsafe_allow_html=True)

        st.markdown(
            f"<div style='margin-top:20px;font-size:11px;font-weight:700;color:#888;"
            f"text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Matching {_grain_noun(result_grain).title()} Records</div>",
            unsafe_allow_html=True,
        )
        display_filtered = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
        st.dataframe(_safe_df(display_filtered.head(1000)), width='stretch', height=320, hide_index=True)
        if len(display_filtered) > 1000:
            st.caption(f"Showing 1,000 of {len(display_filtered):,} rows - download Excel for full list.")
        _dl_btn(display_filtered, "filtered_accounts.xlsx", "dl_filter_table")

    # ── AI Observations ───────────────────────────────────────────────────────
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

    # ── LangSmith feedback ────────────────────────────────────────────────────
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

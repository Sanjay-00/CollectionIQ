"""
Report Builder — assembles a fully self-contained HTML report.
No CDN, no external JS. All CSS inline. Safe for email clients.
"""
import datetime
from report_agent.state import ReportState

YELLOW = "#FFC000"
DARK   = "#0d1117"

BASE_CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
      background:#f0f0f0;color:#111827;font-size:14px;line-height:1.5;}}

/* ── Banner + Header ── */
.banner{{height:5px;background:linear-gradient(90deg,{YELLOW},#FFD740,{YELLOW});}}
.report-header{{background:#111827;padding:26px 36px;display:flex;align-items:center;
                justify-content:space-between;gap:20px;border-bottom:2px solid {YELLOW};}}
.brand-name{{font-size:22px;font-weight:900;color:{YELLOW};letter-spacing:3px;line-height:1;}}
.brand-sub{{font-size:10px;color:#6b7280;letter-spacing:1.5px;margin-top:3px;font-weight:500;}}
.report-title{{font-size:17px;font-weight:700;color:#fff;margin-bottom:3px;}}
.report-meta{{font-size:11px;color:#6b7280;line-height:1.9;}}
.month-badge{{display:inline-block;background:{YELLOW};color:#000;font-size:11px;
              font-weight:800;padding:4px 14px;border-radius:20px;letter-spacing:1px;margin-top:6px;}}

/* ── Content wrapper ── */
.content{{padding:26px 36px;max-width:1200px;margin:0 auto;}}

/* ── Section label ── */
.sec-label{{display:flex;align-items:center;gap:10px;font-size:11px;font-weight:700;
            color:#111827;text-transform:uppercase;letter-spacing:1.8px;margin-bottom:14px;margin-top:28px;}}
.sec-label::before{{content:'';width:4px;height:16px;background:{YELLOW};
                    border-radius:2px;display:inline-block;flex-shrink:0;}}
.sec-label:first-child{{margin-top:0;}}

/* ── KPI comparison cards ── */
.kpi-row{{display:grid;gap:10px;margin-bottom:10px;}}
.kpi-row-5{{grid-template-columns:repeat(5,1fr);}}
.kpi-row-4{{grid-template-columns:repeat(4,1fr);}}
.kpi-card{{background:#fff;border:1px solid #e5e7eb;border-bottom:3px solid {YELLOW};
           border-radius:10px;padding:14px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.05);}}
.kpi-card-name{{font-size:9px;font-weight:700;color:#9ca3af;text-transform:uppercase;
                letter-spacing:1px;display:flex;align-items:center;
                justify-content:space-between;margin-bottom:10px;}}
.kpi-curr-val{{font-size:24px;font-weight:800;color:#111827;letter-spacing:-0.5px;
               line-height:1;margin-bottom:10px;}}
.kpi-compare{{display:flex;align-items:center;justify-content:space-between;
              padding-top:8px;border-top:1px solid #f3f4f6;gap:6px;}}
.kpi-prev-block{{display:flex;flex-direction:column;gap:1px;}}
.kpi-prev-lbl{{font-size:8px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;}}
.kpi-prev-val{{font-size:11px;font-weight:600;color:#6b7280;}}
.mom-chip{{font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;white-space:nowrap;}}
.mom-up     {{color:#16a34a;background:rgba(22,163,74,0.10);}}
.mom-dn     {{color:#dc2626;background:rgba(220,38,38,0.10);}}
.mom-inv-up {{color:#dc2626;background:rgba(220,38,38,0.10);}}
.mom-inv-dn {{color:#16a34a;background:rgba(22,163,74,0.10);}}

/* ── Traffic badges ── */
.tl-green{{color:#fff;background:#16a34a;border-radius:8px;padding:1px 6px;font-size:8px;font-weight:700;}}
.tl-amber{{color:#fff;background:#d97706;border-radius:8px;padding:1px 6px;font-size:8px;font-weight:700;}}
.tl-red  {{color:#fff;background:#dc2626;border-radius:8px;padding:1px 6px;font-size:8px;font-weight:700;}}

/* ── Risk flags ── */
.flags-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;}}
.flag-card{{border-radius:10px;padding:14px 16px;}}
.flag-critical{{background:#fff5f5;border-left:4px solid #dc2626;}}
.flag-high    {{background:#fff7ed;border-left:4px solid #f97316;}}
.flag-medium  {{background:#fffbea;border-left:4px solid #d97706;}}
.flag-title{{font-size:13px;font-weight:700;margin-bottom:3px;}}
.flag-sub  {{font-size:11px;color:#6b7280;margin-bottom:8px;}}
.flag-stats{{display:flex;gap:20px;margin-bottom:8px;}}
.flag-stat-lbl{{font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;}}
.flag-stat-val{{font-size:20px;font-weight:800;}}
.flag-action{{font-size:11px;color:#6b7280;font-style:italic;border-top:1px solid rgba(0,0,0,0.06);padding-top:7px;}}

/* ── Tables ── */
.table-wrap{{border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;}}
table{{width:100%;border-collapse:collapse;font-size:12px;}}
th{{background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;
    font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;white-space:nowrap;}}
td{{padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#374151;}}
tr:last-child td{{border-bottom:none;}}
tr.top-tier td{{background:#f0fdf4;}}
tr.bot-tier td{{background:#fff5f5;}}
.col-coll{{font-weight:800!important;}}
.col-coll-green{{color:#16a34a;}}
.col-coll-amber{{color:#d97706;}}
.col-coll-red  {{color:#dc2626;}}

/* ── Matrix ── */
.matrix-table td{{text-align:center;padding:7px 10px;font-weight:600;font-size:12px;}}
.matrix-table th{{text-align:center;padding:7px 10px;font-size:10px;}}
.matrix-kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px;}}
.matrix-kpi{{background:#fff;border:1px solid #e5e7eb;border-top:3px solid;
             border-radius:8px;padding:12px 14px;text-align:center;}}
.matrix-kpi-lbl{{font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;}}
.matrix-kpi-val{{font-size:22px;font-weight:800;}}

/* ── Two-col layout ── */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.col-label{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
            margin-bottom:8px;padding:5px 10px;border-radius:6px;}}
.col-label-green{{color:#16a34a;background:rgba(22,163,74,0.10);}}
.col-label-red  {{color:#dc2626;background:rgba(220,38,38,0.10);}}

/* ── Narrative ── */
.narrative-box{{background:{DARK};color:#e6edf3;border-radius:10px;
                padding:22px 26px;line-height:1.9;font-size:13px;
                border-left:4px solid {YELLOW};}}

/* ── Action plan ── */
.action-list{{list-style:none;}}
.action-item{{display:flex;gap:14px;align-items:flex-start;
              padding:12px 0;border-bottom:1px solid #f3f4f6;
              font-size:13px;color:#374151;}}
.action-item:last-child{{border-bottom:none;}}
.action-num{{min-width:28px;height:28px;background:{YELLOW};color:#000;border-radius:50%;
             display:flex;align-items:center;justify-content:center;
             font-weight:800;font-size:12px;flex-shrink:0;}}

/* ── Divider ── */
.divider{{height:1px;background:#e5e7eb;margin:28px 0;}}

/* ── Footer ── */
.footer{{text-align:center;color:#9ca3af;font-size:11px;
         padding:18px 36px;border-top:1px solid #e5e7eb;
         background:#fff;margin-top:8px;}}
"""

_INVERSE_KPIS = {"NPA %", "Hard Bucket %"}


def _fmt(val, kind="money"):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if kind == "money":
        if abs(v) >= 1_00_00_000:
            return f"&#8377;{v/1_00_00_000:.2f}Cr"
        if abs(v) >= 1_00_000:
            return f"&#8377;{v/1_00_000:.2f}L"
        return f"&#8377;{v:,.0f}"
    if kind == "pct":
        return f"{v:.1f}%"
    return f"{int(v):,}"


def _tl_badge(traffic):
    cls   = {"green": "tl-green", "amber": "tl-amber", "red": "tl-red"}.get(traffic, "")
    label = {"green": "GOOD",     "amber": "WATCH",    "red": "ALERT"}.get(traffic, "")
    return f'<span class="{cls}">{label}</span>' if cls else ""


def _render_portfolio_health(data: dict, curr_month: str = "", prev_month: str = "") -> str:
    kpis = data.get("kpis", {})

    KPI_TOP = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %"]
    KPI_BOT = ["Hard Bucket %", "Count", "POS", "LCC%", "CMD %"]

    prev_lbl = prev_month or "Prev Month"
    curr_lbl = curr_month or "Curr Month"

    def _card(k):
        if k not in kpis:
            return ""
        v        = kpis[k]
        mom      = v.get("mom", 0)
        inverse  = k in _INVERSE_KPIS
        mom_cls  = ("mom-inv-up" if mom >= 0 else "mom-inv-dn") if inverse else ("mom-up" if mom >= 0 else "mom-dn")
        arrow    = "&#9650;" if mom >= 0 else "&#9660;"
        badge    = _tl_badge(v.get("traffic", ""))
        prev_fmt = v.get("prev_formatted", "&#8212;")
        curr_fmt = v.get("formatted", "&#8212;")

        return f"""
        <div class="kpi-card">
          <div class="kpi-card-name">{k} {badge}</div>
          <div class="kpi-curr-val">{curr_fmt}</div>
          <div class="kpi-compare">
            <div class="kpi-prev-block">
              <span class="kpi-prev-lbl">{prev_lbl}</span>
              <span class="kpi-prev-val">{prev_fmt}</span>
            </div>
            <span class="mom-chip {mom_cls}">{arrow} {abs(mom):.1f}%</span>
          </div>
        </div>"""

    top_cards = "".join(_card(k) for k in KPI_TOP)
    bot_cards = "".join(_card(k) for k in KPI_BOT)

    return (
        f'<div class="sec-label">Portfolio Health Snapshot</div>'
        f'<div class="kpi-row kpi-row-5">{top_cards}</div>'
        f'<div class="kpi-row kpi-row-5">{bot_cards}</div>'
    )


def _render_risk_flags(data: dict) -> str:
    flags = data.get("flags", [])
    if not flags:
        return ""
    SEVERITY_COLOR = {"critical": "#dc2626", "high": "#f97316", "medium": "#d97706"}
    cards = ""
    for f in flags:
        color   = SEVERITY_COLOR.get(f["severity"], "#d97706")
        pos_str = _fmt(f["pos"], "money")
        arr_str = _fmt(f.get("closing_arrears", 0), "money")
        cards += f"""
        <div class="flag-card flag-{f['severity']}">
          <div class="flag-title" style="color:{color};">{f['icon']} {f['title']}</div>
          <div class="flag-sub">{f['subtitle']}</div>
          <div class="flag-stats">
            <div>
              <div class="flag-stat-lbl">Accounts</div>
              <div class="flag-stat-val" style="color:{color};">{f['count']:,}</div>
            </div>
            <div>
              <div class="flag-stat-lbl">POS</div>
              <div class="flag-stat-val" style="color:#111827;">{pos_str}</div>
            </div>
            <div>
              <div class="flag-stat-lbl">Closing Arrears</div>
              <div class="flag-stat-val" style="color:{color};">{arr_str}</div>
            </div>
          </div>
          <div class="flag-action">{f['action']}</div>
        </div>"""
    return (
        f'<div class="sec-label">Critical Risk Flags</div>'
        f'<div class="flags-grid">{cards}</div>'
    )


def _matrix_cell_color(prev_bucket: str, curr_bucket: str, count: int, total: int) -> str:
    from utils import BUCKET_SCORE
    if count == 0:
        return "#f9fafb"
    ps = BUCKET_SCORE.get(prev_bucket, 0)
    cs = BUCKET_SCORE.get(curr_bucket, 0)
    direction = cs - ps
    intensity = min(int(count / max(total, 1) * 600), 200)
    if direction == 0:
        return "#f3f4f6"
    elif direction > 0:
        g = max(255 - intensity, 55)
        return f"rgb(255,{g},{g})"
    else:
        r = max(255 - intensity, 55)
        return f"rgb({r},255,{r})"


def _render_bucket_migration(data: dict) -> str:
    buckets = data.get("buckets", [])
    matrix  = data.get("matrix", {})
    total   = data.get("matched_count", 1) or 1

    header = "<tr><th>From \\ To</th>" + "".join(f"<th>{b}</th>" for b in buckets) + "</tr>"
    rows = ""
    for row_b in buckets:
        row_vals = matrix.get(row_b, {})
        cells = f"<td style='font-weight:700;background:#111827;color:{YELLOW};'>{row_b}</td>"
        for col_b in buckets:
            count = int(row_vals.get(col_b, 0))
            bg    = _matrix_cell_color(row_b, col_b, count, total)
            cells += f'<td style="background:{bg};">{count}</td>'
        rows += f"<tr>{cells}</tr>"

    rr_fwd  = data.get("roll_forward_rate", 0)
    rr_bwd  = data.get("roll_backward_rate", 0)
    rr_npa  = data.get("npa_formation_rate", 0)
    matched = data.get("matched_count", 0)

    kpi_chips = (
        f'<div class="matrix-kpi-row">'
        f'<div class="matrix-kpi" style="border-top-color:#dc2626;">'
        f'<div class="matrix-kpi-lbl">Roll-Forward Rate</div>'
        f'<div class="matrix-kpi-val" style="color:#dc2626;">{rr_fwd}%</div></div>'
        f'<div class="matrix-kpi" style="border-top-color:#16a34a;">'
        f'<div class="matrix-kpi-lbl">Roll-Backward Rate</div>'
        f'<div class="matrix-kpi-val" style="color:#16a34a;">{rr_bwd}%</div></div>'
        f'<div class="matrix-kpi" style="border-top-color:#991b1b;">'
        f'<div class="matrix-kpi-lbl">NPA Formation</div>'
        f'<div class="matrix-kpi-val" style="color:#991b1b;">{rr_npa}%</div></div>'
        f'<div class="matrix-kpi" style="border-top-color:#111827;">'
        f'<div class="matrix-kpi-lbl">Matched Accounts</div>'
        f'<div class="matrix-kpi-val">{matched:,}</div></div>'
        f'</div>'
    )

    table = f'<div class="table-wrap"><table class="matrix-table"><thead>{header}</thead><tbody>{rows}</tbody></table></div>'
    return (
        f'<div class="sec-label">Bucket Migration Matrix</div>'
        f'{table}{kpi_chips}'
    )


def _render_branch_performance(data: dict) -> str:
    def _coll_cls(pct):
        return "col-coll-green" if pct >= 100 else "col-coll-amber" if pct >= 90 else "col-coll-red"

    def _table(items, label, label_cls):
        rows = "".join(
            f'<tr class="{"top-tier" if label_cls == "col-label-green" else "bot-tier"}">'
            f'<td style="font-weight:600;">{b["branch"]}</td>'
            f'<td class="col-coll {_coll_cls(b["coll_pct"])}">{b["coll_pct"]}%</td>'
            f'<td>{b["accounts"]:,}</td>'
            f'<td>&#8377;{b["collection"]}L</td>'
            f'</tr>'
            for b in items
        )
        header = "<tr><th>Branch</th><th>Collection %</th><th>Accounts</th><th>Collected</th></tr>"
        return (
            f'<div>'
            f'<div class="col-label {label_cls}">{label}</div>'
            f'<div class="table-wrap"><table>{header}<tbody>{rows}</tbody></table></div>'
            f'</div>'
        )

    top_tbl = _table(data.get("top5", []),    "&#9650; Top Performers",  "col-label-green")
    bot_tbl = _table(data.get("bottom5", []), "&#9660; Need Attention",  "col-label-red")
    return (
        f'<div class="sec-label">Branch Performance League Table ({data.get("total_branches", 0)} Branches)</div>'
        f'<div class="two-col">{top_tbl}{bot_tbl}</div>'
    )


def _render_executive_rankings(data: dict) -> str:
    def _coll_cls(pct):
        return "col-coll-green" if pct >= 100 else "col-coll-amber" if pct >= 90 else "col-coll-red"

    def _table(items, label, label_cls, tier_cls):
        rows = "".join(
            f'<tr class="{tier_cls}">'
            f'<td style="font-weight:600;">{e["name"]}</td>'
            f'<td class="col-coll {_coll_cls(e["coll_pct"])}">{e["coll_pct"]}%</td>'
            f'<td>{e["strike_rate"]}%</td>'
            f'<td>{e["npa_pct"]}%</td>'
            f'<td>{e["accounts"]:,}</td>'
            f'</tr>'
            for e in items
        )
        header = "<tr><th>Executive</th><th>Collection %</th><th>Strike %</th><th>NPA %</th><th>Accounts</th></tr>"
        return (
            f'<div>'
            f'<div class="col-label {label_cls}">{label}</div>'
            f'<div class="table-wrap"><table>{header}<tbody>{rows}</tbody></table></div>'
            f'</div>'
        )

    top_tbl = _table(data.get("top5", []),    "&#9650; Top Performers", "col-label-green", "top-tier")
    bot_tbl = _table(data.get("bottom5", []), "&#9660; Need Attention", "col-label-red",   "bot-tier")
    return (
        f'<div class="sec-label">Field Executive Rankings ({data.get("total_executives", 0)} Executives)</div>'
        f'<div class="two-col">{top_tbl}{bot_tbl}</div>'
    )


def _render_narrative(narrative: str) -> str:
    paras = "".join(
        f'<p style="margin:0 0 14px 0;">{p.strip()}</p>'
        for p in narrative.split("\n") if p.strip()
    )
    return (
        f'<div class="sec-label">AI Executive Narrative</div>'
        f'<div class="narrative-box">{paras}</div>'
    )


def _render_action_plan(action_plan: str) -> str:
    items_html = ""
    for line in action_plan.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) >= 3 and line[0].isdigit() and line[1] in ".):":
            num, rest = line[0], line[2:].strip()
        elif len(line) >= 4 and line[:2].isdigit() and line[2] in ".):":
            num, rest = line[:2], line[3:].strip()
        else:
            num, rest = "&#8226;", line

        items_html += (
            f'<li class="action-item">'
            f'<span class="action-num">{num}</span>'
            f'<span>{rest}</span>'
            f'</li>'
        )
    return (
        f'<div class="sec-label" style="color:#16a34a;">&#9989; Prioritized Action Plan</div>'
        f'<div style="background:#fff;border-radius:10px;border:1px solid #e5e7eb;'
        f'padding:4px 20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">'
        f'<ul class="action-list">{items_html}</ul>'
        f'</div>'
    )


SECTION_ORDER = [
    "portfolio_health", "risk_flags", "bucket_migration",
    "branch_performance", "executive_rankings",
]


def report_builder_node(state: ReportState) -> ReportState:
    sd          = state.get("section_data", {})
    narrative   = state.get("executive_narrative", "")
    action_plan = state.get("action_plan", "")
    curr_month  = state.get("curr_month", "")
    prev_month  = state.get("prev_month", "") or ""
    filters     = state.get("filters_applied", {})
    timestamp   = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    filter_text = " | ".join(f"{k}: {v}" for k, v in filters.items() if v and v != "All") or "All data"

    body_parts = []
    for name in SECTION_ORDER:
        if name not in sd:
            continue
        try:
            if name == "portfolio_health":
                body_parts.append(_render_portfolio_health(sd[name], curr_month, prev_month))
            elif name == "risk_flags":
                body_parts.append(_render_risk_flags(sd[name]))
            elif name == "bucket_migration":
                body_parts.append(_render_bucket_migration(sd[name]))
            elif name == "branch_performance":
                body_parts.append(_render_branch_performance(sd[name]))
            elif name == "executive_rankings":
                body_parts.append(_render_executive_rankings(sd[name]))
        except Exception:
            pass

    if narrative:
        body_parts.append(_render_narrative(narrative))
    if action_plan:
        body_parts.append(_render_action_plan(action_plan))

    body_html = "\n".join(body_parts)

    prev_label = f" &nbsp;&bull;&nbsp; vs {prev_month}" if prev_month else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shriram Finance - Portfolio Intelligence Report {curr_month}</title>
<style>{BASE_CSS}</style>
</head>
<body>
<div class="banner"></div>
<div class="report-header">
  <div>
    <div class="brand-name">SHRIRAM</div>
    <div class="brand-sub">FINANCE &nbsp;&bull;&nbsp; COLLECTION INTELLIGENCE</div>
  </div>
  <div style="text-align:right;">
    <div class="report-title">Monthly Portfolio Intelligence Report</div>
    <div class="report-meta">
      Filters: {filter_text}<br>
      Generated: {timestamp}
    </div>
    <div class="month-badge">{curr_month}{prev_label}</div>
  </div>
</div>
<div class="content">
{body_html}
</div>
<div class="footer">
  CollectionIQ &bull; Powered by Gemini 2.5 Flash &bull; Shriram Finance Internal Use Only
</div>
</body>
</html>"""

    return {**state, "html_report": html}

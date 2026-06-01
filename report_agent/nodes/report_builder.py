"""
Report Builder — assembles a fully self-contained HTML report.
No CDN, no external JS. All CSS inline. Safe for email clients.
"""
import datetime
from report_agent.state import ReportState

YELLOW = "#FFC000"

BASE_CSS = f"""
body {{font-family: Arial, Helvetica, sans-serif; background: #f5f5f5; margin: 0; padding: 0; color: #111;}}
.report-header {{background: #111; color: {YELLOW}; padding: 24px 32px;}}
.report-header h1 {{font-size: 22px; margin: 4px 0 0 0; color: #fff; font-weight: 700;}}
.report-header .month {{font-size: 14px; color: #aaa; margin-top: 6px;}}
.content {{padding: 24px 32px;}}
.section-card {{background: #fff; border-radius: 8px; margin-bottom: 20px;
               padding: 20px 24px; border-left: 4px solid {YELLOW};
               box-shadow: 0 1px 4px rgba(0,0,0,0.06);}}
.section-title {{font-size: 13px; font-weight: 700; color: {YELLOW}; text-transform: uppercase;
                letter-spacing: 1.5px; margin-bottom: 14px;}}
.kpi-grid {{display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px;}}
.kpi-box {{background: #fafafa; border: 1px solid #e5e7eb; border-radius: 8px;
          padding: 12px 16px; min-width: 140px; flex: 1; text-align: center;}}
.kpi-label {{font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;}}
.kpi-val {{font-size: 22px; font-weight: 700; color: #111;}}
.kpi-mom {{font-size: 11px; margin-top: 4px; color: #6b7280;}}
.mom-up {{color: #16a34a; font-weight: 700;}}
.mom-dn {{color: #dc2626; font-weight: 700;}}
.tl-green  {{color: #fff; background: #16a34a; border-radius: 10px; padding: 1px 8px; font-size: 10px; font-weight: 700;}}
.tl-amber  {{color: #fff; background: #d97706; border-radius: 10px; padding: 1px 8px; font-size: 10px; font-weight: 700;}}
.tl-red    {{color: #fff; background: #dc2626; border-radius: 10px; padding: 1px 8px; font-size: 10px; font-weight: 700;}}
.flag-card {{border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;}}
.flag-critical {{background: #fff5f5; border-left: 4px solid #dc2626;}}
.flag-high     {{background: #fff7ed; border-left: 4px solid #f97316;}}
.flag-medium   {{background: #fffbea; border-left: 4px solid #d97706;}}
table {{width: 100%; border-collapse: collapse; font-size: 13px;}}
th {{background: #111; color: {YELLOW}; padding: 9px 12px; text-align: left; white-space: nowrap;}}
td {{padding: 8px 12px; border-bottom: 1px solid #f0f0f0;}}
tr.top-tier  {{background: #f0fdf4; border-left: 3px solid #16a34a;}}
tr.bot-tier  {{background: #fff5f5; border-left: 3px solid #dc2626;}}
.matrix-table td {{text-align: center; padding: 6px 8px; font-weight: 600; font-size: 12px;}}
.matrix-table th {{text-align: center; padding: 6px 8px; font-size: 11px;}}
.narrative-box {{background: #0d1117; color: #e6edf3; border-radius: 8px;
                padding: 20px 24px; line-height: 1.8; font-size: 14px;}}
.action-list {{list-style: none; margin: 0; padding: 0;}}
.action-item {{padding: 10px 0; border-bottom: 1px solid #e5e7eb; font-size: 13px; display: flex; gap: 10px;}}
.action-num {{color: {YELLOW}; font-weight: 800; font-size: 18px; min-width: 24px;}}
.footer {{text-align: center; color: #aaa; font-size: 11px; padding: 20px 32px; border-top: 1px solid #e5e7eb;}}
"""


def _fmt(val, kind="money"):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if kind == "money":
        if abs(v) >= 1_00_00_000:
            return f"₹{v/1_00_00_000:.2f}Cr"
        if abs(v) >= 1_00_000:
            return f"₹{v/1_00_000:.2f}L"
        return f"₹{v:,.0f}"
    if kind == "pct":
        return f"{v:.1f}%"
    return f"{int(v):,}"


def _tl_badge(traffic):
    cls = {"green": "tl-green", "amber": "tl-amber", "red": "tl-red"}.get(traffic, "")
    label = {"green": "OK", "amber": "WATCH", "red": "ALERT"}.get(traffic, "")
    return f'<span class="{cls}">{label}</span>' if cls else ""


def _render_portfolio_health(data: dict) -> str:
    kpis = data.get("kpis", {})
    KPI_ORDER = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %",
                 "Count", "POS", "LCC%", "CMD %"]
    cards = ""
    for k in KPI_ORDER:
        if k not in kpis:
            continue
        v = kpis[k]
        mom = v.get("mom", 0)
        mom_cls = "mom-up" if mom >= 0 else "mom-dn"
        arrow = "&#9650;" if mom >= 0 else "&#9660;"
        badge = _tl_badge(v.get("traffic", ""))
        cards += f"""
        <div class="kpi-box">
          <div class="kpi-label">{k} {badge}</div>
          <div class="kpi-val">{v.get('formatted', '')}</div>
          <div class="kpi-mom">MoM <span class="{mom_cls}">{arrow} {abs(mom):.1f}%</span></div>
        </div>"""
    return f'<div class="section-card"><div class="section-title">Portfolio Health Snapshot</div><div class="kpi-grid">{cards}</div></div>'


def _render_risk_flags(data: dict) -> str:
    flags = data.get("flags", [])
    if not flags:
        return ""
    content = ""
    for f in flags:
        cls = f"flag-{f['severity']}"
        pos_str = _fmt(f["pos"], "money")
        content += f"""
        <div class="flag-card {cls}">
          <div style="font-size:15px;font-weight:700;">{f['icon']} {f['title']}</div>
          <div style="font-size:12px;color:#555;margin:4px 0 8px 0;">{f['subtitle']}</div>
          <div style="display:flex;gap:24px;margin-bottom:8px;">
            <div><div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Accounts</div>
                 <div style="font-size:22px;font-weight:800;">{f['count']}</div></div>
            <div><div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">POS</div>
                 <div style="font-size:18px;font-weight:700;">{pos_str}</div></div>
          </div>
          <div style="font-size:11px;color:#555;font-style:italic;">{f['action']}</div>
        </div>"""
    return f'<div class="section-card"><div class="section-title">Critical Risk Flags</div>{content}</div>'


def _matrix_cell_color(prev_bucket: str, curr_bucket: str, count: int, total: int) -> str:
    from utils import BUCKET_SCORE
    if count == 0:
        return "#f9fafb"
    ps = BUCKET_SCORE.get(prev_bucket, 0)
    cs = BUCKET_SCORE.get(curr_bucket, 0)
    direction = cs - ps
    intensity = min(int(count / max(total, 1) * 600), 200)
    if direction == 0:
        return f"#f3f4f6"
    elif direction > 0:  # worsened
        r = 255
        g = max(255 - intensity, 55)
        b = max(255 - intensity, 55)
        return f"rgb({r},{g},{b})"
    else:  # improved
        r = max(255 - intensity, 55)
        g = 255
        b = max(255 - intensity, 55)
        return f"rgb({r},{g},{b})"


def _render_bucket_migration(data: dict) -> str:
    buckets = data.get("buckets", [])
    matrix  = data.get("matrix", {})
    total   = data.get("matched_count", 1) or 1

    header = "<tr><th>From \\ To</th>" + "".join(f"<th>{b}</th>" for b in buckets) + "</tr>"
    rows = ""
    for row_b in buckets:
        row_vals = matrix.get(row_b, {})
        cells = f"<td style='font-weight:700;background:#111;color:{YELLOW};'>{row_b}</td>"
        for col_b in buckets:
            count = int(row_vals.get(col_b, 0))
            bg    = _matrix_cell_color(row_b, col_b, count, total)
            cells += f'<td style="background:{bg};">{count}</td>'
        rows += f"<tr>{cells}</tr>"

    kpi_row = (
        f'<div style="display:flex;gap:16px;margin-top:14px;">'
        f'<div class="kpi-box"><div class="kpi-label">Roll-Forward Rate</div>'
        f'<div class="kpi-val" style="color:#dc2626;">{data["roll_forward_rate"]}%</div></div>'
        f'<div class="kpi-box"><div class="kpi-label">Roll-Backward Rate</div>'
        f'<div class="kpi-val" style="color:#16a34a;">{data["roll_backward_rate"]}%</div></div>'
        f'<div class="kpi-box"><div class="kpi-label">NPA Formation</div>'
        f'<div class="kpi-val" style="color:#991b1b;">{data["npa_formation_rate"]}%</div></div>'
        f'<div class="kpi-box"><div class="kpi-label">Matched Accounts</div>'
        f'<div class="kpi-val">{data["matched_count"]:,}</div></div>'
        f'</div>'
    )

    table = f'<div style="overflow-x:auto;"><table class="matrix-table"><thead>{header}</thead><tbody>{rows}</tbody></table></div>'
    return f'<div class="section-card"><div class="section-title">Bucket Migration Matrix</div>{table}{kpi_row}</div>'


def _render_branch_performance(data: dict) -> str:
    def _table(items, title):
        rows = "".join(
            f'<tr class="{"top-tier" if i == 0 else ""}">'
            f'<td style="font-weight:600;">{b["branch"]}</td>'
            f'<td style="font-weight:700;color:#111;">{b["coll_pct"]}%</td>'
            f'<td>{b["accounts"]:,}</td>'
            f'<td>Rs.{b["collection"]}L</td>'
            f'</tr>'
            for i, b in enumerate(items)
        )
        header = "<tr><th>Branch</th><th>Collection %</th><th>Accounts</th><th>Collected</th></tr>"
        return f'<div style="flex:1;min-width:260px;"><div style="font-size:12px;font-weight:700;color:#6b7280;margin-bottom:8px;">{title}</div><table>{header}{rows}</table></div>'

    top_tbl = _table(data.get("top5", []), "Top Performers")
    bot_tbl = _table(data.get("bottom5", []), "Need Attention")
    inner = f'<div style="display:flex;gap:20px;flex-wrap:wrap;">{top_tbl}{bot_tbl}</div>'
    return f'<div class="section-card"><div class="section-title">Branch Performance League Table ({data.get("total_branches", 0)} branches)</div>{inner}</div>'


def _render_executive_rankings(data: dict) -> str:
    def _table(items, title, tier_cls):
        rows = "".join(
            f'<tr class="{tier_cls}">'
            f'<td style="font-weight:600;">{e["name"]}</td>'
            f'<td style="font-weight:700;">{e["coll_pct"]}%</td>'
            f'<td>{e["strike_rate"]}%</td>'
            f'<td>{e["npa_pct"]}%</td>'
            f'<td>{e["accounts"]:,}</td>'
            f'</tr>'
            for e in items
        )
        header = "<tr><th>Executive</th><th>Collection %</th><th>Strike %</th><th>NPA %</th><th>Accounts</th></tr>"
        return f'<div style="flex:1;min-width:280px;"><div style="font-size:12px;font-weight:700;color:#6b7280;margin-bottom:8px;">{title}</div><table>{header}{rows}</table></div>'

    top_tbl = _table(data.get("top5", []), "Top Performers", "top-tier")
    bot_tbl = _table(data.get("bottom5", []), "Need Attention", "bot-tier")
    inner = f'<div style="display:flex;gap:20px;flex-wrap:wrap;">{top_tbl}{bot_tbl}</div>'
    return f'<div class="section-card"><div class="section-title">Field Executive Rankings ({data.get("total_executives", 0)} executives)</div>{inner}</div>'


def _render_narrative(narrative: str) -> str:
    paras = "".join(f'<p style="margin:0 0 14px 0;">{p.strip()}</p>' for p in narrative.split("\n") if p.strip())
    return f'<div class="section-card"><div class="section-title">AI Executive Narrative</div><div class="narrative-box">{paras}</div></div>'


def _render_action_plan(action_plan: str) -> str:
    items_html = ""
    for line in action_plan.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Parse "N. action text" into numbered item
        if len(line) >= 3 and line[0].isdigit() and line[1] in ".):":
            num = line[0]
            rest = line[2:].strip()
        elif len(line) >= 3 and line[:2].isdigit() and line[2] in ".):":
            num = line[:2]
            rest = line[3:].strip()
        else:
            num = ""
            rest = line

        items_html += (
            f'<li class="action-item">'
            f'<span class="action-num">{num}</span>'
            f'<span>{rest}</span>'
            f'</li>'
        )
    return (
        f'<div class="section-card" style="border-left-color:#16a34a;">'
        f'<div class="section-title" style="color:#16a34a;">Prioritized Action Plan</div>'
        f'<ul class="action-list">{items_html}</ul>'
        f'</div>'
    )


_SECTION_RENDERERS = {
    "portfolio_health":   _render_portfolio_health,
    "risk_flags":         _render_risk_flags,
    "bucket_migration":   _render_bucket_migration,
    "branch_performance": _render_branch_performance,
    "executive_rankings": _render_executive_rankings,
}

SECTION_ORDER = [
    "portfolio_health", "risk_flags", "bucket_migration",
    "branch_performance", "executive_rankings",
]


def report_builder_node(state: ReportState) -> ReportState:
    sd           = state.get("section_data", {})
    narrative    = state.get("executive_narrative", "")
    action_plan  = state.get("action_plan", "")
    curr_month   = state.get("curr_month", "")
    filters      = state.get("filters_applied", {})
    timestamp    = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    filter_text  = " | ".join(f"{k}: {v}" for k, v in filters.items() if v and v != "All") or "All data"

    body_parts = []
    for name in SECTION_ORDER:
        if name in sd:
            renderer = _SECTION_RENDERERS.get(name)
            if renderer:
                try:
                    body_parts.append(renderer(sd[name]))
                except Exception:
                    pass  # skip broken section silently

    if narrative:
        body_parts.append(_render_narrative(narrative))
    if action_plan:
        body_parts.append(_render_action_plan(action_plan))

    body_html = "\n".join(body_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shriram Finance - Portfolio Intelligence Report {curr_month}</title>
<style>{BASE_CSS}</style>
</head>
<body>
<div class="report-header">
  <div style="font-size:28px;font-weight:900;letter-spacing:2px;">SHRIRAM <span style="color:#fff;font-weight:400;">FINANCE</span></div>
  <h1>Monthly Portfolio Intelligence Report</h1>
  <div class="month">{curr_month} &nbsp;&bull;&nbsp; Generated: {timestamp} &nbsp;&bull;&nbsp; Filters: {filter_text}</div>
</div>
<div class="content">
{body_html}
</div>
<div class="footer">
  CollectionIQ &bull; Powered by Gemini 2.0 Flash &bull; Shriram Finance Internal Use Only
</div>
</body>
</html>"""

    return {**state, "html_report": html}

"""
Report Builder  -  assembles a fully self-contained HTML report.
Email-safe: all multi-column layouts use <table> instead of CSS grid/flex.
"""
import datetime
from report_agent.state import ReportState

YELLOW = "#FFC000"
DARK   = "#0d1117"

# CSS kept for browser rendering; table layout handles email clients
BASE_CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
      background:#f0f0f0;color:#111827;font-size:14px;line-height:1.5;}}
.content{{padding:26px 36px;max-width:1200px;margin:0 auto;}}
.brand-name{{font-size:22px;font-weight:900;color:{YELLOW};letter-spacing:3px;line-height:1;}}
.brand-sub{{font-size:10px;color:#6b7280;letter-spacing:1.5px;margin-top:3px;font-weight:500;}}
.report-title{{font-size:17px;font-weight:700;color:#fff;margin-bottom:3px;}}
.report-meta{{font-size:11px;color:#6b7280;line-height:1.9;}}
.month-badge{{display:inline-block;background:{YELLOW};color:#000;font-size:11px;
              font-weight:800;padding:4px 14px;border-radius:20px;letter-spacing:1px;margin-top:6px;}}
.kpi-card{{background:#fff;border:1px solid #e5e7eb;border-bottom:3px solid {YELLOW};
           border-radius:10px;padding:14px 16px;}}
.tl-green{{color:#fff;background:#16a34a;border-radius:8px;padding:1px 6px;font-size:8px;font-weight:700;}}
.tl-amber{{color:#fff;background:#d97706;border-radius:8px;padding:1px 6px;font-size:8px;font-weight:700;}}
.tl-red  {{color:#fff;background:#dc2626;border-radius:8px;padding:1px 6px;font-size:8px;font-weight:700;}}
.table-wrap{{border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;}}
table.data{{width:100%;border-collapse:collapse;font-size:12px;}}
table.data th{{background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;
               font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;}}
table.data td{{padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#374151;}}
table.data tr.top-tier td{{background:#f0fdf4;}}
table.data tr.bot-tier td{{background:#fff5f5;}}
table.data .matrix-cell{{text-align:center;padding:7px 10px;font-weight:600;font-size:12px;}}
table.data .matrix-hdr {{text-align:center;padding:7px 10px;font-size:10px;}}
.narrative-box{{background:{DARK};color:#e6edf3;border-radius:10px;padding:22px 26px;
                line-height:1.9;font-size:13px;border-left:4px solid {YELLOW};}}
.action-list{{list-style:none;}}
.footer{{text-align:center;color:#9ca3af;font-size:11px;padding:18px 36px;
         border-top:1px solid #e5e7eb;background:#fff;margin-top:8px;}}
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
    return f'<span class="{cls}" style="font-size:8px;font-weight:700;padding:1px 6px;border-radius:8px;color:#fff;background:{"#16a34a" if traffic=="green" else "#d97706" if traffic=="amber" else "#dc2626"};">{label}</span>' if cls else ""


def _sec_label(title, color="#111827"):
    return (
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin:28px 0 14px 0;">'
        f'<tr>'
        f'<td width="4" style="background:{YELLOW};border-radius:2px;">&nbsp;</td>'
        f'<td style="padding-left:10px;font-size:11px;font-weight:700;color:{color};'
        f'text-transform:uppercase;letter-spacing:1.8px;">{title}</td>'
        f'</tr></table>'
    )


def _kpi_card(k, v, prev_lbl):
    mom      = v.get("mom", 0)
    inverse  = k in _INVERSE_KPIS
    if inverse:
        mom_bg  = "rgba(220,38,38,0.10)"  if mom >= 0 else "rgba(22,163,74,0.10)"
        mom_col = "#dc2626"               if mom >= 0 else "#16a34a"
    else:
        mom_bg  = "rgba(22,163,74,0.10)"  if mom >= 0 else "rgba(220,38,38,0.10)"
        mom_col = "#16a34a"               if mom >= 0 else "#dc2626"

    arrow    = "&#9650;" if mom >= 0 else "&#9660;"
    badge    = _tl_badge(v.get("traffic", ""))
    prev_fmt = v.get("prev_formatted", "&#8212;")
    curr_fmt = v.get("formatted", "&#8212;")

    return (
        f'<div style="background:#fff;border:1px solid #e5e7eb;border-bottom:3px solid {YELLOW};'
        f'border-radius:10px;padding:14px 16px;">'
        # name row
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:10px;">'
        f'<tr>'
        f'<td style="font-size:9px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:1px;">{k}</td>'
        f'<td align="right">{badge}</td>'
        f'</tr></table>'
        # current value
        f'<div style="font-size:24px;font-weight:800;color:#111827;letter-spacing:-0.5px;line-height:1;margin-bottom:10px;">{curr_fmt}</div>'
        # compare row
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="border-top:1px solid #f3f4f6;padding-top:8px;">'
        f'<tr>'
        f'<td valign="middle">'
        f'<div style="font-size:8px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;">{prev_lbl}</div>'
        f'<div style="font-size:11px;font-weight:600;color:#6b7280;">{prev_fmt}</div>'
        f'</td>'
        f'<td align="right" valign="middle">'
        f'<span style="font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;'
        f'background:{mom_bg};color:{mom_col};">{arrow} {abs(mom):.1f}%</span>'
        f'</td>'
        f'</tr></table>'
        f'</div>'
    )


def _kpi_table(cards, n_cols):
    pct = 100 // n_cols
    tds = "".join(
        f'<td width="{pct}%" valign="top" style="padding:4px;">{c}</td>'
        for c in cards
    )
    # pad remaining columns if needed
    remainder = n_cols - len(cards)
    for _ in range(remainder):
        tds += f'<td width="{pct}%"></td>'
    return (
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:8px;">'
        f'<tr>{tds}</tr></table>'
    )


def _render_portfolio_health(data: dict, curr_month: str = "", prev_month: str = "") -> str:
    kpis = data.get("kpis", {})

    KPI_TOP = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %"]
    KPI_BOT = ["Hard Bucket %", "Count", "SOH", "LCC%", "CMD %"]

    prev_lbl = prev_month or "Prev"

    top_cards = [_kpi_card(k, kpis[k], prev_lbl) for k in KPI_TOP if k in kpis]
    bot_cards = [_kpi_card(k, kpis[k], prev_lbl) for k in KPI_BOT if k in kpis]

    return (
        _sec_label("Portfolio Health Snapshot") +
        _kpi_table(top_cards, 5) +
        _kpi_table(bot_cards, 5)
    )


def _render_risk_flags(data: dict) -> str:
    flags = data.get("flags", [])
    if not flags:
        return ""
    SEV_COLOR = {"critical": "#dc2626", "high": "#f97316", "medium": "#d97706"}
    SEV_BG    = {"critical": "#fff5f5", "high": "#fff7ed", "medium": "#fffbea"}

    flag_cards = []
    for f in flags:
        color   = SEV_COLOR.get(f["severity"], "#d97706")
        bg      = SEV_BG.get(f["severity"], "#fffbea")
        pos_str = _fmt(f["pos"], "money")
        arr_str = _fmt(f.get("closing_arrears", 0), "money")
        flag_cards.append(
            f'<div style="background:{bg};border-left:4px solid {color};border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:13px;font-weight:700;color:{color};margin-bottom:3px;">{f["icon"]} {f["title"]}</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">{f["subtitle"]}</div>'
            f'<table cellpadding="0" cellspacing="0" border="0" style="margin-bottom:8px;">'
            f'<tr>'
            f'<td style="padding-right:20px;">'
            f'<div style="font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">Accounts</div>'
            f'<div style="font-size:20px;font-weight:800;color:{color};">{f["count"]:,}</div>'
            f'</td>'
            f'<td style="padding-right:20px;">'
            f'<div style="font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">POS</div>'
            f'<div style="font-size:16px;font-weight:700;color:#111827;">{pos_str}</div>'
            f'</td>'
            f'<td>'
            f'<div style="font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">Closing Arrears</div>'
            f'<div style="font-size:16px;font-weight:700;color:{color};">{arr_str}</div>'
            f'</td>'
            f'</tr></table>'
            f'<div style="font-size:11px;color:#6b7280;font-style:italic;border-top:1px solid rgba(0,0,0,0.06);padding-top:7px;">{f["action"]}</div>'
            f'</div>'
        )

    # 2-column table layout for flags
    rows = ""
    for i in range(0, len(flag_cards), 2):
        pair = flag_cards[i:i+2]
        td1 = f'<td width="50%" valign="top" style="padding:4px;">{pair[0]}</td>'
        td2 = f'<td width="50%" valign="top" style="padding:4px;">{pair[1]}</td>' if len(pair) > 1 else '<td width="50%"></td>'
        rows += f"<tr>{td1}{td2}</tr>"

    return (
        _sec_label("Critical Risk Flags") +
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0">{rows}</table>'
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

    header = (
        f'<tr><th class="matrix-hdr" style="background:#111827;color:{YELLOW};padding:7px 10px;text-align:center;font-size:10px;">From \\ To</th>'
        + "".join(f'<th class="matrix-hdr" style="background:#111827;color:{YELLOW};padding:7px 10px;text-align:center;font-size:10px;">{b}</th>' for b in buckets)
        + "</tr>"
    )
    rows = ""
    for row_b in buckets:
        row_vals = matrix.get(row_b, {})
        cells = f'<td class="matrix-cell" style="font-weight:700;background:#111827;color:{YELLOW};text-align:center;padding:7px 10px;font-size:12px;">{row_b}</td>'
        for col_b in buckets:
            count = int(row_vals.get(col_b, 0))
            bg    = _matrix_cell_color(row_b, col_b, count, total)
            cells += f'<td class="matrix-cell" style="background:{bg};text-align:center;padding:7px 10px;font-weight:600;font-size:12px;">{count}</td>'
        rows += f"<tr>{cells}</tr>"

    rr_items = [
        ("Roll-Forward Rate",  f'{data.get("roll_forward_rate", 0)}%',  "#dc2626"),
        ("Roll-Backward Rate", f'{data.get("roll_backward_rate", 0)}%', "#16a34a"),
        ("NPA Formation",      f'{data.get("npa_formation_rate", 0)}%', "#991b1b"),
        ("Matched Accounts",   f'{data.get("matched_count", 0):,}',     "#111827"),
    ]
    kpi_tds = "".join(
        f'<td width="25%" valign="top" style="padding:4px;">'
        f'<div style="background:#fff;border:1px solid #e5e7eb;border-top:3px solid {c};'
        f'border-radius:8px;padding:12px 14px;text-align:center;">'
        f'<div style="font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">{lbl}</div>'
        f'<div style="font-size:22px;font-weight:800;color:{c};">{val}</div>'
        f'</div></td>'
        for lbl, val, c in rr_items
    )

    return (
        _sec_label("Bucket Migration Matrix") +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;margin-bottom:10px;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead>{header}</thead><tbody>{rows}</tbody></table></div>'
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:14px;">'
        f'<tr>{kpi_tds}</tr></table>'
    )


def _render_branch_performance(data: dict) -> str:
    def _coll_color(pct):
        return "#16a34a" if pct >= 100 else "#d97706" if pct >= 90 else "#dc2626"

    def _table(items, label, label_color, label_bg):
        rows = "".join(
            f'<tr style="background:{"#f0fdf4" if label_color=="#16a34a" else "#fff5f5"};">'
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{b["branch"]}</td>'
            f'<td style="padding:9px 12px;font-weight:800;font-size:12px;color:{_coll_color(b["coll_pct"])};">{b["coll_pct"]}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{b["accounts"]:,}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">&#8377;{b["collection"]}L</td>'
            f'</tr>'
            for b in items
        )
        header = (
            f'<tr><th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">Branch</th>'
            f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;">Coll %</th>'
            f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;">Accounts</th>'
            f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;">Collected</th></tr>'
        )
        return (
            f'<td width="50%" valign="top" style="padding:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{label_color};text-transform:uppercase;'
            f'letter-spacing:1px;background:{label_bg};padding:5px 10px;border-radius:6px;margin-bottom:8px;">{label}</div>'
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
            f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<thead>{header}</thead><tbody>{rows}</tbody></table></div>'
            f'</td>'
        )

    top_td = _table(data.get("top5", []),    "&#9650; Top Performers", "#16a34a", "rgba(22,163,74,0.10)")
    bot_td = _table(data.get("bottom5", []), "&#9660; Need Attention", "#dc2626", "rgba(220,38,38,0.10)")

    return (
        _sec_label(f'Branch Performance League Table ({data.get("total_branches", 0)} Branches)') +
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>{top_td}{bot_td}</tr></table>'
    )


def _render_executive_rankings(data: dict) -> str:
    def _coll_color(pct):
        return "#16a34a" if pct >= 100 else "#d97706" if pct >= 90 else "#dc2626"

    def _table(items, label, label_color, label_bg, tier_bg):
        rows = "".join(
            f'<tr style="background:{tier_bg};">'
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{e["name"]}</td>'
            f'<td style="padding:9px 12px;font-weight:800;font-size:12px;color:{_coll_color(e["coll_pct"])};">{e["coll_pct"]}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{e["strike_rate"]}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{e["npa_pct"]}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{e["accounts"]:,}</td>'
            f'</tr>'
            for e in items
        )
        header = (
            f'<tr>'
            + "".join(
                f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
                for h in ["Executive", "Coll %", "Strike %", "NPA %", "Accounts"]
            )
            + f'</tr>'
        )
        return (
            f'<td width="50%" valign="top" style="padding:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{label_color};text-transform:uppercase;'
            f'letter-spacing:1px;background:{label_bg};padding:5px 10px;border-radius:6px;margin-bottom:8px;">{label}</div>'
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
            f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<thead>{header}</thead><tbody>{rows}</tbody></table></div>'
            f'</td>'
        )

    top_td = _table(data.get("top5", []),    "&#9650; Top Performers", "#16a34a", "rgba(22,163,74,0.10)", "#f0fdf4")
    bot_td = _table(data.get("bottom5", []), "&#9660; Need Attention", "#dc2626", "rgba(220,38,38,0.10)", "#fff5f5")

    return (
        _sec_label(f'Field Executive Rankings ({data.get("total_executives", 0)} Executives)') +
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>{top_td}{bot_td}</tr></table>'
    )


def _render_narrative(narrative: str) -> str:
    bullets = ""
    for line in narrative.split("\n"):
        line = line.strip()
        if not line:
            continue
        text = line.lstrip("- ").strip() if line.startswith("-") else line
        bullets += (
            f'<tr>'
            f'<td width="18" valign="top" style="padding:5px 10px 5px 0;color:{YELLOW};'
            f'font-size:14px;font-weight:900;line-height:1.6;">&#8226;</td>'
            f'<td valign="top" style="padding:5px 0;color:#e6edf3;font-size:13px;line-height:1.6;">{text}</td>'
            f'</tr>'
        )
    return (
        _sec_label("AI Executive Narrative") +
        f'<div style="background:{DARK};border-radius:10px;padding:20px 24px;border-left:4px solid {YELLOW};">'
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0">{bullets}</table>'
        f'</div>'
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
            f'<tr style="border-bottom:1px solid #f3f4f6;">'
            f'<td width="36" valign="top" style="padding:12px 14px 12px 0;">'
            f'<div style="width:28px;height:28px;background:{YELLOW};color:#000;border-radius:50%;'
            f'text-align:center;line-height:28px;font-weight:800;font-size:12px;">{num}</div>'
            f'</td>'
            f'<td valign="middle" style="padding:12px 0;font-size:13px;color:#374151;">{rest}</td>'
            f'</tr>'
        )
    return (
        _sec_label("Prioritized Action Plan", "#16a34a") +
        f'<div style="background:#fff;border-radius:10px;border:1px solid #e5e7eb;padding:4px 20px;">'
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0">{items_html}</table>'
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
<!-- Gold banner -->
<div style="height:5px;background:linear-gradient(90deg,{YELLOW},#FFD740,{YELLOW});"></div>

<!-- Header using table layout (email-safe) -->
<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background:#111827;border-bottom:2px solid {YELLOW};">
  <tr>
    <td style="padding:26px 36px;" valign="middle">
      <div style="font-size:22px;font-weight:900;color:{YELLOW};letter-spacing:3px;line-height:1;">SHRIRAM</div>
      <div style="font-size:10px;color:#6b7280;letter-spacing:1.5px;margin-top:3px;font-weight:500;">
        FINANCE &nbsp;&bull;&nbsp; COLLECTION INTELLIGENCE
      </div>
    </td>
    <td style="padding:26px 36px;text-align:right;" valign="middle">
      <div style="font-size:17px;font-weight:700;color:#fff;margin-bottom:3px;">Monthly Portfolio Intelligence Report</div>
      <div style="font-size:11px;color:#6b7280;line-height:1.9;">
        Filters: {filter_text}<br>Generated: {timestamp}
      </div>
      <div style="display:inline-block;background:{YELLOW};color:#000;font-size:11px;font-weight:800;
                  padding:4px 14px;border-radius:20px;letter-spacing:1px;margin-top:6px;">
        {curr_month}{prev_label}
      </div>
    </td>
  </tr>
</table>

<!-- Content -->
<div class="content">
{body_html}
</div>

<!-- Footer -->
<div class="footer">
  CollectionIQ &bull; Powered by Gemini 2.5 Flash &bull; Shriram Finance Internal Use Only
</div>
</body>
</html>"""

    return {**state, "html_report": html}

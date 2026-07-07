"""
Report Builder  -  assembles a fully self-contained HTML report.
Email-safe: all multi-column layouts use <table> instead of CSS grid/flex.
"""
import datetime
import html
import logging
from report_agent.state import ReportState
from analysis.portfolio_intelligence import OVERDUE_DEMAND_IDENTITY_COLS

logger = logging.getLogger(__name__)

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


def _esc(val) -> str:
    """Escape any string that came from the uploaded data or Gemini output before
    it enters an f-string HTML fragment. Manually-entered LCC fields (customer/
    executive/branch names, vehicle descriptions) can contain &, <, > - unescaped,
    those break the surrounding table markup, and the report is also offered as a
    raw .html download/email attachment (not just rendered in an email client's
    sandboxed viewer), so this isn't just a cosmetic concern."""
    if val is None:
        return ""
    return html.escape(str(val), quote=False)


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
            f'<div style="font-size:13px;font-weight:700;color:{color};margin-bottom:3px;">{f["icon"]} {_esc(f["title"])}</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">{_esc(f["subtitle"])}</div>'
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
            f'<div style="font-size:11px;color:#6b7280;font-style:italic;border-top:1px solid rgba(0,0,0,0.06);padding-top:7px;">{_esc(f["action"])}</div>'
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


def _render_verdict(data: dict) -> str:
    good = data.get("good", [])
    bad  = data.get("bad", [])
    if not good and not bad:
        return ""

    def _list(items):
        if not items:
            return '<div style="font-size:12px;color:#9ca3af;font-style:italic;padding:6px 0;">No notable items.</div>'
        rows = "".join(
            f'<tr>'
            f'<td width="16" valign="top" style="padding:5px 8px 5px 0;font-size:13px;font-weight:900;">&#8226;</td>'
            f'<td valign="top" style="padding:5px 0;font-size:12px;color:#374151;line-height:1.55;">{_esc(it)}</td>'
            f'</tr>'
            for it in items
        )
        return f'<table width="100%" cellpadding="0" cellspacing="0" border="0">{rows}</table>'

    good_td = (
        f'<td width="50%" valign="top" style="padding:4px;">'
        f'<div style="background:#f0fdf4;border-left:4px solid #16a34a;border-radius:10px;padding:14px 16px;height:100%;">'
        f'<div style="font-size:11px;font-weight:800;color:#16a34a;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">&#9650; What&#8217;s Working</div>'
        f'{_list(good)}'
        f'</div></td>'
    )
    bad_td = (
        f'<td width="50%" valign="top" style="padding:4px;">'
        f'<div style="background:#fff5f5;border-left:4px solid #dc2626;border-radius:10px;padding:14px 16px;height:100%;">'
        f'<div style="font-size:11px;font-weight:800;color:#dc2626;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">&#9660; Needs Attention</div>'
        f'{_list(bad)}'
        f'</div></td>'
    )
    return (
        _sec_label("Good vs Bad This Month") +
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>{good_td}{bad_td}</tr></table>'
    )


def _render_concentration(data: dict) -> str:
    img = data.get("image")
    if not img:
        return ""
    return (
        _sec_label("Concentration Map") +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;background:#fff;padding:8px;">'
        f'<img src="{img}" width="100%" style="display:block;border-radius:6px;" alt="Concentration Map"/>'
        f'</div>'
    )


def _render_region_scorecard(data: dict) -> str:
    rows_data = data.get("rows", [])
    if not rows_data:
        return ""
    STATUS_COLOR = {"Worsening": "#dc2626", "Improving": "#16a34a", "Stable": "#6b7280"}

    header = "".join(
        f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
        for h in ["Region", "SMA-2", "SMA-2%", "NPA", "NPA%", "&#916; SMA-2%", "&#916; NPA%", "Collection%", "Strike%", "SOH", "Roll Fwd%", "Roll Bwd%", "Status"]
    )
    rows = ""
    for r in rows_data:
        status      = r.get("Status", "-")
        color       = STATUS_COLOR.get(status, "#6b7280")
        npa_delta   = r.get("Δ NPA%")
        npa_delta_s = f"{npa_delta:+.1f}" if npa_delta is not None else "&#8212;"
        sma2_delta   = r.get("Δ SMA-2%")
        sma2_delta_s = f"{sma2_delta:+.1f}" if sma2_delta is not None else "&#8212;"
        roll_fwd = r.get("Roll Fwd%")
        roll_bwd = r.get("Roll Bwd%")
        rows += (
            f'<tr>'
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{_esc(r.get("Region", ""))}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("SMA-2", 0):,}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("SMA-2%", 0):.1f}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("NPA", 0):,}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("NPA%", 0):.1f}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;color:{color};font-weight:700;">{sma2_delta_s}</td>'
            f'<td style="padding:9px 12px;font-size:12px;color:{color};font-weight:700;">{npa_delta_s}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("Collection%", 0):.1f}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("Strike%", 0):.1f}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">&#8377;{r.get("SOH (Cr)", 0):.2f}Cr</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{f"{roll_fwd:.1f}%" if roll_fwd is not None else "&#8212;"}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{f"{roll_bwd:.1f}%" if roll_bwd is not None else "&#8212;"}</td>'
            f'<td style="padding:9px 12px;font-size:11px;font-weight:700;color:{color};">{status}</td>'
            f'</tr>'
        )
    return (
        _sec_label("Region Scorecard") +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _render_overdue_demand(data: dict) -> str:
    """Top 5 / bottom 5 by Month Demand Collection %, per dimension (Region/
    Branch/Executive) -- NOT every row (see compute_overdue_demand_section's
    docstring: printing every branch/executive was responsible for roughly
    half this report's total row count). A payment clears last month's
    carried-over overdue FIRST, only the remainder counts against this
    month's own EMI demand (see utils.compute_overdue_demand_pct). 100% means
    nothing was outstanding on that side, not that nothing was collected.
    Branch rows carry their Region; Executive rows carry both Branch and Region."""
    def _pct_color(pct):
        return "#16a34a" if pct >= 90 else "#d97706" if pct >= 60 else "#dc2626"

    # Derived from the single source of truth (analysis/portfolio_intelligence.py's
    # OVERDUE_DEMAND_IDENTITY_COLS, Title-case column names) rather than a second
    # hardcoded copy -- {"region": "Region", "branch": "Branch"} used to be
    # maintained separately here, independently of the identical mapping in
    # ui/tabs/portfolio_intelligence.py and report_agent/sections/overdue_demand.py.
    _IDENTITY_LABEL = {c.lower(): c for cols in OVERDUE_DEMAND_IDENTITY_COLS.values() for c in cols}
    # Fixed column widths, keyed by identity-column count (0 = region's own
    # table, 1 = branch's, 2 = executive's) -- with table-layout:fixed below,
    # this is what makes the top5 and bottom5 tables in the same row (and the
    # region/branch/executive tables as a family) line up as a real grid
    # instead of each table silently auto-sizing its columns from its own
    # content/header text width, which is what made the report look
    # misaligned: two tables showing the SAME columns end up with DIFFERENT
    # column widths purely because one has longer names or fewer rows.
    _COL_WIDTHS = {
        0: [25, 15, 15, 15, 15, 15],           # Name, Accounts, Overdue, Month Demand, Overdue%, Demand%
        1: [20, 15, 13, 13, 13, 13, 13],       # + one identity column
        2: [18, 11, 11, 12, 12, 12, 12, 12],   # + two identity columns
    }

    def _table(items, name_label, identity_keys, label, label_color, label_bg):
        widths = _COL_WIDTHS[len(identity_keys)]
        colgroup = "".join(f'<col style="width:{w}%;">' for w in widths)
        identity_headers = "".join(
            f'<th style="background:#111827;color:{YELLOW};padding:9px 10px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{_IDENTITY_LABEL[k]}</th>'
            for k in identity_keys
        )
        _num_th = (
            'style="background:#111827;color:{c};padding:9px 10px;text-align:right;'
            'font-size:10px;font-weight:700;text-transform:uppercase;"'
        ).format(c=YELLOW)
        header = (
            f'<tr><th style="background:#111827;color:{YELLOW};padding:9px 10px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{name_label}</th>'
            f'{identity_headers}'
            f'<th {_num_th}>Accounts</th>'
            f'<th {_num_th}>Overdue</th>'
            f'<th {_num_th}>Month Demand</th>'
            f'<th {_num_th}>Overdue Coll %</th>'
            f'<th {_num_th}>Demand Coll %</th></tr>'
        )
        rows = "".join(
            f'<tr style="background:{"#f0fdf4" if label_color=="#16a34a" else "#fff5f5"};">'
            f'<td style="padding:8px 10px;font-weight:600;font-size:12px;word-break:break-word;">{_esc(r["name"])}</td>'
            + "".join(
                f'<td style="padding:8px 10px;font-size:12px;word-break:break-word;">{_esc(r.get(k, "") or "&#8212;")}</td>'
                for k in identity_keys
            )
            + f'<td style="padding:8px 10px;font-size:12px;text-align:right;">{r["accounts"]:,}</td>'
            f'<td style="padding:8px 10px;font-size:12px;text-align:right;">&#8377;{r["overdue_cr"]:.2f}Cr</td>'
            f'<td style="padding:8px 10px;font-size:12px;text-align:right;">&#8377;{r["demand_cr"]:.2f}Cr</td>'
            f'<td style="padding:8px 10px;font-size:12px;text-align:right;color:{_pct_color(r["overdue_pct"])};">{r["overdue_pct"]:.2f}%</td>'
            f'<td style="padding:8px 10px;font-weight:800;font-size:12px;text-align:right;color:{_pct_color(r["demand_pct"])};">{r["demand_pct"]:.2f}%</td>'
            f'</tr>'
            for r in items
        )
        return (
            f'<td width="50%" valign="top" style="padding:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{label_color};text-transform:uppercase;'
            f'letter-spacing:1px;background:{label_bg};padding:5px 10px;border-radius:6px;margin-bottom:8px;">{label}</div>'
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;overflow-x:auto;">'
            f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;">'
            f'<colgroup>{colgroup}</colgroup>'
            f'<thead>{header}</thead><tbody>{rows}</tbody></table></div>'
            f'</td>'
        )

    dim_labels = [
        (key, key.capitalize(), [c.lower() for c in OVERDUE_DEMAND_IDENTITY_COLS[key]])
        for key in ("region", "branch", "executive")
    ]
    blocks = ""
    for key, name_label, identity_keys in dim_labels:
        dim_data = data.get(key)
        if not dim_data or not (dim_data.get("top5") or dim_data.get("bottom5")):
            continue
        top_td = _table(dim_data.get("top5", []),    name_label, identity_keys, "&#9650; Top 5 by Month Demand Collection %", "#16a34a", "rgba(22,163,74,0.10)")
        bot_td = _table(dim_data.get("bottom5", []), name_label, identity_keys, "&#9660; Bottom 5 by Month Demand Collection %", "#dc2626", "rgba(220,38,38,0.10)")
        blocks += (
            f'<div style="font-size:11px;font-weight:700;color:#374151;margin:12px 0 6px;text-transform:uppercase;letter-spacing:0.6px;">By {name_label} ({dim_data.get("total", 0)} total)</div>'
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:14px;"><tr>{top_td}{bot_td}</tr></table>'
        )
    if not blocks:
        return ""
    return _sec_label("Overdue vs Month Demand Collection") + blocks


def _render_top_accounts(data: dict) -> str:
    rows_data = data.get("rows", [])
    if not rows_data:
        return ""
    summary = data.get("summary", {})
    BUCKET_COLOR = {"NPA": "#dc2626", "SMA-2": "#f97316", "SMA-1": "#d97706", "1-30": "#eab308"}

    header = "".join(
        f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
        for h in ["Loan No", "Customer", "Region", "Branch", "Bucket", "SOH"]
    )
    rows = ""
    for r in rows_data:
        bcolor = BUCKET_COLOR.get(r.get("bucket", ""), "#6b7280")
        rows += (
            f'<tr>'
            f'<td style="padding:8px 12px;font-size:11px;color:#6b7280;">{_esc(r.get("loan_no", ""))}</td>'
            f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(r.get("customer", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("region", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("branch", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:11px;font-weight:700;color:{bcolor};">{_esc(r.get("bucket", ""))}</td>'
            f'<td style="padding:8px 12px;font-weight:700;font-size:12px;">{_fmt(r.get("soh", 0), "money")}</td>'
            f'</tr>'
        )
    summary_line = (
        f'<div style="font-size:11px;color:#6b7280;margin-bottom:10px;">'
        f'These {data.get("n", 0)} accounts total <b>&#8377;{summary.get("total_soh_cr", 0):.2f}Cr</b> '
        f'({summary.get("pct_of_portfolio", 0):.1f}% of portfolio SOH), including <b>{summary.get("npa_count", 0)}</b> already in NPA.'
        f'</div>'
    )
    return (
        _sec_label("Top At-Risk Accounts (by SOH, Delinquent Only)") +
        summary_line +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _render_fleet_exposure(data: dict) -> str:
    top = data.get("top", [])
    if not top:
        return ""

    stat_tds = "".join(
        f'<td width="33%" valign="top" style="padding:4px;">'
        f'<div style="background:#fff;border:1px solid #e5e7eb;border-top:3px solid {YELLOW};'
        f'border-radius:8px;padding:12px 14px;text-align:center;">'
        f'<div style="font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">{lbl}</div>'
        f'<div style="font-size:22px;font-weight:800;color:#111827;">{val}</div>'
        f'</div></td>'
        for lbl, val in [
            ("Fleet Operators (3+ Loans)", f'{data.get("count", 0):,}'),
            ("Total SOH Exposure", f'&#8377;{data.get("total_soh_cr", 0):.2f}Cr'),
            ("Operators w/ NPA Loan", f'{data.get("npa_operators", 0):,}'),
        ]
    )
    header = "".join(
        f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
        for h in ["Customer", "Region", "Branch", "Loans", "NPA Loans", "Total SOH"]
    )
    rows = "".join(
        f'<tr>'
        f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(r["customer"])}</td>'
        f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("region", ""))}</td>'
        f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("branch", ""))}</td>'
        f'<td style="padding:8px 12px;font-size:12px;">{r["loans"]}</td>'
        f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{"#dc2626" if r["npa_loans"] > 0 else "#374151"};">{r["npa_loans"]}</td>'
        f'<td style="padding:8px 12px;font-weight:700;font-size:12px;">&#8377;{r["soh"]:.2f}Cr</td>'
        f'</tr>'
        for r in top
    )
    return (
        _sec_label(f'Fleet Exposure ({data.get("count", 0)} Operators)') +
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:12px;"><tr>{stat_tds}</tr></table>'
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _movement_stat_card(label: str, value: str, color: str = "#111827") -> str:
    return (
        f'<td width="25%" valign="top" style="padding:4px;">'
        f'<div style="background:#fff;border:1px solid #e5e7eb;border-top:3px solid {color};'
        f'border-radius:8px;padding:12px 14px;text-align:center;">'
        f'<div style="font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:20px;font-weight:800;color:{color};">{value}</div>'
        f'</div></td>'
    )


def _movement_delta_color(delta) -> str:
    if delta is None:
        return "#6b7280"
    if delta > 0:
        return "#dc2626"
    if delta < 0:
        return "#16a34a"
    return "#6b7280"


def _movement_fmt(val, is_pct: bool = False) -> str:
    """Plain formatting for a raw curr/prev value - no +/- sign (that's reserved for deltas)."""
    if val is None:
        return "&#8212;"
    return f"{val:.1f}%" if is_pct else f"{int(val):,}"


def _movement_fmt_delta(val, is_pct: bool = False) -> str:
    """Signed formatting for a delta/%change value."""
    if val is None:
        return "&#8212;"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%" if is_pct else f"{sign}{int(val):,}"


def _render_npa_sma2_movement(data: dict) -> str:
    portfolio = data.get("portfolio")
    region    = data.get("region", [])
    b_worst   = data.get("branch_worst", [])
    b_best    = data.get("branch_best", [])
    if (
        not portfolio and not region and not b_worst and not b_best
        and not data.get("exec_worst") and not data.get("exec_best")
    ):
        return ""

    parts = [_sec_label("NPA & SMA-2 Movement  -  This Month vs Last Month")]

    if portfolio:
        cards = [
            _movement_stat_card("NPA (This Month)",  _movement_fmt(portfolio["npa_current"]),  "#dc2626"),
            _movement_stat_card("NPA (Last Month)",   _movement_fmt(portfolio["npa_prev"]),      "#9ca3af"),
            _movement_stat_card("SMA-2 (This Month)", _movement_fmt(portfolio["sma2_current"]),  "#f97316"),
            _movement_stat_card("SMA-2 (Last Month)", _movement_fmt(portfolio["sma2_prev"]),     "#9ca3af"),
            _movement_stat_card("NPA &#916; (Count)",  _movement_fmt_delta(portfolio["npa_delta"]),
                                 _movement_delta_color(portfolio["npa_delta"])),
            _movement_stat_card("SMA-2 &#916; (Count)", _movement_fmt_delta(portfolio["sma2_delta"]),
                                 _movement_delta_color(portfolio["sma2_delta"])),
            _movement_stat_card("NPA &#916;%",  _movement_fmt_delta(portfolio["npa_pct_change"], is_pct=True),
                                 _movement_delta_color(portfolio["npa_pct_change"])),
            _movement_stat_card("SMA-2 &#916;%", _movement_fmt_delta(portfolio["sma2_pct_change"], is_pct=True),
                                 _movement_delta_color(portfolio["sma2_pct_change"])),
        ]
        rows_html = "".join(f'<tr>{"".join(cards[i:i + 4])}</tr>' for i in range(0, len(cards), 4))
        parts.append(f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:16px;">{rows_html}</table>')

    if region:
        header = "".join(
            f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
            for h in ["Region", "Accounts", "NPA (This Mo.)", "NPA (Last Mo.)", "SMA-2 (This Mo.)", "SMA-2 (Last Mo.)", "NPA &#916;", "SMA-2 &#916;", "NPA &#916;%", "SMA-2 &#916;%"]
        )
        rows = ""
        for r in region:
            npa_d, sma2_d = r.get("NPA Δ"), r.get("SMA-2 Δ")
            npa_dpct, sma2_dpct = r.get("NPA Δ%"), r.get("SMA-2 Δ%")
            rows += (
                f'<tr>'
                f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(r.get("RegionName", ""))}</td>'
                f'<td style="padding:8px 12px;font-size:12px;">{r.get("Accounts", 0):,}</td>'
                f'<td style="padding:8px 12px;font-size:12px;">{r.get("NPA (Curr)", 0):,}</td>'
                f'<td style="padding:8px 12px;font-size:12px;">{_movement_fmt(r.get("NPA (Prev)"))}</td>'
                f'<td style="padding:8px 12px;font-size:12px;">{r.get("SMA-2 (Curr)", 0):,}</td>'
                f'<td style="padding:8px 12px;font-size:12px;">{_movement_fmt(r.get("SMA-2 (Prev)"))}</td>'
                f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{_movement_delta_color(npa_d)};">{_movement_fmt_delta(npa_d)}</td>'
                f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{_movement_delta_color(sma2_d)};">{_movement_fmt_delta(sma2_d)}</td>'
                f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{_movement_delta_color(npa_dpct)};">{_movement_fmt_delta(npa_dpct, is_pct=True)}</td>'
                f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{_movement_delta_color(sma2_dpct)};">{_movement_fmt_delta(sma2_dpct, is_pct=True)}</td>'
                f'</tr>'
            )
        parts.append(
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin:22px 0 8px;">By Region</div>'
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;margin-bottom:16px;">'
            f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _mover_table(items, name_col, name_label, label, color, bg, branch_col=None):
        branch_th = '<th style="background:#111827;color:' + YELLOW + ';padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">Branch</th>' if branch_col else ""
        rows = "".join(
            f'<tr>'
            f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(r.get(name_col, ""))}</td>'
            + (f'<td style="padding:8px 12px;font-size:12px;color:#6b7280;">{_esc(r.get(branch_col, ""))}</td>' if branch_col else "")
            + f'<td style="padding:8px 12px;font-size:12px;">{r.get("NPA (Curr)", 0):,}</td>'
            f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{_movement_delta_color(r.get("NPA Δ"))};">{_movement_fmt_delta(r.get("NPA Δ"))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{_movement_delta_color(r.get("NPA Δ%"))};">{_movement_fmt_delta(r.get("NPA Δ%"), is_pct=True)}</td>'
            f'</tr>'
            for r in items
        )
        hdr = (
            f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{name_label}</th>'
            + branch_th
            + "".join(
                f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
                for h in ["NPA (Curr)", "NPA &#916;", "NPA &#916;%"]
            )
        )
        return (
            f'<td width="50%" valign="top" style="padding:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;'
            f'letter-spacing:1px;background:{bg};padding:5px 10px;border-radius:6px;margin-bottom:8px;">{label}</div>'
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
            f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>'
            f'</td>'
        )

    if b_worst or b_best:
        best_td  = _mover_table(b_best,  "Unit", "Branch", "&#9660; Improving Branches (NPA)", "#16a34a", "rgba(22,163,74,0.10)")
        worst_td = _mover_table(b_worst, "Unit", "Branch", "&#9650; Worsening Branches (NPA)", "#dc2626", "rgba(220,38,38,0.10)")
        parts.append(
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin:22px 0 8px;">By Branch (Top Movers)</div>'
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:16px;"><tr>{best_td}{worst_td}</tr></table>'
        )

    e_worst = data.get("exec_worst", [])
    e_best  = data.get("exec_best", [])
    if e_worst or e_best:
        best_td  = _mover_table(e_best,  "MNT NAME", "Executive", "&#9660; Improving Executives (NPA)", "#16a34a", "rgba(22,163,74,0.10)", branch_col="Unit")
        worst_td = _mover_table(e_worst, "MNT NAME", "Executive", "&#9650; Worsening Executives (NPA)", "#dc2626", "rgba(220,38,38,0.10)", branch_col="Unit")
        parts.append(
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin:22px 0 8px;">By Executive (Top Movers)</div>'
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>{best_td}{worst_td}</tr></table>'
        )

    return "".join(parts)


def _render_bucket_migration(data: dict) -> str:
    buckets = data.get("buckets", [])
    matrix  = data.get("matrix", {})
    total   = data.get("matched_count", 1) or 1

    img_html = ""
    if data.get("waterfall_image"):
        img_html = (
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;background:#fff;'
            f'padding:8px;margin-bottom:14px;">'
            f'<img src="{data["waterfall_image"]}" width="100%" style="display:block;border-radius:6px;" alt="Bucket Distribution"/>'
            f'</div>'
        )

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
        img_html +
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
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{_esc(b["branch"])}</td>'
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
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{_esc(e["name"])}</td>'
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


def _render_executive_strike_rankings(data: dict) -> str:
    def _strike_color(pct):
        return "#16a34a" if pct >= 80 else "#d97706" if pct >= 60 else "#dc2626"

    def _table(items, label, label_color, label_bg, tier_bg):
        rows = "".join(
            f'<tr style="background:{tier_bg};">'
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{_esc(e["name"])}</td>'
            f'<td style="padding:9px 12px;font-weight:800;font-size:12px;color:{_strike_color(e["strike_rate"])};">{e["strike_rate"]}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{e["coll_pct"]}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{e["npa_pct"]}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{e["accounts"]:,}</td>'
            f'</tr>'
            for e in items
        )
        header = (
            f'<tr>'
            + "".join(
                f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
                for h in ["Executive", "Strike %", "Coll %", "NPA %", "Accounts"]
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

    top_td = _table(data.get("top5", []),    "&#9650; Best on Strike %", "#16a34a", "rgba(22,163,74,0.10)", "#f0fdf4")
    bot_td = _table(data.get("bottom5", []), "&#9660; Need Attention",   "#dc2626", "rgba(220,38,38,0.10)", "#fff5f5")

    return (
        _sec_label(f'Field Executive Rankings by Strike % ({data.get("total_executives", 0)} Executives)') +
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
            f'<td valign="top" style="padding:5px 0;color:#e6edf3;font-size:13px;line-height:1.6;">{_esc(text)}</td>'
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
            f'<td valign="middle" style="padding:12px 0;font-size:13px;color:#374151;">{_esc(rest)}</td>'
            f'</tr>'
        )
    return (
        _sec_label("Prioritized Action Plan", "#16a34a") +
        f'<div style="background:#fff;border-radius:10px;border:1px solid #e5e7eb;padding:4px 20px;">'
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0">{items_html}</table>'
        f'</div>'
    )


def _render_risk_indicators(data: dict) -> str:
    indicators = data.get("indicators", [])
    if not indicators:
        return ""
    DIR_COLOR = {"Improving": "#16a34a", "Worsening": "#dc2626", "Stable": "#6b7280"}
    header = "".join(
        f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
        for h in ["Signal", "This Month", "Last Month", "Δ", "Direction", "Note"]
    )
    rows = ""
    for ind in indicators:
        color = DIR_COLOR.get(ind.get("Direction", ""), "#9ca3af")
        rows += (
            f'<tr>'
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{ind.get("Signal", "")}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{ind.get("This Month", "")}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{ind.get("Last Month", "")}</td>'
            f'<td style="padding:9px 12px;font-size:12px;font-weight:700;color:{color};">{ind.get("Δ", "")}</td>'
            f'<td style="padding:9px 12px;font-size:11px;font-weight:700;color:{color};">{ind.get("Direction", "")}</td>'
            f'<td style="padding:9px 12px;font-size:11px;color:#6b7280;">{ind.get("Note", "")}</td>'
            f'</tr>'
        )
    return (
        _sec_label("Risk Indicators  -  Early Warning Signals") +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _render_branch_quadrant(data: dict) -> str:
    img     = data.get("image")
    concern = data.get("top_concern", [])
    if not img and not concern:
        return ""

    img_html = ""
    if img:
        img_html = (
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;background:#fff;'
            f'padding:8px;margin-bottom:14px;">'
            f'<img src="{img}" width="100%" style="display:block;border-radius:6px;" alt="Branch Quadrant"/>'
            f'</div>'
        )

    table_html = ""
    if concern:
        header = "".join(
            f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
            for h in ["Rank", "Branch", "Region", "Accounts", "SMA-2%", "NPA%", "Collection%", "Strike%", "Roll Fwd%", "Chronic (3M+)", "SOH"]
        )
        rows = "".join(
            f'<tr>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["Rank"]}</td>'
            f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(c["Branch"])}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{_esc(c.get("Region", "") or "&#8212;")}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["Accounts"]:,}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["SMA-2%"]:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["NPA%"]:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["Collection%"]:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["Strike%"]:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["Roll Fwd%"]:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{c["Chronic (3M+)"]:,}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">&#8377;{c["SOH (Cr)"]:.2f}Cr</td>'
            f'</tr>'
            for c in concern
        )
        table_html = (
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Highest Concern Branches</div>'
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;overflow-x:auto;">'
            f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    return (
        _sec_label(f'Branch Quadrant  -  Collection% vs NPA%  ({data.get("total_branches", 0)} Branches)') +
        img_html + table_html
    )


def _render_executive_recovery(data: dict) -> str:
    top    = data.get("top", [])
    bottom = data.get("bottom", [])
    if not top and not bottom:
        return ""

    def _table(items, label, color, bg):
        rows = "".join(
            f'<tr>'
            f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(r.get("Executive", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{r.get("Accounts", 0):,}</td>'
            f'<td style="padding:8px 12px;font-size:12px;color:#16a34a;font-weight:700;">{r.get("Rescued", 0)}</td>'
            f'<td style="padding:8px 12px;font-size:12px;color:#dc2626;font-weight:700;">{r.get("Slipped", 0)}</td>'
            f'<td style="padding:8px 12px;font-size:12px;font-weight:800;color:{"#16a34a" if r.get("Net Recovery", 0) >= 0 else "#dc2626"};">'
            f'{int(r.get("Net Recovery", 0)):+,}</td>'
            f'</tr>'
            for r in items
        )
        header = "".join(
            f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
            for h in ["Executive", "Accounts", "Rescued", "Slipped", "Net Recovery"]
        )
        return (
            f'<td width="50%" valign="top" style="padding:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;'
            f'letter-spacing:1px;background:{bg};padding:5px 10px;border-radius:6px;margin-bottom:8px;">{label}</div>'
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
            f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
            f'</td>'
        )

    top_td = _table(top,    "&#9650; Best Recovery (Net Rescued)",   "#16a34a", "rgba(22,163,74,0.10)")
    bot_td = _table(bottom, "&#9660; Worst Recovery (Net Slipped)", "#dc2626", "rgba(220,38,38,0.10)")
    return (
        _sec_label(f'Executive Recovery Leaderboard ({data.get("total", 0)} Executives)') +
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>{top_td}{bot_td}</tr></table>'
    )


def _render_product_analysis(data: dict) -> str:
    rows_data = data.get("rows", [])
    if not rows_data:
        return ""
    header = "".join(
        f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
        for h in ["Segment", "Accounts", "SMA-2%", "NPA%", "Collection%", "SOH"]
    )
    rows = ""
    for r in rows_data:
        npa   = r.get("NPA%", 0)
        color = "#dc2626" if npa >= 7 else "#d97706" if npa >= 3 else "#16a34a"
        rows += (
            f'<tr>'
            f'<td style="padding:9px 12px;font-weight:600;font-size:12px;">{_esc(r.get("Segment", ""))}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("Accounts", 0):,}</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("SMA-2%", 0):.1f}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;font-weight:700;color:{color};">{npa:.1f}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">{r.get("Collection%", 0):.1f}%</td>'
            f'<td style="padding:9px 12px;font-size:12px;">&#8377;{r.get("SOH (Cr)", 0):.2f}Cr</td>'
            f'</tr>'
        )
    return (
        _sec_label("Segment-Wise NPA Breakdown") +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _render_repossession(data: dict) -> str:
    rows_data = data.get("rows", [])
    if not rows_data:
        return ""
    header = "".join(
        f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
        for h in ["Loan No", "Customer", "Region", "Branch", "Bucket", "SOH", "Vehicle"]
    )
    rows = ""
    for r in rows_data:
        bucket = r.get("curr_bucket", "")
        bcolor = "#dc2626" if bucket == "NPA" else "#f97316"
        rows += (
            f'<tr>'
            f'<td style="padding:8px 12px;font-size:11px;color:#6b7280;">{_esc(r.get("Loan No", ""))}</td>'
            f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(r.get("Cust Name", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("RegionName", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("Unit", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:11px;font-weight:700;color:{bcolor};">{_esc(bucket)}</td>'
            f'<td style="padding:8px 12px;font-weight:700;font-size:12px;">{_fmt(r.get("SOH", 0) or 0, "money")}</td>'
            f'<td style="padding:8px 12px;font-size:11px;color:#6b7280;">{_esc(r.get("Vehicle Description", ""))}</td>'
            f'</tr>'
        )
    return (
        _sec_label(f'Repossession Candidates ({data.get("total", 0)} Total, Top {len(rows_data)} by SOH)') +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _render_good_customers(data: dict) -> str:
    rows_data = data.get("rows", [])
    if not rows_data:
        from config import GOOD_CUSTOMER_MIN_TENURE_PCT, GOOD_CUSTOMER_MIN_LCC_PCT
        return (
            _sec_label("Good Customers  -  Refinance / Retention Targets") +
            f'<div style="background:#f9fafb;border:1px dashed #d1d5db;border-radius:10px;padding:16px 18px;'
            f'font-size:12px;color:#6b7280;">'
            f'No accounts meet the good customer criteria ({GOOD_CUSTOMER_MIN_TENURE_PCT}%+ tenure completed '
            f'AND LCC% &#8805; {GOOD_CUSTOMER_MIN_LCC_PCT}%) in the current selection.'
            f'</div>'
        )
    header = "".join(
        f'<th style="background:#111827;color:{YELLOW};padding:9px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">{h}</th>'
        for h in ["Customer", "Region", "Branch", "Tenure Completed", "LCC%", "SOH"]
    )
    rows = ""
    for r in rows_data:
        tenure = r.get("Tenure Completed %") or 0
        rows += (
            f'<tr>'
            f'<td style="padding:8px 12px;font-weight:600;font-size:12px;">{_esc(r.get("Cust Name", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("RegionName", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{_esc(r.get("Unit", ""))}</td>'
            f'<td style="padding:8px 12px;font-size:12px;color:#16a34a;font-weight:700;">{tenure:.1f}%</td>'
            f'<td style="padding:8px 12px;font-size:12px;">{r.get("LCC%", 0) or 0:.1f}%</td>'
            f'<td style="padding:8px 12px;font-weight:700;font-size:12px;">{_fmt(r.get("SOH", 0) or 0, "money")}</td>'
            f'</tr>'
        )
    return (
        _sec_label(f'Good Customers  -  Refinance / Retention Targets ({data.get("total", 0)} Total)') +
        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">'
        f'<table class="data" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


SECTION_ORDER = [
    "portfolio_health", "verdict", "risk_flags", "risk_indicators",
    "bucket_migration", "npa_sma2_movement", "branch_quadrant", "concentration",
    "region_scorecard", "overdue_demand", "product_analysis", "top_accounts", "fleet_exposure",
    "repossession", "good_customers", "branch_performance",
    "executive_recovery", "executive_rankings", "executive_strike_rankings",
]

_RENDERERS = {
    "verdict":                    _render_verdict,
    "risk_flags":                 _render_risk_flags,
    "risk_indicators":            _render_risk_indicators,
    "bucket_migration":           _render_bucket_migration,
    "npa_sma2_movement":          _render_npa_sma2_movement,
    "branch_quadrant":            _render_branch_quadrant,
    "concentration":              _render_concentration,
    "region_scorecard":           _render_region_scorecard,
    "overdue_demand":             _render_overdue_demand,
    "product_analysis":           _render_product_analysis,
    "top_accounts":               _render_top_accounts,
    "fleet_exposure":             _render_fleet_exposure,
    "repossession":               _render_repossession,
    "good_customers":             _render_good_customers,
    "branch_performance":         _render_branch_performance,
    "executive_recovery":         _render_executive_recovery,
    "executive_rankings":         _render_executive_rankings,
    "executive_strike_rankings":  _render_executive_strike_rankings,
}


def report_builder_node(state: ReportState) -> ReportState:
    sd          = state.get("section_data", {})
    narrative   = state.get("executive_narrative", "")
    action_plan = state.get("action_plan", "")
    curr_month  = state.get("curr_month", "")
    prev_month  = state.get("prev_month", "") or ""
    filters     = state.get("filters_applied", {})
    timestamp   = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    filter_text = _esc(" | ".join(f"{k}: {v}" for k, v in filters.items() if v and v != "All") or "All data")
    curr_month  = _esc(curr_month)
    prev_month  = _esc(prev_month)

    # Verdict-first: AI executive narrative + action plan lead the report,
    # so a lead gets the TL;DR before scrolling into supporting detail tables.
    body_parts = []
    if narrative:
        body_parts.append(_render_narrative(narrative))
    if action_plan:
        body_parts.append(_render_action_plan(action_plan))

    for name in SECTION_ORDER:
        if name not in sd:
            continue
        try:
            if name == "portfolio_health":
                body_parts.append(_render_portfolio_health(sd[name], curr_month, prev_month))
            else:
                body_parts.append(_RENDERERS[name](sd[name]))
        except Exception as e:
            logger.warning("Report section '%s' failed to render: %s", name, e)

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

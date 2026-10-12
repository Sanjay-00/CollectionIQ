"""
One report model (report_agent/story.py) -> HTML (email-safe), PDF and an
Excel annex. Formatting and red shading are decided here once, so the three
formats always agree.
"""
from __future__ import annotations

import datetime
import html as _html
import os
import re
from io import BytesIO

import pandas as pd

from ui.components import format_value as fmt, heat_bg, heat_range, is_blank as _blank

YELLOW, INK, MUTED = "#FFC000", "#111827", "#4b5563"
_TONE = {"bad": "#dc2626", "good": "#16a34a", "info": "#d97706", "plain": "#9ca3af"}
# Verdict banner: (edge, background) by tone.
_VERDICT = {"bad": ("#dc2626", "#fef2f2"), "good": ("#16a34a", "#f0fdf4"), "mixed": ("#d97706", "#fffbeb"),
            "info": ("#6b7280", "#f3f4f6")}
FLAG_BG, FLAG_EDGE = "#fff1f2", "#dc2626"          # a flagged row (e.g. the worst quarter of executives)


# ── Cell formatting: ui.components.format_value, shared with the on-screen tables ──

def cell_text(row, col: dict) -> str:
    """The text of one cell; a % column with a "count" reads "11.9% (37)",
    with a "count" and a "base" "11.9% (37 of 310)" ("-" when the base is 0),
    and one with an "amount" (₹ Cr) reads "34.6% (₹121.40 Cr)"."""
    v = row.get(col["key"])
    base = row.get(col["base"]) if col.get("base") else None
    if col.get("base") and not _blank(base) and int(base) == 0:
        return "-"
    text = fmt(v, col["fmt"])
    count = row.get(col["count"]) if col.get("count") else None
    if col.get("count") and not _blank(v) and not _blank(count):
        text += f" ({int(count):,}" + (f" of {int(base):,})" if not _blank(base) else ")")
    amount = row.get(col["amount"]) if col.get("amount") else None
    if col.get("amount") and not _blank(v) and not _blank(amount):
        text += f" (₹{float(amount):,.2f} Cr)"
    return text


def _align(c: dict) -> str:
    return c.get("align") or ("left" if c["fmt"] == "text" else "right")


def _grouped(block: dict) -> list[bool]:
    """Rows that are a group label (e.g. "ASSET QUALITY"), drawn as a band across the table."""
    g = block.get("group_key")
    if not g or g not in block["df"].columns:
        return [False] * len(block["df"])
    return [bool(v) for v in block["df"][g].fillna(False)]


def _flagged(block: dict) -> list[bool]:
    f = block.get("flag")
    if not f or f not in block["df"].columns:
        return [False] * len(block["df"])
    return [bool(v) for v in block["df"][f].fillna(False)]


def _cell_colors(block: dict) -> list[list[tuple[str | None, str | None]]]:
    """(background, text colour) for each body cell: red shading by rank on
    heat columns (by "heat_key" when given, e.g. only the cells where a loan
    got worse), red/green on change columns (a rise is worse unless the
    column says good_if_up)."""
    df, cols = block["df"], block["columns"]
    hk = {c["key"]: c.get("heat_key", c["key"]) for c in cols if c.get("heat")}
    ranges = {k: heat_range(df[h].tolist()) for k, h in hk.items() if h in df.columns}
    out = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v, bg, fg = row.get(c["key"]), None, None
            shade = row.get(hk[c["key"]]) if c["key"] in ranges else None
            if not _blank(shade):
                bg = heat_bg(shade, ranges[c["key"]])
                fg = "#7f1d1d" if bg else None
            if c["fmt"] in ("pp", "count_change") and not _blank(v) and float(v) != 0:
                worse = (float(v) > 0) != bool(c.get("good_if_up"))
                fg = "#dc2626" if worse else "#16a34a"
            tone = row.get(c["tone_key"]) if c.get("tone_key") else None
            if not _blank(tone):                             # a precomputed better / worse
                fg = "#dc2626" if tone else "#16a34a"
            cells.append((bg, fg))
        out.append(cells)
    return out


# ── HTML (email-safe: tables and inline styles only) ─────────────────────────

def _e(s) -> str:
    return _html.escape(str(s))


def _html_block(b: dict) -> str:
    t = b["type"]
    if t == "heading":
        note = f'<div style="font-size:12px;color:{MUTED};margin:2px 0 0 12px;">{_e(b["note"])}</div>' if b["note"] else ""
        return (f'<div style="margin:26px 0 8px 0;"><div style="border-left:4px solid {YELLOW};padding-left:8px;'
                f'font-size:16px;font-weight:800;color:{INK};">{_e(b["text"])}</div>{note}</div>')
    if t == "bullets":
        rows = ""
        for it in b["items"]:
            detail = f'<div style="font-size:12px;color:{MUTED};margin-top:2px;">{_e(it["detail"])}</div>' if it.get("detail") else ""
            rows += (f'<tr><td style="border-left:4px solid {_TONE.get(it["tone"], "#9ca3af")};padding:6px 10px;'
                     f'background:#fafafa;"><div style="font-size:13px;font-weight:600;color:{INK};">{_e(it["text"])}</div>'
                     f'{detail}</td></tr><tr><td style="height:5px;"></td></tr>')
        return f'<table width="100%" cellpadding="0" cellspacing="0" border="0">{rows}</table>'
    if t == "kpis":
        cells = ""
        for it in b["items"]:
            change = ""
            if it.get("change"):
                color = "#dc2626" if it.get("worse") else ("#16a34a" if it.get("worse") is False else MUTED)
                change = f'<div style="font-size:11px;color:{color};font-weight:700;">{_e(it["change"])} vs last month</div>'
            cells += (f'<td valign="top" style="border:1px solid #e5e7eb;border-top:3px solid {YELLOW};padding:8px 10px;'
                      f'background:#fff;"><div style="font-size:10.5px;color:{MUTED};font-weight:700;text-transform:uppercase;">'
                      f'{_e(it["label"])}</div><div style="font-size:18px;font-weight:800;color:{INK};">{_e(it["value"])}</div>'
                      f'{change}</td><td style="width:6px;"></td>')
        title = f'<div style="font-size:12px;font-weight:700;color:{INK};margin:8px 0 4px 0;">{_e(b["title"])}</div>' if b["title"] else ""
        return f'{title}<table cellpadding="0" cellspacing="0" border="0" width="100%"><tr>{cells}</tr></table>'
    if t == "table":
        df, cols = b["df"], b["columns"]
        colors, flags, groups = _cell_colors(b), _flagged(b), _grouped(b)
        head = "".join(f'<th style="background:#111;color:{YELLOW};padding:6px 8px;font-size:11px;'
                       f'text-align:{_align(c)};">{_e(c.get("label", c["key"]))}</th>'
                       for c in cols)
        body = ""
        for i, (_, row) in enumerate(df.iterrows()):
            if groups[i]:                    # a group label: one band across the table
                body += (f'<tr><td colspan="{len(cols)}" style="padding:6px 8px;font-size:11px;font-weight:800;'
                         f'letter-spacing:0.6px;color:{INK};background:#f3f4f6;border-top:1px solid #e5e7eb;">'
                         f'{_e(row.get(cols[0]["key"]))}</td></tr>')
                continue
            tds = ""
            for j, (c, (bg, fg)) in enumerate(zip(cols, colors[i])):
                bg = bg or (FLAG_BG if flags[i] else None)
                style = (f'padding:5px 8px;font-size:12px;border-bottom:1px solid #eee;'
                         f'{"white-space:nowrap;" if c["fmt"] != "text" else ""}'
                         f'text-align:{_align(c)};'
                         f'{"font-weight:700;" if c.get("bold") else ""}'
                         f'{"border-left:4px solid " + FLAG_EDGE + ";" if flags[i] and j == 0 else ""}'
                         f'{"background:" + bg + ";" if bg else ""}{"color:" + fg + ";font-weight:600;" if fg else ""}')
                tds += f'<td style="{style}">{_e(cell_text(row, c))}</td>'
            body += f'<tr style="background:{"#fff" if i % 2 == 0 else "#fafafa"};">{tds}</tr>'
        if b.get("total"):
            row = pd.Series(b["total"])
            tds = "".join(
                f'<td style="padding:5px 8px;font-size:12px;font-weight:800;border-top:2px solid {YELLOW};'
                f'background:#fffbea;text-align:{_align(c)};">{_e(cell_text(row, c))}</td>'
                for c in cols)
            body += f"<tr>{tds}</tr>"
        note = f'<div style="font-size:11px;color:{MUTED};margin-top:4px;">{_e(b["note"])}</div>' if b["note"] else ""
        title = (f'<div style="font-size:12px;font-weight:700;color:{INK};margin:10px 0 4px 0;">{_e(b["title"])}</div>'
                 if b["title"] not in ("", None) and not b.get("hide_title") else "")
        return (f'{title}<table width="100%" cellpadding="0" cellspacing="0" border="0" '
                f'style="border-collapse:collapse;border:1px solid #e5e7eb;"><tr>{head}</tr>{body}</table>{note}')
    if t == "glance":
        return _html_glance(b)
    if t == "callout":
        edge, bg = _VERDICT.get(b["tone"], _VERDICT["info"])
        return (f'<div style="border-left:4px solid {edge};background:{bg};padding:8px 12px;margin:8px 0;'
                f'font-size:13px;font-weight:600;color:{INK};">{_e(b["text"])}</div>')
    return ""


def _html_glance(b: dict) -> str:
    edge, bg = _VERDICT.get(b["verdict"]["tone"], _VERDICT["info"])
    verdict = (f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr><td style="border-left:6px solid '
               f'{edge};background:{bg};padding:12px 16px;font-size:16px;font-weight:700;color:{INK};line-height:1.4;">'
               f'{_e(b["verdict"]["text"])}</td></tr></table><div style="height:10px;"></div>')
    tiles = _html_block({"type": "kpis", "title": "", "items": b["tiles"]})

    def column(title: str, color: str, items: list[str], mark: str) -> str:
        rows = "".join(f'<tr><td valign="top" style="width:18px;color:{color};font-weight:800;font-size:13px;'
                       f'padding:3px 0;">{mark if mark else i + 1}</td><td style="font-size:12.5px;color:{INK};'
                       f'padding:3px 0;line-height:1.35;">{item}</td></tr>' for i, item in enumerate(items))
        return (f'<td valign="top" width="33%" style="border:1px solid #e5e7eb;border-top:4px solid {color};'
                f'padding:10px 12px;background:#fff;"><div style="font-size:11px;font-weight:800;color:{color};'
                f'letter-spacing:0.8px;margin-bottom:4px;">{title}</div>'
                f'<table cellpadding="0" cellspacing="0" border="0" width="100%">{rows}</table></td>')
    focus = [f'<b>{_e(f["title"])}</b><br><span style="color:{MUTED};">{_e(f["detail"])}</span>' for f in b["focus"]]
    cols = (column("WHAT WENT WELL", "#16a34a", [_e(x) for x in b["good"]], "&#10003;")
            + '<td style="width:8px;"></td>'
            + column("WHAT WENT WRONG", "#dc2626", [_e(x) for x in b["bad"]], "&#10007;")
            + '<td style="width:8px;"></td>'
            + column("WHERE TO FOCUS", "#b45309", focus or ["Nothing urgent."], ""))
    return (f'{verdict}{tiles}<div style="height:10px;"></div>'
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>{cols}</tr></table>')


def _mark_duplicate_titles(blocks: list[dict]) -> list[dict]:
    """Hide a table's title when the heading right above says the same."""
    out, last = [], ""
    for b in blocks:
        if b["type"] == "heading":
            last = b["text"]
        elif b["type"] == "table" and b["title"].lower() == last.lower():
            b = {**b, "hide_title": True}
        out.append(b)
    return out


def to_html(model: dict) -> str:
    when = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    ai = ""
    if model.get("ai_summary"):
        ai = (f'<div style="border:1px solid #fde68a;background:#fffbeb;border-radius:6px;padding:10px 14px;margin-top:14px;">'
              f'<div style="font-size:11px;font-weight:800;color:#92400e;">AI SUMMARY (wording only; every figure checked '
              f'against the tables)</div><div style="font-size:13px;color:{INK};margin-top:4px;">{_e(model["ai_summary"])}</div></div>')
    notes = ""
    if model["data_notes"]:
        notes = (f'<div style="font-size:11px;color:{MUTED};margin-top:6px;">Data checks: '
                 f'{_e(" ".join(model["data_notes"]))}</div>')
    body = "".join(_html_block(b) for b in _mark_duplicate_titles(model["blocks"]))
    footer = "Generated by CollectionIQ" + (" · AI wording by Gemini" if model.get("ai_summary") else "")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(model['title'])} {_e(model['month'])}</title></head>
<body style="margin:0;background:#f3f4f6;font-family:Calibri,Arial,Helvetica,sans-serif;color:{INK};">
<div style="max-width:1100px;margin:0 auto;background:#fff;">
<table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#111827;border-bottom:3px solid {YELLOW};"><tr>
<td style="padding:20px 28px;"><div style="font-size:20px;font-weight:900;color:{YELLOW};letter-spacing:3px;">COLLECTIONIQ</div>
<div style="font-size:10px;color:#9ca3af;letter-spacing:1px;">PORTFOLIO &amp; COLLECTION REPORT</div></td>
<td style="padding:20px 28px;text-align:right;"><div style="font-size:17px;font-weight:700;color:#fff;">{_e(model['title'])}</div>
<div style="font-size:12px;color:#d1d5db;">{_e(model['subtitle'])} · {_e(model['scope'])} · {model['accounts']:,} loans</div>
<div style="font-size:11px;color:#9ca3af;">Generated {when}</div></td></tr></table>
<div style="padding:8px 28px 28px 28px;">{notes}{ai}{body}
<div style="margin-top:30px;border-top:1px solid #e5e7eb;padding-top:8px;font-size:11px;color:#9ca3af;">{footer} · Internal use only</div>
</div></div></body></html>"""


# ── PDF (reportlab) ──────────────────────────────────────────────────────────

_FONT = None


def _pdf_font() -> tuple[str, str, bool]:
    """(regular, bold, unicode_ok). A system font with the ₹ ▲ ▼ → glyphs when
    available (Segoe UI on Windows, DejaVu on Linux); otherwise Helvetica, and
    those symbols are written as text instead."""
    global _FONT
    if _FONT:
        return _FONT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [
        (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for reg, bold in candidates:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("ReportFont", reg))
                pdfmetrics.registerFont(TTFont("ReportFont-Bold", bold))
                _FONT = ("ReportFont", "ReportFont-Bold", True)
                return _FONT
            except Exception:
                pass
    _FONT = ("Helvetica", "Helvetica-Bold", False)
    return _FONT


def _pdf_text(s: str, unicode_ok: bool) -> str:
    s = _e(s)
    if not unicode_ok:
        s = s.replace("₹", "Rs ").replace("▲", "+").replace("▼", "-").replace("→", "->").replace("·", "-")
    return s


def to_pdf(model: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import CondPageBreak, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table

    reg, bold, uni = _pdf_font()
    T = lambda s: _pdf_text(s, uni)
    st = {
        "title": ParagraphStyle("t", fontName=bold, fontSize=16, leading=20, textColor=colors.HexColor(INK)),
        "sub": ParagraphStyle("s", fontName=reg, fontSize=9, textColor=colors.HexColor(MUTED)),
        "h": ParagraphStyle("h", fontName=bold, fontSize=12.5, textColor=colors.HexColor(INK), spaceBefore=10, spaceAfter=2),
        "note": ParagraphStyle("n", fontName=reg, fontSize=8.5, textColor=colors.HexColor(MUTED), spaceAfter=4),
        "b": ParagraphStyle("b", fontName=bold, fontSize=9.5, textColor=colors.HexColor(INK), leading=12),
        "bd": ParagraphStyle("bd", fontName=reg, fontSize=8.5, textColor=colors.HexColor(MUTED), leading=11),
        "cell": ParagraphStyle("c", fontName=reg, fontSize=8, leading=10),
        "head": ParagraphStyle("hd", fontName=bold, fontSize=8, leading=10, textColor=colors.HexColor(YELLOW)),
        "kl": ParagraphStyle("kl", fontName=bold, fontSize=7.5, textColor=colors.HexColor(MUTED)),
        "kv": ParagraphStyle("kv", fontName=bold, fontSize=13, textColor=colors.HexColor(INK), leading=16),
    }
    page_w = landscape(A4)[0] - 24 * mm
    story = [Paragraph(T(f"COLLECTIONIQ · {model['title']}"), st["title"]),
             Paragraph(T(f"{model['subtitle']} · {model['scope']} · {model['accounts']:,} loans · "
                         f"generated {datetime.datetime.now():%d %b %Y}"), st["sub"]), Spacer(1, 4)]
    if model["data_notes"]:
        story.append(Paragraph(T("Data checks: " + " ".join(model["data_notes"])), st["note"]))
    if model.get("ai_summary"):
        story += [Paragraph(T("AI summary (wording only; every figure checked against the tables)"), st["kl"]),
                  Paragraph(T(model["ai_summary"]), st["bd"]), Spacer(1, 4)]

    def kpi_table(items: list[dict]):
        cells = []
        for it in items:
            change = ""
            if it.get("change"):
                color = "#dc2626" if it.get("worse") else ("#16a34a" if it.get("worse") is False else MUTED)
                change = f'<br/><font size="7.5" color="{color}">{T(it["change"])} vs last month</font>'
            cells.append(Paragraph(f'<font size="7" color="{MUTED}">{T(it["label"].upper())}</font><br/>'
                                   f'<b>{T(it["value"])}</b>{change}', st["kv"]))
        w = page_w / max(len(cells), 1)
        return Table([cells], colWidths=[w] * len(cells), style=[
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
            ("LINEABOVE", (0, 0), (-1, 0), 2, colors.HexColor(YELLOW)), ("VALIGN", (0, 0), (-1, -1), "TOP")])

    def pdf_glance(b: dict) -> list:
        edge, bg = _VERDICT.get(b["verdict"]["tone"], _VERDICT["info"])
        verdict = Table([[Paragraph(f"<b>{T(b['verdict']['text'])}</b>",
                                    ParagraphStyle("v", parent=st["b"], fontSize=11.5, leading=15))]],
                        colWidths=[page_w], style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                                                   ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor(edge)),
                                                   ("TOPPADDING", (0, 0), (-1, -1), 7),
                                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 7)])

        def column(title: str, color: str, items: list[str]) -> list:
            return ([Paragraph(f'<font color="{color}"><b>{title}</b></font>', st["kl"])]
                    + [Paragraph(item, st["bd"]) for item in items])
        focus = [f"<b>{i}. {T(f['title'])}</b><br/>{T(f['detail'])}" for i, f in enumerate(b["focus"], start=1)]
        # A bullet, not ✓ / ✗: Segoe UI and DejaVu draw those as empty boxes.
        cols = Table([[column("WHAT WENT WELL", "#16a34a", [f'<font color="#16a34a">•</font> {T(x)}' for x in b["good"]]),
                       column("WHAT WENT WRONG", "#dc2626", [f'<font color="#dc2626">•</font> {T(x)}' for x in b["bad"]]),
                       column("WHERE TO FOCUS", "#b45309", focus or ["Nothing urgent."])]],
                     colWidths=[page_w / 3] * 3, style=[
                         ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                         ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                         ("LINEABOVE", (0, 0), (0, 0), 3, colors.HexColor("#16a34a")),
                         ("LINEABOVE", (1, 0), (1, 0), 3, colors.HexColor("#dc2626")),
                         ("LINEABOVE", (2, 0), (2, 0), 3, colors.HexColor("#b45309"))])
        return [verdict, Spacer(1, 6), kpi_table(b["tiles"]), Spacer(1, 6), cols, Spacer(1, 6)]

    pending: list = []          # a heading waiting to be kept with the block after it
    last_heading = ""

    def _emit(flowables: list) -> None:
        """Keep the waiting heading on the same page as the block after it."""
        nonlocal pending
        story.append(KeepTogether(pending + flowables))
        pending = []

    for b in model["blocks"]:
        if b["type"] == "heading":
            last_heading = b["text"]
            pending = [Paragraph(T(b["text"]), st["h"])] + ([Paragraph(T(b["note"]), st["note"])] if b["note"] else [])
            continue
        before = len(story)
        if b["type"] == "bullets":
            for it in b["items"]:
                text = f'<font color="{_TONE.get(it["tone"], "#9ca3af")}">■</font> {T(it["text"])}'
                rows = [[Paragraph(text, st["b"])]]
                if it.get("detail"):
                    rows.append([Paragraph(T(it["detail"]), st["bd"])])
                story.append(Table(rows, colWidths=[page_w], style=[("LEFTPADDING", (0, 0), (-1, -1), 2),
                                                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        elif b["type"] == "kpis":
            if b["title"]:
                story.append(Paragraph(T(b["title"]), st["b"]))
            story += [kpi_table(b["items"]), Spacer(1, 4)]
        elif b["type"] == "glance":
            story += pdf_glance(b)
        elif b["type"] == "callout":
            edge, bg = _VERDICT.get(b["tone"], _VERDICT["info"])
            story += [Table([[Paragraph(f"<b>{T(b['text'])}</b>", st["b"])]], colWidths=[page_w], style=[
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge))]), Spacer(1, 4)]
        elif b["type"] == "table":
            df, cols = b["df"], b["columns"]
            colors_, flags = _cell_colors(b), _flagged(b)
            right = ParagraphStyle("cr", parent=st["cell"], alignment=2)
            head_r = ParagraphStyle("hr", parent=st["head"], alignment=2)

            def para(text: str, c: dict, bold: bool = False, color: str | None = None):
                text = T(text)
                text = f"<b>{text}</b>" if bold or c.get("bold") else text
                text = f'<font color="{color}">{text}</font>' if color else text
                return Paragraph(text, st["cell"] if _align(c) == "left" else right)

            data = [[Paragraph(T(c.get("label", c["key"])), st["head"] if _align(c) == "left" else head_r)
                     for c in cols]]
            for (_, row), row_colors in zip(df.iterrows(), colors_):
                data.append([para(cell_text(row, c), c, bold=bool(fg), color=fg)
                             for c, (_, fg) in zip(cols, row_colors)])
            # Width by the longest cell, but never narrower than the longest
            # header word or cell word, so nothing breaks mid-word ("Loa / ns").
            def weight(c) -> float:
                texts = [cell_text(r, c) for _, r in df.head(50).iterrows()]
                if b.get("total"):
                    texts.append(cell_text(pd.Series(b["total"]), c))
                words = str(c.get("label", c["key"])).split() + [w for t in texts for w in t.split()]
                return max(max(len(w) for w in words) * 1.15, *(len(t) for t in texts), 4) + 2   # + cell padding
            weights = [min(weight(c), 60) for c in cols]
            widths = [page_w * w / sum(weights) for w in weights]
            style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
                     ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5e7eb")),
                     ("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")])]
            for r, grouped in enumerate(_grouped(b), start=1):
                if grouped:                  # a group label: one band across the table
                    data[r] = [Paragraph(f"<b>{T(df.iloc[r - 1][cols[0]['key']])}</b>", st["cell"])] + [""] * (len(cols) - 1)
                    style += [("SPAN", (0, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f3f4f6"))]
            for r, (row_colors, flagged) in enumerate(zip(colors_, flags), start=1):
                if flagged:
                    style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor(FLAG_BG)),
                              ("LINEBEFORE", (0, r), (0, r), 3, colors.HexColor(FLAG_EDGE))]
                for ci, (bg, _) in enumerate(row_colors):
                    if bg:
                        style.append(("BACKGROUND", (ci, r), (ci, r), colors.HexColor(bg)))
            if b.get("total"):
                trow = pd.Series(b["total"])
                data.append([para(cell_text(trow, c), c, bold=True) for c in cols])
                style += [("LINEABOVE", (0, len(data) - 1), (-1, len(data) - 1), 1.5, colors.HexColor(YELLOW)),
                          ("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), colors.HexColor("#fffbea"))]
            tbl = Table(data, colWidths=widths, repeatRows=1, style=style)
            show_title = b["title"] and b["title"].lower() != last_heading.lower()
            parts = [Paragraph(T(b["title"]), st["b"]), tbl] if show_title else [tbl]
            if b["note"]:
                parts.append(Paragraph(T(b["note"]), st["note"]))
            # One keep-together per heading+table (never nested); a short
            # table on its own is kept whole, a long one may split. A long
            # table under a heading isn't glued to it whole (a page-tall table
            # would leave the page before it blank): the heading starts a new
            # page only when its first rows wouldn't fit under it.
            if pending and len(df) > 15:
                story += [CondPageBreak(55 * mm), *pending, *parts]
                pending = []
            elif pending or len(df) > 20:
                story.extend(parts)
            else:
                story.append(KeepTogether(parts))
            story.append(Spacer(1, 6))
        if pending:                     # keep the waiting heading with what was just added
            added = story[before:]
            del story[before:]
            _emit(added)
    if pending:
        story.extend(pending)

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(reg, 7.5)
        canvas.setFillColor(colors.HexColor("#9ca3af"))
        canvas.drawString(12 * mm, 7 * mm, _pdf_text(f"CollectionIQ · {model['title']} · {model['subtitle']} · Internal use only", uni)
                          .replace("&amp;", "&"))
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                      topMargin=10 * mm, bottomMargin=12 * mm,
                      title=f"{model['title']} {model['month']}").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ── Excel annex ──────────────────────────────────────────────────────────────

_BAD_SHEET = re.compile(r"[\[\]:*?/\\]")


def annex_frame(df: pd.DataFrame, cols: list[dict]) -> pd.DataFrame:
    """A table as the reader saw it, with numbers left as numbers: the same
    columns and headers as the report, a % column split into "<label> %" and
    its "(loans)" / "(of loans)" / "(₹ Cr)" columns. Never the engine's own
    column names (base, n, _roll_valid, MNT NAME...)."""
    out = {}
    for c in cols:
        key, label = c["key"], c.get("label", c["key"])
        if key not in df.columns:
            continue
        if c["fmt"] == "pct":
            out[label if label.endswith("%") else f"{label} %"] = df[key].round(2)
            for part, suffix in (("count", "(loans)"), ("base", "(of loans)"), ("amount", "(₹ Cr)")):
                if c.get(part) in df.columns:
                    out[f"{label} {suffix}"] = df[c[part]]
        elif c["fmt"] == "pp":
            out[f"{label} (pts)"] = pd.to_numeric(df[key], errors="coerce").round(2)
        elif c["fmt"] == "rs_cr":
            out[f"{label} (₹ Cr)"] = (pd.to_numeric(df[key], errors="coerce") / 1e7).round(2)
        elif c["fmt"] == "cr":
            out[f"{label} (₹ Cr)"] = pd.to_numeric(df[key], errors="coerce").round(2)
        else:
            out[label] = df[key]
    return pd.DataFrame(out)


def to_excel(model: dict) -> bytes:
    """Summary sheet, then every table in full (not just the rows shown in the
    report), then every call list with its proof columns."""
    from ui.components import _dateonly, style_excel_sheet
    sheets: list[tuple[str, pd.DataFrame]] = []
    summary = [{"Section": "Report", "Item": f"{model['title']} · {model['subtitle']} · {model['scope']}", "Detail": ""}]
    for b in model["blocks"]:
        if b["type"] == "glance":
            summary.append({"Section": "Verdict", "Item": b["verdict"]["text"], "Detail": ""})
            summary += [{"Section": "Went well", "Item": x, "Detail": ""} for x in b["good"]]
            summary += [{"Section": "Went wrong", "Item": x, "Detail": ""} for x in b["bad"]]
            summary += [{"Section": "Where to focus", "Item": f["title"], "Detail": f["detail"]} for f in b["focus"]]
            summary += [{"Section": "KPIs", "Item": it["label"],
                         "Detail": f'{it["value"]}' + (f' ({it["change"]} vs last month)' if it.get("change") else "")}
                        for it in b["tiles"]]
        elif b["type"] == "callout":
            summary.append({"Section": "Note", "Item": b["text"], "Detail": ""})
        elif b["type"] == "bullets":
            summary += [{"Section": "Summary", "Item": it["text"], "Detail": it.get("detail", "")} for it in b["items"]]
        elif b["type"] == "kpis":
            summary += [{"Section": b["title"] or "KPIs", "Item": it["label"],
                         "Detail": f'{it["value"]}' + (f' ({it["change"]} vs last month)' if it.get("change") else "")}
                        for it in b["items"]]
        elif b["type"] == "table" and b["sheet"] is not False and b.get("annex") is not None:
            sheets.append((b["sheet"], b["annex"]))
        elif b["type"] == "table" and b["sheet"] is not False:
            full = b["full"]
            if b.get("total"):
                full = pd.concat([full, pd.DataFrame([b["total"]])[[c for c in full.columns if c in b["total"]]]],
                                 ignore_index=True)
            sheets.append((b["sheet"], annex_frame(full, b["columns"])))
    sheets.insert(0, ("Summary", pd.DataFrame(summary)))
    for name, df in model.get("annex", []):
        sheets.append((name, df))

    buf, used = BytesIO(), set()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets:
            base = _BAD_SHEET.sub(" ", str(name))[:31] or "Sheet"
            sheet, k = base, 2
            while sheet in used:
                sheet = f"{base[:28]} {k}"; k += 1
            used.add(sheet)
            out = _dateonly(df)
            out.to_excel(writer, index=False, sheet_name=sheet)
            style_excel_sheet(writer.sheets[sheet], out)
    return buf.getvalue()

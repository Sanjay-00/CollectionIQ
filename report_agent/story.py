"""
The monthly report as a MODEL: an ordered list of blocks (headings, bullets,
KPI rows, tables), built from the same engines as the app. report_agent/
render.py turns one model into HTML, PDF and an Excel annex, so every format
says exactly the same thing.

Presets, by reader:
  - "Leadership": the whole portfolio on 1-2 pages.
  - "Regional" (default): Leadership plus branches, roll rates by branch,
    units needing attention, why it's happening, and alerts.
  - "Branch": one branch (the sidebar's Branch filter) with its executives
    and call lists.
"""
from __future__ import annotations

import pandas as pd

PRESETS = ["Regional", "Leadership", "Branch"]
TABLE_ROWS_ON_PAGE = 15        # longer tables show their worst rows; the Excel annex has all


# ── Block helpers ────────────────────────────────────────────────────────────

def heading(text: str, note: str = "") -> dict:
    return {"type": "heading", "text": text, "note": note}


def bullets(items: list[dict]) -> dict:
    """items: {"text", "tone": bad/good/info/plain, "detail"?}"""
    return {"type": "bullets", "items": items}


def kpis(title: str, items: list[dict]) -> dict:
    """items: {"label", "value" (formatted), "change" (str|None), "worse" (bool|None)}"""
    return {"type": "kpis", "title": title, "items": items}


def table(title: str, df: pd.DataFrame, columns: list[dict], note: str = "",
          full: pd.DataFrame | None = None, sheet: str | None = None, total: dict | None = None) -> dict:
    """columns: {"key", "label"?, "fmt": text|int|pct|pp|cr|rs_cr|count_change, "heat"?: bool,
    "good_if_up"?: bool, "count"?: column}. `full` is the untruncated frame for
    the Excel annex. `total`: an optional Total row (dict), worked out from ALL
    the data in view, never just the rows shown."""
    return {"type": "table", "title": title, "df": df.reset_index(drop=True), "columns": columns,
            "note": note, "full": df if full is None else full, "sheet": sheet or title[:31], "total": total}


def _unit_total(c: pd.DataFrame, p: pd.DataFrame | None, name_col: str, label: str = "Total") -> dict | None:
    """Every loan in view as one Total row (sum over sum), with last month's
    delinquency and the change -- matching the unit tables' columns."""
    from analysis.summary import portfolio_total
    t = portfolio_total(c, p)
    if not t:
        return None
    row = {name_col: label, "Accounts": int(t["Accounts"]), "Delinquent": int(t["Delinquent"]),
           "Delinquency%": float(t["Delinquency%"]), "SMA-2": int(t["SMA-2"]), "SMA-2%": float(t["SMA-2%"]),
           "NPA": int(t["NPA"]), "NPA%": float(t["NPA%"]), "NPA% (SOH)": float(t["NPA% (SOH)"]),
           "NPA SOH (Cr)": float(t["NPA SOH (Cr)"]),
           "Collection%": float(t["Collection%"]), "Strike%": float(t["Strike%"])}
    if "Prev Delinquency%" in t:
        row["Prev Delinquency%"], row["Δ Delinquency%"] = t["Prev Delinquency%"], t["Δ Delinquency%"]
    return row


# ── Formatting used inside sentences ─────────────────────────────────────────

def _move(change, unit: str = " pts") -> str:
    if change is None or pd.isna(change):
        return ""
    return "no change" if change == 0 else f"{'up' if change > 0 else 'down'} {abs(change):.1f}{unit}"


def _kpi(metrics: dict, key: str, label: str, kind: str, worse_if_up: bool, prev: dict | None = None) -> dict:
    """A % metric's change is shown in points (59.0% -> 60.1% is "▲ 1.10 pts"),
    an amount's or count's as a % change -- the same convention as the app."""
    from utils import fmt_value
    val, mom = metrics[key]
    value = f"{val:.2f}%" if kind == "pct" else fmt_value(val, kind)
    if kind == "pct":
        before = (prev or {}).get(key)
        change = None if before is None else round(val - before, 2)
        text = None if change is None else ("no change" if change == 0 else f"{'▲' if change > 0 else '▼'} {abs(change):.2f} pts")
    else:
        change = mom
        text = None if mom is None else f"{'▲' if mom >= 0 else '▼'} {abs(mom):.1f}%"
    return {"label": label, "value": value, "change": text,
            "worse": None if not change else (change > 0) == worse_if_up}


# ── Sections ─────────────────────────────────────────────────────────────────

def _summary(ac: dict, moves: dict) -> list[dict]:
    items = [{"text": h["title"], "detail": h["detail"], "tone": h["tone"]} for h in ac["headlines"]]
    for level in ("Region", "Branch"):
        for side, tone in (("worse", "bad"), ("better", "good")):
            u = (moves.get(level) or {}).get(side) or []
            if u:
                top = u[0]
                items.append({"tone": tone, "text": (
                    f"{level} with the biggest {'rise' if side == 'worse' else 'fall'} in delinquency: "
                    f"{top['name']} at {top['now']:.1f}% ({_move(top['change'])})")})
    return [heading("Summary", "Built from the data by fixed rules; every figure appears in a table below."),
            bullets(items)]


def _scoreboard(metrics: dict, prev_values: dict | None) -> list[dict]:
    def k(key: str, label: str, kind: str, worse_if_up: bool) -> dict:
        return _kpi(metrics, key, label, kind, worse_if_up, prev_values)
    return [
        heading("Scoreboard", "This month, with the change against last month."),
        kpis("Collections", [
            k("Month Demand", "Month Demand", "money", False),
            k("Total Collection", "Total Collection", "money", False),
            k("Collection %", "Collection %", "pct", False),
            k("Strike %", "Strike %", "pct", False),
            k("LCC%", "LCC %", "pct", False),
        ]),
        kpis("Book and risk", [
            k("Count", "Accounts", "count", False),
            k("SOH", "SOH", "money", True),
            k("Delinquency %", "Delinquency %", "pct", True),
            k("SMA-2 %", "SMA-2 %", "pct", True),
            k("NPA %", "NPA %", "pct", True),
            k("NPA % (SOH)", "NPA % (SOH)", "pct", True),
            k("Hard Bucket %", "Hard Bucket %", "pct", True),
        ]),
    ]


def _early_warning(ac: dict) -> list[dict]:
    from analysis import roll_flow as rf
    s = ac.get("roll_summary")
    if not s:
        return [heading("Early Warning"), bullets([{"tone": "plain", "text": "Needs last month's file to compare."}])]
    rows = [{"Step": step, "Loans": s[step]["n"], "% of loans": s[step]["pct"],
             "SOH (Cr)": s[step]["soh_cr"], "% of SOH": s[step]["soh_pct"],
             "Out of": f"{s[step]['base']:,} loans / ₹{s[step]['base_soh_cr']:,.2f} Cr"} for step in rf.step_labels()]
    blocks = [heading("Early Warning: Loans That Slipped",
                      "Loans in each bucket last month that are in a worse bucket now, by loans and by last month's SOH."),
              table("Roll steps", pd.DataFrame(rows), [
                  {"key": "Step", "fmt": "text"},
                  {"key": "% of loans", "label": "Loans", "fmt": "pct", "count": "Loans"},
                  {"key": "% of SOH", "label": "SOH", "fmt": "pct", "amount": "SOH (Cr)"},
                  {"key": "Out of", "fmt": "text"}], sheet="Early warning")]
    out = ac.get("outlook")
    if out:
        npa, nd = out["new_npa"], out["new_defaulters"]
        blocks.append(bullets([
            {"tone": "bad", "text": f"Next month, if this month's roll rates repeat: about {npa['n']:,} new NPAs "
                                    f"(likely {npa['low']:,} to {npa['high']:,}), about ₹{npa['soh'] / 1e7:,.2f} Cr SOH."},
            {"tone": "bad", "text": f"About {nd['n']:,} new defaulters (likely {nd['low']:,} to {nd['high']:,}), "
                                    f"about ₹{nd['soh'] / 1e7:,.2f} Cr SOH."}]))
    return blocks


_UNIT_COLS = [  # the shared reading order: size, delinquency (now, last month, change), buckets, collection
    {"key": "Accounts", "fmt": "int"},
    {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct", "count": "Delinquent", "heat": True},
    {"key": "Prev Delinquency%", "label": "Last Month", "fmt": "pct"},
    {"key": "Δ Delinquency%", "label": "Change", "fmt": "pp"},
    {"key": "SMA-2%", "label": "SMA-2", "fmt": "pct", "count": "SMA-2", "heat": True},
    {"key": "NPA%", "label": "NPA", "fmt": "pct", "count": "NPA", "heat": True},
    {"key": "NPA% (SOH)", "label": "NPA by SOH", "fmt": "pct", "amount": "NPA SOH (Cr)", "heat": True},
    {"key": "Collection%", "label": "Collection", "fmt": "pct", "good_if_up": True},
]


def _present(cols: list[dict], df: pd.DataFrame) -> list[dict]:
    return [c for c in cols if c["key"] in df.columns]


def _regions(c, p) -> list[dict]:
    from analysis.summary import unit_table
    reg = unit_table(c, p, "Region")
    if reg.empty:
        return []
    reg = reg.sort_values("Delinquency%", ascending=False)
    cols = _present([{"key": "Region", "fmt": "text"}, *_UNIT_COLS], reg)
    return [heading("Regions", "Worst delinquency first."),
            table("Regions", reg, cols, sheet="Regions", total=_unit_total(c, p, "Region"))]


def _branches(c, p) -> list[dict]:
    from analysis.summary import unit_table
    br = unit_table(c, p, "Branch")
    if br.empty:
        return []
    br = br.sort_values("Delinquency%", ascending=False)
    cols = _present([{"key": "Branch", "fmt": "text"}, {"key": "Region", "fmt": "text"}, *_UNIT_COLS], br)
    shown = br.head(TABLE_ROWS_ON_PAGE)
    note = "Worst delinquency first." + (f" Showing the worst {len(shown)} of {len(br)}; all are in the Excel annex."
                                          if len(br) > len(shown) else "")
    label = "Total" if len(shown) == len(br) else f"Total (all {len(br)} branches)"
    return [heading("Branches", note), table("Branches", shown, cols, full=br, sheet="Branches",
                                             total=_unit_total(c, p, "Branch", label))]


def _roll_by(c, grain: str, title: str) -> list[dict]:
    from analysis import roll_flow as rf
    by = rf.roll_steps_by(c, grain)
    if by.empty:
        return []
    name = {"branch": "Branch", "executive": "Executive", "region": "Region"}[grain]
    keep = [name, *(["Branch"] if grain == "executive" else []), *(["Region"] if grain != "region" else []),
            "Matched Accounts", "STD → Behind | %", "STD → Behind | Accounts", "STD → Behind | SOH %",
            "STD → Behind | SOH (Cr)", "1-30 → SMA-1+ | %", "1-30 → SMA-1+ | Accounts", "1-30 → SMA-1+ | SOH %",
            "1-30 → SMA-1+ | SOH (Cr)", "SMA-2 → NPA | %", "SMA-2 → NPA | Accounts", "SMA-2 → NPA | SOH %",
            "SMA-2 → NPA | SOH (Cr)",
            "Back to STD | %", "Back to STD | Accounts"]
    df = by[[k for k in keep if k in by.columns]]
    cols = [{"key": name, "fmt": "text"}, *([{"key": "Branch", "fmt": "text"}] if grain == "executive" else []),
            *([{"key": "Region", "fmt": "text"}] if grain != "region" else []),
            {"key": "Matched Accounts", "label": "Loans", "fmt": "int"},
            {"key": "STD → Behind | %", "label": "New defaulters", "fmt": "pct", "count": "STD → Behind | Accounts", "heat": True},
            {"key": "STD → Behind | SOH %", "label": "by SOH", "fmt": "pct", "amount": "STD → Behind | SOH (Cr)", "heat": True},
            {"key": "1-30 → SMA-1+ | %", "label": "1-30 → SMA-1+", "fmt": "pct", "count": "1-30 → SMA-1+ | Accounts", "heat": True},
            {"key": "1-30 → SMA-1+ | SOH %", "label": "by SOH", "fmt": "pct", "amount": "1-30 → SMA-1+ | SOH (Cr)", "heat": True},
            {"key": "SMA-2 → NPA | %", "label": "SMA-2 → NPA", "fmt": "pct", "count": "SMA-2 → NPA | Accounts", "heat": True},
            {"key": "SMA-2 → NPA | SOH %", "label": "by SOH", "fmt": "pct", "amount": "SMA-2 → NPA | SOH (Cr)", "heat": True},
            {"key": "Back to STD | %", "label": "Recovered", "fmt": "pct", "count": "Back to STD | Accounts", "good_if_up": True}]
    shown = df.head(TABLE_ROWS_ON_PAGE)
    note = "Most new defaulters first." + (f" Showing {len(shown)} of {len(df)}; all are in the Excel annex."
                                           if len(df) > len(shown) else "")
    total = None
    s = rf.roll_steps_summary(c)
    if s:
        unit = {"branch": "branches", "executive": "executives", "region": "regions"}[grain]
        total = {name: "Total" if len(shown) == len(df) else f"Total (all {len(df)} {unit})",
                 "Matched Accounts": len(rf._matched(c))}
        for step in ("STD → Behind", "1-30 → SMA-1+", "SMA-2 → NPA", "Back to STD"):
            total[f"{step} | %"], total[f"{step} | Accounts"] = s[step]["pct"], s[step]["n"]
            total[f"{step} | SOH %"], total[f"{step} | SOH (Cr)"] = s[step]["soh_pct"], s[step]["soh_cr"]
    return [heading(title, note), table(title, shown, _present(cols, df), full=df, sheet=title[:31], total=total)]


def _attention(ac: dict) -> list[dict]:
    items = []
    for grain, label in (("branch", "Branch"), ("executive", "Executive")):
        for u in ac["attention"][grain]["worst"]:
            region = f", {u['region']}" if u.get("region") else ""
            items.append({"tone": "bad", "text": f"{label} {u['name']}{region} ({u['accounts']:,} loans)",
                          "detail": "; ".join(u["reasons"])})
        for u in ac["attention"][grain]["improving"]:
            items.append({"tone": "good", "text": f"{label} {u['name']} improving", "detail": u["why"]})
    if not items:
        return []
    return [heading("Needs Attention", "Units well above the portfolio's rates, with the reasons. "
                                       "Units too small to judge fairly are left out."), bullets(items)]


def _why(c, p, curr_month) -> list[dict]:
    from analysis.portfolio_intelligence import compute_region_scorecard
    from analysis.root_cause import compute_chronic_shock_split, compute_insurance_split, compute_region_why_table
    blocks = []
    why = compute_region_why_table(c, compute_region_scorecard(c, p), as_of=curr_month)
    if not why.empty:
        df = why[[k for k in ("Region", "Dominant Driver", "Driver Share %", "Delinquent Accounts") if k in why.columns]]
        blocks += [heading("Why It's Happening", "The cause behind the biggest share of each region's delinquent loans."),
                   table("Main driver by region", df, [
                       {"key": "Region", "fmt": "text"}, {"key": "Dominant Driver", "label": "Main driver", "fmt": "text"},
                       {"key": "Driver Share %", "label": "Share of delinquent loans", "fmt": "pct"},
                       {"key": "Delinquent Accounts", "label": "Delinquent loans", "fmt": "int"}], sheet="Why by region")]
    ins, arr = compute_insurance_split(c), compute_chronic_shock_split(c)
    items = []
    if not ins.empty:
        n = ins["Delinquent Accounts"].sum()
        items.append({"tone": "info", "text": "Of all delinquent loans: " + ", ".join(
            f"{g} {ins[g].sum() / n * 100:.1f}%" for g in ("Insurance-Only", "Installment-Only", "Both", "Other")
            if g in ins.columns)})
    if not arr.empty:
        from analysis.root_cause import ARREARS_GROUPS
        n = arr["Delinquent Accounts"].sum()
        items.append({"tone": "info", "text": "Paying vs not paying: " + ", ".join(
            f"{g} {arr[g].sum() / n * 100:.1f}%" for g in ARREARS_GROUPS if g in arr.columns)})
    if items:
        blocks += [bullets(items)]
    return blocks


def _business(c, curr_month) -> list[dict]:
    from analysis.summary import business_snapshot
    b = business_snapshot(c, curr_month)
    if not b.get("month"):
        return []
    mom = lambda v: None if v is None else f"{'▲' if v >= 0 else '▼'} {abs(v):.1f}%"
    items = [{"label": f"New loans ({b['month']})", "value": f"{b['accounts']:,}", "change": mom(b["accounts_mom_pct"]),
              "worse": None if not b["accounts_mom_pct"] else b["accounts_mom_pct"] < 0},
             {"label": "Amount funded", "value": f"₹{b['funded_cr']:,.2f} Cr", "change": mom(b["funded_mom_pct"]),
              "worse": None if not b["funded_mom_pct"] else b["funded_mom_pct"] < 0},
             {"label": "Average ticket", "value": f"₹{b['avg_ticket_l']:,.2f} L", "change": None, "worse": None}]
    lines = []
    if b["top_segments"]:
        lines.append({"tone": "plain", "text": "Top segments: " + ", ".join(
            f"{x['Segment']} ({x['Accounts']:,})" for x in b["top_segments"])})
    if b["top_branches"]:
        lines.append({"tone": "plain", "text": "Top branches: " + ", ".join(
            f"{x['Branch']} ({x['Accounts This Month']:,})" for x in b["top_branches"])})
    return [heading("New Business"), kpis("", items), *([bullets(lines)] if lines else [])]


def _alerts(alerts, alerts_prev) -> list[dict]:
    from analysis.summary import alert_snapshot
    rows = alert_snapshot(alerts, alerts_prev)
    if not rows:
        return []
    df = pd.DataFrame([{"Alert": r["title"], "Severity": r["severity"].title(), "Loans": r["count"],
                        "Last Month": r["prev"], "Change": None if r["prev"] is None else r["count"] - r["prev"]}
                       for r in rows])
    return [heading("Risk Alerts"), table("Risk alerts", df, [
        {"key": "Alert", "fmt": "text"}, {"key": "Severity", "fmt": "text"}, {"key": "Loans", "fmt": "int"},
        {"key": "Last Month", "fmt": "int"}, {"key": "Change", "fmt": "count_change"}], sheet="Alerts")]


def _priorities(ac: dict) -> list[dict]:
    if not ac["focus"]:
        return []
    df = pd.DataFrame([{"#": f["rank"], "Group": f["name"], "Loans": f["loans"], "SOH (Rs)": f["soh"],
                        "Likely to slip if ignored": (f["risk"] or {}).get("n"),
                        "What to do": f["action"]} for f in ac["focus"]])
    return [heading("What To Do First", "Today's loans, cheapest and most urgent first. \"Likely to slip\" "
                                        "uses this month's roll rate for each loan's bucket."),
            table("Priorities", df, [
                {"key": "#", "fmt": "int"}, {"key": "Group", "fmt": "text"}, {"key": "Loans", "fmt": "int"},
                {"key": "SOH (Rs)", "label": "SOH", "fmt": "rs_cr"},
                {"key": "Likely to slip if ignored", "fmt": "int"}, {"key": "What to do", "fmt": "text"}],
                sheet="Priorities")]


def _executives(c, p) -> list[dict]:
    from analysis.executive_scorecard import compute_executive_scorecard
    ex = compute_executive_scorecard(c, min_accounts=1, df_prev=p)
    if ex.empty:
        return []
    ex = ex.sort_values("Delinquency %", ascending=False)
    cols = _present([{"key": "MNT NAME", "label": "Executive", "fmt": "text"}, {"key": "Accounts", "fmt": "int"},
                     {"key": "Delinquency %", "label": "Delinquent", "fmt": "pct", "count": "Delinquent", "heat": True},
                     {"key": "Prev Delinquency %", "label": "Last Month", "fmt": "pct"},
                     {"key": "Δ Delinquency %", "label": "Change", "fmt": "pp"},
                     {"key": "NPA %", "label": "NPA", "fmt": "pct", "count": "NPA", "heat": True},
                     {"key": "Collection %", "fmt": "pct", "good_if_up": True},
                     {"key": "Strike Rate %", "label": "Strike %", "fmt": "pct", "good_if_up": True}], ex)
    total = _unit_total(c, p, "MNT NAME")
    if total:      # the executive table spells its columns with a space ("Delinquency %")
        total = {**total, "Delinquency %": total["Delinquency%"], "NPA %": total["NPA%"],
                 "Collection %": total["Collection%"], "Strike Rate %": total["Strike%"],
                 "Prev Delinquency %": total.get("Prev Delinquency%"), "Δ Delinquency %": total.get("Δ Delinquency%")}
    return [heading("Executives", "Worst delinquency first."), table("Executives", ex, cols, sheet="Executives",
                                                                     total=total)]


def _call_lists(ac: dict) -> list[dict]:
    blocks = []
    for t in ac["calls"]:
        df = t["loans"]
        cols = [{"key": k, "fmt": "rs_cr" if k == "SOH" else ("pct" if k == "Arrears / EMI" else "text")}
                for k in ("Loan No", "Cust Name", "Executive", "Bucket Now", "Arrears / EMI", "SOH") if k in df.columns]
        cols = [{"key": "Arrears / EMI", "fmt": "num"} if c["key"] == "Arrears / EMI" else c for c in cols]
        shown = df.head(10)
        blocks.append(table(f"Call list: {t['name']}", shown, cols, full=df, sheet=f"Call {len(blocks) + 1}",
                            note=f"Top {len(shown)} of {len(df):,} by SOH; the full list with proof columns is in the Excel annex."))
    return [heading("Call Lists")] + blocks if blocks else []


def _definitions() -> list[dict]:
    from ui.glossary import GLOSSARY
    keep = ["STD → Behind", "1-30 → SMA-1+", "SMA-1 → SMA-2+", "SMA-2 → NPA", "Back to STD", "Insurance-Only",
            "NPA% (SOH)", "Not Paying 3M+", "Hard Bucket", "Delinquent Accounts", "Concern Score"]
    rows = [{"Term": k, "Meaning": GLOSSARY[k]} for k in keep if k in GLOSSARY]
    rows += [
        {"Term": "Delinquency %", "Meaning": "Loans with any EMI or charge overdue (Arrears/EMI above 0), as % of all loans."},
        {"Term": "NPA %", "Meaning": "Loans 3 or more EMIs overdue, as % of all loans (NPA % (SOH): by money)."},
        {"Term": "Forecast", "Meaning": "Today's loans in each bucket x this month's rate of moving out of that bucket; "
                                        "the range covers about 95% of likely outcomes."},
    ]
    return [heading("Definitions"), table("Definitions", pd.DataFrame(rows),
                                          [{"key": "Term", "fmt": "text"}, {"key": "Meaning", "fmt": "text"}],
                                          sheet="Definitions")]


# ── Optional detail sections (off by default) ────────────────────────────────

_RENAME = {"RegionName": "Region", "Unit": "Branch", "MNT NAME": "Executive", "curr_bucket": "Bucket",
           "SegmentName": "Segment", "FUEL_TYPE": "Fuel Type"}
_RUPEE_COLS = {"SOH", "Closing Arrears", "Loan Amount", "Last Receipt Amount", "Arrears against Inst+Exp",
               "Month Due-Inst", "Month Due-Exp"}


_COUNT_FOR = {"Delinquency%": "Delinquent", "Delinquency %": "Delinquent"}
_AMOUNT_FOR = {"NPA% (SOH)": "NPA SOH (Cr)", "NPA % (SOH)": "NPA SOH (Cr)"}   # % by SOH -> its ₹ Cr


def _count_column(pct_col: str, columns) -> str | None:
    """The count behind a % column, if the table has it ("NPA%" -> "NPA")."""
    cand = _COUNT_FOR.get(pct_col) or pct_col.replace("%", "").strip()
    return cand if cand in columns and cand != pct_col else None


def _auto_columns(df: pd.DataFrame, heat=(), good=()) -> list[dict]:
    """A column spec from the column names: % -> pct (with its count in the
    same cell when the table has it, and the count column then not shown on
    its own), (Cr) -> crore, rupee amounts -> ₹, counts -> int, else text."""
    counts = {c: _count_column(c, df.columns) for c in df.columns if "%" in c}
    amounts = {c: a for c, a in _AMOUNT_FOR.items() if c in df.columns and a in df.columns}
    hidden = {v for v in counts.values() if v} | set(amounts.values())
    cols = []
    for c in df.columns:
        if c in hidden:
            continue
        num = pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        if num and ("%" in c):
            f = "pct"
        elif num and "(Cr)" in c:
            f = "cr"
        elif num and c in _RUPEE_COLS:
            f = "inr"
        elif num and (c in ("Arrears / EMI",) or "(L)" in c):
            f = "num"
        elif num:
            f = "int"
        else:
            f = "text"
        cols.append({"key": c, "fmt": f, "heat": c in heat, "good_if_up": c in good, "count": counts.get(c),
                     "amount": amounts.get(c)})
    return cols


def _list_table(title: str, df: pd.DataFrame, keep: list[str], n: int = 20, note: str = "",
                heat=(), good=(), sheet: str | None = None) -> dict:
    full = df.rename(columns=_RENAME)
    keep = [_RENAME.get(k, k) for k in keep if _RENAME.get(k, k) in full.columns]
    shown = full[keep].head(n)
    extra = f" Showing {len(shown)} of {len(full):,}; all are in the Excel annex." if len(full) > len(shown) else ""
    return table(title, shown, _auto_columns(shown, heat, good), note=(note + extra).strip(),
                 full=full, sheet=sheet or title[:31])


def _x_overdue(c, p, m) -> list[dict]:
    from analysis.portfolio_intelligence import compute_overdue_demand_scorecard
    od = compute_overdue_demand_scorecard(c)
    good = ("Overdue Collection %", "Month Demand Collection %", "Overall Collection %")
    keep = ["Accounts", "Overdue (Cr)", "Overdue Collection %", "Month Demand (Cr)", "Month Demand Collection %",
            "Overall Collection %"]
    blocks = [heading("Overdue vs Month Demand", "A payment clears last month's overdue first; only the rest counts "
                                                 "against this month's demand. Lowest overdue collection first.")]
    for level in ("region", "branch"):
        df = od.get(level, pd.DataFrame())
        if not df.empty:
            name = "Region" if level == "region" else "Branch"
            blocks.append(_list_table(f"Overdue vs demand by {level}", df, [name, *(["Region"] if level == "branch" else []), *keep],
                                      n=15, good=good))
    return blocks if len(blocks) > 1 else []


def _x_business(c, p, m) -> list[dict]:
    from analysis.new_business import compute_new_advances_by_dimension, compute_new_advances_trend
    by = compute_new_advances_by_dimension(c, as_of=m).get("region", pd.DataFrame())
    trend = compute_new_advances_trend(c, as_of=m, months=12)
    blocks = [heading("New Business Detail")]
    if not by.empty:
        blocks.append(_list_table("New business by region", by, ["Region", "Accounts This Month", "Funded (Cr)",
                                                                "Avg Ticket (L)", "Prev Accounts", "Accounts MoM %"],
                                  good=("Accounts MoM %",)))
    if not trend.empty:
        blocks.append(_list_table("New business, last 12 months", trend.sort_values("Month", ascending=False),
                                  ["Month", "Accounts", "Funded (Cr)", "Avg Ticket (L)"], n=12))
    return blocks if len(blocks) > 1 else []


def _x_products(c, p, m) -> list[dict]:
    from utils import segment_column, unit_metrics
    heat = ("SMA-2%", "NPA%", "NPA% (SOH)")
    blocks = [heading("Segments and Sourcing", "Highest NPA first.")]
    for col, label in ((segment_column(c), "Segment"), ("FUEL_TYPE", "Fuel Type"), ("SRC Name", "Source")):
        if not col or col not in c.columns:
            continue
        df = unit_metrics(c, [col]).rename(columns={col: label})
        if not df.empty:
            blocks.append(_list_table(f"By {label.lower()}", df.sort_values("NPA%", ascending=False),
                                      [label, "Accounts", "SMA-2", "SMA-2%", "NPA", "NPA%", "NPA% (SOH)", "NPA SOH (Cr)",
                                       "Collection%", "SOH (Cr)"], n=15, heat=heat, good=("Collection%",)))
    return blocks if len(blocks) > 1 else []


def _x_top_accounts(c, p, m) -> list[dict]:
    from analysis.exposure import compute_top_accounts
    df, summ = compute_top_accounts(c, n=50)
    if df.empty:
        return []
    return [heading("Top At-Risk Accounts", f"The largest SOH among delinquent loans: the top 20 hold "
                                            f"₹{df.head(20)['SOH'].sum() / 1e7:,.2f} Cr."),
            _list_table("Top at-risk accounts", df, ["Loan No", "Cust Name", "RegionName", "Unit", "MNT NAME",
                                                     "curr_bucket", "Arrears / EMI", "SOH"])]


def _x_fleet(c, p, m) -> list[dict]:
    from analysis.exposure import compute_fleet_exposure
    from config import FLEET_MIN_LOANS
    fl = compute_fleet_exposure(c)
    if not fl.get("count"):
        return []
    return [heading("Fleet Operators", f"Customers (same mobile number) with {FLEET_MIN_LOANS} or more loans."),
            bullets([{"tone": "info", "text": f"{fl['count']:,} fleet operators, ₹{fl['total_soh_cr']:,.2f} Cr SOH; "
                                              f"{fl['npa_operators']:,} of them have at least one NPA loan."}]),
            _list_table("Largest fleet operators", fl["top_df"], list(fl["top_df"].columns))]


def _x_repossession(c, p, m) -> list[dict]:
    from analysis.exposure import compute_repossession_list
    from config import REPOSSESSION_BUCKETS, REPOSSESSION_EXCLUDE_STATUSES, REPOSSESSION_WINDOW_MONTHS
    df = compute_repossession_list(c, as_of=m)
    if df.empty:
        return []
    df = df.sort_values("SOH", ascending=False)
    return [heading("Repossession Candidates",
                    f"{', '.join(REPOSSESSION_BUCKETS)} loans agreed in the last {REPOSSESSION_WINDOW_MONTHS} months, "
                    f"excluding {', '.join(REPOSSESSION_EXCLUDE_STATUSES)}: {len(df):,} loans, "
                    f"₹{df['SOH'].sum() / 1e7:,.2f} Cr SOH. Largest first."),
            _list_table("Repossession candidates", df, ["Loan No", "Cust Name", "RegionName", "Unit", "MNT NAME",
                                                        "curr_bucket", "Arrears / EMI", "SOH", "Vehicle Description"])]


def _x_good(c, p, m) -> list[dict]:
    from analysis.exposure import compute_good_customers
    from config import GOOD_CUSTOMER_MIN_LCC_PCT, GOOD_CUSTOMER_MIN_TENURE_PCT
    df = compute_good_customers(c)
    if df.empty:
        return []
    return [heading("Good Customers", f"Refinance and relationship candidates: {GOOD_CUSTOMER_MIN_TENURE_PCT}%+ of "
                                      f"tenure done and LCC {GOOD_CUSTOMER_MIN_LCC_PCT}%+ ({len(df):,} loans)."),
            _list_table("Good customers", df, ["Loan No", "Cust Name", "RegionName", "Unit", "Tenure Completed %",
                                               "LCC%", "SOH"], good=("LCC%",))]


def _x_recovery(c, p, m) -> list[dict]:
    from analysis.portfolio_intelligence import compute_executive_recovery
    df = compute_executive_recovery(c)
    if df.empty:
        return []
    keep = ["Executive", "Region", "Accounts", "Rescued", "Slipped", "Net Recovery", "Collection%"]
    return [heading("Executive Recovery", "Rescued: moved from SMA-1/SMA-2/NPA to a better bucket. "
                                          "Slipped: moved to any worse bucket."),
            _list_table("Most rescued", df, keep, n=10, good=("Collection%",), sheet="Executive recovery"),
            _list_table("Most slipped", df.sort_values("Net Recovery"), keep, n=10, good=("Collection%",),
                        sheet="Recovery (worst)")]


def _x_rankings(c, p, m) -> list[dict]:
    from analysis.executive_scorecard import compute_executive_scorecard
    df = compute_executive_scorecard(c, df_prev=p)
    if df.empty:
        return []
    keep = ["Executive (Branch)", "Accounts", "Collection %", "Strike Rate %", "Delinquency %", "NPA %"]
    good = ("Collection %", "Strike Rate %")
    return [heading("Executive Rankings", "By Collection %."),
            _list_table("Best collection", df, keep, n=10, good=good, heat=("Delinquency %", "NPA %"),
                        sheet="Executives (best)"),
            _list_table("Lowest collection", df.sort_values("Collection %"), keep, n=10, good=good,
                        heat=("Delinquency %", "NPA %"), sheet="Executives (lowest)")]


# key: (label shown in the app, builder)
OPTIONAL_SECTIONS = {
    "overdue": ("Overdue vs Month Demand", _x_overdue),
    "business": ("New business detail (by region, 12-month trend)", _x_business),
    "products": ("Segments and sourcing", _x_products),
    "top_accounts": ("Top at-risk accounts", _x_top_accounts),
    "fleet": ("Fleet operators", _x_fleet),
    "repossession": ("Repossession candidates", _x_repossession),
    "good": ("Good customers", _x_good),
    "recovery": ("Executive recovery", _x_recovery),
    "rankings": ("Executive rankings", _x_rankings),
}


# ── The report ───────────────────────────────────────────────────────────────

def build_report(df_curr: pd.DataFrame, df_prev: pd.DataFrame, curr_month: str, prev_month: str | None,
                 preset: str = "Regional", filters: dict | None = None,
                 alerts: list | None = None, alerts_prev: list | None = None,
                 extras: list[str] | tuple = ()) -> dict:
    """The report model: {title, subtitle, meta, data_notes, blocks}. `extras`:
    keys of OPTIONAL_SECTIONS to add before the definitions."""
    from analysis import action_center
    from analysis.summary import biggest_moves
    from utils import compute_metrics
    filters = filters or {}
    if preset == "Branch" and filters.get("Branch", "All") in ("", "All", None):
        raise ValueError("Pick one branch in the sidebar's Branch filter for a Branch report.")
    ac = action_center.build(df_curr, df_prev)
    metrics = compute_metrics(df_curr, df_prev)
    prev_values = ({k: v[0] for k, v in compute_metrics(df_prev, df_prev.iloc[0:0]).items()}
                   if df_prev is not None and len(df_prev) else None)
    moves = {lvl: biggest_moves(df_curr, df_prev, lvl, 1) for lvl in ("Region", "Branch")}

    blocks = _summary(ac, moves) + _scoreboard(metrics, prev_values) + _early_warning(ac)
    if preset == "Leadership":
        blocks += _regions(df_curr, df_prev) + _business(df_curr, curr_month) + _priorities(ac)
    elif preset == "Regional":
        blocks += (_regions(df_curr, df_prev) + _branches(df_curr, df_prev)
                   + _roll_by(df_curr, "branch", "Roll Rates by Branch") + _attention(ac)
                   + _why(df_curr, df_prev, curr_month) + _business(df_curr, curr_month)
                   + _alerts(alerts, alerts_prev) + _priorities(ac))
    else:  # Branch
        blocks += (_executives(df_curr, df_prev) + _roll_by(df_curr, "executive", "Roll Rates by Executive")
                   + _priorities(ac) + _call_lists(ac))
    for key in extras:
        if key in OPTIONAL_SECTIONS:
            blocks += OPTIONAL_SECTIONS[key][1](df_curr, df_prev, curr_month)
    blocks += _definitions()

    scope = " | ".join(f"{k}: {v}" for k, v in filters.items() if v not in (None, "", "All", "None")) or "Whole portfolio"
    return {
        "title": f"{preset} Report",
        "subtitle": f"{curr_month}" + (f" vs {prev_month}" if prev_month else ""),
        "scope": scope,
        "preset": preset,
        "month": curr_month,
        "data_notes": list(df_curr.attrs.get("data_fixes", []) or []),
        "accounts": int(metrics["Count"][0]),
        "blocks": blocks,
        "ai_summary": None,
        # Every call list in full, with the columns that prove each loan's
        # place on it, for the Excel annex (regional managers forward these).
        "annex": [(f"Call {i} {t['name'].split(':')[0]}", t["loans"]) for i, t in enumerate(ac["calls"], start=1)],
    }

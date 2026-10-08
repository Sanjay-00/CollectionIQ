"""
The monthly report as a MODEL: an ordered list of blocks, built from the same
engines as the app. report_agent/render.py turns one model into HTML, PDF or
an Excel annex, so every format says exactly the same thing.

It reads as a story, not a dump: every report opens with "At a glance" (a
verdict line, KPI tiles, what went well, what went wrong, where to focus),
then ranked bars (each unit's delinquency, its change, and an On track /
Watch / Act now status against the report's own average), then where loans
are slipping, why, and what to do first. Full tables live in the Excel annex.

Levels, by reader (report_agent/packs.py makes one file per unit in view):
  - "Leadership": everything in view, high level: zones and regions compared.
  - "Zone": one per zone: its regions compared, best and worst branches.
  - "Regional" (default): one per region, in detail: branches, executives,
    slipping, why, what to do first.
  - "Branch": one per branch: its executives, slipping, call lists.
"""
from __future__ import annotations

import pandas as pd

PRESETS = ["Leadership", "Zone", "Regional", "Branch"]
DEFAULT_PRESET = "Regional"
TABLE_ROWS_ON_PAGE = 15        # longer tables show their worst rows; the Excel annex has all
BARS_ALL_UP_TO = 12            # up to this many units: all as bars; more: the worst and the best few
BARS_EACH_SIDE = 6


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


def glance(verdict: dict, tiles: list[dict], good: list[str], bad: list[str], focus: list[dict]) -> dict:
    """The opening story. verdict: {"tone": good/bad/mixed/info, "text"}; tiles:
    like kpis items; good / bad: short sentences; focus: {"title", "detail"}."""
    return {"type": "glance", "verdict": verdict, "tiles": tiles, "good": good, "bad": bad, "focus": focus}


def bars(title: str, rows: list[dict], note: str = "", value_head: str = "", extra_heads: tuple = (),
         full: pd.DataFrame | None = None, sheet: str | None = None, top: float | None = None) -> dict:
    """A ranked bar chart. rows: {"label", "sub"?, "value" (0-100), "text", "change"?, "worse"?,
    "status"?: red/amber/green, "extra"?: [str]}. Bars share one scale (the
    largest value, or `top` so two blocks side by side compare), so lengths
    compare across rows."""
    top = top or max([r["value"] for r in rows if r.get("value") is not None] or [0])
    return {"type": "bars", "title": title, "rows": rows, "note": note, "max": top or 1,
            "value_head": value_head, "extra_heads": list(extra_heads),
            "full": full, "sheet": sheet or title[:31]}


def actions(title: str, items: list[dict], note: str = "") -> dict:
    """Numbered things to do. items: {"rank", "title", "loans", "soh_cr", "action", "risk"?}"""
    return {"type": "actions", "title": title, "items": items, "note": note}


def callout(text: str, tone: str = "info") -> dict:
    return {"type": "callout", "text": text, "tone": tone}


# ── Formatting used inside sentences ─────────────────────────────────────────

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

_STATUS_TEXT = {"red": "Act now", "amber": "Watch", "green": "On track"}
ACT_NOW_GAP_PTS = 5.0       # delinquency this many points above the report's own average: Act now
_SIGNAL = {"Collection %": 0.5, "Delinquency %": 0.25, "NPA %": 0.25}   # smaller moves read as "steady"


def _status(value, bench) -> str | None:
    """On track (at or below the report's average delinquency), Watch (above
    it), Act now (ACT_NOW_GAP_PTS or more above it)."""
    if value is None or bench is None or pd.isna(value) or pd.isna(bench):
        return None
    if value >= bench + ACT_NOW_GAP_PTS:
        return "red"
    return "amber" if value > bench else "green"


def _pp(change) -> str | None:
    if change is None or pd.isna(change):
        return None
    return "no change" if round(change, 1) == 0 else f"{'▲' if change > 0 else '▼'} {abs(change):.1f} pts"


def _f(v, digits: int = 1) -> str:
    return "-" if v is None or pd.isna(v) else f"{float(v):.{digits}f}%"


def _ranked(title: str, note: str, df: pd.DataFrame, make_row, sheet: str) -> list[dict]:
    """df sorted worst first -> one bars block, or the worst and best few when long."""
    if len(df) <= BARS_ALL_UP_TO:
        return [heading(title, note), bars(title, [make_row(r) for _, r in df.iterrows()], full=df, sheet=sheet,
                                           value_head="Delinquent", extra_heads=("NPA", "Collection"))]
    worst, best = df.head(BARS_EACH_SIDE), df.tail(BARS_EACH_SIDE).iloc[::-1]
    note += f" The {BARS_EACH_SIDE} worst and {BARS_EACH_SIDE} best of {len(df)}; every one is in the Excel annex."
    worst_rows = [make_row(r) for _, r in worst.iterrows()]
    top = max(r["value"] for r in worst_rows)           # one scale for both, so the bars compare
    return [heading(title, note),
            bars("Needing the most help", worst_rows, full=df, sheet=sheet, top=top,
                 value_head="Delinquent", extra_heads=("NPA", "Collection")),
            bars("Doing best", [make_row(r) for _, r in best.iterrows()], value_head="Delinquent", top=top,
                 extra_heads=("NPA", "Collection"))]


def _bench_note(bench: float) -> str:
    return (f"Delinquency, worst first; this report's average is {bench:.1f}%. On track: at or below it. "
            f"Watch: above it. Act now: {ACT_NOW_GAP_PTS:.0f}+ pts above it.")


def _units_ranked(c, p, grain: str, title: str, bench: float) -> list[dict]:
    """Zones, regions or branches as ranked bars of delinquency."""
    from analysis.summary import unit_table
    t = unit_table(c, p, grain)
    if len(t) < 2:
        return []
    t = t.sort_values("Delinquency%", ascending=False).reset_index(drop=True)
    parent = {"Region": "Zone", "Branch": "Region"}.get(grain)
    show_parent = bool(parent and parent in t.columns and t[parent].nunique() > 1)

    def row(r) -> dict:
        change = r.get("Δ Delinquency%")
        return {"label": str(r[grain]),
                "sub": f"{int(r['Accounts']):,} loans" + (f" · {r[parent]}" if show_parent else ""),
                "value": float(r["Delinquency%"]), "text": _f(r["Delinquency%"]),
                "change": _pp(change), "worse": None if change is None or pd.isna(change) else change > 0,
                "status": _status(r["Delinquency%"], bench),
                "extra": [_f(r.get("NPA%")), _f(r.get("Collection%"))]}
    return _ranked(title, _bench_note(bench), t, row, sheet=title[:31])


def _executives_ranked(c, p, bench: float) -> list[dict]:
    from analysis.executive_scorecard import compute_executive_scorecard
    ex = compute_executive_scorecard(c, df_prev=p)          # executives big enough to judge fairly
    if len(ex) < 2:
        return []
    ex = ex.sort_values("Delinquency %", ascending=False).reset_index(drop=True)
    many_branches = "Unit" in ex.columns and ex["Unit"].nunique() > 1

    def row(r) -> dict:
        change = r.get("Δ Delinquency %")
        return {"label": str(r["MNT NAME"]),
                "sub": f"{int(r['Accounts']):,} loans" + (f" · {r['Unit']}" if many_branches else ""),
                "value": float(r["Delinquency %"]), "text": _f(r["Delinquency %"]),
                "change": _pp(change), "worse": None if change is None or pd.isna(change) else change > 0,
                "status": _status(r["Delinquency %"], bench),
                "extra": [_f(r.get("NPA %")), _f(r.get("Collection %"))]}
    return _ranked("Executives Ranked", _bench_note(bench), ex, row, sheet="Executives")


def _glance(c, p, metrics: dict, prev_values: dict | None, ac: dict, child: str | None) -> list[dict]:
    """The opening story: a verdict, six KPI tiles, what went well, what went
    wrong, and where to focus. Every figure comes from the same engines as the
    sections after it."""
    from analysis.summary import biggest_moves
    tiles = [_kpi(metrics, "Collection %", "Collection", "pct", False, prev_values),
             _kpi(metrics, "Delinquency %", "Delinquency", "pct", True, prev_values),
             _kpi(metrics, "NPA %", "NPA", "pct", True, prev_values),
             _kpi(metrics, "NPA % (SOH)", "NPA by SOH", "pct", True, prev_values),
             {**_kpi(metrics, "SOH", "SOH", "money", True, prev_values), "worse": None},   # neither good nor bad
             _kpi(metrics, "Count", "Loans", "count", False, prev_values)]

    good, bad = [], []
    if prev_values:
        for key, label in (("Collection %", "collection"), ("Delinquency %", "delinquency"), ("NPA %", "NPA")):
            before = prev_values.get(key)
            if before is None:
                continue
            now = metrics[key][0]
            change = now - before
            if abs(change) < _SIGNAL[key]:
                continue
            better = (change > 0) if key == "Collection %" else (change < 0)
            (good if better else bad).append(
                f"{label} {'up' if change > 0 else 'down'} {abs(change):.1f} pts to {now:.1f}%")
    if not prev_values:
        verdict = {"tone": "info", "text": "No last month to compare with: this month's position only."}
    elif good and bad:
        verdict = {"tone": "mixed", "text": f"Mixed month: {', '.join(good)}, but {', '.join(bad)}."}
    elif bad:
        verdict = {"tone": "bad", "text": f"A tough month: {', '.join(bad)}."}
    elif good:
        verdict = {"tone": "good", "text": f"A good month: {', '.join(good)}."}
    else:
        verdict = {"tone": "info", "text": "A steady month: collection, delinquency and NPA all within a small move of last month."}

    moves = {"worse": [], "better": []}
    if child and p is not None and len(p):
        try:
            moves = biggest_moves(c, p, child, 1)
        except (KeyError, ValueError):
            pass
    if moves["worse"] and verdict["tone"] in ("bad", "mixed"):
        w = moves["worse"][0]
        verdict["text"] += f" {w['name']} is the main drag (delinquency {_pp(w['change'])})."

    # The verdict and the tiles already carry the headline moves, so these
    # lists lead with what they don't: which unit, how many loans, how much.
    well, wrong = [], []
    if moves["better"]:
        m = moves["better"][0]
        well.append(f"{m['name']} improved most: delinquency {_pp(m['change'])} to {m['now']:.1f}%")
    if moves["worse"]:
        m = moves["worse"][0]
        wrong.append(f"{m['name']} slipped most: delinquency {_pp(m['change'])} to {m['now']:.1f}%")
    cured = (ac.get("roll_summary") or {}).get("Back to STD")
    if cured and cured["n"]:
        well.append(f"{cured['n']:,} loans recovered to STD (₹{cured['soh_cr']:,.2f} Cr)")
    for h in ac["headlines"]:
        text = f"{h['title']} (₹{h['soh'] / 1e7:,.2f} Cr)" if h.get("soh") else h["title"]
        (well if h["tone"] == "good" else wrong if h["tone"] == "bad" else []).append(text)
    well = list(dict.fromkeys(well + [g[0].upper() + g[1:] for g in good]))      # no line twice
    wrong = list(dict.fromkeys(wrong + [b[0].upper() + b[1:] for b in bad]))
    focus = [{"title": f["name"], "detail": f"{f['loans']:,} loans, ₹{f['soh'] / 1e7:,.2f} Cr. {f['action']}"}
             for f in ac["focus"][:3]]
    return [heading("At a Glance"), glance(verdict, tiles, well[:3] or ["Nothing stood out."],
                                           wrong[:3] or ["Nothing stood out."], focus)]


def _slipping(c, ac: dict, child: str | None) -> list[dict]:
    """The roll steps as bars, the outlook if they repeat, and where the new
    defaulters come from (by the next level down)."""
    from analysis import roll_flow as rf
    s = ac.get("roll_summary")
    if not s:
        return [heading("Where Loans Are Slipping"), callout("Needs last month's file to compare.")]
    cure = rf.step_labels()[-1]
    rows = [{"label": step, "sub": f"of {s[step]['base']:,} loans", "value": s[step]["pct"],
             "text": f"{s[step]['pct']:.1f}% ({s[step]['n']:,})",
             "extra": [f"₹{s[step]['soh_cr']:,.2f} Cr", _f(s[step]["soh_pct"])],
             "color": "green" if step == cure else "red"} for step in rf.step_labels()]
    blocks = [heading("Where Loans Are Slipping", "Of the loans in each bucket last month, the share now in a worse "
                                                  "bucket (and the share back to STD)."),
              bars("Roll steps", rows, value_head="Loans", extra_heads=("SOH", "By SOH"), sheet="Roll steps",
                   full=pd.DataFrame([{"Step": k, **v} for k, v in s.items()]))]
    out = ac.get("outlook")
    if out:
        npa, nd = out["new_npa"], out["new_defaulters"]
        blocks.append(callout(f"If this month's roll rates repeat, next month brings about {npa['n']:,} new NPAs "
                              f"(₹{npa['soh'] / 1e7:,.2f} Cr) and {nd['n']:,} new defaulters "
                              f"(₹{nd['soh'] / 1e7:,.2f} Cr).", "bad"))
    if child:
        by = rf.roll_steps_by(c, child)
        col = "STD → Behind | %"
        if len(by) > 1 and col in by.columns:
            name = {"zone": "Zone", "region": "Region", "branch": "Branch", "executive": "Executive"}[child]
            top = by.sort_values(col, ascending=False).head(8)
            blocks.append(bars(f"Where new defaulters come from, by {name.lower()}", [
                {"label": str(r[name]), "sub": f"{int(r['Matched Accounts']):,} loans", "value": float(r[col]),
                 "text": f"{r[col]:.1f}% ({int(r['STD → Behind | Accounts']):,})", "color": "red"}
                for _, r in top.iterrows()], value_head="Paid up last month, behind now", full=by,
                sheet=f"New defaulters by {name.lower()}"))
    return blocks


def _why(c, p, curr_month, by_region: bool) -> list[dict]:
    """What's overdue (insurance vs installment) and how stuck the loans are
    (paying vs not), as shares of delinquent loans; by region when there are several."""
    from analysis.root_cause import ARREARS_GROUPS, compute_chronic_shock_split, compute_insurance_split
    blocks = []
    for frame, groups, title in ((compute_insurance_split(c), ("Insurance-Only", "Installment-Only", "Both", "Other"),
                                  "What's overdue"),
                                 (compute_chronic_shock_split(c), ARREARS_GROUPS, "Paying vs not paying")):
        if frame.empty:
            continue
        total = frame["Delinquent Accounts"].sum()
        rows = [{"label": g, "value": frame[g].sum() / total * 100,
                 "text": f"{frame[g].sum() / total * 100:.1f}% ({int(frame[g].sum()):,})", "color": "amber"}
                for g in groups if g in frame.columns and total]
        if rows:
            blocks.append(bars(title, rows, value_head="Of delinquent loans", full=frame, sheet=title))
    if by_region:
        from analysis.portfolio_intelligence import compute_region_scorecard
        from analysis.root_cause import compute_region_why_table
        why = compute_region_why_table(c, compute_region_scorecard(c, p), as_of=curr_month)
        if len(why) > 1:
            blocks.append(bars("Main cause, by region", [
                {"label": str(r["Region"]), "sub": str(r["Dominant Driver"]), "value": float(r["Driver Share %"]),
                 "text": _f(r["Driver Share %"]), "color": "amber"} for _, r in why.iterrows()],
                value_head="Share of its delinquent loans", full=why, sheet="Why by region"))
    return [heading("Why Loans Are Behind", "Shares of the delinquent loans.")] + blocks if blocks else []


def _focus(ac: dict) -> list[dict]:
    if not ac["focus"]:
        return []
    items = [{"rank": f["rank"], "title": f["name"], "loans": f["loans"], "soh_cr": f["soh"] / 1e7,
              "action": f["action"], "risk": (f.get("risk") or {}).get("n")} for f in ac["focus"]]
    return [heading("Where To Focus First", "Today's loans, cheapest and most urgent first. \"Likely to slip\" "
                                            "uses this month's roll rate for each loan's bucket."),
            actions("Where to focus first", items)]

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
    return [heading("Who to Call First", "The largest loans in each group; the full lists are in the Excel annex.")] + blocks if blocks else []


def _definitions() -> list[dict]:
    from ui.glossary import GLOSSARY
    keep = ["STD → Behind", "1-30 → SMA-1+", "SMA-1 → SMA-2+", "SMA-2 → NPA", "Back to STD", "Insurance-Only",
            "NPA% (SOH)", "Not Paying 3M+", "Hard Bucket", "Delinquent Accounts", "Concern Score"]
    rows = [{"Term": _STATUS_TEXT[s], "Meaning": m} for s, m in (
        ("green", "Delinquency at or below this report's own average."),
        ("amber", "Delinquency above this report's average."),
        ("red", f"Delinquency {ACT_NOW_GAP_PTS:.0f} or more points above this report's average."))]
    rows += [{"Term": k, "Meaning": GLOSSARY[k]} for k in keep if k in GLOSSARY]
    rows += [
        {"Term": "Delinquency %", "Meaning": "Loans with any EMI or charge overdue (Arrears/EMI above 0), as % of all loans."},
        {"Term": "NPA %", "Meaning": "Loans 3 or more EMIs overdue, as % of all loans (NPA % (SOH): by money)."},
        {"Term": "Forecast", "Meaning": "Today's loans in each bucket x this month's rate of moving out of that bucket; "
                                        "the range covers about 95% of likely outcomes."},
    ]
    return [heading("How to Read This"), table("Definitions", pd.DataFrame(rows),
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

# Every main section, in the order it appears in a report (key -> label). Keys
# must not repeat an OPTIONAL_SECTIONS key (e.g. "business" is the optional detail).
MAIN_SECTIONS = {
    "glance": "At a glance", "zones": "Zones compared", "regions": "Regions ranked",
    "branches": "Branches ranked", "executives": "Executives ranked",
    "slipping": "Where loans are slipping", "why": "Why loans are behind",
    "new_business": "New business", "focus": "Where to focus first",
    "call_lists": "Who to call first", "definitions": "How to read this",
}
# What each level's report holds (the report tab ticks these when a level is picked).
PRESET_SECTIONS = {
    "Leadership": ["glance", "zones", "regions", "slipping", "new_business", "focus"],
    "Zone": ["glance", "regions", "branches", "slipping", "focus"],
    "Regional": ["glance", "branches", "executives", "slipping", "why", "new_business", "focus", "definitions"],
    "Branch": ["glance", "executives", "slipping", "focus", "call_lists", "definitions"],
}
# The level just below each report's own: whose moves the verdict names, and
# whose new defaulters "Where loans are slipping" breaks down.
_CHILD = {"Leadership": ("Region", "region"), "Zone": ("Region", "region"),
          "Regional": ("Branch", "branch"), "Branch": ("Executive", "executive")}


def build_report(df_curr: pd.DataFrame, df_prev: pd.DataFrame, curr_month: str, prev_month: str | None,
                 preset: str = DEFAULT_PRESET, filters: dict | None = None,
                 alerts: list | None = None, alerts_prev: list | None = None,
                 extras: list[str] | tuple = (), sections: list[str] | tuple | None = None,
                 unit: str | None = None) -> dict:
    """The report model: {title, subtitle, scope, blocks, ...}.
    sections: keys of MAIN_SECTIONS and OPTIONAL_SECTIONS to include (shown in
    MAIN_SECTIONS order, the optional ones before "How to read this"); None
    means the level's own sections plus `extras` (keys of OPTIONAL_SECTIONS).
    unit: the zone / region / branch this file is about (packs.py splits by it).
    alerts / alerts_prev are accepted for callers that pass them; the focus
    list already covers the alert groups."""
    from analysis import action_center
    from utils import compute_metrics
    filters = filters or {}
    preset = preset if preset in PRESET_SECTIONS else DEFAULT_PRESET
    ac = action_center.build(df_curr, df_prev)
    metrics = compute_metrics(df_curr, df_prev)
    prev_values = ({k: v[0] for k, v in compute_metrics(df_prev, df_prev.iloc[0:0]).items()}
                   if df_prev is not None and len(df_prev) else None)
    bench = float(metrics["Delinquency %"][0])
    child_level, child_grain = _CHILD[preset]

    chosen = set(PRESET_SECTIONS[preset] + list(extras) if sections is None else sections)
    build = {
        "glance": lambda: _glance(df_curr, df_prev, metrics, prev_values, ac, child_level),
        "zones": lambda: _units_ranked(df_curr, df_prev, "Zone", "Zones Compared", bench),
        "regions": lambda: _units_ranked(df_curr, df_prev, "Region", "Regions Ranked", bench),
        "branches": lambda: _units_ranked(df_curr, df_prev, "Branch", "Branches Ranked", bench),
        "executives": lambda: _executives_ranked(df_curr, df_prev, bench),
        "slipping": lambda: _slipping(df_curr, ac, child_grain),
        "why": lambda: _why(df_curr, df_prev, curr_month,
                            by_region="RegionName" in df_curr.columns and df_curr["RegionName"].nunique() > 1),
        "new_business": lambda: _business(df_curr, curr_month),
        "focus": lambda: _focus(ac),
        "call_lists": lambda: _call_lists(ac),
    }
    blocks = []
    for key in MAIN_SECTIONS:
        if key in chosen and key in build:
            blocks += build[key]()
    for key, (_, fn) in OPTIONAL_SECTIONS.items():     # detail sections, before "How to read this"
        if key in chosen:
            blocks += fn(df_curr, df_prev, curr_month)
    if "definitions" in chosen:
        blocks += _definitions()

    scope = " | ".join(f"{k}: {v}" for k, v in filters.items() if v not in (None, "", "All", "None")) or "Whole portfolio"
    return {
        "title": f"{preset} Report" + (f": {unit}" if unit else ""),
        "subtitle": f"{curr_month}" + (f" vs {prev_month}" if prev_month else ""),
        "scope": scope,
        "preset": preset,
        "unit": unit,
        "month": curr_month,
        "data_notes": list(df_curr.attrs.get("data_fixes", []) or []),
        "accounts": int(metrics["Count"][0]),
        "blocks": blocks,
        "ai_summary": None,
        # Every call list in full, with the columns that prove each loan's
        # place on it, for the Excel annex (managers forward these).
        "annex": [(f"Call {i} {t['name'].split(':')[0]}", t["loans"]) for i, t in enumerate(ac["calls"], start=1)],
    }

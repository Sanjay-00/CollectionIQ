"""
The monthly report as a MODEL: an ordered list of blocks, built from the same
engines as the app. report_agent/render.py turns one model into HTML, PDF or
an Excel annex, so every format says exactly the same thing.

It reads as a story, not a dump: every report opens with "At a glance" (a
verdict line, KPI tiles, what went well, what went wrong, where to focus),
then scorecards (the same columns and Total row as Portfolio Intelligence),
roll forward (the bucket matrix and roll steps by unit, as on the Migration
tab), why loans are behind, and what to do first. Full tables live in the
Excel annex.

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
ALL_ROWS_UP_TO = 25            # a scorecard this long shows every row; longer: the worst and the best few
ROWS_EACH_SIDE = 10


# The level just below each report's own: (its name, its roll_flow grain, its
# column in the file). Whose moves the verdict names, and what "Change by",
# roll steps, "Why loans are behind" and the new book break down by.
_CHILD = {"Leadership": ("Region", "region", "RegionName"), "Zone": ("Region", "region", "RegionName"),
          "Regional": ("Branch", "branch", "Unit"), "Branch": ("Executive", "executive", "MNT NAME")}


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
          full: pd.DataFrame | None = None, sheet: str | None | bool = None, total: dict | None = None,
          flag: str | None = None, annex: pd.DataFrame | None = None) -> dict:
    """columns: {"key", "label"?, "fmt": text|int|pct|pp|cr|inr|rs_cr|num|count_change, "heat"?: bool,
    "heat_key"?: column shaded by instead of key, "good_if_up"?: bool, "bold"?: bool,
    "count"?: column, "base"?: column (with count: "9.9% (33 of 332)"), "amount"?: ₹ Cr column}.
    `full` is the untruncated frame for the Excel annex; sheet=False: no annex
    sheet (the annex already has it). `total`: an optional Total row (dict),
    worked out from ALL the data in view, never just the rows shown. `flag`: a
    True/False column; True rows are tinted red. `annex`: the Excel sheet as
    given, when the on-screen cells (e.g. "57.8% → 60.0%") aren't numbers."""
    return {"type": "table", "title": title, "df": df.reset_index(drop=True), "columns": columns,
            "note": note, "full": df if full is None else full,
            "sheet": False if sheet is False else (sheet or title[:31]), "total": total, "flag": flag,
            "annex": annex}


def glance(verdict: dict, tiles: list[dict], good: list[str], bad: list[str], focus: list[dict]) -> dict:
    """The opening story. verdict: {"tone": good/bad/mixed/info, "text"}; tiles:
    like kpis items; good / bad: short sentences; focus: {"title", "detail"}."""
    return {"type": "glance", "verdict": verdict, "tiles": tiles, "good": good, "bad": bad, "focus": focus}


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

_SIGNAL = {"Collection %": 0.5, "Delinquency %": 0.25, "NPA %": 0.25}   # smaller moves read as "steady"


def _pp(change) -> str | None:
    if change is None or pd.isna(change):
        return None
    return "no change" if round(change, 1) == 0 else f"{'▲' if change > 0 else '▼'} {abs(change):.1f} pts"


def _varies(df: pd.DataFrame, col: str) -> bool:
    """A column worth showing: present and not the same on every row (a
    single-region report needn't say "JALGAON" on every line)."""
    return col in df.columns and df[col].nunique(dropna=True) > 1


def _scorecard_columns(name: str, extra: list[str] = (), amounts: bool = False) -> list[dict]:
    """The Portfolio Intelligence column set (ui/tabs/portfolio_intelligence.unit_columns).
    amounts: the month's demand and collected (₹ lakh) just before Collection, so
    the % reads as "this much collected of this much due"."""
    cols = [
        {"key": name, "fmt": "text", "bold": True},
        *[{"key": e, "fmt": "text"} for e in extra],
        {"key": "Accounts", "label": "Loans", "fmt": "int"},
        {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct", "count": "Delinquent", "heat": True},
        {"key": "Δ Delinquency%", "label": "vs last month", "fmt": "pp"},
        {"key": "SMA-2%", "label": "SMA-2", "fmt": "pct", "count": "SMA-2", "heat": True},
        {"key": "NPA%", "label": "NPA", "fmt": "pct", "count": "NPA", "heat": True},
        {"key": "NPA% (SOH)", "label": "NPA by SOH", "fmt": "pct", "amount": "NPA SOH (Cr)", "heat": True},
        {"key": "Hard Bucket%", "label": "Hard bucket", "fmt": "pct", "count": "Hard Bucket", "heat": True},
        {"key": "Collection%", "label": "Collection", "fmt": "pct"},
        {"key": "Strike%", "label": "Strike", "fmt": "pct"},
        {"key": "Roll Fwd%", "label": "Rolled forward", "fmt": "pct", "count": "Slipped", "heat": True},
    ]
    if amounts:
        at = next(i for i, c in enumerate(cols) if c["key"] == "Collection%")
        cols[at:at] = [{"key": "Demand", "label": "Demand", "fmt": "rs_l"},
                       {"key": "Collected", "label": "Collected", "fmt": "rs_l"}]
    return cols


def _total_row(c, p, name: str, extra=()) -> dict | None:
    """Every loan in this report as one unit (sum over sum), never the rows' average."""
    from analysis.summary import portfolio_total
    t = portfolio_total(c, p)
    return {**t, name: "Total", **{e: "" for e in extra}} if t else None


def _without_branch(names: pd.Series) -> pd.Series:
    """"SNEHA JOSHI (BHUSAWAL)" -> "SNEHA JOSHI", for a table that shows the branch in its own column."""
    return names.astype(str).str.replace(r"\s*\([^()]*\)$", "", regex=True)


_SCORE_NOTE = ("Worst delinquency first. Each % shows its loans in brackets (NPA by SOH: its ₹ Cr). "
               "Darker red: worse than the others in this table. Total: every loan in this report.")


def _scorecard(title: str, t: pd.DataFrame, name: str, extra: list[str], total: dict | None,
               sheet: str, note: str = _SCORE_NOTE, flag_worst: bool = False, amounts: bool = False) -> list[dict]:
    """One scorecard table, worst delinquency first; when longer than
    ALL_ROWS_UP_TO, the worst and the best ROWS_EACH_SIDE (every row is still
    in the Excel annex). flag_worst tints the worst quarter red."""
    t = t.sort_values("Delinquency%", ascending=False, kind="stable").reset_index(drop=True)
    cols = [c for c in _scorecard_columns(name, extra, amounts) if c["key"] in t.columns]
    if len(t) <= ALL_ROWS_UP_TO:
        flag = None
        if flag_worst and len(t) >= 4:
            t = t.assign(_worst=[i < round(len(t) / 4) for i in range(len(t))])
            flag = "_worst"
            note += " Tinted rows: the worst quarter by delinquency."
        return [heading(title, note), table(title, t, cols, total=total, sheet=sheet, flag=flag)]
    worst, best = t.head(ROWS_EACH_SIDE), t.tail(ROWS_EACH_SIDE).iloc[::-1]
    note += f" The {ROWS_EACH_SIDE} worst and {ROWS_EACH_SIDE} best of {len(t)}; every one is in the Excel annex."
    return [heading(title, note),
            table("Needing the most help", worst, cols, full=t, sheet=sheet),
            table("Doing best", best, cols, total=total, sheet=False)]


def _units_scorecard(c, p, grain: str, title: str) -> list[dict]:
    """Zones, regions or branches with every headline metric and a Total row."""
    from analysis.summary import unit_table
    t = unit_table(c, p, grain)
    if len(t) < 2:
        return []
    extra = [x for x in ("Zone", "Region") if x != grain and _varies(t, x)]
    return _scorecard(title, t, grain, extra, _total_row(c, p, grain, extra), sheet=title[:31])


def _executives_scorecard(c, p) -> list[dict]:
    """Executives (big enough to judge fairly) with the same columns, the worst quarter tinted."""
    from analysis.summary import unit_table
    t = unit_table(c, p, "Executive")
    if len(t) < 2:
        return []
    extra = [x for x in ("Branch", "Region") if _varies(t, x)]
    if "Branch" in extra:                  # "SNEHA JOSHI (BHUSAWAL)": the branch has its own column
        t = t.assign(Executive=_without_branch(t["Executive"]))
    return _scorecard("Executive Scorecard", t, "Executive", extra, _total_row(c, p, "Executive", extra),
                      sheet="Executives", flag_worst=True, amounts=True,
                      note=_SCORE_NOTE + " Demand and Collected are this month's, in ₹ lakh; Collection is "
                                         "Collected as % of Demand. Executives with too few loans to judge "
                                         "fairly are left out.")


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


# ── This month vs last month ─────────────────────────────────────────────────

MATURED_STATUS = "MAT"      # Loan Status of a matured loan: tenure over, money still owed


def _is_matured(df: pd.DataFrame) -> pd.Series:
    return df["Loan Status"].astype(str).str.strip().str.upper() == MATURED_STATUS


def _matured(df: pd.DataFrame) -> dict:
    """Matured accounts (Loan Status MAT): count and SOH (₹ Cr). Past its tenure,
    all that's owed is overdue, so their SOH is also their arrears."""
    from utils import to_num
    if df is None or not len(df) or "Loan Status" not in df.columns:
        return {}
    m = df[_is_matured(df)]
    return {"n": len(m), "soh_cr": to_num(m, "SOH", fill=0).sum() / 1e7}


def _newly_matured(c: pd.DataFrame, p: pd.DataFrame) -> int | None:
    """Loans that are MAT now but weren't last month (both files needed)."""
    need = {"Loan Status", "Loan No"}
    if p is None or not len(p) or not need <= set(c.columns) or not need <= set(p.columns):
        return None
    was = set(p.loc[_is_matured(p), "Loan No"].astype(str))
    return int((~c.loc[_is_matured(c), "Loan No"].astype(str).isin(was)).sum())


def _blank_num(v) -> bool:
    return v is None or pd.isna(v)


def _mom_row(group: str, metric: str, before, now, kind: str, worse_if_up: bool | None,
             extra_before=None, extra_now=None, extra_cr: bool = False) -> dict:
    """One comparison row. kind: pct (change in points), money (₹ Cr, change in
    %), count (change in %). extra_*: what's behind a % in brackets, loans
    ("11.9% (37)") or, with extra_cr, ₹ Cr ("37.9% (₹15.89 Cr)"). worse_if_up
    None: neither good nor bad (no colour)."""
    def show(v, extra):
        if _blank_num(v):
            return "-"
        if kind == "pct":
            if _blank_num(extra):
                return f"{v:.1f}%"
            return f"{v:.1f}% (₹{extra:,.2f} Cr)" if extra_cr else f"{v:.1f}% ({extra:,.0f})"
        return f"₹{v:,.2f} Cr" if kind == "money" else f"{v:,.0f}"
    change, worse = "-", None
    if not _blank_num(before) and not _blank_num(now):
        if kind == "pct":
            d = round(now - before, 2)
            change = "no change" if d == 0 else f"{'▲' if d > 0 else '▼'} {abs(d):.2f} pts"
        elif before:
            d = round((now - before) / before * 100, 1)
            change = "no change" if d == 0 else f"{'▲' if d > 0 else '▼'} {abs(d):.1f}%"
        else:
            d = 0 if now == before else 1
            change = "-" if now == before else "new"
        if worse_if_up is not None and d:
            worse = (d > 0) == worse_if_up
    return {"Metric": metric, "Last month": show(before, extra_before), "This month": show(now, extra_now),
            "Change": change, "_worse": worse, "_group": False, "_section": group}


def _month_vs_month(c, p, curr_month: str, prev_month: str | None, ac: dict) -> list[dict]:
    """The month in one table, last month beside this month: portfolio,
    matured accounts, asset quality, collections, and this month's movement."""
    from analysis.summary import business_snapshot, portfolio_total
    from utils import compute_overdue_demand_pct, insurance_only_mask, non_starter_mask
    now = portfolio_total(c, None)
    if not now:
        return []
    has_prev = p is not None and len(p) > 0
    before = portfolio_total(p, None) if has_prev else {}
    b = lambda k: before.get(k) if before else None        # noqa: E731
    pct = lambda n, d: (n / d * 100) if d else None        # noqa: E731
    biz = business_snapshot(c, curr_month)
    mat_now, mat_before = _matured(c), (_matured(p) if has_prev else {})
    rows: list[dict] = []

    def group(label: str) -> None:
        rows.append({"Metric": label.upper(), "Last month": "", "This month": "", "Change": "", "_worse": None,
                     "_group": True, "_section": label})

    group("Portfolio")
    rows += [_mom_row("Portfolio", "Loans in book", b("Accounts"), now["Accounts"], "count", None),
             _mom_row("Portfolio", "SOH", b("SOH (Cr)"), now["SOH (Cr)"], "money", None),
             _mom_row("Portfolio", "POS", (b("POS") or 0) / 1e7 if before else None, now["POS"] / 1e7, "money", None)]
    if biz.get("month"):
        rows += [_mom_row("Portfolio", "New loans", biz.get("prev_accounts") if biz.get("has_prev") else None,
                          biz.get("accounts"), "count", False),
                 _mom_row("Portfolio", "Amount funded", biz.get("prev_funded_cr") if biz.get("has_prev") else None,
                          biz.get("funded_cr"), "money", False)]
    if mat_now:
        group("Matured accounts")
        mb = lambda k: mat_before.get(k) if mat_before else None   # noqa: E731
        rows += [_mom_row("Matured accounts", "Matured accounts (MAT)", mb("n"), mat_now["n"], "count", True),
                 _mom_row("Matured accounts", "Matured, share of loans",
                          pct(mb("n"), b("Accounts")) if mat_before else None, pct(mat_now["n"], now["Accounts"]),
                          "pct", True),
                 _mom_row("Matured accounts", "Matured SOH still owed", mb("soh_cr"), mat_now["soh_cr"], "money", True)]
        fresh = _newly_matured(c, p)
        if fresh is not None:
            rows.append({"Metric": "Newly matured this month", "Last month": "", "This month": f"{fresh:,}",
                         "Change": "", "_worse": None, "_group": False, "_section": "Matured accounts"})
    group("Asset quality")
    for t in (before, now):                              # unit_metrics has the SMA-1 count, not its %
        if t:
            t["SMA-1%"] = pct(t.get("SMA-1", 0), t["Accounts"])
    for label, key, count in (("Delinquent", "Delinquency%", "Delinquent"), ("SMA-1", "SMA-1%", "SMA-1"),
                              ("SMA-2", "SMA-2%", "SMA-2"), ("NPA", "NPA%", "NPA"),
                              ("Hard bucket (6+ EMIs)", "Hard Bucket%", "Hard Bucket"),
                              ("Not paying 3M+", "Not Paying 3M+%", "Not Paying 3M+")):
        if key in now:
            rows.append(_mom_row("Asset quality", label, b(key), now[key], "pct", True, b(count), now.get(count)))
    for label, mask in (("Non-starters", non_starter_mask), ("Insurance charge only", insurance_only_mask)):
        n_now = int(mask(c).sum())
        n_before = int(mask(p).sum()) if has_prev else None
        rows.append(_mom_row("Asset quality", label, pct(n_before, b("Accounts")) if has_prev else None,
                             pct(n_now, now["Accounts"]), "pct", True, n_before, n_now))
    rows.append(_mom_row("Asset quality", "NPA by SOH", b("NPA% (SOH)"), now["NPA% (SOH)"], "pct", True,
                         b("NPA SOH (Cr)"), now.get("NPA SOH (Cr)"), extra_cr=True))
    group("Collections")
    rows += [_mom_row("Collections", "Demand", (b("Demand") or 0) / 1e7 if before else None, now["Demand"] / 1e7,
                      "money", None),
             _mom_row("Collections", "Collected", (b("Collected") or 0) / 1e7 if before else None,
                      now["Collected"] / 1e7, "money", False),
             _mom_row("Collections", "Collection %", b("Collection%"), now["Collection%"], "pct", False),
             _mom_row("Collections", "Month demand collection %",
                      compute_overdue_demand_pct(p)["demand_pct"] if has_prev else None,
                      compute_overdue_demand_pct(c)["demand_pct"], "pct", False),
             _mom_row("Collections", "Strike %", b("Strike%"), now["Strike%"], "pct", False)]
    s = ac.get("roll_summary")
    if s:
        group("Movement this month")
        for label, step in (("New defaulters (STD → behind)", "STD → Behind"), ("New NPAs (SMA-2 → NPA)", "SMA-2 → NPA"),
                            ("Back to STD", "Back to STD")):
            rows.append({"Metric": label, "Last month": "", "Change": "", "_group": False, "_section": "Movement",
                         "This month": f"{s[step]['n']:,} (₹{s[step]['soh_cr']:,.2f} Cr)",
                         "_worse": None})
    df = pd.DataFrame(rows).astype({"_worse": object})
    cols = [{"key": "Metric", "fmt": "text", "bold": True},
            {"key": "Last month", "label": f"Last month ({prev_month})" if prev_month else "Last month",
             "fmt": "text", "align": "right"},
            {"key": "This month", "label": f"This month ({curr_month})", "fmt": "text", "align": "right"},
            {"key": "Change", "fmt": "text", "align": "right", "tone_key": "_worse"}]
    note = ("Same loans in view both months. % metrics change in points, amounts and counts in %. "
            "Red: worse than last month; green: better.")
    if not has_prev:
        note = "Last month's file isn't loaded, so only this month is shown. " + note
    blk = table("This month vs last month", df, cols, sheet="Month vs month")
    blk["group_key"] = "_group"
    return [heading("This Month vs Last Month", note), blk]


def _unit_changes(c, p, child_level: str) -> list[dict]:
    """Each unit one level down, last month beside this month on delinquency,
    SMA-2, NPA and collection: who moved, not just that the total moved."""
    from analysis.summary import portfolio_total, unit_table
    from utils import _unit_key
    if p is None or not len(p):
        return []
    grain = child_level
    now, before = unit_table(c, p, grain), unit_table(p, p.iloc[0:0], grain)
    if len(now) < 2 or before.empty:
        return []
    prev = {_unit_key(k): r for k, r in before.set_index(grain).iterrows()}
    t = now.copy()
    for key in ("SMA-2%", "Collection%"):
        t[f"Prev {key}"] = [prev[_unit_key(k)][key] if _unit_key(k) in prev else None for k in t[grain]]
    t["Prev NPA%"] = [prev[_unit_key(k)]["NPA%"] if _unit_key(k) in prev else None for k in t[grain]]
    t["Δ SMA-2%"] = t["SMA-2%"] - t["Prev SMA-2%"]
    t["Δ Collection%"] = t["Collection%"] - t["Prev Collection%"]
    arrow = lambda a, z: "-" if pd.isna(a) or pd.isna(z) else f"{a:.1f}% → {z:.1f}%"   # noqa: E731
    for name, cur in (("Delinquent", "Delinquency%"), ("SMA-2", "SMA-2%"), ("NPA", "NPA%"), ("Collection", "Collection%")):
        t[name] = [arrow(a, z) for a, z in zip(t[f"Prev {cur}"], t[cur])]
    t = t.sort_values("Δ Delinquency%", ascending=False, na_position="last", kind="stable").reset_index(drop=True)
    if grain == "Executive" and _varies(t, "Branch"):
        t = t.assign(Executive=_without_branch(t["Executive"]))
    extra = [x for x in ("Branch",) if grain == "Executive" and _varies(t, x)]
    ra = {"fmt": "text", "align": "right"}
    cols = [{"key": grain, "fmt": "text", "bold": True}, *[{"key": e, "fmt": "text"} for e in extra],
            {"key": "Accounts", "label": "Loans", "fmt": "int"},
            {"key": "Delinquent", **ra}, {"key": "Δ Delinquency%", "label": "Δ", "fmt": "pp"},
            {"key": "SMA-2", **ra}, {"key": "Δ SMA-2%", "label": "Δ", "fmt": "pp"},
            {"key": "NPA", **ra}, {"key": "Δ NPA%", "label": "Δ", "fmt": "pp"},
            {"key": "Collection", **ra}, {"key": "Δ Collection%", "label": "Δ", "fmt": "pp", "good_if_up": True}]
    tn, tb = portfolio_total(c, p), portfolio_total(p, None)
    total = None
    if tn and tb:
        total = {grain: "Total", **{e: "" for e in extra}, "Accounts": tn["Accounts"]}
        for name, cur in (("Delinquent", "Delinquency%"), ("SMA-2", "SMA-2%"), ("NPA", "NPA%"), ("Collection", "Collection%")):
            total[name] = arrow(tb[cur], tn[cur])
        total.update({"Δ Delinquency%": tn["Delinquency%"] - tb["Delinquency%"], "Δ SMA-2%": tn["SMA-2%"] - tb["SMA-2%"],
                      "Δ NPA%": tn["NPA%"] - tb["NPA%"], "Δ Collection%": tn["Collection%"] - tb["Collection%"]})
    # The Excel sheet: numbers, last month / this month / change for each metric.
    metrics = (("Delinquent", "Delinquency%", "Δ Delinquency%"), ("SMA-2", "SMA-2%", "Δ SMA-2%"),
               ("NPA", "NPA%", "Δ NPA%"), ("Collection", "Collection%", "Δ Collection%"))
    rows = [t]
    if total:
        rows.append(pd.DataFrame([{grain: "Total", "Accounts": tn["Accounts"],
                                   **{f"Prev {cur}": tb[cur] for _, cur, _ in metrics},
                                   **{cur: tn[cur] for _, cur, _ in metrics},
                                   **{d: total[d] for _, _, d in metrics}}]))
    both = pd.concat(rows, ignore_index=True)
    annex = pd.DataFrame({grain: both[grain], **{e: both.get(e) for e in extra}, "Loans": both["Accounts"]})
    for name, cur, d in metrics:
        annex[f"{name} last month %"] = pd.to_numeric(both[f"Prev {cur}"], errors="coerce").round(2)
        annex[f"{name} this month %"] = pd.to_numeric(both[cur], errors="coerce").round(2)
        annex[f"{name} change (pts)"] = pd.to_numeric(both[d], errors="coerce").round(2)
    noun = {"Region": "region", "Branch": "branch", "Executive": "executive", "Zone": "zone"}[grain]
    title = f"Change by {noun}"
    note = (f"Each {noun}: last month → this month, and the change in points. Biggest rise in delinquency first. "
            "Red: worse; green: better.")
    if len(t) > ALL_ROWS_UP_TO:
        note += f" The {ROWS_EACH_SIDE} biggest rises and {ROWS_EACH_SIDE} biggest falls of {len(t)}; all are in the Excel annex."
        return [table(title, t.head(ROWS_EACH_SIDE), cols, note=note, sheet=title, annex=annex),
                table(f"{title}: most improved", t.tail(ROWS_EACH_SIDE).iloc[::-1], cols, total=total, sheet=False)]
    return [table(title, t, cols, note=note, total=total, sheet=title, annex=annex)]


# ── New book: loans disbursed since config.NEW_BOOK_FROM ─────────────────────

def new_book_since() -> pd.Timestamp:
    from config import NEW_BOOK_FROM
    return pd.Timestamp(NEW_BOOK_FROM)


def _book_table(nb: pd.DataFrame, col: str, label: str, sort: str) -> tuple[pd.DataFrame, dict]:
    """One row per value of `col` in the new book: loans, funded, SOH and how
    they pay today; plus the Total row (every new-book loan, sum over sum)."""
    from utils import non_starter_mask, to_num, unit_metrics
    extra = nb.assign(_funded=to_num(nb, "Loan Amount", fill=0), _ns=non_starter_mask(nb))
    m = unit_metrics(nb, [col]).rename(columns={col: label})
    sums = extra.groupby(col)[["_funded", "_ns"]].sum()
    m["Funded (Cr)"] = m[label].map(sums["_funded"]).fillna(0) / 1e7
    m["Non-starters"] = m[label].map(sums["_ns"]).fillna(0).astype(int)
    m["Non-starters%"] = m["Non-starters"] / m["Accounts"] * 100
    t = unit_metrics(nb, []).iloc[0].to_dict()
    total = {**t, label: "Total", "Funded (Cr)": extra["_funded"].sum() / 1e7,
             "Non-starters": int(extra["_ns"].sum()), "Non-starters%": extra["_ns"].sum() / len(nb) * 100}
    m.loc[m["Demand"] <= 0, "Collection%"] = None          # nothing due yet (e.g. this month's loans): "-", not 0%
    if extra["_ns"].sum() == 0:                              # no column of zeros
        m, total = m.drop(columns=["Non-starters%"]), {k: v for k, v in total.items() if k != "Non-starters%"}
    if not total.get("Demand"):
        total["Collection%"] = None
    m = (m.sort_values(label) if sort == label else m.sort_values(sort, ascending=False, kind="stable"))
    return m.reset_index(drop=True), total


def _book_columns(label: str, df: pd.DataFrame) -> list[dict]:
    return [c for c in _all_book_columns(label) if c["key"] in df.columns]


def _all_book_columns(label: str) -> list[dict]:
    return [{"key": label, "fmt": "text", "bold": True}, {"key": "Accounts", "label": "Loans", "fmt": "int"},
            {"key": "Funded (Cr)", "label": "Funded", "fmt": "cr"}, {"key": "SOH (Cr)", "label": "SOH", "fmt": "cr"},
            {"key": "Delinquency%", "label": "Delinquent", "fmt": "pct", "count": "Delinquent", "heat": True},
            {"key": "SMA-2%", "label": "SMA-2", "fmt": "pct", "count": "SMA-2", "heat": True},
            {"key": "NPA%", "label": "NPA", "fmt": "pct", "count": "NPA", "heat": True},
            {"key": "Non-starters%", "label": "Non-starters", "fmt": "pct", "count": "Non-starters", "heat": True},
            {"key": "Collection%", "label": "Collection", "fmt": "pct"}]


def _new_book(c: pd.DataFrame, curr_month: str, preset: str) -> list[dict]:
    """Every loan disbursed since NEW_BOOK_FROM: how it pays today against the
    whole book, by disbursal month, by the next level down, by segment, and
    the loans already going bad."""
    from utils import non_starter_mask, segment_column, to_num, unit_metrics
    start = new_book_since()
    title = f"New Book (disbursed since {start:%b %Y})"
    if "Ag_Date" not in c.columns:
        return []
    ag = pd.to_datetime(c["Ag_Date"], errors="coerce")
    nb = c[ag >= start].assign(_month=ag[ag >= start].dt.strftime("%Y-%m"))
    if nb.empty:
        return [heading(title), callout(f"No loan in this report was disbursed since {start:%d %b %Y}.", "info")]
    whole, new = unit_metrics(c, []).iloc[0], unit_metrics(nb, []).iloc[0]
    ns = non_starter_mask(nb)
    funded = to_num(nb, "Loan Amount", fill=0).sum() / 1e7
    blocks = [heading(title, f"Every loan disbursed from {start:%d %b %Y} to {curr_month}, and how it pays today. "
                             "Red: worse than the others in the table."),
              bullets([{"tone": "info", "text": f"{len(nb):,} loans disbursed, ₹{funded:,.2f} Cr funded; "
                                                 f"₹{new['SOH (Cr)']:,.2f} Cr still outstanding."},
                       {"tone": "bad" if new["Delinquency%"] > whole["Delinquency%"] else "good",
                        "text": f"Delinquent {new['Delinquency%']:.1f}% ({int(new['Delinquent']):,}) against "
                                f"{whole['Delinquency%']:.1f}% for the whole book; NPA {new['NPA%']:.1f}% "
                                f"({int(new['NPA']):,}) against {whole['NPA%']:.1f}%."},
                       {"tone": "bad" if ns.any() else "good",
                        "text": f"{int(ns.sum()):,} non-starters (never paid the first EMI), "
                                f"₹{to_num(nb[ns], 'SOH', fill=0).sum() / 1e7:,.2f} Cr SOH."}])]
    df, total = _book_table(nb, "_month", "Disbursed", sort="Disbursed")
    blocks.append(table("By disbursal month", df, _book_columns("Disbursed", df), total=total,
                        note="Read down the Delinquent and NPA columns: newer months going bad faster than "
                             "older ones point to a sourcing or underwriting problem.", sheet="New book by month"))
    label, _, col = _CHILD[preset]
    if col in nb.columns and nb[col].nunique() > 1:
        df, total = _book_table(nb, col, label, sort="Delinquency%")
        blocks.append(table(f"By {label.lower()}", df, _book_columns(label, df), total=total,
                            note="Worst delinquency first.", sheet=f"New book by {label.lower()}"))
    seg = segment_column(nb)
    if seg and nb[seg].nunique() > 1:
        df, total = _book_table(nb, seg, "Segment", sort="Delinquency%")
        blocks.append(table("By segment", df, _book_columns("Segment", df), total=total,
                            note="Worst delinquency first.", sheet="New book by segment"))
    bucket = nb["curr_bucket"] if "curr_bucket" in nb.columns else pd.Series("", index=nb.index)
    bad = nb[bucket.isin(["SMA-2", "NPA"]) | ns]
    if len(bad):
        bad = bad.assign(**{"Disbursed on": pd.to_datetime(bad["Ag_Date"], errors="coerce").dt.strftime("%d %b %Y"),
                            "Non-starter": ns[bad.index].map({True: "Yes", False: ""}),
                            "SOH": to_num(bad, "SOH", fill=0)}).sort_values("SOH", ascending=False)
        # Not _RENAME: these are whole-file rows, and the file has both Segment and
        # SegmentName, so renaming SegmentName -> Segment would make two "Segment" columns.
        bad = bad.rename(columns={"Cust Name": "Customer", "Unit": "Branch", "MNT NAME": "Executive",
                                  "curr_bucket": "Bucket", "RegionName": "Region"})
        keep = [k for k in ("Loan No", "Customer", "Region", "Branch", "Executive", "Disbursed on", "Bucket",
                            "Arrears / EMI", "Non-starter", "SOH")
                if k in bad.columns and (k not in ("Region", "Branch") or _varies(bad, k))
                and (k != "Non-starter" or ns.any())]
        fmts = {"Arrears / EMI": "num", "SOH": "inr"}
        cols = [{"key": k, "label": {"Arrears / EMI": "EMIs overdue"}.get(k, k), "fmt": fmts.get(k, "text")}
                for k in keep]
        shown = bad.head(20)
        more = "; all are in the Excel annex." if len(bad) > len(shown) else "."
        blocks.append(table("New loans already going bad", shown, cols, full=bad, sheet="New book going bad",
                            note=f"SMA-2, NPA or never paid the first EMI; largest SOH first. "
                                 f"Showing {len(shown)} of {len(bad):,}{more}"))
    return blocks


_STEP_LABEL = {"STD → Behind": "New defaulters (STD → behind)", "1-30 → SMA-1+": "1-30 → SMA-1 or worse",
               "SMA-1 → SMA-2+": "SMA-1 → SMA-2 or worse", "SMA-2 → NPA": "SMA-2 → NPA", "Back to STD": "Back to STD"}


def _bucket_matrix(c) -> tuple[pd.DataFrame, list[dict], dict] | None:
    """Last month's bucket (rows) x this month's (columns): each cell the % of
    the row's loans (loans in brackets), red only where the loan got worse;
    then rolled forward / stayed / rolled back. Returns (df, columns, total)."""
    from analysis import roll_flow as rf
    from utils import BUCKET_SCORE
    counts = rf.migration_matrix(c, "count")
    buckets = list(counts.columns)
    rows = []
    for b in counts.index:
        n = int(counts.loc[b].sum())
        if not n:
            continue
        row, moved = {"Last month": b, "Loans": n}, {"fwd": 0, "stay": 0, "back": 0}
        for col in buckets:
            k = int(counts.loc[b, col])
            step = BUCKET_SCORE[col] - BUCKET_SCORE[b]
            row[col], row[f"{col} %"] = k, k / n * 100
            row[f"{col} worse"] = k / n * 100 if step > 0 else None
            moved["fwd" if step > 0 else "back" if step < 0 else "stay"] += k
        row.update({"Fwd": moved["fwd"], "Stay": moved["stay"], "Back": moved["back"]})
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for k in ("Fwd", "Stay", "Back"):
        df[f"{k} %"] = df[k] / df["Loans"] * 100
    n = int(df["Loans"].sum())
    total = {"Last month": "Total", "Loans": n}
    for k in [*buckets, "Fwd", "Stay", "Back"]:
        total[k] = int(df[k].sum())
        total[f"{k} %"] = total[k] / n * 100
    cols = [{"key": "Last month", "label": "Last month \\ Now", "fmt": "text", "bold": True},
            {"key": "Loans", "fmt": "int"},
            *[{"key": f"{b} %", "label": b, "fmt": "pct", "count": b, "heat": True, "heat_key": f"{b} worse"}
              for b in buckets],
            {"key": "Fwd %", "label": "Rolled forward", "fmt": "pct", "count": "Fwd", "heat": True},
            {"key": "Stay %", "label": "Stayed", "fmt": "pct", "count": "Stay"},
            {"key": "Back %", "label": "Rolled back", "fmt": "pct", "count": "Back"}]
    return df, cols, total


def _roll_forward(c, ac: dict, child: str | None, name: str = "") -> list[dict]:
    """The bucket matrix with rolled forward / stayed / back, the outlook if
    this month's rates repeat, and the roll steps by the next level down."""
    from analysis import roll_flow as rf
    s = ac.get("roll_summary")
    if not s:
        return [heading("Roll Forward"), callout("Needs last month's file to compare.")]
    blocks = [heading("Roll Forward", "Where last month's loans are now. Each row is a bucket last month; each cell "
                                      "is the share of those loans in that bucket now (loans in brackets). "
                                      "Red: moved to a worse bucket.")]
    matrix = _bucket_matrix(c)
    if matrix:
        df, cols, total = matrix
        blocks.append(table("Bucket movement", df, cols, total=total, sheet="Bucket movement"))
    out = ac.get("outlook")
    if out:
        npa, nd = out["new_npa"], out["new_defaulters"]
        blocks.append(callout(f"If this month's roll rates repeat, next month brings about {npa['n']:,} new NPAs "
                              f"(₹{npa['soh'] / 1e7:,.2f} Cr) and {nd['n']:,} new defaulters "
                              f"(₹{nd['soh'] / 1e7:,.2f} Cr).", "bad"))
    by = rf.roll_steps_by(c, child) if child else pd.DataFrame()
    if len(by) > 1:
        extra = [x for x in ("Branch", "Region") if x != name and _varies(by, x)]
        steps = rf.step_labels()
        cols = [{"key": name, "fmt": "text", "bold": True}, *[{"key": e, "fmt": "text"} for e in extra],
                {"key": "Matched Accounts", "label": "Loans", "fmt": "int"},
                *[{"key": f"{lb} | %", "label": _STEP_LABEL.get(lb, lb), "fmt": "pct", "count": f"{lb} | Accounts",
                   "base": f"{lb} | Base", "heat": lb != steps[-1]} for lb in steps]]
        total = {name: "Total", **{e: "" for e in extra}, "Matched Accounts": int(matrix[2]["Loans"]) if matrix else None}
        for lb in steps:
            total.update({f"{lb} | %": s[lb]["pct"], f"{lb} | Accounts": s[lb]["n"], f"{lb} | Base": s[lb]["base"]})
        blocks.append(table(f"Roll steps by {name.lower()}", by, cols, total=total,
                            note=f"Each cell: % of the {name.lower()}'s loans in that bucket last month that moved "
                                 f"(loans that moved of loans in the bucket). Most new defaulters first.",
                            sheet=f"Roll steps by {name.lower()}"))
    return blocks


_WHY_LABEL = {"Insurance-Only": "Insurance charge only", "Installment-Only": "EMI only", "Both": "EMI and insurance",
              "Other": "Other charges", "Hard + Not Paying": "6+ EMIs, not paying",
              "Hard + Still Paying": "6+ EMIs, still paying", "Early Delinquency": "Under 6 EMIs"}
SILENT_LABEL = "Was not paying, now under 6 EMIs"   # root_cause.WAS_SILENT (its name carries the threshold)


def _why(c, preset: str) -> list[dict]:
    """One table by the next level down: what's unpaid (insurance vs EMI) and
    how far behind (6+ EMIs paying or not), each as a share of its delinquent loans."""
    from analysis.root_cause import ARREARS_GROUPS, WAS_SILENT, compute_chronic_shock_split, compute_insurance_split
    labels = {**_WHY_LABEL, WAS_SILENT: SILENT_LABEL}
    name, _, col = _CHILD[preset]
    if col not in c.columns:
        return []
    ins, chronic = compute_insurance_split(c, col), compute_chronic_shock_split(c, col)
    if ins.empty:
        return []
    groups = ["Insurance-Only", "Installment-Only", "Both", "Other"]
    t = ins[[col, "Delinquent Accounts", *groups]]
    if not chronic.empty:
        t = t.merge(chronic[[col, *[g for g in ARREARS_GROUPS if g in chronic.columns]]], on=col, how="left")
        groups += [g for g in ARREARS_GROUPS if g in chronic.columns]
    groups = [g for g in groups if t[g].sum() > 0]               # no column of zeros
    t = t.rename(columns={col: name}).sort_values("Delinquent Accounts", ascending=False).reset_index(drop=True)
    total = {name: "Total", "Delinquent Accounts": int(t["Delinquent Accounts"].sum())}
    for g in groups:
        t[f"{g} %"] = t[g] / t["Delinquent Accounts"] * 100
        total[g] = int(t[g].sum())
        total[f"{g} %"] = total[g] / total["Delinquent Accounts"] * 100 if total["Delinquent Accounts"] else None
    cols = [{"key": name, "fmt": "text", "bold": True},
            {"key": "Delinquent Accounts", "label": "Delinquent loans", "fmt": "int"},
            *[{"key": f"{g} %", "label": labels.get(g, g), "fmt": "pct", "count": g, "heat": True}
              for g in groups]]
    return [heading("Why Loans Are Behind", "Each % is a share of the delinquent loans (loans in brackets). The "
                                            "first columns say what is unpaid; the last ones how far behind and "
                                            "whether the customer still pays."),
            table("Why loans are behind", t, cols, total=total if len(t) > 1 else None, sheet="Why loans are behind")]


def _focus(ac: dict) -> list[dict]:
    if not ac["focus"]:
        return []
    df = pd.DataFrame([{"#": i, "Group": f["name"], "Loans": f["loans"], "SOH (Cr)": f["soh"] / 1e7,
                        "Likely to slip": (f.get("risk") or {}).get("n"), "What to do": f["action"]}
                       for i, f in enumerate(ac["focus"], start=1)])
    cols = [{"key": "#", "fmt": "int", "bold": True}, {"key": "Group", "label": "Where", "fmt": "text", "bold": True},
            {"key": "Loans", "fmt": "int"}, {"key": "SOH (Cr)", "label": "SOH", "fmt": "cr"},
            {"key": "Likely to slip", "label": "Likely to slip if left", "fmt": "int"},
            {"key": "What to do", "fmt": "text"}]
    return [heading("Where To Focus First", "Today's loans, cheapest and most urgent first. \"Likely to slip\" "
                                            "uses this month's roll rate for each loan's bucket."),
            table("Where to focus first", df, cols, sheet="Where to focus")]

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
        labels = {"Cust Name": "Customer", "Bucket Now": "Bucket", "Arrears / EMI": "EMIs overdue"}
        fmts = {"SOH": "inr", "Arrears / EMI": "num"}
        cols = [{"key": k, "label": labels.get(k, k), "fmt": fmts.get(k, "text")}
                for k in ("Loan No", "Cust Name", "Executive", "Bucket Now", "Arrears / EMI", "SOH") if k in df.columns]
        shown = df.head(10)
        blocks.append(table(f"Call list: {t['name']}", shown, cols, sheet=False,       # the annex has each list in full
                            note=f"Top {len(shown)} of {len(df):,} by SOH; the full list with proof columns is in the Excel annex."))
    return [heading("Who to Call First", "The largest loans in each group; the full lists are in the Excel annex.")] + blocks if blocks else []


# "How to read this" explains a term only when a table in the report shows it
# (as a column header or a table title), in this order.
_MEANING = {
    "Delinquent": "Loans with any EMI or charge overdue, as % of the unit's loans (loans in brackets).",
    "vs last month": "Change in delinquency since last month, in percentage points. Red: got worse.",
    "SMA-2": "Loans in the SMA-2 bucket, one step from NPA.",
    "NPA": "Loans 3 or more EMIs overdue, as % of the unit's loans.",
    "NPA by SOH": "NPA measured by money: SOH of NPA loans as % of the unit's SOH (₹ Cr in brackets).",
    "Hard bucket": "Loans 6 or more EMIs overdue right now.",
    "Demand": "This month's collection demand: instalment, expense and bounce charges due.",
    "Collected": "Collected this month, excluding reserve collection.",
    "Collection": "Collected this month (excluding reserve collection) as % of the month's demand "
                  "(instalment, expense and bounce charges).",
    "Strike": "Share of loans the file marks Strike = Y: a collection made this month.",
    "Rolled forward": "Loans in a worse bucket than last month, as % of loans in both months' files.",
    "Bucket movement": "Rows: each loan's bucket last month. Columns: its bucket now. Each row adds to 100%.",
    "New defaulters (STD → behind)": "Paid up (STD) last month, behind now. The cheapest to bring back.",
    "1-30 → SMA-1 or worse": "Were 1-30 days behind last month, now SMA-1 or worse: slipping further.",
    "SMA-1 → SMA-2 or worse": "Were SMA-1 last month, now SMA-2 or worse: one step from NPA.",
    "SMA-2 → NPA": "Were SMA-2 last month, NPA now: this month's new NPAs.",
    "Back to STD": "Were behind last month, fully paid up (STD) now: recovered loans.",
    "Insurance charge only": "EMI paid; only the insurance or expense charge (over ₹3,000) is unpaid. "
                             "A cash or WCL adjustment fixes it.",
    "EMI only": "EMI unpaid, no insurance or expense charge unpaid.",
    "EMI and insurance": "Both the EMI and the insurance or expense charge are unpaid.",
    "Other charges": "A small insurance charge (₹3,000 or less) or only bounce or penal charges.",
    "6+ EMIs, not paying": "6 or more EMIs overdue and nothing paid in 3 months: legal or repossession candidates.",
    "6+ EMIs, still paying": "6 or more EMIs overdue, but paying something: worth a restructuring talk.",
    SILENT_LABEL: "Went 3+ months without paying and crossed 6 EMIs overdue at some point, but has paid part since "
                  "and is now under 6 EMIs. Call before they go quiet again.",
    "Under 6 EMIs": "Behind, but under 6 EMIs overdue. Normal field follow-up.",
    "Loans in book": "Every loan in this report's files: running, matured (MAT) and S&S.",
    "POS": "Principal outstanding: the loan amount still to be repaid, excluding interest and charges.",
    "Matured accounts (MAT)": "Loans whose tenure is over (Loan Status MAT) but money is still owed.",
    "Matured SOH still owed": "SOH of the matured accounts: what is still outstanding on loans past their tenure.",
    "Newly matured this month": "Loans that are MAT now but were not last month.",
    "Not paying 3M+": "No payment in the last 3 months and more than 6 EMIs overdue.",
    "Δ": "Change since last month, in percentage points. Red: worse; green: better.",
    "Non-starters": "Never paid the first EMI (Non Starter = Y) and still behind.",
    "Month demand collection %": "Of this month's fresh demand (instalment, expense, bounce), the share collected "
                                 "after last month's overdue was cleared first.",
    "Funded": "Loan amount disbursed.",
    "Disbursed": "Month of the agreement date (Ag_Date).",
    "Likely to slip if left": "Loans in the group times this month's roll rate for their bucket.",
    "EMIs overdue": "Arrears divided by the EMI: 2.5 means two and a half EMIs unpaid.",
}


PTS_TERM = "pts (percentage points)"
PTS_MEANING = ("The plain difference between two percentages. NPA 37.2% last month and 37.8% now is "
               "▲ 0.6 pts (37.8 minus 37.2), not ▲ 1.5% (that would be the growth of NPA itself). "
               "Every % in this report changes in pts; amounts and counts (SOH, loans, demand) change in %.")


def _shows_points(blocks: list[dict]) -> bool:
    """Does any block show a change in points ("▲ 0.56 pts")?"""
    for b in blocks:
        if b["type"] == "glance" and any("pts" in (t.get("change") or "") for t in b["tiles"]):
            return True
        if b["type"] == "table" and (any(c["fmt"] == "pp" for c in b["columns"])
                                     or ("Change" in b["df"].columns
                                         and b["df"]["Change"].astype(str).str.contains("pts").any())):
            return True
    return False


def _definitions(blocks: list[dict]) -> list[dict]:
    shown = {b["title"] for b in blocks if b["type"] == "table"} | {
        c.get("label", c["key"]) for b in blocks if b["type"] == "table" for c in b["columns"]}
    shown |= {m for b in blocks if b["type"] == "table" and "Metric" in b["df"].columns for m in b["df"]["Metric"]}
    terms = [t for t in _MEANING if t in shown]
    if _shows_points(blocks):                     # last: how every change in the report is written
        terms.append(PTS_TERM)
    if not terms:
        return []
    rows = pd.DataFrame([{"Term": t, "Meaning": _MEANING.get(t, PTS_MEANING)} for t in terms])
    return [heading("How to Read This"), table("Definitions", rows, [{"key": "Term", "fmt": "text", "bold": True},
                                                                   {"key": "Meaning", "fmt": "text"}],
                                              sheet="Definitions")]


# ── Optional detail sections (off by default) ────────────────────────────────

_RENAME = {"RegionName": "Region", "Unit": "Branch", "MNT NAME": "Executive", "curr_bucket": "Bucket",
           "SegmentName": "Segment", "FUEL_TYPE": "Fuel Type"}
_RUPEE_COLS = {"SOH", "Closing Arrears", "Loan Amount", "Last Receipt Amount", "Arrears against Inst+Exp",
               "Month Due-Inst", "Month Due-Exp"}


_LABEL = {"Arrears / EMI": "EMIs overdue", "Cust Name": "Customer"}      # plain headers in detail tables
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
                     "amount": amounts.get(c), **({"label": _LABEL[c]} if c in _LABEL else {})})
    return cols


def _list_table(title: str, df: pd.DataFrame, keep: list[str], n: int = 20, note: str = "",
                heat=(), good=(), sheet: str | None = None) -> dict:
    full = df.rename(columns=_RENAME)
    keep = [_RENAME.get(k, k) for k in keep if _RENAME.get(k, k) in full.columns]
    # A place column that's the same on every row (one region's report) says nothing.
    keep = [k for k in keep if k not in ("Zone", "Region", "Branch") or k == keep[0] or _varies(full, k)]
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
        if len(df) > 1:                    # a one-row breakdown just repeats the report's own total
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
    heat = ("Delinquency%", "SMA-2%", "NPA%", "NPA% (SOH)")
    blocks = [heading("Segments", "Highest NPA first. Segment names that differ only in case, spacing or "
                                  "truncation are counted as one segment.")]
    # Segment in full (every segment, it's the one that matters); fuel type as a short second view.
    for col, label, n in ((segment_column(c), "Segment", 40), ("FUEL_TYPE", "Fuel Type", 15)):
        if not col or col not in c.columns:
            continue
        df = unit_metrics(c, [col]).rename(columns={col: label})
        if len(df) > 1:
            blocks.append(_list_table(f"By {label.lower()}", df.sort_values("NPA%", ascending=False),
                                      [label, "Accounts", "Delinquency%", "Delinquent", "SMA-2", "SMA-2%", "NPA", "NPA%",
                                       "NPA% (SOH)", "NPA SOH (Cr)", "Collection%", "SOH (Cr)"],
                                      n=n, heat=heat, good=("Collection%",)))
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
    "products": ("Segments", _x_products),
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
    "glance": "At a glance", "month_vs_month": "This month vs last month", "zones": "Zone scorecard", "regions": "Region scorecard",
    "branches": "Branch scorecard", "executives": "Executive scorecard",
    "slipping": "Roll forward", "why": "Why loans are behind",
    "new_business": "New business", "new_book": f"New book (since {new_book_since():%b %Y})", "focus": "Where to focus first",
    "call_lists": "Who to call first", "definitions": "How to read this",
}
# What each level's report holds (the report tab ticks these when a level is picked).
PRESET_SECTIONS = {
    "Leadership": ["glance", "month_vs_month", "zones", "regions", "slipping", "new_business", "new_book", "focus"],
    "Zone": ["glance", "month_vs_month", "regions", "branches", "slipping", "new_book", "focus"],
    "Regional": ["glance", "month_vs_month", "branches", "executives", "slipping", "why", "new_business", "new_book", "focus", "definitions"],
    "Branch": ["glance", "month_vs_month", "executives", "slipping", "new_book", "focus", "call_lists", "definitions"],
}
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
    from utils import frame_cache
    with frame_cache():                  # each number of each frame worked out once per report
        return _build_report(df_curr, df_prev, curr_month, prev_month, preset, filters, extras, sections, unit)


def _build_report(df_curr, df_prev, curr_month, prev_month, preset, filters, extras, sections, unit) -> dict:
    from analysis import action_center
    from utils import compute_metrics
    filters = filters or {}
    preset = preset if preset in PRESET_SECTIONS else DEFAULT_PRESET
    ac = action_center.build(df_curr, df_prev, with_overview=False)    # the report shows no pulse / attention
    metrics = compute_metrics(df_curr, df_prev)
    prev_values = ({k: v[0] for k, v in compute_metrics(df_prev, df_prev.iloc[0:0]).items()}
                   if df_prev is not None and len(df_prev) else None)
    child_level, child_grain, _ = _CHILD[preset]

    chosen = set(PRESET_SECTIONS[preset] + list(extras) if sections is None else sections)
    build = {
        "glance": lambda: _glance(df_curr, df_prev, metrics, prev_values, ac, child_level),
        "month_vs_month": lambda: (_month_vs_month(df_curr, df_prev, curr_month, prev_month, ac)
                                   + _unit_changes(df_curr, df_prev, child_level)),
        "zones": lambda: _units_scorecard(df_curr, df_prev, "Zone", "Zone Scorecard"),
        "regions": lambda: _units_scorecard(df_curr, df_prev, "Region", "Region Scorecard"),
        "branches": lambda: _units_scorecard(df_curr, df_prev, "Branch", "Branch Scorecard"),
        "executives": lambda: _executives_scorecard(df_curr, df_prev),
        "slipping": lambda: _roll_forward(df_curr, ac, child_grain, child_level),
        "why": lambda: _why(df_curr, preset),
        "new_business": lambda: _business(df_curr, curr_month),
        "new_book": lambda: _new_book(df_curr, curr_month, preset),
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
        blocks += _definitions(blocks)

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

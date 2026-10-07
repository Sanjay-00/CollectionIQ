"""
Dashboard summary: where the portfolio moved, league tables by any headline
metric, and compact business / alert snapshots. Built only from the shared
engines (utils.unit_metrics, analysis/roll_flow.py, the existing new-advances
and alert functions), so every number matches the detail tab it links to.
"""
import pandas as pd

from analysis import roll_flow as rf
from config import ACTION_MIN_ACCOUNTS, DASHBOARD_TOP_N
from utils import _unit_key, unit_metrics

_GRAIN = {  # grain: (group columns, display name column)
    "Region": (["RegionName"], "Region"),
    "Branch": (["Unit"], "Branch"),
    "Executive": (["MNT NAME", "Unit"], "Executive"),
}
_MIN = {"Region": 1, "Branch": ACTION_MIN_ACCOUNTS["branch"], "Executive": ACTION_MIN_ACCOUNTS["executive"]}

# League-table metrics: label -> (source column, a higher value is worse).
# "New defaulters %" comes from the roll analysis (needs last month's file).
LEAGUE_METRICS = {
    "NPA %": ("NPA%", True),
    "NPA % (SOH)": ("NPA% (SOH)", True),
    "Delinquency %": ("Delinquency%", True),
    "SMA-2 %": ("SMA-2%", True),
    "Hard Bucket %": ("Hard Bucket%", True),
    "New defaulters %": ("STD → Behind | %", True),
    "Collection %": ("Collection%", False),
    "Strike %": ("Strike%", False),
}


# The count behind each league metric, shown beside it as "11.9% (37)".
LEAGUE_COUNTS = {"NPA %": "NPA", "Delinquency %": "Delinquent", "SMA-2 %": "SMA-2",
                 "Hard Bucket %": "Hard Bucket", "New defaulters %": "STD → Behind | Accounts"}
# The money behind a % by SOH, shown beside it as "34.6% (₹121.40 Cr)".
LEAGUE_AMOUNTS = {"NPA % (SOH)": "NPA SOH (Cr)"}


def _units(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """unit_metrics per unit with a display name, region and a normalized key
    (case/space-insensitive) to match the same unit in last month's file."""
    cols, _ = _GRAIN[grain]
    if df is None or df.empty or not set(cols) <= set(df.columns):
        return pd.DataFrame()
    m = unit_metrics(df, cols)
    if m.empty:
        return m
    m["_key"] = list(zip(*[m[c].map(_unit_key) for c in cols]))
    m["Name"] = [f"{n} ({u})" for n, u in zip(m["MNT NAME"], m["Unit"])] if grain == "Executive" else m[cols[0]].astype(str)
    if grain != "Region" and "RegionName" in df.columns:
        region = df.groupby(cols)["RegionName"].first()
        keys = list(zip(*[m[c] for c in cols])) if len(cols) > 1 else list(m[cols[0]])
        m["Region"] = [region.get(k) for k in keys]
    return m


def league_table(df_curr: pd.DataFrame, df_prev: pd.DataFrame, metric: str, grain: str,
                 worst: bool = True, n: int = DASHBOARD_TOP_N) -> tuple[pd.DataFrame, int]:
    """Top `n` units by `metric` (worst or best first) with last month's value,
    the change and the rank both months. Units smaller than the minimum for
    their level aren't ranked. Returns (table, number of units ranked)."""
    col, higher_is_worse = LEAGUE_METRICS[metric]
    now = _units(df_curr, grain)
    if now.empty:
        return pd.DataFrame(), 0
    if col.endswith("| %"):            # a roll-analysis metric
        roll = rf.roll_steps_by(df_curr, grain.lower())
        if roll.empty:
            return pd.DataFrame(), 0
        name_cols = ["Executive", "Branch"] if grain == "Executive" else [grain]
        roll_key = list(zip(*[roll[c].map(_unit_key) for c in name_cols]))
        count_col = LEAGUE_COUNTS[metric]
        now = now.merge(pd.DataFrame({"_key": roll_key, col: roll[col].values, count_col: roll[count_col].values}),
                        on="_key", how="inner")
    now = now[now["Accounts"] >= _MIN[grain]]
    if now.empty:
        return pd.DataFrame(), 0
    descending = higher_is_worse == worst
    now = now.sort_values(col, ascending=not descending, kind="stable").reset_index(drop=True)
    now["Rank"] = range(1, len(now) + 1)

    prev_val, prev_rank = {}, {}
    if col in ("STD → Behind | %",):
        pass                            # needs a third month; shown as "-"
    else:
        prev = _units(df_prev, grain)
        if not prev.empty:
            prev = prev[prev["Accounts"] >= _MIN[grain]].sort_values(col, ascending=not descending, kind="stable")
            prev_val = dict(zip(prev["_key"], prev[col]))
            prev_rank = dict(zip(prev["_key"], range(1, len(prev) + 1)))
    top = now.head(n)
    out = pd.DataFrame({
        grain: top["Name"].values,
        "Accounts": top["Accounts"].values,
        "Now": top[col].values,
        "Count": top[LEAGUE_COUNTS[metric]].values if metric in LEAGUE_COUNTS else [None] * len(top),
        "Amount (Cr)": top[LEAGUE_AMOUNTS[metric]].values if metric in LEAGUE_AMOUNTS else [None] * len(top),
        "Delinquent": top["Delinquent"].values,
        # Always shown beside the chosen metric, for context.
        "Delinquency %": top["Delinquency%"].values,
        "Last Month": [prev_val.get(k) for k in top["_key"]],
        "Rank": top["Rank"].values,
        "Rank Last Month": [prev_rank.get(k) for k in top["_key"]],
    })
    if "Region" in top.columns:
        out.insert(1, "Region", top["Region"].values)
    out["Change"] = [None if p is None else round(v - p, 2) for v, p in zip(out["Now"], out["Last Month"])]
    return out, len(now)


def biggest_moves(df_curr: pd.DataFrame, df_prev: pd.DataFrame, grain: str, n: int,
                  metric: str = "Delinquency%") -> dict:
    """Units whose `metric` rose / fell most since last month (points), among
    units big enough to judge. {"worse": [...], "better": [...]}."""
    now, prev = _units(df_curr, grain), _units(df_prev, grain)
    if now.empty or prev.empty:
        return {"worse": [], "better": []}
    now = now[now["Accounts"] >= _MIN[grain]]
    before = dict(zip(prev["_key"], prev[metric]))
    count = {"Delinquency%": "Delinquent", "NPA%": "NPA", "SMA-2%": "SMA-2"}.get(metric)
    rows = [{"name": r["Name"], "region": r.get("Region"), "now": float(r[metric]),
             "count": int(r[count]) if count else None,
             "change": round(float(r[metric]) - before[r["_key"]], 2)}
            for _, r in now.iterrows() if r["_key"] in before]
    rows = [r for r in rows if r["change"] != 0]
    return {"worse": sorted([r for r in rows if r["change"] > 0], key=lambda r: -r["change"])[:n],
            "better": sorted([r for r in rows if r["change"] < 0], key=lambda r: r["change"])[:n]}


def unit_table(df_curr: pd.DataFrame, df_prev: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Every headline metric for each region / branch / executive, with the
    COUNT behind each % (so a screen can show "11.9% (37)"), last month's
    delinquency and NPA, and roll counts. Units below the app's usual minimum
    for their level are left out (MIN_ACCOUNTS_DIMENSION_BREAKDOWN for
    branches, MIN_ACCOUNTS_EXECUTIVE for executives)."""
    from config import MIN_ACCOUNTS_DIMENSION_BREAKDOWN, MIN_ACCOUNTS_EXECUTIVE
    now = _units(df_curr, grain)
    if now.empty:
        return pd.DataFrame()
    floor = {"Region": 1, "Branch": MIN_ACCOUNTS_DIMENSION_BREAKDOWN, "Executive": MIN_ACCOUNTS_EXECUTIVE}[grain]
    now = now[now["Accounts"] >= floor]
    prev = _units(df_prev, grain)
    before = prev.set_index("_key")[["Delinquency%", "NPA%"]].to_dict("index") if not prev.empty else {}
    out = pd.DataFrame({grain: now["Name"].values})
    if grain == "Executive":
        out["Branch"] = now["Unit"].astype(str).values
    if grain != "Region" and "Region" in now.columns:
        out["Region"] = now["Region"].values
    for col in ("Accounts", "Delinquent", "Delinquency%"):
        out[col] = now[col].values
    p = [before.get(k) for k in now["_key"]]
    out["Prev Delinquency%"] = [None if b is None else b["Delinquency%"] for b in p]
    out["Δ Delinquency%"] = [None if b is None else round(v - b["Delinquency%"], 2) for v, b in zip(now["Delinquency%"], p)]
    for col in ("SMA-2", "SMA-2%", "NPA", "NPA%", "NPA% (SOH)", "NPA SOH (Cr)"):
        out[col] = now[col].values
    out["Δ NPA%"] = [None if b is None else round(v - b["NPA%"], 2) for v, b in zip(now["NPA%"], p)]
    for col in ("Hard Bucket", "Hard Bucket%", "Collection%", "Strike%", "SOH (Cr)"):
        out[col] = now[col].values
    if "Roll Fwd%" in now.columns:
        out["Roll Fwd%"] = now["Roll Fwd%"].values
        out["Slipped"] = now["Slipped"].values
        out["Rescued"] = now["Rescued"].values
    return out.reset_index(drop=True)


def portfolio_total(df_curr: pd.DataFrame, df_prev: pd.DataFrame | None) -> dict:
    """Every loan in view as one unit (sum over sum, never an average of rows):
    all of unit_metrics' columns, plus last month's delinquency and the change
    when last month's file is there. {} when there are no loans. The Total row
    of every region / branch / executive table, on screen and in the report."""
    m = unit_metrics(df_curr, [])
    if m.empty:
        return {}
    t = m.iloc[0].to_dict()
    pm = unit_metrics(df_prev, []) if df_prev is not None and len(df_prev) else pd.DataFrame()
    if not pm.empty:
        t["Prev Delinquency%"] = float(pm.iloc[0]["Delinquency%"])
        t["Δ Delinquency%"] = round(t["Delinquency%"] - t["Prev Delinquency%"], 2)
    return t


def business_snapshot(df_curr: pd.DataFrame, as_of) -> dict:
    """This month's new loans vs last month, with the top segments and branches."""
    from analysis.new_business import compute_new_advances, compute_new_advances_by_dimension
    adv = compute_new_advances(df_curr, as_of=as_of)
    by = compute_new_advances_by_dimension(df_curr, as_of=as_of).get("branch", pd.DataFrame())
    seg = adv.get("segment", pd.DataFrame())
    return {
        **{k: adv.get(k) for k in ("month", "accounts", "funded_cr", "avg_ticket_l", "has_prev",
                                   "prev_accounts", "prev_funded_cr", "accounts_mom_pct", "funded_mom_pct")},
        "top_segments": [] if seg.empty else seg.head(3)[["Segment", "Accounts", "Funded (Cr)"]].to_dict("records"),
        "top_branches": [] if by.empty else by.head(3)[["Branch", "Accounts This Month", "Funded (Cr)"]].to_dict("records"),
    }


def alert_snapshot(alerts: list, alerts_prev: list) -> list[dict]:
    """Each alert's count now and last month, worst severity first."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    prev = {a["title"]: a["count"] for a in (alerts_prev or [])}
    rows = [{"title": a["title"], "count": int(a["count"]), "severity": a.get("severity", "medium"),
             "prev": prev.get(a["title"])} for a in (alerts or [])]
    return sorted(rows, key=lambda r: (order.get(r["severity"], 9), -r["count"]))

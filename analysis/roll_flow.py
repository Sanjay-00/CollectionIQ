"""
Roll / flow analysis by count AND by SOH: which loans moved between buckets
from last month to this month, measured both as a share of loans and as a
share of last month's SOH (the exposure that was sitting in that bucket).

The early steps matter most to a collection manager: a loan that was STD last
month and is behind now (a new defaulter), and a 1-30 DPD loan that slipped to
SMA-1 or worse, are the cheapest to bring back and the ones that otherwise
become NPA two or three months later.

Works on the current-month frame after app.py's prev-month carry-over
(prev_bucket, prev_SOH, matched by Loan No). Only loans present in BOTH
months with a real bucket in both are counted; units are this month's
region / branch / executive (the book a manager owns today).
"""
import pandas as pd

from config import MIN_ACCOUNTS_DIMENSION_BREAKDOWN, MIN_ACCOUNTS_EXECUTIVE, ROLL_STEPS
from utils import BUCKET_SCORE, PREV_CARRYOVER_COLS, _pct_col, to_num

VALID_BUCKETS = ["STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA"]
_PREV_SOH = PREV_CARRYOVER_COLS["SOH"]          # "prev_SOH"
CURED = "Back to STD"


def has_roll_data(df: pd.DataFrame) -> bool:
    return "prev_bucket" in df.columns and "curr_bucket" in df.columns


def _matched(df: pd.DataFrame) -> pd.DataFrame:
    """Loans with a real bucket in both months, with last month's SOH and the
    direction of travel. Last month's SOH falls back to this month's when the
    previous file has no SOH column."""
    if not has_roll_data(df) or df.empty:
        return pd.DataFrame()
    m = df[df["prev_bucket"].isin(VALID_BUCKETS) & df["curr_bucket"].isin(VALID_BUCKETS)].copy()
    prev_soh = to_num(m, _PREV_SOH) if _PREV_SOH in m.columns else pd.Series(float("nan"), index=m.index)
    m["_prev_soh"] = prev_soh.fillna(to_num(m, "SOH", fill=0)).fillna(0.0)
    m["_worse"] = m["curr_bucket"].map(BUCKET_SCORE) > m["prev_bucket"].map(BUCKET_SCORE)
    m["_cured"] = (m["prev_bucket"] != "STD") & (m["curr_bucket"] == "STD")
    return m


# ── Step flags: one column pair per early-warning step ───────────────────────

def _step_flags(m: pd.DataFrame) -> pd.DataFrame:
    """Per loan: for each step, is it in the step's base (was in the from-bucket
    last month) and did it roll (moved to a worse bucket)? Plus the cure step
    (was behind last month, STD now). Counts and last-month SOH for each."""
    f = pd.DataFrame(index=m.index)
    f["_n"] = 1
    f["_soh"] = m["_prev_soh"]
    for label, from_bucket in ROLL_STEPS.items():
        base = m["prev_bucket"] == from_bucket
        rolled = base & m["_worse"]
        f[f"{label}|base"] = base.astype(int)
        f[f"{label}|base_soh"] = m["_prev_soh"].where(base, 0.0)
        f[f"{label}|n"] = rolled.astype(int)
        f[f"{label}|soh"] = m["_prev_soh"].where(rolled, 0.0)
    behind = m["prev_bucket"] != "STD"
    f[f"{CURED}|base"] = behind.astype(int)
    f[f"{CURED}|base_soh"] = m["_prev_soh"].where(behind, 0.0)
    f[f"{CURED}|n"] = m["_cured"].astype(int)
    f[f"{CURED}|soh"] = m["_prev_soh"].where(m["_cured"], 0.0)
    return f


def step_labels() -> list[str]:
    """The early-warning steps in order, then the cure step."""
    return [*ROLL_STEPS, CURED]


def _summarise(sums: pd.DataFrame) -> pd.DataFrame:
    """Raw sums -> per step: base loans/SOH, rolled loans/SOH and both %."""
    out = pd.DataFrame(index=sums.index)
    out["Matched Accounts"] = sums["_n"].astype(int)
    out["Matched SOH (Cr)"] = (sums["_soh"] / 1e7).round(2)
    for label in step_labels():
        out[f"{label} | Base"] = sums[f"{label}|base"].astype(int)
        out[f"{label} | Base SOH (Cr)"] = (sums[f"{label}|base_soh"] / 1e7).round(2)
        out[f"{label} | Accounts"] = sums[f"{label}|n"].astype(int)
        out[f"{label} | SOH (Cr)"] = (sums[f"{label}|soh"] / 1e7).round(2)
        out[f"{label} | %"] = _pct_col(sums[f"{label}|n"], sums[f"{label}|base"])
        out[f"{label} | SOH %"] = _pct_col(sums[f"{label}|soh"], sums[f"{label}|base_soh"])
    return out


def roll_steps_summary(df: pd.DataFrame) -> dict | None:
    """Portfolio-wide (or drill-down slice) numbers for the early-warning
    cards: {step: {base, base_soh_cr, n, soh_cr, pct, soh_pct}}. None when
    there's no previous month to compare with."""
    m = _matched(df)
    if m.empty:
        return None
    row = _summarise(_step_flags(m).sum().to_frame().T).iloc[0]
    return {
        label: {
            "base": int(row[f"{label} | Base"]), "base_soh_cr": float(row[f"{label} | Base SOH (Cr)"]),
            "n": int(row[f"{label} | Accounts"]), "soh_cr": float(row[f"{label} | SOH (Cr)"]),
            "pct": float(row[f"{label} | %"]), "soh_pct": float(row[f"{label} | SOH %"]),
        }
        for label in step_labels()
    }


_GRAINS = {
    # grain: (group columns, identity columns shown, min matched accounts)
    "region": (["RegionName"], {"RegionName": "Region"}, MIN_ACCOUNTS_DIMENSION_BREAKDOWN),
    "branch": (["Unit"], {"Unit": "Branch"}, MIN_ACCOUNTS_DIMENSION_BREAKDOWN),
    "executive": (["MNT NAME", "Unit"], {"MNT NAME": "Executive", "Unit": "Branch"}, MIN_ACCOUNTS_EXECUTIVE),
}


def roll_steps_by(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """One row per region / branch / executive (this month's labels) with every
    step's base, rolled count and SOH and both %. Executives are name + branch.
    Branch and executive rows carry their Region. Sorted with the most new
    defaulters (by %) first -- the earliest, most fixable slip."""
    cols, identity, min_n = _GRAINS[grain]
    m = _matched(df)
    if m.empty or not set(cols) <= set(m.columns):
        return pd.DataFrame()
    flags = _step_flags(m)
    sums = flags.groupby([m[c] for c in cols], sort=True).sum()
    out = _summarise(sums)
    out = out[out["Matched Accounts"] >= min_n].reset_index()
    if out.empty:
        return out
    if grain != "region" and "RegionName" in m.columns:
        region = m.groupby(cols, sort=True)["RegionName"].first()
        keys = list(zip(*[out[c] for c in cols])) if len(cols) > 1 else list(out[cols[0]])
        out.insert(len(cols), "Region", [region.get(k) for k in keys])
    out = out.rename(columns=identity)
    first = next(iter(ROLL_STEPS))
    out = out.sort_values([f"{first} | %", f"{first} | Accounts"], ascending=False, kind="stable")
    return out.reset_index(drop=True)


def migration_matrix(df: pd.DataFrame, measure: str = "count") -> pd.DataFrame:
    """Last month's bucket (rows) x this month's bucket (columns).
    measure: "count" (loans), "count_pct" (% of the row's loans), "soh" (Cr of
    last month's SOH) or "soh_pct" (% of the row's last-month SOH). Each % row
    adds to 100: of what sat in that bucket last month, where did it go."""
    m = _matched(df)
    if m.empty:
        return pd.DataFrame(0.0, index=VALID_BUCKETS, columns=VALID_BUCKETS)
    if measure in ("soh", "soh_pct"):
        mat = pd.crosstab(m["prev_bucket"], m["curr_bucket"], values=m["_prev_soh"], aggfunc="sum")
    else:
        mat = pd.crosstab(m["prev_bucket"], m["curr_bucket"])
    mat = mat.reindex(index=VALID_BUCKETS, columns=VALID_BUCKETS).fillna(0.0).astype(float)
    if measure.endswith("_pct"):
        totals = mat.sum(axis=1)
        mat = mat.div(totals.where(totals > 0), axis=0).mul(100).fillna(0.0).round(1)
    elif measure == "soh":
        mat = (mat / 1e7).round(2)
    return mat


_CALL_LIST_COLS = [
    "Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit", "MNT NAME",
    "prev_bucket", "curr_bucket", "Arrears / EMI", "Closing Arrears", "SOH",
    "Last Receipt Date", "Last Receipt Amount", "Ag_Date",
]


def loans_that_rolled(df: pd.DataFrame, from_bucket: str) -> pd.DataFrame:
    """The loans behind a step: in `from_bucket` last month, a worse bucket
    now. Largest SOH first -- the biggest exposure to call about first."""
    m = _matched(df)
    if m.empty:
        return pd.DataFrame()
    hit = m[(m["prev_bucket"] == from_bucket) & m["_worse"]]
    cols = [c for c in _CALL_LIST_COLS if c in hit.columns]
    out = hit[cols].assign(SOH=to_num(hit, "SOH")).sort_values("SOH", ascending=False)
    return out.rename(columns={"prev_bucket": "Last Month", "curr_bucket": "Now",
                               "RegionName": "Region", "Unit": "Branch", "MNT NAME": "Executive"}).reset_index(drop=True)

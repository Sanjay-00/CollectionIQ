"""
Root-Cause Diagnostics  -  pre-computed, pure pandas, no LLM.

Answers "why" a region/branch's Collection%/NPA%/SMA-2% is moving the way it
is, not just "what the numbers are" -- built ON TOP of the existing Portfolio
Intelligence outputs (compute_region_scorecard, compute_branch_quadrant,
compute_bucket_waterfall, compute_fleet_exposure), not a replacement for them.

Every function here takes the SAME already-processed df (post
utils.load_and_validate / assign_buckets, so curr_bucket/SOH already exist)
that every other analysis/*.py module consumes.
"""
import io

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

from utils import to_num, account_count, is_yes
from config import (
    HARD_BUCKET_ARREARS_EMI_MIN,
    FLEET_MIN_LOANS,
    RECENT_ADVANCES_MONTHS,
    RECENT_ADVANCES_COHORT_START,
)


def _safe_div(num: float, den: float, scale: float = 100.0) -> float:
    return round(float(num) / float(den) * scale, 2) if den else 0.0


# ── Shared flag cleaning ────────────────────────────────────────────────────
# Real, verified-on-production-data contamination (a June LCC closing extract,
# not hypothetical): a handful of raw LCC columns pick up values from an
# adjacent column on some rows (a column-shift issue at the source), or leak
# in an Excel date serial where a category label was expected. Every
# downstream root-cause function must see the SAME cleaned flags -- computed
# here once, not re-filtered ad hoc per function, so no two callers can drift
# into two different definitions of "a valid Non Starter/Strike/CoLending
# value." Unrecognized values become NaN (excluded from any %, not silently
# miscounted as one side or the other) rather than trusted as-is.
_VALID_FLAG_VALUES = {
    "Non Starter":      {"Y", "YES", "N", "NO"},
    "Strike":           {"Y", "YES", "N", "NO"},
    "CoLending_Loans":  {"Y", "N"},
    "NPA Status":       {"LOAN DPD", "CUST DPD"},
}


def clean_contaminated_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Whitelist known-good values for the flag columns above; anything else
    (a stray number, a segment code, a date serial that leaked in) becomes NaN.
    Returns (cleaned_df, contamination_counts) -- the counts are for a visible
    data-quality footnote, so a caller never has to silently trust a raw
    .value_counts() on these columns again."""
    df = df.copy()
    contamination: dict[str, int] = {}
    for col, valid in _VALID_FLAG_VALUES.items():
        if col not in df.columns:
            continue
        normalized = df[col].astype(str).str.strip().str.upper()
        is_valid = normalized.isin(valid)
        is_present = df[col].notna()
        bad = int((is_present & ~is_valid).sum())
        if bad:
            contamination[col] = bad
        df[col] = df[col].where(is_valid, other=pd.NA)
    return df, contamination


# ── Item 1: Insurance-vs-installment split, by branch ───────────────────────

def compute_insurance_split(df: pd.DataFrame, group_col: str = "Unit") -> pd.DataFrame:
    """For every delinquent account (Arrears/EMI > 0), split into:
      - insurance_only: ARREARS AGAINST INST <= 0 AND ARREARS AGAINST EXP > 0
        (customer is current on the loan itself; a WCL/cash adjustment on the
        insurance/expense leg clears this -- not a credit problem)
      - installment_only: the reverse (real repayment shortfall, no expense arrears)
      - both: genuine shortfall on both legs
    One row per group_col value (default: branch/Unit), plus the region it
    rolls up to for a report table that can show either grain."""
    required = {"Arrears / EMI", "ARREARS AGAINST INST", "ARREARS AGAINST EXP", group_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    delinquent = df[to_num(df, "Arrears / EMI") > 0].copy()
    if delinquent.empty:
        return pd.DataFrame()

    inst = to_num(delinquent, "ARREARS AGAINST INST")
    exp = to_num(delinquent, "ARREARS AGAINST EXP")
    delinquent["_insurance_only"] = (inst <= 0) & (exp > 0)
    delinquent["_installment_only"] = (inst > 0) & (exp <= 0)
    delinquent["_both"] = (inst > 0) & (exp > 0)

    rows = []
    group_cols = [group_col] + (["RegionName"] if "RegionName" in df.columns and group_col != "RegionName" else [])
    for keys, grp in delinquent.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = account_count(grp)
        row = dict(zip(group_cols, keys))
        row.update({
            "Delinquent Accounts": n,
            "Insurance-Only": int(grp["_insurance_only"].sum()),
            "Insurance-Only %": _safe_div(grp["_insurance_only"].sum(), n),
            "Installment-Only": int(grp["_installment_only"].sum()),
            "Installment-Only %": _safe_div(grp["_installment_only"].sum(), n),
            "Both": int(grp["_both"].sum()),
            "Both %": _safe_div(grp["_both"].sum(), n),
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Insurance-Only %", ascending=False).reset_index(drop=True)


# ── Item 2: Chronic vs. shock four-way split, by branch ─────────────────────

def compute_chronic_shock_split(df: pd.DataFrame, group_col: str = "Unit") -> pd.DataFrame:
    """Cross a BEHAVIORAL flag (No Coll 3 Months and >6 EMI -- a zero-payment
    streak that at some point crossed 6 EMIs of arrears) against a SNAPSHOT
    flag (currently Arrears/EMI >= HARD_BUCKET_ARREARS_EMI_MIN, today's
    severity regardless of history). The two can diverge -- a customer can hit
    one without the other -- so treating them as the same signal hides which
    of four genuinely different populations an account belongs to:
      - Chronic + Hard:      write-off/legal candidates
      - Chronic + Not Hard:  partial payer after a long silence, worth a call
                              before they relapse
      - Shock + Hard:        sudden deterioration, a restructuring candidate
      - Neither:             normal delinquency
    """
    required = {"Arrears / EMI", group_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    delinquent = df[to_num(df, "Arrears / EMI") > 0].copy()
    if delinquent.empty:
        return pd.DataFrame()

    chronic = is_yes(delinquent, "No Coll 3 Months and >6 EMI") if "No Coll 3 Months and >6 EMI" in delinquent.columns else pd.Series(False, index=delinquent.index)
    hard = to_num(delinquent, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN

    delinquent["_chronic_hard"] = chronic & hard
    delinquent["_chronic_not_hard"] = chronic & ~hard
    delinquent["_shock_hard"] = ~chronic & hard
    delinquent["_neither"] = ~chronic & ~hard

    rows = []
    group_cols = [group_col] + (["RegionName"] if "RegionName" in df.columns and group_col != "RegionName" else [])
    for keys, grp in delinquent.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = account_count(grp)
        row = dict(zip(group_cols, keys))
        row.update({
            "Delinquent Accounts": n,
            "Chronic + Hard": int(grp["_chronic_hard"].sum()),
            "Chronic + Hard %": _safe_div(grp["_chronic_hard"].sum(), n),
            "Chronic + Not Hard": int(grp["_chronic_not_hard"].sum()),
            "Chronic + Not Hard %": _safe_div(grp["_chronic_not_hard"].sum(), n),
            "Shock + Hard": int(grp["_shock_hard"].sum()),
            "Shock + Hard %": _safe_div(grp["_shock_hard"].sum(), n),
            "Neither": int(grp["_neither"].sum()),
            "Neither %": _safe_div(grp["_neither"].sum(), n),
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Chronic + Hard %", ascending=False).reset_index(drop=True)


# ── Item 5: Per-region "why" synthesis ───────────────────────────────────────

_DRIVER_LABELS = {
    "insurance_only_share":  "Insurance-driven delinquency",
    "chronic_share":         "Chronic non-payer buildup",
    "shock_share":           "Sudden-shock deterioration",
    "fleet_share":           "Fleet-operator concentration",
    "recent_vintage_share":  "Recent-advance (sourcing/underwriting) quality",
}


def compute_region_why_table(df_curr: pd.DataFrame, region_scorecard: pd.DataFrame) -> pd.DataFrame:
    """One row per region: NPA%/Collection%/Δ NPA% (from the existing
    compute_region_scorecard) plus a single DOMINANT DRIVER tag, picked as
    whichever candidate driver claims the largest share of that region's
    delinquent book:
      - insurance_only_share:  ARREARS AGAINST INST<=0 AND ARREARS AGAINST EXP>0
      - chronic_share:         No Coll 3 Months and >6 EMI
      - shock_share:           Arrears/EMI>=HARD_BUCKET_ARREARS_EMI_MIN AND NOT chronic
      - fleet_share:           delinquent loan belongs to a customer with
                                >= FLEET_MIN_LOANS loans (computed on the FULL
                                book, not region-scoped -- a fleet owner's
                                loans can span regions/branches)
      - recent_vintage_share:  Ag_Date within RECENT_ADVANCES_MONTHS of today
    This is the one-page deliverable for a manager: not a pile of charts, a
    single "here's why, specifically" table, per region.
    """
    required = {"Arrears / EMI", "RegionName", "Cust Mob No", "Loan No"}
    if df_curr.empty or not required.issubset(df_curr.columns) or region_scorecard.empty:
        return pd.DataFrame()

    # Fleet ownership is a PORTFOLIO-WIDE rollup (customer, not region/branch --
    # semantic_model.py's own callout: a fleet owner's loans can span branches).
    loans_per_customer = df_curr.groupby("Cust Mob No")["Loan No"].transform("nunique")
    is_fleet_loan = loans_per_customer >= FLEET_MIN_LOANS

    recent_cutoff = pd.Timestamp.today() - pd.DateOffset(months=RECENT_ADVANCES_MONTHS)
    ag_date = pd.to_datetime(df_curr.get("Ag_Date"), errors="coerce") if "Ag_Date" in df_curr.columns else None

    rows = []
    for region, grp in df_curr.groupby("RegionName"):
        delinquent = grp[to_num(grp, "Arrears / EMI") > 0]
        n = account_count(delinquent)
        if n == 0:
            continue

        inst = to_num(delinquent, "ARREARS AGAINST INST") if "ARREARS AGAINST INST" in delinquent.columns else pd.Series(0, index=delinquent.index)
        exp = to_num(delinquent, "ARREARS AGAINST EXP") if "ARREARS AGAINST EXP" in delinquent.columns else pd.Series(0, index=delinquent.index)
        insurance_only = ((inst <= 0) & (exp > 0)).sum()

        chronic = is_yes(delinquent, "No Coll 3 Months and >6 EMI") if "No Coll 3 Months and >6 EMI" in delinquent.columns else pd.Series(False, index=delinquent.index)
        hard = to_num(delinquent, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN
        shock = (~chronic & hard).sum()

        fleet = is_fleet_loan.loc[delinquent.index].sum()

        recent_vintage = 0
        if ag_date is not None:
            recent_vintage = (ag_date.loc[delinquent.index] >= recent_cutoff).sum()

        shares = {
            "insurance_only_share": _safe_div(insurance_only, n, scale=1.0),
            "chronic_share":        _safe_div(chronic.sum(), n, scale=1.0),
            "shock_share":          _safe_div(shock, n, scale=1.0),
            "fleet_share":          _safe_div(fleet, n, scale=1.0),
            "recent_vintage_share": _safe_div(recent_vintage, n, scale=1.0),
        }
        dominant_key = max(shares, key=shares.get)

        sc_row = region_scorecard[region_scorecard["Region"] == region]
        rows.append({
            "Region": region,
            "NPA%": float(sc_row["NPA%"].iloc[0]) if not sc_row.empty else None,
            "Δ NPA%": float(sc_row["Δ NPA%"].iloc[0]) if not sc_row.empty and sc_row["Δ NPA%"].iloc[0] is not None else None,
            "Collection%": float(sc_row["Collection%"].iloc[0]) if not sc_row.empty else None,
            "Status": sc_row["Status"].iloc[0] if not sc_row.empty else "-",
            "Dominant Driver": _DRIVER_LABELS[dominant_key],
            "Driver Share %": round(shares[dominant_key] * 100, 1),
            "Delinquent Accounts": n,
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Worsening regions first -- that's where a manager's attention goes first.
    status_order = {"Worsening": 0, "Stable": 1, "-": 2, "Improving": 3}
    out["_sort"] = out["Status"].map(status_order).fillna(2)
    return out.sort_values(["_sort", "NPA%"], ascending=[True, False]).drop(columns="_sort").reset_index(drop=True)


# ── Recent Advances (Nov'25 onward) cohort profile ──────────────────────────
# A SEPARATE bucket scheme from utils.BUCKET_ORDER/curr_bucket (STD/1-30 DPD/
# SMA-1/SMA-2/NPA/NA) -- this one matches the Excel Automation project's own
# bucket_analysis.yaml thresholds on raw Arrears/EMI exactly (0-1/1-2/2-3/NPA,
# the same buckets that project already emails to branches daily), so a
# manager sees the SAME bucket definitions here as in that report, not a
# second, subtly different scheme.
_RECENT_BUCKET_ORDER = ["STD", "0-1", "1-2", "2-3", "NPA"]


def _assign_recent_bucket(arrears_emi: pd.Series) -> pd.Series:
    v = pd.to_numeric(arrears_emi, errors="coerce")
    return pd.Series(
        np.select(
            [v.isna(), v <= 0, v < 1, v < 2, v < 3],
            ["NA", "STD", "0-1", "1-2", "2-3"],
            default="NPA",
        ),
        index=arrears_emi.index,
    )


def compute_recent_advances_summary(df_curr: pd.DataFrame, cohort_start: str = RECENT_ADVANCES_COHORT_START) -> dict:
    """Headline: portfolio-wide count/SOH vs. the Nov'25+ (or whatever
    cohort_start is) cohort's count/SOH, and the cohort's share of each."""
    if df_curr.empty or "Ag_Date" not in df_curr.columns:
        return {}
    ag_date = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
    cohort = df_curr[ag_date >= pd.Timestamp(cohort_start)]

    portfolio_count = account_count(df_curr)
    portfolio_soh = to_num(df_curr, "SOH").sum() if "SOH" in df_curr.columns else 0.0
    cohort_count = account_count(cohort)
    cohort_soh = to_num(cohort, "SOH").sum() if "SOH" in cohort.columns else 0.0

    return {
        "cohort_start": cohort_start,
        "portfolio_count": portfolio_count,
        "portfolio_soh_cr": round(portfolio_soh / 1e7, 2),
        "cohort_count": cohort_count,
        "cohort_soh_cr": round(cohort_soh / 1e7, 2),
        "cohort_count_pct": _safe_div(cohort_count, portfolio_count),
        "cohort_soh_pct": _safe_div(cohort_soh, portfolio_soh),
    }


def _cohort_df(df_curr: pd.DataFrame, cohort_start: str) -> pd.DataFrame:
    if df_curr.empty or "Ag_Date" not in df_curr.columns:
        return pd.DataFrame()
    ag_date = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
    return df_curr[ag_date >= pd.Timestamp(cohort_start)].copy()


def compute_recent_advances_bucket_summary(df_curr: pd.DataFrame, cohort_start: str = RECENT_ADVANCES_COHORT_START) -> pd.DataFrame:
    """Portfolio-wide bucket-wise split of the Nov'25+ cohort: one row per
    bucket (STD/0-1/1-2/2-3/NPA), Count/Count%/SOH(Cr)/SOH% -- % is of the
    COHORT total, not the whole portfolio. Loans with no parseable Arrears/EMI
    are excluded from bucketing but still counted in the cohort total via the
    'NA' share implicitly missing from the 100% (surfaced in the caller if needed)."""
    cohort = _cohort_df(df_curr, cohort_start)
    if cohort.empty or "Arrears / EMI" not in cohort.columns:
        return pd.DataFrame()

    cohort = cohort.copy()
    cohort["_bucket"] = _assign_recent_bucket(cohort["Arrears / EMI"])
    graded = cohort[cohort["_bucket"] != "NA"]
    total_count = account_count(graded)
    total_soh = to_num(graded, "SOH").sum() if "SOH" in graded.columns else 0.0

    rows = []
    for bucket in _RECENT_BUCKET_ORDER:
        sub = graded[graded["_bucket"] == bucket]
        n = account_count(sub)
        soh = to_num(sub, "SOH").sum() if "SOH" in sub.columns else 0.0
        rows.append({
            "Bucket": bucket,
            "Count": n,
            "Count %": _safe_div(n, total_count),
            "SOH (Cr)": round(soh / 1e7, 2),
            "SOH %": _safe_div(soh, total_soh),
        })
    return pd.DataFrame(rows)


def compute_recent_advances_bucket_by_group(
    df_curr: pd.DataFrame, group_col: str = "Unit", cohort_start: str = RECENT_ADVANCES_COHORT_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same bucket-wise split as compute_recent_advances_bucket_summary, but
    one row per group_col value (Region/Branch) instead of one portfolio-wide
    row. Returns (count_pct_wide, soh_pct_wide) -- each wide-format: rows =
    group value (+ Region, if grouping by Branch/Unit), columns = bucket
    order + Total, values = that bucket's % share of the group's own cohort."""
    cohort = _cohort_df(df_curr, cohort_start)
    required = {"Arrears / EMI", group_col}
    if cohort.empty or not required.issubset(cohort.columns):
        return pd.DataFrame(), pd.DataFrame()

    cohort = cohort.copy()
    cohort["_bucket"] = _assign_recent_bucket(cohort["Arrears / EMI"])
    graded = cohort[cohort["_bucket"] != "NA"]
    if graded.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = [group_col] + (["RegionName"] if "RegionName" in graded.columns and group_col != "RegionName" else [])

    count_rows, soh_rows = [], []
    for keys, grp in graded.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n_total = account_count(grp)
        soh_total = to_num(grp, "SOH").sum() if "SOH" in grp.columns else 0.0
        count_row = dict(zip(group_cols, keys))
        soh_row = dict(zip(group_cols, keys))
        for bucket in _RECENT_BUCKET_ORDER:
            sub = grp[grp["_bucket"] == bucket]
            n = account_count(sub)
            soh = to_num(sub, "SOH").sum() if "SOH" in sub.columns else 0.0
            count_row[bucket] = n
            count_row[f"{bucket} %"] = _safe_div(n, n_total)
            soh_row[f"{bucket} (Cr)"] = round(soh / 1e7, 2)
            soh_row[f"{bucket} %"] = _safe_div(soh, soh_total)
        count_row["Total Count"] = n_total
        soh_row["Total SOH (Cr)"] = round(soh_total / 1e7, 2)
        count_rows.append(count_row)
        soh_rows.append(soh_row)

    count_df = pd.DataFrame(count_rows).sort_values("Total Count", ascending=False).reset_index(drop=True) if count_rows else pd.DataFrame()
    soh_df = pd.DataFrame(soh_rows).sort_values("Total SOH (Cr)", ascending=False).reset_index(drop=True) if soh_rows else pd.DataFrame()
    return count_df, soh_df


# ── Match against the daily due-date-missed feed ────────────────────────────
# A DIFFERENT file/schema from the master LCC (see core/rc_processor.py and
# core/transformer.py in the sibling Excel Automation project): one row per
# CURRENTLY-missed-due-date loan, refreshed daily, with no SOH/POS column at
# all -- so it can only ever answer "is this cohort loan on today's live
# missed-due-date list," never contribute to the SOH-based bucket profile
# above. Loan No / LOAN NO is the verified stable join key across both files.

_DAILY_FEED_LOAN_COL_CANDIDATES = ["LOAN NO", "Loan No", "LOAN_NO"]


def load_daily_missed_feed(file) -> tuple[pd.DataFrame | None, str | None]:
    """Read the Excel Automation due-date-missed daily extract (.xlsb/.xlsx/.xls).
    Returns (df, error) -- df is None on failure. These files often carry
    pivot/summary sheets AHEAD of the data (e.g. the 26 Sept extract has a
    pivot as its first sheet), so this picks the sheet that actually has a
    Loan No column -- the largest one if several do -- instead of assuming
    sheet 0. Column names are stripped but NOT renamed to the master LCC's
    schema (LOAN NO, REGIONNAME, ... stay as-is): it is read only for the Loan
    No match and region counts, never merged wholesale into the master."""
    try:
        fname = getattr(file, "name", "").lower()
        if fname.endswith(".xlsb"):
            engine = "pyxlsb"
        elif fname.endswith(".xls"):
            engine = "xlrd"
        else:
            engine = "openpyxl"
        if hasattr(file, "seek"):
            file.seek(0)
        book = pd.ExcelFile(file, engine=engine)
        best = None
        for sheet in book.sheet_names:
            candidate = book.parse(sheet)
            candidate.columns = candidate.columns.astype(str).str.strip()
            if any(c in candidate.columns for c in _DAILY_FEED_LOAN_COL_CANDIDATES):
                if best is None or len(candidate) > len(best):
                    best = candidate
    except Exception as e:
        return None, f"Could not read file: {e}"
    if best is None:
        return None, "Could not find a sheet with a Loan No column (expected 'LOAN NO') in this file."
    return best, None


def compute_recent_advances_daily_match(
    df_curr: pd.DataFrame, daily_df: pd.DataFrame, cohort_start: str = RECENT_ADVANCES_COHORT_START,
) -> dict:
    """Of the master's Nov'25+ cohort, how many Loan Nos also appear on
    TODAY's live due-date-missed list -- a freshness cross-check against
    master's own (possibly weeks/months-stale) snapshot, not a second bucket
    computation. Also reports daily-feed loans NOT found in the cohort at
    all, which is itself a signal: likely a loan disbursed AFTER the master's
    closing date, not yet reflected in the snapshot being analyzed."""
    cohort = _cohort_df(df_curr, cohort_start)
    if cohort.empty or "Loan No" not in cohort.columns:
        return {}
    loan_col = next((c for c in _DAILY_FEED_LOAN_COL_CANDIDATES if c in daily_df.columns), None)
    if loan_col is None:
        return {}

    cohort_loans = set(cohort["Loan No"].astype(str))
    daily_loans = set(daily_df[loan_col].astype(str))
    matched = cohort_loans & daily_loans
    unmatched_daily = daily_loans - cohort_loans

    return {
        "cohort_count": len(cohort_loans),
        "daily_feed_count": len(daily_loans),
        "matched_count": len(matched),
        "pct_of_cohort_on_daily_list": _safe_div(len(matched), len(cohort_loans)),
        "pct_of_daily_list_in_cohort": _safe_div(len(matched), len(daily_loans)),
        "unmatched_daily_count": len(unmatched_daily),
        "unmatched_daily_loan_nos": sorted(unmatched_daily),
    }


# ── Region-wise delinquency status of the cohort (the manager's table) ───────
# Denominator (total cohort loans per region) comes from the MASTER LCC; the
# numerators (delinquent / PNPA / NPA) come from the daily due-date-missed feed
# itself, using the feed's own Arrears/EMI -- the feed is today's live status,
# and has no SOH/POS, so counts only. PNPA = SMA-2 = 2 <= Arrears/EMI < 3,
# NPA = Arrears/EMI >= 3: the same thresholds as the Bucket column the
# Excel Automation reports use (verified: reproduces 621 PNPA / 314 NPA on the
# 26 Sept feed). All % are of the region's TOTAL cohort, not of its delinquents.

# The master LCC and the daily feed spell some regions differently.
_REGION_ALIASES = {"CHHATRAPATI SAMBHAJI NAGAR": "CS NAGAR"}
_DAILY_FEED_REGION_CANDIDATES = ["REGIONNAME", "RegionName", "REGION"]
_DAILY_FEED_ARREARS_CANDIDATES = ["ARREARS / EMI", "Arrears / EMI"]


def _canon_region(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().replace(_REGION_ALIASES)


def _pct_or_none(num: float, den: float) -> float | None:
    return round(float(num) / float(den) * 100, 2) if den else None


_GRAIN_SPECS: dict[str, dict] = {
    "region": {
        "master_cols": ["RegionName"],
        "daily_candidates": [["REGIONNAME"], ["RegionName"], ["REGION"]],
        "aliases": _REGION_ALIASES,
        "out_cols": ["Region"],
    },
    "branch": {
        "master_cols": ["Unit"],
        "daily_candidates": [["UNIT"], ["Unit"]],
        "aliases": None,
        "out_cols": ["Branch"],
    },
    "executive": {
        # Composite key -- the SAME convention registry/semantic_model.py uses
        # for the executive entity: an executive name recurs across branches,
        # so name alone doesn't uniquely identify one (verified on real data:
        # (RE NAME, UNIT) tuples overlap the master's (MNT NAME, Unit) tuples;
        # RE NAME alone would silently merge different people at different
        # branches who happen to share a name).
        "master_cols": ["MNT NAME", "Unit"],
        "daily_candidates": [["RE NAME", "UNIT"], ["RE NAME", "Unit"]],
        "aliases": None,
        "out_cols": ["Executive", "Branch"],
    },
}


def _canon_cols(df: pd.DataFrame, cols: list[str], aliases: dict | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(str).str.strip().str.upper()
        if aliases:
            s = s.replace(aliases)
        out[c] = s
    return out


def compute_recent_advances_status_by_grain(
    df_curr: pd.DataFrame, daily_df: pd.DataFrame, grain: str = "region",
    cohort_start: str = RECENT_ADVANCES_COHORT_START,
) -> pd.DataFrame:
    """One row per Region/Branch/Executive (grain) + a Grand Total row: total
    cohort loans (master), delinquent/PNPA(SMA-2)/NPA counts from today's
    due-date-missed feed's own Arrears/EMI, and each as a % of that group's
    total cohort. Sorted WORST first (highest Delinquent %) -- the row order
    itself is the priority order for a manager, and the UI/Excel layers color
    it red (worst) to green (best) on top of this. A feed group with no
    matching master group (name mismatch / not in this master) keeps its
    counts but gets a blank total and %, rather than a misleading number."""
    spec = _GRAIN_SPECS[grain]
    cohort = _cohort_df(df_curr, cohort_start)
    if cohort.empty or "Loan No" not in cohort.columns or not set(spec["master_cols"]).issubset(cohort.columns):
        return pd.DataFrame()
    daily_cols = next((c for c in spec["daily_candidates"] if set(c).issubset(daily_df.columns)), None)
    arrears_col = next((c for c in _DAILY_FEED_ARREARS_CANDIDATES if c in daily_df.columns), None)
    loan_col = next((c for c in _DAILY_FEED_LOAN_COL_CANDIDATES if c in daily_df.columns), None)
    if not (daily_cols and arrears_col and loan_col):
        return pd.DataFrame()

    out_cols = spec["out_cols"]
    m_keys = _canon_cols(cohort, spec["master_cols"], spec["aliases"])
    m_keys.columns = out_cols
    totals = m_keys.assign(_loan=cohort["Loan No"].values).groupby(out_cols)["_loan"].nunique()

    f_keys = _canon_cols(daily_df, daily_cols, spec["aliases"])
    f_keys.columns = out_cols
    feed = f_keys.assign(
        _loan=daily_df[loan_col].astype(str).values,
        _a=pd.to_numeric(daily_df[arrears_col], errors="coerce").values,
    ).drop_duplicates("_loan")
    feed["_pnpa"] = (feed["_a"] >= 2) & (feed["_a"] < 3)
    feed["_npa"] = feed["_a"] >= 3
    by_grp = feed.groupby(out_cols).agg(delinq=("_loan", "size"), pnpa=("_pnpa", "sum"), npa=("_npa", "sum"))

    label = pd.Timestamp(cohort_start).strftime("%b'%y")
    total_col = f"Total {label} Onward Cases"
    rows = []
    for key in totals.index.union(by_grp.index):
        key_tuple = key if isinstance(key, tuple) else (key,)
        total = int(totals.get(key, 0)) or None
        d = int(by_grp["delinq"].get(key, 0))
        p = int(by_grp["pnpa"].get(key, 0))
        n = int(by_grp["npa"].get(key, 0))
        row = dict(zip(out_cols, key_tuple))
        row.update({
            total_col: total,
            "Delinquent Cases": d, "Delinquent %": _pct_or_none(d, total),
            "PNPA (SMA-2) Cases": p, "PNPA %": _pct_or_none(p, total),
            "NPA Cases": n, "NPA %": _pct_or_none(n, total),
        })
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("Delinquent %", ascending=False, na_position="last").reset_index(drop=True)

    grand_total = int(out[total_col].fillna(0).sum())
    gd, gp, gn = (int(out[c].sum()) for c in ("Delinquent Cases", "PNPA (SMA-2) Cases", "NPA Cases"))
    grand = {c: "" for c in out_cols}
    grand[out_cols[0]] = "Grand Total"
    grand.update({
        total_col: grand_total,
        "Delinquent Cases": gd, "Delinquent %": _pct_or_none(gd, grand_total),
        "PNPA (SMA-2) Cases": gp, "PNPA %": _pct_or_none(gp, grand_total),
        "NPA Cases": gn, "NPA %": _pct_or_none(gn, grand_total),
    })
    return pd.concat([out, pd.DataFrame([grand])], ignore_index=True)


def compute_recent_advances_region_status(
    df_curr: pd.DataFrame, daily_df: pd.DataFrame, cohort_start: str = RECENT_ADVANCES_COHORT_START,
) -> pd.DataFrame:
    """Region-grain convenience wrapper -- see compute_recent_advances_status_by_grain."""
    return compute_recent_advances_status_by_grain(df_curr, daily_df, grain="region", cohort_start=cohort_start)




# ── Standalone, manager-presentable Excel export ─────────────────────────────
# openpyxl (not xlsxwriter -- that's only present in this venv as some other
# package's transitive dependency, not a pinned requirement here) to match
# ui/components.py::_excel_bytes, the app's one other Excel writer, so there's
# a single pinned Excel-writing dependency for the whole project.

_HEADER_FILL = PatternFill("solid", fgColor="1A3660")
_HEADER_FONT = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
_TITLE_FONT = Font(bold=True, size=16, color="1A3660", name="Calibri")
_SUBTITLE_FONT = Font(italic=True, size=10, color="6B7280", name="Calibri")
_SECTION_FONT = Font(bold=True, size=12, color="111827", name="Calibri")
_WARN_FONT = Font(bold=True, size=10, color="92400E", name="Calibri")

_STATUS_STYLE = {
    "Worsening": (PatternFill("solid", fgColor="FEE2E2"), Font(color="991B1B", bold=True)),
    "Improving": (PatternFill("solid", fgColor="DCFCE7"), Font(color="166534", bold=True)),
    "Stable":    (PatternFill("solid", fgColor="FEF3C7"), Font(color="92400E", bold=True)),
}


def _write_df_sheet(ws, df: pd.DataFrame, status_col: str | None = None) -> None:
    """Write `df` starting at row 1 with a styled header, auto-sized columns,
    a frozen header row, and (if status_col is given) color-coded status cells."""
    if df.empty:
        ws.cell(row=1, column=1, value="No data available.")
        return

    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=1, column=j, value=col)
        c.font = _HEADER_FONT
        c.fill = _HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for i, (_, r) in enumerate(df.iterrows(), start=2):
        for j, col in enumerate(df.columns, start=1):
            val = r[col]
            if isinstance(val, float) and pd.isna(val):
                val = None
            cell = ws.cell(row=i, column=j, value=val)
            if status_col and col == status_col and val in _STATUS_STYLE:
                fill, font = _STATUS_STYLE[val]
                cell.fill = fill
                cell.font = font

    for j, col in enumerate(df.columns, start=1):
        # Not `.astype(str).str.len()` -- pandas' string-backed dtype (seen on
        # a column of all-None, e.g. "Δ NPA%" with no previous month loaded)
        # propagates NA through that chain instead of stringifying None to
        # "None", so .max() on an all-null column returns NaN, not 0 -- and
        # int(NaN) crashes. Filtering to non-null values directly sidesteps
        # the dtype-dependent stringification behavior entirely.
        lengths = [len(str(v)) for v in df[col] if pd.notna(v)]
        max_len = max([len(str(col))] + lengths) if lengths else len(str(col))
        ws.column_dimensions[get_column_letter(j)].width = min(max(max_len + 2, 10), 36)

    ws.freeze_panes = "A2"


# Excel's own default 3-color-scale palette (red/yellow/green) -- so this
# reads as a native conditional format, not a custom one-off color choice.
_COLORSCALE_RED = "F8696B"
_COLORSCALE_YELLOW = "FFEB84"
_COLORSCALE_GREEN = "63BE7B"


def _apply_worst_first_colorscale(ws, n_data_rows: int, pct_col_letters: list[str]) -> None:
    """Red (highest/worst) -> yellow -> green (lowest/best) on the given
    columns, over rows 2..n_data_rows+1 -- i.e. EXCLUDING the Grand Total row,
    which callers append as the last row: an aggregate isn't a peer to rank
    against the entities it's a total of, and including it would also skew
    the min/max the color scale computes."""
    if n_data_rows <= 1:
        return
    last_row = n_data_rows  # data rows are 2..(n_data_rows+1); Grand Total is row n_data_rows+1
    rule = ColorScaleRule(
        start_type="min", start_color=_COLORSCALE_GREEN,
        mid_type="percentile", mid_value=50, mid_color=_COLORSCALE_YELLOW,
        end_type="max", end_color=_COLORSCALE_RED,
    )
    for col in pct_col_letters:
        ws.conditional_formatting.add(f"{col}2:{col}{last_row}", rule)


def _write_status_sheet(ws, df: pd.DataFrame) -> None:
    """Like _write_df_sheet, plus the worst-first red->green color scale on
    every '... %' column (Delinquent %/PNPA %/NPA %), excluding Grand Total."""
    _write_df_sheet(ws, df)
    if df.empty:
        return
    pct_cols = [c for c in df.columns if c.endswith("%")]
    col_letters = [get_column_letter(df.columns.get_loc(c) + 1) for c in pct_cols]
    # Grand Total is the last row (see compute_recent_advances_status_by_grain).
    _apply_worst_first_colorscale(ws, n_data_rows=len(df) - 1, pct_col_letters=col_letters)


def build_root_cause_workbook(
    why_df: pd.DataFrame,
    insurance_split_df: pd.DataFrame,
    chronic_shock_df: pd.DataFrame,
    contamination: dict[str, int],
    generated_for: str = "",
    recent_summary: dict | None = None,
    recent_bucket_df: pd.DataFrame | None = None,
    recent_by_region_count: pd.DataFrame | None = None,
    recent_by_region_soh: pd.DataFrame | None = None,
    recent_by_branch_count: pd.DataFrame | None = None,
    recent_by_branch_soh: pd.DataFrame | None = None,
    daily_match: dict | None = None,
    region_status: pd.DataFrame | None = None,
    branch_status: pd.DataFrame | None = None,
    executive_status: pd.DataFrame | None = None,
) -> bytes:
    """One styled workbook, presentation-ready for a manager: a Summary/cover
    sheet (title, generation date, data-quality notice) plus one sheet per
    diagnostic table. Pure function -- returns raw .xlsx bytes, no Streamlit
    dependency, so it's independently testable. The recent_* / daily_match
    args are all optional -- the workbook degrades gracefully to just the
    three original root-cause sheets when the cohort analysis hasn't been run."""
    wb = Workbook()
    cover = wb.active
    cover.title = "Summary"

    cover["A1"] = "Root Cause Diagnostics"
    cover["A1"].font = _TITLE_FONT
    subtitle = f"Generated {pd.Timestamp.now():%d %b %Y, %H:%M}"
    if generated_for:
        subtitle += f"  -  {generated_for}"
    cover["A2"] = subtitle
    cover["A2"].font = _SUBTITLE_FONT

    row = 4
    cover.cell(row=row, column=1, value="Contents").font = _SECTION_FONT
    row += 1
    contents = ["Region Diagnosis (the \"why\" table)", "Insurance vs. Installment Split (by branch)", "Chronic vs. Shock Split (by branch)"]
    if recent_bucket_df is not None and not recent_bucket_df.empty:
        contents += ["Recent Advances Summary", "Recent Advances Bucket Split", "Recent Advances by Region", "Recent Advances by Branch"]
    status_labels = {"Region": region_status, "Branch": branch_status, "Executive": executive_status}
    for label, df in status_labels.items():
        if df is not None and not df.empty:
            contents.append(f"Recent Advances: {label} Delinquency Status (worst first)")
    if daily_match:
        contents.append("Recent Advances: Daily Feed Match")
    for label in contents:
        cover.cell(row=row, column=1, value=f"•  {label}")
        row += 1

    if contamination:
        row += 1
        cover.cell(row=row, column=1, value="Data Quality Notice").font = _WARN_FONT
        row += 1
        cover.cell(
            row=row, column=1,
            value="Unrecognized values were found and excluded from the flags below "
                  "(not silently trusted as Yes/No):",
        )
        row += 1
        for col, n in contamination.items():
            cover.cell(row=row, column=1, value=f"  {col}: {n} rows")
            row += 1

    cover.column_dimensions["A"].width = 78

    _write_df_sheet(wb.create_sheet("Region Diagnosis"), why_df, status_col="Status")
    _write_df_sheet(wb.create_sheet("Insurance Split"), insurance_split_df)
    _write_df_sheet(wb.create_sheet("Chronic vs Shock"), chronic_shock_df)

    if recent_bucket_df is not None and not recent_bucket_df.empty:
        if recent_summary:
            ws = wb.create_sheet("Recent Advances Summary")
            ws["A1"] = f"Recent Advances Cohort (from {recent_summary.get('cohort_start', '')})"
            ws["A1"].font = _SECTION_FONT
            labels = [
                ("Portfolio Loans", recent_summary.get("portfolio_count")),
                ("Portfolio SOH (Cr)", recent_summary.get("portfolio_soh_cr")),
                ("Cohort Loans", recent_summary.get("cohort_count")),
                ("Cohort SOH (Cr)", recent_summary.get("cohort_soh_cr")),
                ("Cohort % of Portfolio (Count)", recent_summary.get("cohort_count_pct")),
                ("Cohort % of Portfolio (SOH)", recent_summary.get("cohort_soh_pct")),
            ]
            for i, (label, val) in enumerate(labels, start=3):
                ws.cell(row=i, column=1, value=label).font = Font(bold=True)
                ws.cell(row=i, column=2, value=val)
            ws.column_dimensions["A"].width = 34
        _write_df_sheet(wb.create_sheet("Recent Advances Bucket Split"), recent_bucket_df)
        if recent_by_region_count is not None and not recent_by_region_count.empty:
            _write_df_sheet(wb.create_sheet("Recent Adv by Region (Count)"), recent_by_region_count)
        if recent_by_region_soh is not None and not recent_by_region_soh.empty:
            _write_df_sheet(wb.create_sheet("Recent Adv by Region (SOH)"), recent_by_region_soh)
        if recent_by_branch_count is not None and not recent_by_branch_count.empty:
            _write_df_sheet(wb.create_sheet("Recent Adv by Branch (Count)"), recent_by_branch_count)
        if recent_by_branch_soh is not None and not recent_by_branch_soh.empty:
            _write_df_sheet(wb.create_sheet("Recent Adv by Branch (SOH)"), recent_by_branch_soh)

    if region_status is not None and not region_status.empty:
        _write_status_sheet(wb.create_sheet("Recent Adv Region Status"), region_status)
    if branch_status is not None and not branch_status.empty:
        _write_status_sheet(wb.create_sheet("Recent Adv Branch Status"), branch_status)
    if executive_status is not None and not executive_status.empty:
        _write_status_sheet(wb.create_sheet("Recent Adv Executive Status"), executive_status)

    if daily_match:
        ws = wb.create_sheet("Recent Adv Daily Feed Match")
        ws["A1"] = "Recent Advances Cohort vs. Today's Due-Date-Missed Feed"
        ws["A1"].font = _SECTION_FONT
        labels = [
            ("Cohort Loans", daily_match.get("cohort_count")),
            ("Daily Feed Loans", daily_match.get("daily_feed_count")),
            ("Matched (on both)", daily_match.get("matched_count")),
            ("% of Cohort Currently on Missed-Due-Date List", daily_match.get("pct_of_cohort_on_daily_list")),
            ("% of Daily Feed Found in Cohort", daily_match.get("pct_of_daily_list_in_cohort")),
            ("Daily Feed Loans NOT in Cohort (likely disbursed after this snapshot)", daily_match.get("unmatched_daily_count")),
        ]
        for i, (label, val) in enumerate(labels, start=3):
            ws.cell(row=i, column=1, value=label).font = Font(bold=True)
            ws.cell(row=i, column=2, value=val)
        ws.column_dimensions["A"].width = 58

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

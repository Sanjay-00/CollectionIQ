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

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

from utils import (
    to_num, account_count, is_yes, recent_advances_cutoff, bucket_from_arrears_emi, fleet_loan_mask,
    canonical_region, delinquency_by, attach_prev_delinquency, _unit_key, _safe_pct, order_unit_columns,
    insurance_only_mask, bucket_from_arrears_emi as _bucket_rule, parse_date_columns,
)
from config import (
    HARD_BUCKET_ARREARS_EMI_MIN,
    NOT_PAYING_3M_FLAG_COL,
    RECENT_ADVANCES_COHORT_START,
)


# The one "x as % of y" rule (0.0 when y is 0), shared with utils.
_safe_div = _safe_pct


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
      - insurance_only: utils.insurance_only_mask (EMI paid, insurance/expense
        arrears above config.INSURANCE_EXP_ARREARS_MIN) -- a cash/WCL
        adjustment clears it, not a credit problem
      - installment_only: installment arrears, no expense arrears
      - both: shortfall on both legs
      - other: none of the above (e.g. a small insurance charge under the
        threshold, or only bounce/penal charges), so the four always add up
        to every delinquent loan
    One row per group_col value (default: branch/Unit), plus the region it
    rolls up to for a report table that can show either grain."""
    required = {"Arrears / EMI", "ARREARS AGAINST INST", "ARREARS AGAINST EXP", group_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    delinquent = df[to_num(df, "Arrears / EMI") > 0].copy()
    if delinquent.empty:
        return pd.DataFrame()

    inst = to_num(delinquent, "ARREARS AGAINST INST", fill=0)
    exp = to_num(delinquent, "ARREARS AGAINST EXP", fill=0)
    delinquent["_insurance_only"] = insurance_only_mask(delinquent)
    delinquent["_installment_only"] = (inst > 0) & (exp <= 0)
    delinquent["_both"] = (inst > 0) & (exp > 0)
    delinquent["_other"] = ~(delinquent["_insurance_only"] | delinquent["_installment_only"] | delinquent["_both"])

    rows = []
    group_cols = [group_col] + (["RegionName"] if "RegionName" in df.columns and group_col != "RegionName" else [])
    # Only the columns the loop reads (copying ~100 per branch was the cost).
    delinquent = delinquent[[*group_cols, *[c for c in delinquent.columns if c == "Loan No" or c.startswith("_")]]]
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
            "Other": int(grp["_other"].sum()),
            "Other %": _safe_div(grp["_other"].sum(), n),
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Insurance-Only %", ascending=False).reset_index(drop=True)


# ── Item 2: Deep arrears, still paying vs not paying (four-way split), by branch ─────────────────────

# The four groups of compute_chronic_shock_split, named for what the data
# shows (not a prediction), so a manager can read them without a glossary.
HARD_NOT_PAYING = "Hard + Not Paying"
HARD_STILL_PAYING = "Hard + Still Paying"
WAS_SILENT = f"Was Silent, Now Under {HARD_BUCKET_ARREARS_EMI_MIN} EMIs"
EARLY_DELINQUENCY = "Early Delinquency"
ARREARS_GROUPS = [HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, EARLY_DELINQUENCY]


def add_unit_delinquency(table: pd.DataFrame, df_curr: pd.DataFrame, df_prev: pd.DataFrame | None,
                         group_col: str = "Unit") -> pd.DataFrame:
    """Put the unit's total Accounts before Delinquent Accounts and its
    Delinquency% (delinquent / all its loans, from the full book, so it
    matches Portfolio Intelligence) after, plus last month's % and the
    change when there's a prev file."""
    if table.empty or group_col not in table.columns or "Delinquent Accounts" not in table.columns:
        return table
    curr = delinquency_by(df_curr, [group_col])
    out = table.copy()
    hits = [curr.get((_unit_key(u),)) for u in out[group_col]]
    out.insert(out.columns.get_loc("Delinquent Accounts"), "Accounts",
               pd.Series([h[0] if h else None for h in hits], index=out.index, dtype="Int64"))
    pct = [_safe_pct(h[1], h[0]) if h else None for h in hits]
    out.insert(out.columns.get_loc("Delinquent Accounts") + 1, "Delinquency%", pd.Series(pct, index=out.index, dtype=float))
    return attach_prev_delinquency(out, df_prev, {group_col: group_col})


def compute_chronic_shock_split(df: pd.DataFrame, group_col: str = "Unit") -> pd.DataFrame:
    """Cross a BEHAVIORAL flag (No Coll 3 Months and >6 EMI -- a zero-payment
    streak that at some point crossed 6 EMIs of arrears) against a SNAPSHOT
    flag (currently Arrears/EMI >= HARD_BUCKET_ARREARS_EMI_MIN, today's
    severity regardless of history). The two can diverge -- a customer can hit
    one without the other -- so treating them as the same signal hides which
    of four genuinely different populations an account belongs to:
      - HARD_NOT_PAYING (chronic + hard):     legal/write-off candidates
      - HARD_STILL_PAYING (hard, not chronic): deep arrears but still paying
                                               something; restructuring candidate
      - WAS_SILENT (chronic, not hard):        part-paid after a long silence,
                                               worth a call before they slip back
      - EARLY_DELINQUENCY (neither):           normal early delinquency
    """
    required = {"Arrears / EMI", group_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    delinquent = df[to_num(df, "Arrears / EMI") > 0].copy()
    if delinquent.empty:
        return pd.DataFrame()

    chronic = is_yes(delinquent, NOT_PAYING_3M_FLAG_COL)
    hard = to_num(delinquent, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN

    delinquent["_chronic_hard"] = chronic & hard
    delinquent["_chronic_not_hard"] = chronic & ~hard
    delinquent["_shock_hard"] = ~chronic & hard
    delinquent["_neither"] = ~chronic & ~hard

    rows = []
    group_cols = [group_col] + (["RegionName"] if "RegionName" in df.columns and group_col != "RegionName" else [])
    # Only the columns the loop reads (copying ~100 per branch was the cost).
    delinquent = delinquent[[*group_cols, *[c for c in delinquent.columns if c == "Loan No" or c.startswith("_")]]]
    for keys, grp in delinquent.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = account_count(grp)
        row = dict(zip(group_cols, keys))
        row["Delinquent Accounts"] = n
        for label, flag in ((HARD_NOT_PAYING, "_chronic_hard"), (HARD_STILL_PAYING, "_shock_hard"),
                            (WAS_SILENT, "_chronic_not_hard"), (EARLY_DELINQUENCY, "_neither")):
            row[label] = int(grp[flag].sum())
            row[f"{label} %"] = _safe_div(grp[flag].sum(), n)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(f"{HARD_NOT_PAYING} %", ascending=False).reset_index(drop=True)


# ── Item 5: Per-region "why" synthesis ───────────────────────────────────────

_DRIVER_LABELS = {
    "insurance_only_share":  "Insurance-driven delinquency",
    "chronic_share":         "Not paying for 3+ months",
    "shock_share":           "Deep arrears, still paying",
    "fleet_share":           "Fleet-operator concentration",
    "recent_vintage_share":  "Recent-advance (sourcing/underwriting) quality",
}


def compute_region_why_table(df_curr: pd.DataFrame, region_scorecard: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """One row per region: NPA%/Collection%/Δ NPA% (from the existing
    compute_region_scorecard) plus a single DOMINANT DRIVER tag, picked as
    whichever candidate driver claims the largest share of that region's
    delinquent book:
      - insurance_only_share:  ARREARS AGAINST INST<=0 AND ARREARS AGAINST EXP>0
      - chronic_share:         No Coll 3 Months and >6 EMI
      - shock_share:           Arrears/EMI>=HARD_BUCKET_ARREARS_EMI_MIN AND NOT chronic
      - fleet_share:           delinquent loan belongs to a fleet operator
                                (utils.fleet_loan_mask: >= FLEET_MIN_LOANS
                                loans across the whole upload, not just this
                                view; blank mobile numbers never count)
      - recent_vintage_share:  Ag_Date within RECENT_ADVANCES_MONTHS of the
                                reporting month (as_of), same window as the
                                Alerts tab and AI Query
    This is the one-page deliverable for a manager: not a pile of charts, a
    single "here's why, specifically" table, per region.
    """
    required = {"Arrears / EMI", "RegionName", "Cust Mob No", "Loan No"}
    if df_curr.empty or not required.issubset(df_curr.columns) or region_scorecard.empty:
        return pd.DataFrame()

    # Fleet status from the customer's whole-upload loan count (a fleet owner's
    # loans can span branches/regions); blank mobiles are never fleet.
    is_fleet_loan = fleet_loan_mask(df_curr)

    recent_cutoff = recent_advances_cutoff(as_of)
    ag_date = pd.to_datetime(df_curr.get("Ag_Date"), errors="coerce") if "Ag_Date" in df_curr.columns else None

    rows = []
    for region, grp in df_curr.groupby("RegionName"):
        delinquent = grp[to_num(grp, "Arrears / EMI") > 0]
        n = account_count(delinquent)
        if n == 0:
            continue

        insurance_only = insurance_only_mask(delinquent).sum()

        chronic = is_yes(delinquent, NOT_PAYING_3M_FLAG_COL)
        hard = to_num(delinquent, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN
        shock = (~chronic & hard).sum()

        fleet = is_fleet_loan.loc[delinquent.index].sum()

        recent_vintage = 0
        if ag_date is not None:
            recent_vintage = (ag_date.loc[delinquent.index] >= recent_cutoff).sum()

        shares = {
            "insurance_only_share": _safe_div(insurance_only, n),
            "chronic_share":        _safe_div(chronic.sum(), n),
            "shock_share":          _safe_div(shock, n),
            "fleet_share":          _safe_div(fleet, n),
            "recent_vintage_share": _safe_div(recent_vintage, n),
        }
        dominant_key = max(shares, key=shares.get)

        sc_row = region_scorecard[region_scorecard["Region"] == region]

        def _sc(col):
            # Delinquency% and last month's figures come from the region
            # scorecard too, so they match Portfolio Intelligence exactly.
            if sc_row.empty or col not in sc_row.columns or pd.isna(sc_row[col].iloc[0]):
                return None
            return float(sc_row[col].iloc[0])

        rows.append({
            "Region": region,
            "Accounts": account_count(grp),
            "NPA%": float(sc_row["NPA%"].iloc[0]) if not sc_row.empty else None,
            "Δ NPA%": float(sc_row["Δ NPA%"].iloc[0]) if not sc_row.empty and sc_row["Δ NPA%"].iloc[0] is not None else None,
            "Collection%": float(sc_row["Collection%"].iloc[0]) if not sc_row.empty else None,
            "Status": sc_row["Status"].iloc[0] if not sc_row.empty else "-",
            "Dominant Driver": _DRIVER_LABELS[dominant_key],
            "Driver Share %": round(shares[dominant_key], 1),
            "Delinquent Accounts": n,
            "Delinquency%": _sc("Delinquency%"),
            "Prev Delinquency%": _sc("Prev Delinquency%"),
            "Δ Delinquency%": _sc("Δ Delinquency%"),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if out["Prev Delinquency%"].isna().all():   # no previous file
        out = out.drop(columns=["Prev Delinquency%", "Δ Delinquency%"])
    out = out[order_unit_columns(out.columns, ["Region", "Status"])]
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


_RECENT_BUCKET_LABEL = {"1-30 DPD": "0-1", "SMA-1": "1-2", "SMA-2": "2-3"}


def _assign_recent_bucket(arrears_emi: pd.Series) -> pd.Series:
    """utils.bucket_from_arrears_emi (the app's one bucket rule), shown with
    the EMI-band labels this report uses (0-1/1-2/2-3)."""
    return _bucket_rule(arrears_emi).replace(_RECENT_BUCKET_LABEL)


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
    # Only the columns read below: filtering ~100 columns per group and bucket was the cost.
    graded = graded[[*group_cols, "_bucket", *[c for c in ("Loan No", "SOH") if c in graded.columns]]]

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
    # Same date conversion as the LCC upload (.xlsb lists store dates as
    # serial numbers like 46236).
    best, _ = parse_date_columns(best)
    return best, None


# ── Recent Advances delinquency status (master LCC x today's missed list) ────
# The manager's question: of the Nov'25-onward loans still RUNNING in the
# monthly LCC, how many are on today's due-date-missed list, and in which
# bucket? Both sides of every % are the SAME loans:
#   - Running Loans (denominator): the LCC's own running Nov'25+ loans.
#   - Delinquent / buckets (numerator): those same loans, matched by Loan No,
#     that appear on the list -- bucketed by the LIST's Arrears/EMI (today's
#     position) with the app's one bucket rule (utils.bucket_from_arrears_emi).
# Region/branch/executive always come from the LCC row, so both sides group
# identically: a list loan outside this LCC (another zone, a branch missing
# from this extract) or hidden by a sidebar filter can never inflate a %.
# Those loans are counted separately in compute_recent_advances_daily_match.

STATUS_BUCKETS = ["1-30 DPD", "SMA-1", "SMA-2", "NPA"]
_OTHER_BUCKET = "Other"  # on the list but Arrears/EMI blank or <= 0

# The master LCC and the daily feed can spell a region differently
# (config.REGION_NAME_ALIASES, shared with the upload loader).
_DAILY_FEED_REGION_CANDIDATES = ["REGIONNAME", "RegionName", "REGION"]
_DAILY_FEED_ARREARS_CANDIDATES = ["ARREARS / EMI", "Arrears / EMI"]


def _canon_region(s: pd.Series) -> pd.Series:
    return canonical_region(s.astype(str).str.strip().str.upper())


def _loan_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _running_cohort(df_curr: pd.DataFrame, cohort_start: str) -> pd.DataFrame:
    """Loans agreed on/after cohort_start that are still running (Loan Status
    RUN), one row per Loan No -- the status table's denominator."""
    cohort = _cohort_df(df_curr, cohort_start)
    if cohort.empty or "Loan No" not in cohort.columns:
        return pd.DataFrame()
    if "Loan Status" in cohort.columns:
        cohort = cohort[cohort["Loan Status"].astype(str).str.strip().str.upper() == "RUN"]
    return cohort.drop_duplicates("Loan No")


def _feed_by_loan(daily_df: pd.DataFrame) -> pd.DataFrame | None:
    """The daily list keyed by normalised Loan No (arrears, region, unit,
    executive), or None when it has no Loan No / Arrears-EMI column."""
    loan_col = _first_col(daily_df, _DAILY_FEED_LOAN_COL_CANDIDATES)
    arrears_col = _first_col(daily_df, _DAILY_FEED_ARREARS_CANDIDATES)
    if loan_col is None or arrears_col is None:
        return None
    region_col = _first_col(daily_df, _DAILY_FEED_REGION_CANDIDATES)

    def _text(col):
        return daily_df[col].astype(str).str.strip().str.upper().to_numpy() if col in daily_df.columns else ""

    out = pd.DataFrame(
        {
            "arrears": pd.to_numeric(daily_df[arrears_col], errors="coerce").to_numpy(),
            "region": _canon_region(daily_df[region_col]).to_numpy() if region_col else "",
            "unit": _text("UNIT"),
            "exec_code": _text("RE CODE"),
            "exec_name": _text("RE NAME"),
        },
        index=_loan_key(daily_df[loan_col]).to_numpy(),
    )
    return out[~out.index.duplicated()]


def compute_recent_advances_daily_match(
    df_curr: pd.DataFrame, daily_df: pd.DataFrame,
    cohort_start: str = RECENT_ADVANCES_COHORT_START, df_all: pd.DataFrame | None = None,
) -> dict:
    """Headline + an explanation for EVERY list loan the status table doesn't
    count. df_curr is the sidebar-filtered LCC; df_all is the whole upload
    (defaults to df_curr), used only to tell "hidden by a filter" apart from
    "not in this LCC at all". Every list loan lands in exactly one of:
      matched            -> counted in the status table
      in_view_not_counted-> in the LCC view but not a running Nov'25+ loan there
                            (e.g. LCC Ag_Date before the cutoff, or not RUN)
      hidden_by_filter   -> in the LCC, but outside the current sidebar filters
      missing_from_lcc   -> not in the LCC, though its region IS covered by it
      other_scope        -> not in the LCC, region not covered (another zone)"""
    cohort = _running_cohort(df_curr, cohort_start)
    feed = _feed_by_loan(daily_df)
    if cohort.empty or feed is None:
        return {}
    df_all = df_curr if df_all is None else df_all

    cohort_keys = _loan_key(cohort["Loan No"])
    view_keys = set(_loan_key(df_curr["Loan No"]))
    all_keys = set(_loan_key(df_all["Loan No"])) if "Loan No" in df_all.columns else view_keys
    lcc_regions = set(_canon_region(df_all["RegionName"])) if "RegionName" in df_all.columns else set()

    keys = feed.index.to_series()
    matched = keys.isin(set(cohort_keys))
    in_view = keys.isin(view_keys)
    in_lcc = keys.isin(all_keys)
    in_view_not_counted = in_view & ~matched
    hidden_by_filter = in_lcc & ~in_view
    missing_from_lcc = ~in_lcc & feed["region"].isin(lcc_regions)
    other_scope = ~in_lcc & ~missing_from_lcc

    # Matched loans whose executive on the list differs from the LCC's (the
    # table stays on the LCC executive so both sides of its % group alike).
    exec_changed = 0
    if matched.any():
        lcc = cohort.set_index(cohort_keys.to_numpy())
        on_list = feed[matched.to_numpy()]
        for lcc_col, feed_col in (("MNT CODE", "exec_code"), ("MNT NAME", "exec_name")):
            if lcc_col in lcc.columns and (on_list[feed_col] != "").all():
                lcc_exec = lcc.loc[on_list.index, lcc_col].astype(str).str.strip().str.upper()
                exec_changed = int((lcc_exec.to_numpy() != on_list[feed_col].to_numpy()).sum())
                break

    n_cohort = len(cohort_keys)
    return {
        "cohort_count": n_cohort,
        "daily_feed_count": len(feed),
        "matched_count": int(matched.sum()),
        "pct_of_cohort_on_daily_list": _safe_div(int(matched.sum()), n_cohort),
        "in_view_not_counted_count": int(in_view_not_counted.sum()),
        "in_view_not_counted_loan_nos": sorted(keys[in_view_not_counted]),
        "hidden_by_filter_count": int(hidden_by_filter.sum()),
        "missing_from_lcc_count": int(missing_from_lcc.sum()),
        "missing_from_lcc_loan_nos": sorted(keys[missing_from_lcc]),
        "missing_from_lcc_by_branch": feed.loc[missing_from_lcc.to_numpy(), "unit"].value_counts().to_dict(),
        "other_scope_count": int(other_scope.sum()),
        "exec_changed_count": exec_changed,
    }


_GRAIN_SPECS: dict[str, dict] = {
    "region":    {"master_cols": ["RegionName"],        "out_cols": ["Region"]},
    "branch":    {"master_cols": ["Unit", "RegionName"], "out_cols": ["Branch", "Region"]},
    # Executive = (name, branch): the same name recurs across branches
    # (registry/semantic_model.py's executive entity uses the same key).
    "executive": {"master_cols": ["MNT NAME", "Unit"],  "out_cols": ["Executive", "Branch"]},
}


def _with_pcts(df: pd.DataFrame, count_cols: list[str]) -> pd.DataFrame:
    total = df["Running Loans"]
    for c in count_cols:
        df[f"{c} %"] = (df[c] / total.where(total > 0) * 100).round(2)
    return df


def compute_recent_advances_status_by_grain(
    df_curr: pd.DataFrame, daily_df: pd.DataFrame, grain: str = "region",
    cohort_start: str = RECENT_ADVANCES_COHORT_START,
) -> pd.DataFrame:
    """One row per region/branch/executive + a Grand Total row: Running Loans
    (running Nov'25+ loans in the LCC), how many of them are on the list
    (Delinquent), and the bucket split -- each count with a % of Running Loans.
    Sorted worst first (highest Delinquent %). An "Other" bucket column only
    appears when some list loan has no positive Arrears/EMI."""
    spec = _GRAIN_SPECS[grain]
    cohort = _running_cohort(df_curr, cohort_start)
    feed = _feed_by_loan(daily_df)
    if cohort.empty or feed is None or not set(spec["master_cols"]).issubset(cohort.columns):
        return pd.DataFrame()

    keys = _loan_key(cohort["Loan No"])
    on_list = keys.isin(set(feed.index))
    bucket = bucket_from_arrears_emi(keys.map(feed["arrears"]))
    bucket = bucket.mask(bucket.isin(["STD", "NA"]), _OTHER_BUCKET).where(on_list, "")
    buckets = STATUS_BUCKETS + ([_OTHER_BUCKET] if (bucket == _OTHER_BUCKET).any() else [])

    out_cols = spec["out_cols"]
    labels = [cohort[c].astype(str).str.strip().rename(o) for c, o in zip(spec["master_cols"], out_cols)]
    table = pd.DataFrame({"Running Loans": cohort.groupby(labels).size(), "Delinquent": on_list.groupby(labels).sum()})
    per_bucket = pd.get_dummies(bucket).reindex(columns=buckets, fill_value=0).groupby(labels).sum()
    table = table.join(per_bucket).astype(int).reset_index()

    count_cols = ["Delinquent"] + buckets
    table = _with_pcts(table, count_cols).sort_values("Delinquent %", ascending=False, na_position="last")

    grand = {c: "" for c in out_cols}
    grand[out_cols[0]] = "Grand Total"
    grand.update({c: int(table[c].sum()) for c in ["Running Loans"] + count_cols})
    table = pd.concat([table, _with_pcts(pd.DataFrame([grand]), count_cols)], ignore_index=True)

    ordered = out_cols + ["Running Loans"] + [x for c in count_cols for x in (c, f"{c} %")]
    return table[ordered]


# ── Standalone, manager-presentable Excel export ─────────────────────────────
# openpyxl (not xlsxwriter -- that's only present in this venv as some other
# package's transitive dependency, not a pinned requirement here) to match
# ui/components.py::_excel_bytes, the app's one other Excel writer, so there's
# a single pinned Excel-writing dependency for the whole project.

# Same blue header / white text as every other download (ui/components.py).
_HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
_HEADER_FONT = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
_TITLE_FONT = Font(bold=True, size=16, color="1F4E78", name="Calibri")
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


# White -> Excel's default red: delinquency has no "safe" level, so the
# lowest value is plain white, never green (same rule as the app's tables).
_COLORSCALE_RED = "F8696B"
_COLORSCALE_WHITE = "FFFFFF"


def _apply_worst_first_colorscale(ws, n_data_rows: int, pct_col_letters: list[str]) -> None:
    """White (lowest) -> red (highest/worst) on the given
    columns, over rows 2..n_data_rows+1 -- i.e. EXCLUDING the Grand Total row,
    which callers append as the last row: an aggregate isn't a peer to rank
    against the entities it's a total of, and including it would also skew
    the min/max the color scale computes."""
    if n_data_rows <= 1:
        return
    last_row = n_data_rows  # data rows are 2..(n_data_rows+1); Grand Total is row n_data_rows+1
    rule = ColorScaleRule(
        start_type="min", start_color=_COLORSCALE_WHITE,
        end_type="max", end_color=_COLORSCALE_RED,
    )
    for col in pct_col_letters:
        ws.conditional_formatting.add(f"{col}2:{col}{last_row}", rule)


def _write_status_sheet(ws, df: pd.DataFrame) -> None:
    """Like _write_df_sheet, plus the worst-first white->red color scale on
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
    contents = ["Region Diagnosis (the \"why\" table)", "Insurance vs. Installment Split (by branch)", "Deep Arrears: Still Paying vs Not Paying (by branch)"]
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
    _write_df_sheet(wb.create_sheet("Paying vs Not Paying"), chronic_shock_df)

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
        ws["A1"] = "Running Recent Advances vs. Today's Due-Date-Missed List"
        ws["A1"].font = _SECTION_FONT
        labels = [
            ("Running loans (LCC)", daily_match.get("cohort_count")),
            ("Loans on the missed-due-date list", daily_match.get("daily_feed_count")),
            ("Counted: running loans on the list", daily_match.get("matched_count")),
            ("% of running loans on the list", daily_match.get("pct_of_cohort_on_daily_list")),
            ("Not counted: in LCC but not a running recent loan there (check Ag_Date / status)", daily_match.get("in_view_not_counted_count")),
            ("Not counted: hidden by the dashboard filters", daily_match.get("hidden_by_filter_count")),
            ("Not counted: region covered by this LCC, but loan missing from it", daily_match.get("missing_from_lcc_count")),
            ("Not counted: region/zone not in this LCC", daily_match.get("other_scope_count")),
            ("Delinquent loans with a different executive on the list (shown under LCC executive)", daily_match.get("exec_changed_count")),
        ]
        for i, (label, val) in enumerate(labels, start=3):
            ws.cell(row=i, column=1, value=label).font = Font(bold=True)
            ws.cell(row=i, column=2, value=val)
        ws.column_dimensions["A"].width = 86

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

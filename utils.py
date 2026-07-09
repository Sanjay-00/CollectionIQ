import datetime
import html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from config import HARD_BUCKET_ARREARS_EMI_MIN, LCC_DATE_MIN_YEAR, LCC_DATE_MAX_YEAR, SEGMENT_NAME_PREFIX_MATCH_CHARS

YELLOW = "#FFC000"

_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
# pandas Timestamp is nanosecond-precision and bounded (~1677 to ~2262); a serial
# number outside that range raises OutOfBoundsDatetime and used to crash the
# ENTIRE upload for every user of that file, not just null out the one bad row.
# One garbage cell (a placeholder sentinel, a fat-fingered huge number) took the
# whole app down. Bound the serial range BEFORE the arithmetic that can overflow,
# not after, so an out-of-range value becomes NaT like any other bad date instead
# of an unrecoverable crash.
# NOT (pd.Timestamp.max - _EXCEL_EPOCH).days -- that subtraction itself overflows,
# since pd.Timedelta's range (~106751 days) is narrower than the gap between the
# 1899 epoch and pd.Timestamp.max (~2262), so computing the bound that way raises
# the exact OutOfBoundsDatetime this code exists to prevent.
_MAX_SERIAL = pd.Timedelta.max.days - 1


def _parse_date_column(col: pd.Series) -> tuple[pd.Series, int, int]:
    """Per-CELL-aware date parsing -- correct regardless of what mix of raw
    types a column contains. Replaces a fragile whole-column "guess the type"
    heuristic that got this wrong in 3 separate, real ways in one session:

    1. pyxlsb hands back numeric Excel serials for MOST cells but raw TEXT
       date strings for any cell the source workbook formatted as text --
       the old heuristic committed to ONE strategy for the whole column,
       silently nulling whichever cells didn't fit it (observed: ~2,300+
       text-formatted Ag_Date cells lost on a real file).
    2. A faster-engine swap under evaluation (calamine) hands back real
       datetime objects for MOST cells but raw numeric serials for any cell
       lacking explicit date-format metadata -- the mirror-image problem
       (observed: ~10,000+ cells would have been lost the same way).
    3. openpyxl hands back an ALREADY-CORRECT datetime64 column for a
       cleanly-formatted .xlsx. pd.to_numeric() on that doesn't fail, it
       silently casts every date to a microsecond-since-epoch integer, which
       the old heuristic misread as "100% Excel serials" and reinterpreted
       as literal out-of-range day-counts. Observed: 100% of a real file's
       Ag_Date column (13,205/13,205 rows) vanished this way.

    Every one of those is the SAME root mistake: deciding one parsing
    strategy for an entire column based on a majority-vote guess, instead of
    checking what each CELL actually is. This function classifies each cell
    by its own real type (already-a-datetime object, numeric, or text) and
    parses it with the matching strategy -- fully vectorized via boolean
    masks, no per-row Python loop, so this isn't a performance regression
    versus the old column-level heuristic.

    Returns (parsed_series, unexpected_failure_count, total_non_blank_count).
    unexpected_failure_count excludes cells that were already blank to begin
    with (a legitimate, expected outcome -- e.g. Last Receipt Date on a loan
    with no payment yet -- not a parsing failure), so a caller can
    distinguish "this column has some genuinely missing data" from
    "something in this column failed to parse and needs attention".
    total_non_blank_count is the denominator for turning that into a rate."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col, 0, int(col.notna().sum())

    raw_blank = col.isna() | col.astype(str).str.strip().str.lower().isin(["", "nan", "none", "nat"])

    is_dt_obj = col.map(lambda v: isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)))
    # bool is a subclass of int, so pd.to_numeric() silently accepts a stray
    # True/False in a date column and converts it into a (wrong) serial date
    # instead of failing -- exclude it so it falls through to the string
    # branch, which correctly reports it as a parse failure.
    is_bool_obj = col.map(lambda v: isinstance(v, bool))
    numeric = pd.to_numeric(col, errors="coerce")
    is_numeric = numeric.notna() & ~is_dt_obj & ~is_bool_obj

    result = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")

    if is_dt_obj.any():
        result.loc[is_dt_obj] = pd.to_datetime(col[is_dt_obj], errors="coerce")

    if is_numeric.any():
        in_range = (numeric > 0) & (numeric <= _MAX_SERIAL)
        serial_mask = is_numeric & in_range
        if serial_mask.any():
            result.loc[serial_mask] = _EXCEL_EPOCH + pd.to_timedelta(numeric[serial_mask], unit="D")
        # numeric but out of range stays NaT -- same crash-prevention bound as before.

    remaining = ~is_dt_obj & ~is_numeric
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(col[remaining], errors="coerce")

    unexpected_failures = int((result.isna() & ~raw_blank).sum())
    return result, unexpected_failures, int((~raw_blank).sum())

# Columns that MUST exist for calculations to work
CRITICAL_COLS = [
    "Loan No", "RegionName", "Unit", "Loan Status", "Ag_Date",
    "Arrears / EMI", "Month Receipt Amount", "Month Collection (Excluding Reserve Collection)",
    "Net Collection Demand Inst+Exp+BC",
    "POS", "LCC%", "Strike", "Closing Arrears",
    "Month Due-Inst", "Month Due-Exp", "Total Cum Collection",
]

# Known column name variations across different LCC extracts
COL_ALIASES = {
    
    "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by":
        "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by R":
        "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "Cum Coll (Inst+Exp+BC)": "Cum Coll (Inst+Exp)",
    "MONTH DUE P":"MONTH DUE PC",
}

# All expected columns (used for reference only  -  missing ones show a warning, not error)
REQUIRED_COLS = [
    "SNo", "Loan No", "CHANNEL", "BU", "StateName", "Zone", "RegionName", "Unit",
    "Ag_Date", "SRC Code", "SRC Name", "MNT CODE", "MNT NAME", "Due Dt", "Tenure",
    "Loan Status", "Loan Amount", "Veh ID", "Cust Name", "Guar Name", "Cust Mob No",
    "Guar Mob No", "Segment", "SegmentName", "Make", "Vehicle Description",
    "Year Of Manufacture", "Arrear Opening", "ARREARS OPEN AGAINST INST",
    "ARREARS OPEN AGAINST EXP", "ARREARS OPEN AGAINST BC", "ARREARS OPEN AGAINST PC",
    "OPENING RESERVE COLLECTION", "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)",
    "MONTH DUE PC", "Month Receipt Amount", "MONTHCOLL INST", "MONTHCOLL EXP",
    "MONTHCOLL BC", "MONTHCOLL PC", "Month Collection (Excluding Reserve Collection)",
    "Closing Arrears", "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "Cum Due-Inst", "Cum Due-Exp", "CUM DUE (BC)", "Cum Due PC", "Cum Coll (Inst+Exp)",
    "Total Cum Collection", "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
    "ARREARS AGAINST BC", "ARREARS AGAINST PC", "CLOSING RESERVE COLLECTION",
    "Arrears against Inst+Exp", "Uncleared Cheque/Amount Not remitted by RE", "LCC%",
    "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC", "POS", "scheme",
    "Non Starter", "Strike", "RCEndors(>90Days)", "RTO / INSURANCE",
    "NET Collection Demand Inst+Exp", "Net Collection Demand Inst+Exp+BC",
    "NET COLLECTION", "NET COLLECTION EXCLUDING RESERVE COLL", "Last Receipt Date",
    "Last Receipt Amount", "ParentLDueDate", "No Coll 3 Months and >6 EMI",
    "NACHStatus", "SaleType", "CoLending_Loans", "CUSTOMER_STATUS", "LGL_FLAG",
    "LGL_DESCRIPTION", "TyreFlag", "FUEL_TYPE",
]

BUCKET_ORDER = ["STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA", "NA"]
# "NA" deliberately has NO entry here -- it means "Arrears / EMI was missing/
# unparseable that period," not a real delinquency state, so it must never be
# treated as comparable to a real bucket. It used to map to -1 (lower than
# every real score), which meant a loan going from "NA" (unknown) to STD (the
# HEALTHIEST real bucket) registered as "rolled forward" (worsened) purely
# because -1 < 0 -- a false deterioration signal with zero basis, confirmed on
# real production data (an executive with 0% NPA/SMA-2 showing 100% Roll Fwd%,
# entirely from 2 loans whose prior bucket was "NA"). Every consumer below maps
# through this dict and treats an unmapped key as NaN, then excludes NaN rows
# from its own "valid comparison" mask (`curr_score.notna() & prev_score.notna()`)
# -- so leaving "NA" out here makes it automatically un-comparable everywhere,
# in one place, rather than patching each consumer's mask individually.
# agents/data_executor.py's independent _BUCKET_SCORE (bucket_worse_than/
# bucket_better_than AI Query filters) already omits "NA" the same way.
BUCKET_SCORE = {"STD": 0, "1-30 DPD": 1, "SMA-1": 2, "SMA-2": 3, "NPA": 4}
BUCKET_COLORS = {
    "STD":      "#16a34a",
    "1-30 DPD": "#FFC000",
    "SMA-1":    "#f97316",
    "SMA-2":    "#ef4444",
    "NPA":      "#991b1b",
    "NA":       "#9ca3af",
}

# Numeric columns carried over from the previous-month file into the current-month
# DataFrame as prev_* columns (matched per Loan No), so the AI can compute
# month-over-month change/reduction queries (e.g. "regions by max SOH reduction",
# "branches with biggest drop in insurance cases").
#
# Keys are the source column names in the prev file; values are the eval-safe
# target names (no spaces or slashes) so the plan engine's derive expressions and
# the column validator work on them directly. This is a curated set that covers the
# vast majority of "change vs last month" questions  -  extend it here in ONE place
# to support a new prev metric; nothing else needs to change.
PREV_CARRYOVER_COLS = {
    "SOH":                  "prev_SOH",
    "POS":                  "prev_POS",
    "Closing Arrears":      "prev_Closing_Arrears",
    "ClosingPC":            "prev_ClosingPC",
    "Arrears / EMI":        "prev_Arrears_EMI",
    "ARREARS AGAINST INST": "prev_Arrears_Inst",
    "ARREARS AGAINST EXP":  "prev_Arrears_Exp",
    # Flow columns (this month's transactional collection/demand, not a stock
    # balance) -- still meaningful to carry forward matched by Loan No for
    # "collection % this month vs previous" comparisons. Without these, the
    # collection_pct ratio metric (registry/ontology.py) silently has no
    # previous-period version under time.compare.
    "Month Collection (Excluding Reserve Collection)": "prev_Month_Collection",
    "Net Collection Demand Inst+Exp+BC":               "prev_Net_Collection_Demand",
    # Not numeric (Y/N flag), but carried forward the same way -- needed for the
    # strike_pct count_ratio metric's previous-period version (registry/ontology.py).
    "Strike":               "prev_Strike",
}


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v = pd.to_numeric(df["Arrears / EMI"], errors="coerce")
    df["curr_bucket"] = np.select(
        [v.isna(), v <= 0, v < 1, v < 2, v < 3],
        ["NA",     "STD",  "1-30 DPD", "SMA-1", "SMA-2"],
        default="NPA",
    )
    df["curr_score"] = df["curr_bucket"].map(BUCKET_SCORE)
    # SOH = Sum of Hire = POS + Closing Arrears = total exposure if customer defaults
    _pos = pd.to_numeric(df["POS"], errors="coerce").fillna(0) if "POS" in df.columns else pd.Series(0.0, index=df.index)
    _arr = pd.to_numeric(df["Closing Arrears"], errors="coerce").fillna(0) if "Closing Arrears" in df.columns else pd.Series(0.0, index=df.index)
    df["SOH"] = _pos + _arr

    # Overdue-first collection waterfall: a payment clears "Arrear Opening" (the
    # amount already overdue coming INTO this month) before any of it counts
    # against this month's own EMI demand -- never the two other way round, and
    # never averaged across loans (a customer's overpayment can't offset a
    # different customer's shortfall). Computed once per loan here, exactly like
    # SOH above, so the dashboard KPI cards, the region/branch/executive
    # breakdown table, the report section, and the AI Query registry metric are
    # all reading the same two columns instead of five independent
    # reimplementations of the same waterfall.
    #
    # "Overdue" clips a negative Arrear Opening (a customer already in credit)
    # to 0 -- there's nothing outstanding to collect, so it must not contribute
    # a negative amount that would silently inflate ANOTHER loan's apparent
    # collection rate once summed into the same branch/region total.
    #
    # "MonthDemandExclPC" deliberately excludes MONTH DUE PC (penal charges) --
    # explicit business call: penal charges are not treated as core EMI demand
    # for this metric, same Inst+Exp+BC scope elsewhere would use, minus BC's
    # own carve-out reasoning not applying here (BC is a real due amount; PC is
    # a penalty, not principal/EMI demand).
    _overdue = pd.to_numeric(df.get("Arrear Opening"), errors="coerce").fillna(0).clip(lower=0) \
        if "Arrear Opening" in df.columns else pd.Series(0.0, index=df.index)
    _demand = (
        pd.to_numeric(df.get("Month Due-Inst"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("Month Due-Exp"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("MONTH DUE (BC)"), errors="coerce").fillna(0)
    ) if "Month Due-Inst" in df.columns else pd.Series(0.0, index=df.index)
    _paid = pd.to_numeric(df.get("Month Collection (Excluding Reserve Collection)"), errors="coerce").fillna(0) \
        if "Month Collection (Excluding Reserve Collection)" in df.columns else pd.Series(0.0, index=df.index)

    df["Overdue"] = _overdue
    df["MonthDemandExclPC"] = _demand
    df["OverdueCollected"] = np.minimum(_paid, _overdue)
    df["DemandCollected"] = np.minimum((_paid - _overdue).clip(lower=0), _demand)

    # Per-loan % versions of the same two figures -- needed so a loan_table
    # ("case wise") AI Query can SELECT these directly as plain columns. The
    # aggregate ratio metrics (registry/ontology.py's overdue_collection_pct /
    # month_demand_collection_pct) only work through group_aggregate, which a
    # loan_table query's plain column-select never invokes -- without a real
    # per-row column, the Logical Planner has nothing to put in display_columns
    # and silently drops the request instead of erroring. Same zero-denominator
    # -> 100% rule as compute_overdue_demand_pct (nothing owed = fully clear).
    df["Overdue Collection %"] = (
        df["OverdueCollected"] / df["Overdue"].replace(0, np.nan) * 100
    ).fillna(100.0).round(2)
    df["Month Demand Collection %"] = (
        df["DemandCollected"] / df["MonthDemandExclPC"].replace(0, np.nan) * 100
    ).fillna(100.0).round(2)
    return df


def compute_overdue_demand_pct(df: pd.DataFrame) -> dict:
    """Overdue Collection % and Month Demand Collection % for whatever slice of
    `df` is passed in (portfolio total, or a region/branch/executive group) --
    the single shared implementation of the waterfall business rule assign_buckets
    computes per loan above (Overdue/MonthDemandExclPC/OverdueCollected/DemandCollected).

    Sums first, divides once (never averages a per-loan %, which would let a
    handful of tiny loans skew a group's real rate).

    Deliberate rule, confirmed with the business: when the group's total Overdue
    (or total MonthDemandExclPC) is zero -- nothing was ever owed on that side --
    the % is 100, not 0 and not "N/A". Nothing outstanding is the same accounting
    outcome as everything outstanding having been collected.
    """
    cols = ("Overdue", "MonthDemandExclPC", "OverdueCollected", "DemandCollected")
    if df.empty or not all(c in df.columns for c in cols):
        return {
            "overdue_pct": 100.0, "demand_pct": 100.0,
            "overdue_total": 0.0, "overdue_collected": 0.0,
            "demand_total": 0.0, "demand_collected": 0.0,
        }

    overdue_total     = float(df["Overdue"].sum())
    overdue_collected = float(df["OverdueCollected"].sum())
    demand_total      = float(df["MonthDemandExclPC"].sum())
    demand_collected  = float(df["DemandCollected"].sum())

    overdue_pct = 100.0 if overdue_total == 0 else round(overdue_collected / overdue_total * 100, 2)
    demand_pct  = 100.0 if demand_total == 0 else round(demand_collected / demand_total * 100, 2)

    return {
        "overdue_pct": overdue_pct, "demand_pct": demand_pct,
        "overdue_total": overdue_total, "overdue_collected": overdue_collected,
        "demand_total": demand_total, "demand_collected": demand_collected,
    }


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded column names to our standard names.

    Pass 1  -  Strip leading/trailing spaces from every column.
    Pass 2  -  Apply COL_ALIASES (exact known truncations / variations).
    Pass 3  -  Case-insensitive exact match (handles capitalisation differences).
    Pass 4  -  Prefix match for truncated long names.
              Targets sorted longest-first so the more-specific column wins
              when a shorter column name is a prefix of a longer one
              (e.g. "Net Collection Demand Inst+Exp+BC" beats "NET Collection
              Demand Inst+Exp" when the file column is ambiguously truncated).
    """
    all_standard = list(dict.fromkeys(CRITICAL_COLS + REQUIRED_COLS))  # critical first, no dupes

    # Pass 1  -  strip spaces
    df.columns = pd.Index([str(c).strip() for c in df.columns])

    # Pass 2  -  known aliases
    df.rename(columns={k: v for k, v in COL_ALIASES.items() if k in df.columns}, inplace=True)

    # Build case-insensitive lookup; longest targets first so more-specific
    # columns win over shorter prefix-collision siblings.
    target_ci: dict[str, str] = {}
    for t in sorted(all_standard, key=len, reverse=True):
        tl = t.lower()
        if tl not in target_ci:
            target_ci[tl] = t

    rename_map: dict[str, str] = {}
    claimed: set[str] = {c for c in df.columns if c in all_standard}

    for col in df.columns:
        if col in all_standard:
            claimed.add(col)
            continue

        col_lower = col.lower()

        # Pass 3  -  case-insensitive exact match
        if col_lower in target_ci and target_ci[col_lower] not in claimed:
            target = target_ci[col_lower]
            rename_map[col] = target
            claimed.add(target)
            continue

        # Pass 4  -  prefix match (file col is truncated version of target)
        # Skip short columns (< 15 chars) to avoid false matches.
        if len(col) < 15:
            continue

        for tl, target in sorted(target_ci.items(), key=lambda x: len(x[0]), reverse=True):
            if target in claimed:
                continue
            if len(col_lower) < len(tl) and tl.startswith(col_lower):
                rename_map[col] = target
                claimed.add(target)
                break

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    return df


def _reorder_to_template(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to the canonical LCC template sequence -- REQUIRED_COLS'
    own order, confirmed column-for-column against a real ZONAG-format extract --
    rather than whatever order THIS particular upload happened to have. Different
    regional files (multi-file uploads) or a re-exported sheet can hand back the
    same ~85 columns in a different sequence; without this, an AI Query result
    that includes every column (e.g. "give me all cases with all columns")
    rendered them in a visibly jumbled, upload-dependent order instead of the
    familiar template layout. Anything not in REQUIRED_COLS (an unexpected
    column, or a later template addition) is kept, just appended after in its
    original relative order -- never dropped. Called right after
    _normalize_columns(), before assign_buckets(), so the derived business
    columns (SOH, curr_bucket, Overdue, ...) land after the reordered raw
    block, not interleaved into it."""
    known_order = [c for c in REQUIRED_COLS if c in df.columns]
    extra_cols  = [c for c in df.columns if c not in REQUIRED_COLS]
    return df[known_order + extra_cols]


@__import__("streamlit").cache_data(show_spinner=False)
def load_and_validate(file) -> tuple[pd.DataFrame, list[str]]:
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
        df = pd.read_excel(file, engine=engine, sheet_name=0)
    except Exception as e:
        return None, [f"Could not read file: {e}"]

    # Normalize column names (strips spaces, fixes capitalisation, maps truncated names)
    df = _normalize_columns(df)
    df = _reorder_to_template(df)

    # If critical columns are missing from sheet 0, try a sheet named "LCC"
    missing_critical = [c for c in CRITICAL_COLS if c not in df.columns]
    if missing_critical:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            xl   = pd.ExcelFile(file, engine=engine)
            lcc  = next((s for s in xl.sheet_names if str(s).strip().upper() == "LCC"), None)
            if lcc:
                if hasattr(file, "seek"):
                    file.seek(0)
                df = pd.read_excel(file, engine=engine, sheet_name=lcc)
                df = _normalize_columns(df)
                df = _reorder_to_template(df)
                missing_critical = [c for c in CRITICAL_COLS if c not in df.columns]
        except Exception:
            pass  # fall through to the error below

    # Hard-fail if critical columns still missing after sheet fallback
    if missing_critical:
        return None, [
            f"Missing critical column(s): {', '.join(missing_critical)}"
        ]

    date_parse_warnings = []
    for date_col in ["Ag_Date", "Last Receipt Date", "ParentLDueDate"]:
        if date_col not in df.columns:
            continue
        df[date_col], unexpected_failures, total_non_blank = _parse_date_column(df[date_col])

        # Business-plausibility bound, applied AFTER per-cell parsing regardless
        # of which strategy produced a given cell: a stray numeric value that
        # isn't actually a date (e.g. a rupee amount that ended up in this
        # column in the source extract) can be numerically small enough to
        # parse into a "valid" but nonsense Timestamp (e.g. year 2170) without
        # ever overflowing pandas' Timestamp range, so the crash-prevention
        # bound inside _parse_date_column doesn't catch it. Reject anything
        # outside a real loan portfolio's plausible date range instead of
        # letting a wrong-but-well-formed date reach business logic (e.g. Last
        # Receipt Date feeds "paid this month" AI Query filters directly).
        implausible = df[date_col].notna() & ~df[date_col].dt.year.between(LCC_DATE_MIN_YEAR, LCC_DATE_MAX_YEAR)
        unexpected_failures += int(implausible.sum())
        df[date_col] = df[date_col].where(~implausible, pd.NaT)

        # Safety net for whatever edge case NEITHER this function nor its 3
        # known predecessors anticipated: surface it as a visible warning (same
        # pattern as missing_optional_cols below) instead of it silently
        # vanishing into the data a 4th time. Threshold on the FRACTION of
        # originally non-blank cells that failed, not a raw count, so this
        # scales correctly from a 400-row sample file to a 60k-row production
        # one, and never fires on a column that's just legitimately sparse
        # (e.g. Last Receipt Date on loans with no payment yet).
        if total_non_blank > 0 and unexpected_failures / total_non_blank > 0.05:
            date_parse_warnings.append(
                f"{date_col}: {unexpected_failures} of {total_non_blank} value(s) could not be "
                f"parsed as valid dates (showing blank instead of a wrong date)."
            )
    df.attrs["date_parse_warnings"] = date_parse_warnings

    # Due Dt is a numeric EMI due day (5, 10, 15, 20)  -  keep as number
    df["Due Dt"] = pd.to_numeric(df["Due Dt"], errors="coerce")

    # Additive cash flows - zero is the correct default when missing
    for col in ["Month Receipt Amount", "Month Collection (Excluding Reserve Collection)", "NET COLLECTION", "Cum Coll (Inst+Exp)", "Total Cum Collection"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # These must NOT be filled - missing means unknown, not zero (fillna(0) distorts ratios)
    for col in [
        "NET Collection Demand Inst+Exp", "Net Collection Demand Inst+Exp+BC",
        "POS", "Arrears / EMI",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["LCC%"] = pd.to_numeric(df["LCC%"], errors="coerce")
    df["Strike"] = df["Strike"].astype(str).str.strip().str.upper()
    if "Unit" in df.columns:
        df["Unit"] = df["Unit"].astype(str).str.strip().str.upper()
    if "MNT NAME" in df.columns:
        # Manually retyped each month, so the same executive shows up as "Sunil Waghmare" one
        # month and "SUNIL WAGHMARE " the next -- normalize once here rather than in every
        # consumer that groups by this column. astype(str).notna() mask keeps real blanks as
        # NaN instead of turning them into the literal string "NAN".
        mask = df["MNT NAME"].notna()
        df.loc[mask, "MNT NAME"] = df.loc[mask, "MNT NAME"].astype(str).str.strip().str.upper()

    # SegmentName/Segment: the source system truncates the same real segment
    # at different lengths across rows (confirmed on real data -- see
    # config.py::SEGMENT_NAME_PREFIX_MATCH_CHARS), which otherwise splits one
    # segment's NPA/SOH numbers across several rows in every segment-wise
    # breakdown instead of one.
    for col in ["SegmentName", "Segment"]:
        if col in df.columns:
            df[col] = normalize_truncated_names(df[col])

    # Mobile numbers: if even one row is blank, Excel/pandas silently upgrades the
    # whole column to float64, so every number renders as "9876543210.0" (or worse,
    # scientific notation) everywhere it's displayed or exported. Strip that artifact.
    for col in ["Cust Mob No", "Guar Mob No"]:
        if col in df.columns:
            df[col] = clean_mobile(df[col])

    df = assign_buckets(df)

    # Drop duplicate Loan Nos  -  keep the first occurrence.
    # Duplicates inflate every count-based metric (NPA count, roll rates, etc.).
    # A real extract shouldn't have dupes at all, so the count is surfaced via
    # df.attrs (survives a plain return, doesn't touch the errs contract that
    # _load_and_concat treats as fatal-per-file) rather than dropped silently.
    dropped_duplicates = 0
    if "Loan No" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["Loan No"])
        dropped_duplicates = before - len(df)
    df.attrs["dropped_duplicate_loans"] = dropped_duplicates

    # Non-critical columns (REQUIRED_COLS minus CRITICAL_COLS) can be missing or
    # fail to normalize-match without ever raising an error -- this module's own
    # docstring says that's "surfaced as a warning," but no such warning existed
    # anywhere in the codebase. Without it, a missing column like CoLending_Loans
    # or LGL_FLAG is invisible: a co-lending query silently reports "no co
    # -lending accounts" instead of "column not found," identical in outward
    # behavior to a real zero-count answer. Surfaced via df.attrs (same
    # survives-a-plain-return pattern as dropped_duplicate_loans above) rather
    # than raised, since a missing optional column is legitimately not fatal.
    df.attrs["missing_optional_cols"] = [
        c for c in REQUIRED_COLS if c not in CRITICAL_COLS and c not in df.columns
    ]

    return df, []


def apply_filters(df: pd.DataFrame, region: str, branch: str, status: str, segment: tuple = ()) -> pd.DataFrame:
    if len(df.columns) == 0:
        return df
    if region != "All" and "RegionName" in df.columns:
        df = df[df["RegionName"] == region]
    if branch != "All" and "Unit" in df.columns:
        df = df[df["Unit"] == branch]
    if status != "All" and "Loan Status" in df.columns:
        df = df[df["Loan Status"] == status]
    if segment:
        _seg_col = next((c for c in ["SegmentName", "Segment"] if c in df.columns), None)
        if _seg_col:
            df = df[df[_seg_col].isin(segment)]
    return df


def _safe_pct(num, den):
    if den == 0:
        return 0.0
    return round(num / den * 100, 2)


def to_num(df: pd.DataFrame, col: str, fill: float | None = None) -> pd.Series:
    """Coerce `df[col]` to numeric, index-aligned to `df` even when `col` is absent
    (so it's always safe to use in boolean masks or arithmetic against other columns).
    Missing values (and a missing column entirely) become `fill`, or stay NaN if
    `fill` is None."""
    if col not in df.columns:
        return pd.Series(fill if fill is not None else float("nan"), index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.fillna(fill) if fill is not None else s


def account_count(df: pd.DataFrame, col: str = "Loan No") -> int:
    """Distinct-loan count, falling back to row count when `col` is absent."""
    return df[col].nunique() if col in df.columns else len(df)


def normalize_truncated_names(series: pd.Series, prefix_chars: int = SEGMENT_NAME_PREFIX_MATCH_CHARS) -> pd.Series:
    """Merge values that are truncated-at-different-lengths variants of the
    same real name (e.g. "Passenger Commerc" / "Passenger Commerci" /
    "Passenger Commercial") under whichever variant is longest -- see
    config.py::SEGMENT_NAME_PREFIX_MATCH_CHARS for the full rationale and the
    real-data evidence behind the 15-character default. Leaves NaN untouched."""
    mask = series.notna()
    if not mask.any():
        return series
    values = series[mask].astype(str).str.strip()
    prefix = values.str[:prefix_chars]
    canonical = values.groupby(prefix).transform(lambda s: max(s, key=len))
    result = series.copy()
    result[mask] = canonical
    return result


def clean_mobile(series: pd.Series) -> pd.Series:
    """Strip the trailing ".0" that pandas adds when a numeric-looking column
    (e.g. a mobile number) has any missing values and gets upgraded to float64.
    Leaves already-clean strings and NaN (-> "") untouched."""
    def _fmt(v):
        if pd.isna(v):
            return ""
        s = str(v).strip()
        return s[:-2] if s.endswith(".0") else s
    return series.apply(_fmt)


def is_yes(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask for a Y/N flag column, index-aligned to `df` (False when `col` is absent).

    Some monthly LCC extracts spell the flag out as "Yes" instead of "Y" (seen in
    Non Starter/Strike columns), so both spellings count as true.
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    normalized = df[col].astype(str).str.strip().str.upper()
    return normalized.isin(["Y", "YES"])


def compute_strike_pct(df: pd.DataFrame) -> float:
    """% of accounts current on their installment obligation (Strike=Y), among accounts
    with a valid Y/N Strike value.

    Single source of truth for the dashboard (compute_metrics) and Portfolio Intelligence
    (analysis/portfolio_intelligence.py::compute_pulse_kpis), which both call this instead
    of reimplementing it. The AI Query path (registry/ontology.py's strike_pct METRIC) is a
    separate, declarative definition consumed by the general compiler
    (compiler/measures.py's count_ratio handler) rather than a direct function call, so it
    can't share this implementation directly - but it must match this definition, and
    tests/test_metric_consistency.py checks the two stay in sync.
    """
    if df.empty or "Strike" not in df.columns:
        return 0.0
    strike_valid = df[df["Strike"].astype(str).str.strip().str.upper().isin(["Y", "N"])]
    if strike_valid.empty:
        return 0.0
    return _safe_pct(is_yes(strike_valid, "Strike").sum(), len(strike_valid))


def compute_hard_bucket_pct(df: pd.DataFrame) -> float:
    """% of accounts >= HARD_BUCKET_ARREARS_EMI_MIN EMIs overdue.

    Single source of truth for the dashboard (compute_metrics) and every Portfolio
    Intelligence table that reports Hard Bucket% (compute_pulse_kpis, compute_region_scorecard,
    compute_branch_quadrant). See compute_strike_pct's docstring re: the AI Query path.
    """
    total = account_count(df)
    if total == 0:
        return 0.0
    return _safe_pct((to_num(df, "Arrears / EMI") >= HARD_BUCKET_ARREARS_EMI_MIN).sum(), total)


def _mom_pct(curr, prev):
    if prev == 0:
        return 0.0
    return round((curr - prev) / abs(prev) * 100, 2)


def compute_metrics(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    _zero = {
        "Month Demand": 0.0, "Total Collection": 0.0, "Collection %": 0.0,
        "Strike %": 0.0, "NPA %": 0.0, "Hard Bucket %": 0.0, "SMA-2 %": 0.0,
        "Count": 0, "SOH": 0.0, "LCC%": 0.0, "CMD %": 0.0,
    }

    def _calc(df):
        if df.empty or "Loan No" not in df.columns:
            return _zero.copy()
        n_accounts = df["Loan No"].nunique()
        demand = df["Net Collection Demand Inst+Exp+BC"].sum(min_count=1)
        demand = 0.0 if pd.isna(demand) else demand
        collection = df["Month Collection (Excluding Reserve Collection)"].sum()
        _soh_col = "SOH" if "SOH" in df.columns else "POS"
        pos = df[_soh_col].sum(min_count=1)
        pos = 0.0 if pd.isna(pos) else pos
        cum_coll_total = df["Total Cum Collection"].sum()
        # "Cum Coll (Inst+Exp)" is optional (REQUIRED_COLS, not CRITICAL_COLS) -
        # a file missing it must not crash the whole Dashboard tab.
        cum_coll_inst_exp = to_num(df, "Cum Coll (Inst+Exp)", fill=0).sum()

        strike_pct = compute_strike_pct(df)

        npa_pct = _safe_pct(
            df[df["curr_bucket"] == "NPA"]["Loan No"].nunique(),
            n_accounts,
        )
        sma2_pct = _safe_pct(
            df[df["curr_bucket"] == "SMA-2"]["Loan No"].nunique() if "curr_bucket" in df.columns else 0,
            n_accounts,
        )
        hard_pct = compute_hard_bucket_pct(df)
        _cum_due = sum(
            pd.to_numeric(df[c], errors="coerce").fillna(0).sum()
            for c in ("Cum Due-Inst", "Cum Due-Exp")if c in df.columns
        )
        # LCC% = Cum Coll (Inst+Exp) / (Cum Due-Inst + Cum Due-Exp) -- matches the
        # documented business definition (agents/domain_expert.py) and the AI Query
        # path's lcc_pct METRIC (registry/ontology.py). "Total Cum Collection" is a
        # broader figure (includes BC/other components) and is the wrong numerator.
        lcc_avg = _safe_pct(to_num(df, "Cum Coll (Inst+Exp)", fill=0).sum(), _cum_due)
        lcc_avg = round(lcc_avg, 2) if not pd.isna(lcc_avg) else 0.0
        # Capped at 100 -- matches the documented definition (agents/domain_expert.py:
        # "LCC% = ... Capped at 100, max value is 100") and the AI Query path's
        # lcc_pct METRIC (registry/ontology.py, cap=100). A customer who has paid
        # ahead of cumulative dues can otherwise push this over 100%.
        lcc_avg = min(lcc_avg, 100.0)
        cmd_pct = _safe_pct(cum_coll_total, cum_coll_inst_exp)

        return {
            "Month Demand": demand,
            "Total Collection": collection,
            "Collection %": _safe_pct(collection, demand),
            "Strike %": strike_pct,
            "NPA %": npa_pct,
            "Hard Bucket %": hard_pct,
            "SMA-2 %": sma2_pct,
            "Count": n_accounts,
            "SOH": pos,
            "LCC%": lcc_avg,
            "CMD %": cmd_pct,
        }

    curr = _calc(df_curr)
    prev = _calc(df_prev)
    result = {}
    for k in curr:
        result[k] = (curr[k], _mom_pct(curr[k], prev[k]))
    return result


def fmt_value(val, kind: str) -> str:
    if kind == "money":
        if abs(val) >= 1_00_00_000:
            return f"₹{val / 1_00_00_000:.2f}Cr"
        if abs(val) >= 1_00_000:
            return f"₹{val / 1_00_000:.2f}L"
        return f"₹{val:,.0f}"
    if kind == "pct":
        return f"{val:.2f}"
    if kind == "count":
        if abs(val) >= 1_00_00_000:
            return f"{val / 1_00_00_000:.2f}Cr"
        if abs(val) >= 1_00_000:
            return f"{val / 1_00_000:.2f}L"
        return f"{int(val):,}"
    return str(val)


def build_status_bar_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "curr_bucket" not in df.columns or "Loan No" not in df.columns:
        return go.Figure()
    counts = (
        df.groupby("curr_bucket")["Loan No"]
        .nunique()
        .reindex(BUCKET_ORDER, fill_value=0)
    )
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=YELLOW,
        text=counts.values.tolist(),
        textposition="outside",
        textfont=dict(size=12, color="#000000"),
    ))
    fig.update_layout(
        title=dict(text="Loan Count by DPD Bucket", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def build_branch_bar_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "Unit" not in df.columns:
        return go.Figure()

    grp = df.groupby("Unit").agg(
        demand=("Net Collection Demand Inst+Exp+BC", "sum"),
        collection=("Month Collection (Excluding Reserve Collection)", "sum"),
    )
    grp["coll_pct"] = grp.apply(
        lambda r: _safe_pct(r["collection"], r["demand"]), axis=1
    )
    grp = grp.sort_values("coll_pct", ascending=True).tail(12)

    fig = go.Figure(go.Bar(
        x=grp["coll_pct"].tolist(),
        y=grp.index.tolist(),
        orientation="h",
        marker_color=YELLOW,
        text=[f"{v:.0f}" for v in grp["coll_pct"]],
        textposition="outside",
        textfont=dict(size=11, color="#000000"),
    ))
    max_val = grp["coll_pct"].max() if len(grp) > 0 else 100
    fig.update_layout(
        title=dict(text="Collection % by Branch", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            range=[0, max_val * 1.12],
            title=dict(text="Collection %", font=dict(color="#000000")),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def build_closing_pc_chart(df: pd.DataFrame) -> go.Figure:
    """Arrears exposure by DPD bucket  -  SUM(Closing Arrears) per bucket.
    Shows how much money is stuck at each risk level."""
    if df.empty or "Closing Arrears" not in df.columns or "curr_bucket" not in df.columns:
        return go.Figure()

    df = df.copy()
    df["Closing Arrears"] = pd.to_numeric(df["Closing Arrears"], errors="coerce").fillna(0)

    exposure = (
        df.groupby("curr_bucket")["Closing Arrears"]
        .sum()
        .reindex(BUCKET_ORDER, fill_value=0)
    )

    # Format labels in Indian units
    def _fmt(v):
        if v >= 1_00_00_000:
            return f"₹{v/1_00_00_000:.1f}Cr"
        if v >= 1_00_000:
            return f"₹{v/1_00_000:.1f}L"
        return f"₹{v:,.0f}"

    labels = [_fmt(v) for v in exposure.values]

    colors = [BUCKET_COLORS.get(b, YELLOW) for b in exposure.index]

    fig = go.Figure(go.Bar(
        x=exposure.index.tolist(),
        y=exposure.values.tolist(),
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont=dict(size=12, color="#000000"),
    ))
    fig.update_layout(
        title=dict(text="Closing Arrears by DPD Bucket", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def build_html_export(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    metrics: dict,
    fig_status: go.Figure,
    fig_branch: go.Figure,
    fig_trend: go.Figure,
    filters: dict,
    *,
    curr_month: str = "",
    alerts: list = None,
    scorecard_df=None,
    roll_rate_meta: dict = None,
) -> str:
    import datetime as _dt

    def _esc(val) -> str:
        """Escape any string that came from the uploaded data before it enters an
        f-string HTML fragment. Manually-entered LCC fields (executive/branch/
        customer names) can contain &, <, > - unescaped, those break the
        surrounding table markup, and this HTML is also offered as a raw
        download, not just rendered inline."""
        if val is None:
            return ""
        return html.escape(str(val), quote=False)

    def _fig_html(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    KPI_TOP = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %"]
    KPI_BOT = ["Count", "SOH", "LCC%", "CMD %"]
    KINDS = {
        "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
        "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct",
        "Count": "count", "SOH": "money", "LCC%": "pct", "CMD %": "pct",
    }
    INVERSE = {"NPA %", "Hard Bucket %"}

    def _card(label, value, mom, unit="", inverse=False):
        arrow = "&#9650;" if mom >= 0 else "&#9660;"
        color = ("#CC0000" if mom >= 0 else "#00A651") if inverse else ("#00A651" if mom >= 0 else "#CC0000")
        return (
            f'<div style="background:#fff;border:1px solid #e5e7eb;border-bottom:3px solid {YELLOW};'
            f'border-radius:10px;padding:16px 14px;min-width:120px;flex:1;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px;">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:#111827;line-height:1;letter-spacing:-0.5px;">{value}{unit}</div>'
            f'<div style="font-size:11px;margin-top:8px;color:#9ca3af;">MoM <span style="color:{color};font-weight:700;">{arrow} {abs(mom):.2f}%</span></div>'
            f'</div>'
        )

    def _cards(keys, style=""):
        cards = "".join(
            _card(k, fmt_value(metrics[k][0], KINDS[k]), metrics[k][1],
                  unit="%" if KINDS[k] == "pct" else "",
                  inverse=k in INVERSE)
            for k in keys
        )
        return f'<div style="display:flex;gap:10px;flex-wrap:wrap;{style}">{cards}</div>'

    def _section(title):
        return (
            f'<div style="display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700;'
            f'color:#111827;margin:28px 0 14px 0;">'
            f'<span style="width:4px;height:18px;background:{YELLOW};border-radius:2px;display:inline-block;flex-shrink:0;"></span>'
            f'{title}</div>'
        )

    filter_info = " | ".join(f"<b>{_esc(k)}:</b> {_esc(v)}" for k, v in filters.items() if v != "All") or "All data"
    generated_at = _dt.datetime.now().strftime("%d %b %Y, %I:%M %p")
    month_label = _esc(curr_month or filters.get("Year Month", ""))

    # ── Smart Alerts section ──────────────────────────────────────────────────
    alerts_html = ""
    if alerts:
        SEVERITY_COLOR = {"critical": "#dc2626", "high": "#f97316", "medium": "#d97706"}
        cards_html = ""
        for alert in alerts:
            is_clear = alert["count"] == 0
            color = "#16a34a" if is_clear else SEVERITY_COLOR.get(alert["severity"], "#d97706")
            pos_fmt = fmt_value(alert["pos"], "money") if not is_clear else " - "
            arr_fmt = fmt_value(alert["closing_arrears"], "money") if not is_clear else " - "
            cards_html += (
                f'<div style="background:#fff;border:1px solid #e5e7eb;border-left:4px solid {color};'
                f'border-radius:10px;padding:14px 16px;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                f'<span style="font-size:18px;">{"✅" if is_clear else alert["icon"]}</span>'
                f'<span style="font-size:13px;font-weight:700;color:{color};">{alert["title"]}</span></div>'
                f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">{alert["subtitle"]}</div>'
                f'<div style="display:flex;gap:14px;">'
                f'<div><div style="font-size:9px;color:#9ca3af;font-weight:600;text-transform:uppercase;">Accounts</div>'
                f'<div style="font-size:22px;font-weight:800;color:{color};">{alert["count"]}</div></div>'
                f'<div><div style="font-size:9px;color:#9ca3af;font-weight:600;text-transform:uppercase;">SOH</div>'
                f'<div style="font-size:16px;font-weight:700;color:#111;">{pos_fmt}</div></div>'
                f'<div><div style="font-size:9px;color:#9ca3af;font-weight:600;text-transform:uppercase;">Arrears</div>'
                f'<div style="font-size:16px;font-weight:700;color:{color};">{arr_fmt}</div></div>'
                f'</div>'
                f'<div style="font-size:11px;color:#6b7280;font-style:italic;margin-top:8px;border-top:1px solid #f3f4f6;padding-top:6px;">'
                f'{"✓ All clear" if is_clear else alert["action"]}</div>'
                f'</div>'
            )
        alerts_html = (
            _section("Smart Alerts") +
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">{cards_html}</div>'
        )

    # ── Executive Scorecard section ───────────────────────────────────────────
    scorecard_html = ""
    if scorecard_df is not None and len(scorecard_df) > 0:
        top5 = scorecard_df[scorecard_df["Tier"] == "top"].head(5)
        bot5 = scorecard_df[scorecard_df["Tier"] == "bottom"].sort_values("Collection %").head(5)

        def _exec_rows(sub):
            rows = ""
            for _, row in sub.iterrows():
                coll      = row["Collection %"]
                npa_count = int(row.get("NPA", 0))
                sma2      = int(row.get("SMA-2", 0))
                coll_color = "#16a34a" if coll > 100 else "#d97706" if coll >= 90 else "#dc2626"
                npa_color  = "#dc2626" if npa_count > 0 else "#16a34a"
                sma2_color = "#d97706" if sma2 > 0 else "#16a34a"
                rows += (
                    f'<tr style="border-bottom:1px solid #f3f4f6;">'
                    f'<td style="padding:7px 10px;font-size:12px;font-weight:600;color:#111;">{_esc(row["Executive (Branch)"])}</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row["Accounts"]}</td>'
                    f'<td style="padding:7px 10px;font-size:13px;font-weight:800;color:{coll_color};text-align:center;">{coll}%</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row["Strike Rate %"]}%</td>'
                    f'<td style="padding:7px 10px;font-size:12px;font-weight:700;color:{sma2_color};text-align:center;">{row.get("SMA-2 %", 0)}%</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row.get("NPA %", 0)}%</td>'
                    f'<td style="padding:7px 10px;font-size:12px;font-weight:700;color:{npa_color};text-align:center;">{npa_count}</td>'
                    f'<td style="padding:7px 10px;font-size:12px;font-weight:700;color:{sma2_color};text-align:center;">{sma2}</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row.get("Total POS (L)", 0)}</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row.get("Total SOH (L)", 0)}</td>'
                    f'</tr>'
                )
            return rows

        def _exec_table(title, sub, color):
            headers = ["Executive", "Accounts", "Coll %", "Strike %", "SMA-2 %", "NPA %", "NPA", "SMA-2", "POS (L)", "SOH (L)"]
            th = "".join(
                f'<th style="padding:7px 10px;text-align:{"left" if i==0 else "center"};font-size:10px;'
                f'color:#6b7280;font-weight:700;text-transform:uppercase;">{h}</th>'
                for i, h in enumerate(headers)
            )
            return (
                f'<div style="flex:1;">'
                f'<div style="font-size:11px;font-weight:700;color:{color};text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px;padding:5px 10px;background:{color}18;border-radius:6px;">{title}</div>'
                f'<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr style="background:#f9fafb;">{th}</tr></thead>'
                f'<tbody>{_exec_rows(sub)}</tbody>'
                f'</table></div>'
            )

        scorecard_html = (
            _section(f"Executive Performance ({len(scorecard_df)} Executives)") +
            f'<div style="display:flex;gap:16px;background:#fff;border:1px solid #e5e7eb;'
            f'border-radius:10px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            + _exec_table("Top Performers", top5, "#16a34a")
            + f'<div style="width:1px;background:#e5e7eb;flex-shrink:0;"></div>'
            + _exec_table("Need Attention", bot5, "#dc2626")
            + f'</div>'
        )

    # ── Roll-Rate section ─────────────────────────────────────────────────────
    roll_html = ""
    if roll_rate_meta and roll_rate_meta.get("matched_count", 0) > 0:
        rr = roll_rate_meta
        rr_items = [
            ("Roll-Forward Rate", f'{rr["roll_forward_rate"]:.1f}%',  "#dc2626", "Accounts that worsened"),
            ("Roll-Backward Rate", f'{rr["roll_backward_rate"]:.1f}%',  "#16a34a", "Returned to STD"),
            ("NPA Formation",     f'{rr["npa_formation_rate"]:.1f}%', "#991b1b", "New NPA this month"),
            ("Matched Accounts",  f'{rr["matched_count"]:,}',         "#111827", "In both months"),
        ]
        rr_cards = "".join(
            f'<div style="background:#fff;border:1px solid #e5e7eb;border-top:3px solid {c};'
            f'border-radius:10px;padding:14px 16px;flex:1;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;">{lbl}</div>'
            f'<div style="font-size:24px;font-weight:800;color:{c};">{val}</div>'
            f'<div style="font-size:11px;color:#9ca3af;margin-top:4px;">{tip}</div>'
            f'</div>'
            for lbl, val, c, tip in rr_items
        )
        roll_html = (
            _section("Bucket Migration / Roll-Rate") +
            f'<div style="display:flex;gap:10px;">{rr_cards}</div>'
        )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CollectionIQ  -  Regional Collection Dashboard {month_label}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Inter',Arial,sans-serif;background:#f2f2f2;color:#111827;}}
  .banner{{height:4px;background:linear-gradient(90deg,{YELLOW},#FFD740,{YELLOW});background-size:200%;animation:shimmer 3s linear infinite;}}
  @keyframes shimmer{{0%{{background-position:-200% 0}}100%{{background-position:200% 0}}}}
  .header{{background:#fff;border-bottom:3px solid {YELLOW};padding:14px 32px;display:flex;align-items:center;gap:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}}
  .logo-box{{background:#111;border-radius:8px;padding:8px 14px;}}
  .logo-main{{font-size:16px;font-weight:900;color:{YELLOW};letter-spacing:2px;line-height:1;}}
  .logo-sub{{font-size:9px;color:#888;letter-spacing:1px;margin-top:2px;}}
  .header-title{{font-size:17px;font-weight:700;color:#111;}}
  .header-sub{{font-size:11px;color:#999;margin-top:2px;}}
  .month-badge{{margin-left:auto;background:{YELLOW};color:#000;font-size:11px;font-weight:800;padding:5px 14px;border-radius:20px;letter-spacing:1px;white-space:nowrap;}}
  .content{{padding:20px 32px 40px;max-width:1400px;margin:0 auto;}}
  .meta-bar{{display:flex;align-items:center;justify-content:space-between;background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 16px;margin-bottom:20px;font-size:12px;color:#6b7280;}}
  .chart-card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:6px;box-shadow:0 2px 6px rgba(0,0,0,0.06);}}
  .footer{{text-align:center;padding:20px;font-size:11px;color:#9ca3af;border-top:1px solid #e5e7eb;margin-top:8px;}}
</style>
</head>
<body>
<div class="banner"></div>
<div class="header">
  <div class="logo-box">
    <div class="logo-main">COLLECTION</div>
    <div class="logo-sub">IQ</div>
  </div>
  <div>
    <div class="header-title">Regional Collection Dashboard</div>
    <div class="header-sub">Credit &amp; Collection Risk Monitoring &nbsp;&middot;&nbsp; CollectionIQ</div>
  </div>
  {f'<div class="month-badge">{month_label}</div>' if month_label else ''}
</div>

<div class="content">
  <div class="meta-bar">
    <span>&#128269; <b>Filters:</b> &nbsp;{filter_info}</span>
    <span>Generated: {generated_at}</span>
  </div>

  {_section("Key Performance Indicators")}
  {_cards(KPI_TOP, "margin-bottom:10px;")}
  {_cards(KPI_BOT)}

  {_section("Portfolio Analysis")}
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
    <div class="chart-card">{_fig_html(fig_status)}</div>
    <div class="chart-card">{_fig_html(fig_branch)}</div>
    <div class="chart-card">{_fig_html(fig_trend)}</div>
  </div>

  {alerts_html}
  {scorecard_html}
  {roll_html}
</div>

<div class="footer">
  Generated by <b>CollectionIQ</b> &nbsp;&middot;&nbsp; Regional Collection Dashboard &nbsp;&middot;&nbsp; {generated_at}
</div>
</body>
</html>"""
    return page_html
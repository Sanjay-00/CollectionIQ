import datetime
import html
import re
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from config import (
    HARD_BUCKET_ARREARS_EMI_MIN, LCC_DATE_MIN_YEAR, LCC_DATE_MAX_YEAR, SEGMENT_NAME_PREFIX_MATCH_CHARS,
    RECENT_ADVANCES_MONTHS, REGION_NAME_ALIASES, INSURANCE_EXP_ARREARS_MIN, NOT_PAYING_3M_FLAG_COL,
    KNOWN_DATE_COLUMNS, NOT_DATE_COLUMNS, DATE_COLUMN_MIN_PARSED_SHARE,
)
from dateutil.relativedelta import relativedelta

YELLOW = "#FFC000"

# registry/ontology.py writes this in a concept's condition value instead of a
# fixed date ("Ag_Date >= <start of the recent-advances window>").
CUTOFF_PLACEHOLDER = "__CUTOFF_1Y__"


def recent_advances_cutoff(as_of=None, months: int = RECENT_ADVANCES_MONTHS) -> pd.Timestamp:
    """Start of the "recent advances" window: `months` before the file's
    reporting month (as_of), or before today only when no reporting month is
    known. Every surface -- Alerts, AI Query, Investigator, Root Cause -- must
    anchor here, or re-analysing an older file gives each tab a different
    window for the same rule. Date-level (no time of day), so a loan
    disbursed exactly on the boundary date is always included."""
    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.date.today())
    return ref.normalize() - relativedelta(months=months)


def count_agreed_after_month(df: pd.DataFrame, reporting_month) -> int:
    """Loans whose Ag_Date falls AFTER the reporting month -- impossible in a
    closing extract, and in practice a day/month swap at the source (a real Aug
    extract had 98, each a valid past date once day and month were swapped)."""
    if "Ag_Date" not in df.columns or reporting_month is None:
        return 0
    month_end = pd.Timestamp(reporting_month).to_period("M").end_time
    return int((pd.to_datetime(df["Ag_Date"], errors="coerce") > month_end).sum())


def resolve_dynamic_values(obj, as_of=None):
    """Return a copy of a condition list / plan with every CUTOFF_PLACEHOLDER
    replaced by recent_advances_cutoff(as_of). Walks nested dicts/lists, so a
    whole compiled step-plan can be resolved in one pass."""
    if isinstance(obj, dict):
        return {k: resolve_dynamic_values(v, as_of) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_dynamic_values(v, as_of) for v in obj]
    if isinstance(obj, str) and obj == CUTOFF_PLACEHOLDER:
        return recent_advances_cutoff(as_of)
    return obj

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

    is_dt_obj = col.map(lambda v: isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date))) #various date formats
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
        result.loc[remaining] = _parse_text_dates(col[remaining])

    unexpected_failures = int((result.isna() & ~raw_blank).sum())
    return result, unexpected_failures, int((~raw_blank).sum())


# Text-date formats seen in LCC extracts, tried BEFORE pandas' format-less
# parse. That fallback parses cell-by-cell via dateutil (slow on a large
# file, and the source of a "Could not infer format" warning per column).
# Unambiguous formats first; numeric d/m/y vs m/d/y order is chosen per column.
_UNAMBIGUOUS_DATE_FORMATS = (
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d",
    "%d-%b-%Y", "%d-%b-%y", "%d %b %Y", "%d-%B-%Y",
)
_DAY_FIRST_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y")
_MONTH_FIRST_FORMATS = ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y")
_NUMERIC_DATE_RE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}")


def _column_is_day_first(text: pd.Series) -> bool:
    """Day-first only when the column itself proves it: some cell's FIRST
    number is > 12 (can't be a month) and no cell's SECOND number is. A
    column with no deciding cell (every value like 03/04/2024) stays
    month-first -- the same order pandas' own format inference chose before
    this parser existed, so no existing result silently changes."""
    parts = text.str.extract(_NUMERIC_DATE_RE)
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    return bool((first > 12).any() and not (second > 12).any())


def _parse_text_dates(col: pd.Series) -> pd.Series:
    """Vectorized text-date parse: each known format is applied to whatever
    is still unparsed; only cells matching none of them fall through to the
    old per-cell parse (warning suppressed -- unparseable cells are already
    counted and surfaced by _parse_date_column's failure count)."""
    text = col.astype(str).str.strip()
    day_first = _column_is_day_first(text)
    formats = _UNAMBIGUOUS_DATE_FORMATS + (_DAY_FIRST_FORMATS if day_first else _MONTH_FIRST_FORMATS)

    result = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    todo = pd.Series(True, index=col.index)
    for fmt in formats:
        if not todo.any():
            break
        parsed = pd.to_datetime(text[todo], format=fmt, errors="coerce")
        hit = parsed.index[parsed.notna()]
        if len(hit):
            result.loc[hit] = parsed.loc[hit]
            todo.loc[hit] = False

    if todo.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format")
            result.loc[todo] = pd.to_datetime(text[todo], errors="coerce", dayfirst=day_first)
    return result

_DATE_NAME_RE = re.compile(r"(?:^|[^A-Z])DATE(?:[^A-Z]|$)|(?:^|[\s_])DT$", re.IGNORECASE)


def is_date_column_name(name) -> bool:
    """A column whose name says it holds dates: the word "date" ("Last
    Received Date", "NPA_DATE", "AG DATE"; not "UPDATED_BY"), a camel-case
    "...Date" ("NpaDate"), or a trailing "DT" ("NPA DT"), minus
    config.NOT_DATE_COLUMNS ("Due Dt")."""
    n = str(name).strip()
    if n.upper() in {c.upper() for c in NOT_DATE_COLUMNS}:
        return False
    return bool(_DATE_NAME_RE.search(n) or re.search(r"[a-z]Date", n))   # also "NpaDate", "LastRecDate"


def parse_date_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert every date column to a real date, whatever the file stored
    (Excel serial, text, or date): config.KNOWN_DATE_COLUMNS always, plus any
    column is_date_column_name finds. Dates outside LCC_DATE_MIN_YEAR..
    LCC_DATE_MAX_YEAR become blank (e.g. a rupee amount in a date column).
    Returns (df, warnings) -- one warning per column with >5% unreadable
    values, or per name-matched column left unconverted because most of its
    values aren't dates."""
    warnings_out: list[str] = []
    known = {c.upper() for c in KNOWN_DATE_COLUMNS}
    for col in [c for c in df.columns if str(c).upper() in known or is_date_column_name(c)]:
        parsed, failures, non_blank = _parse_date_column(df[col])
        implausible = parsed.notna() & ~parsed.dt.year.between(LCC_DATE_MIN_YEAR, LCC_DATE_MAX_YEAR)
        failures += int(implausible.sum())
        parsed = parsed.where(~implausible, pd.NaT)
        if str(col).upper() not in known and non_blank and (non_blank - failures) / non_blank < DATE_COLUMN_MIN_PARSED_SHARE:
            warnings_out.append(f"{col}: looks like a date column by name, but most values aren't dates, so it was left as-is.")
            continue
        df[col] = parsed
        if non_blank and failures / non_blank > 0.05:
            warnings_out.append(
                f"{col}: {failures} of {non_blank} value(s) could not be "
                f"parsed as valid dates (showing blank instead of a wrong date)."
            )
    return df, warnings_out


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
    # ZONAG Aug-2026 extract naming (confirmed same meaning by the business).
    "EMI Accrued": "VehEMI Accrued",
    "Cum Due (PC)": "Cum Due PC",
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


def bucket_from_arrears_emi(arrears_emi: pd.Series) -> pd.Series:
    """The app's one DPD-bucket rule: Arrears/EMI -> STD/1-30 DPD/SMA-1/SMA-2/NPA
    (NA when missing). Every bucket anywhere in the app must come from here."""
    v = pd.to_numeric(arrears_emi, errors="coerce")
    return pd.Series(
        np.select(
            [v.isna(), v <= 0, v < 1, v < 2, v < 3],
            ["NA",     "STD",  "1-30 DPD", "SMA-1", "SMA-2"],
            default="NPA",
        ),
        index=arrears_emi.index,
    )


def _region_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


_REGION_BY_KEY = {
    k: v for alias, v in REGION_NAME_ALIASES.items()
    for k in (_region_key(pd.Series([alias]))[0], _region_key(pd.Series([v]))[0])
}


def canonical_region(s: pd.Series) -> pd.Series:
    """Map known spellings of one region (config.REGION_NAME_ALIASES) to its
    single name; every other value is returned unchanged."""
    return _region_key(s).map(_REGION_BY_KEY).fillna(s)


def repair_upload(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fix two known source-extract faults, and describe each fix so the app
    can show it (never a silent change):
      - rows where CoLending_Loans and CUSTOMER_STATUS traded places (ALIVE/DEAD
        under co-lending AND Y/N under customer status -- only a swap produces
        that pair, so swapping back is not a guess);
      - alternate spellings of the same region."""
    notes: list[str] = []
    if {"CoLending_Loans", "CUSTOMER_STATUS"}.issubset(df.columns):
        co = df["CoLending_Loans"].astype(str).str.strip().str.upper()
        cs = df["CUSTOMER_STATUS"].astype(str).str.strip().str.upper()
        swapped = co.isin(["ALIVE", "DEAD"]) & cs.isin(["Y", "N", "YES", "NO"])
        if swapped.any():
            df = df.copy()
            df.loc[swapped, ["CoLending_Loans", "CUSTOMER_STATUS"]] = (
                df.loc[swapped, ["CUSTOMER_STATUS", "CoLending_Loans"]].to_numpy()
            )
            n_colending = int(cs[swapped].isin(["Y", "YES"]).sum())
            notes.append(
                f"Fixed {int(swapped.sum()):,} row(s) where CoLending_Loans and CUSTOMER_STATUS were swapped "
                f"in the source file ({n_colending:,} of them are co-lending loans)."
            )
    if "RegionName" in df.columns:
        fixed = canonical_region(df["RegionName"])
        changed = fixed.astype(str) != df["RegionName"].astype(str)
        if changed.any():
            renames = df.loc[changed, "RegionName"].astype(str).str.strip().value_counts()
            df = df.assign(RegionName=fixed)
            notes.append("Merged region spellings: " + ", ".join(
                f"{old} → {canonical_region(pd.Series([old]))[0]} ({n:,} rows)" for old, n in renames.items()
            ) + ".")
    return df, notes


CUSTOMER_LOAN_COUNT = "Customer Loan Count"
CUSTOMER_SOH = "Customer SOH"


def add_customer_loan_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add CUSTOMER_LOAN_COUNT: how many distinct loans this loan's customer
    (Cust Mob No) holds in the WHOLE upload. Computed once at load time, before
    any sidebar filter, so "fleet operator" is a fact about the customer, not
    about the current view -- filtering to one branch must not demote a
    customer whose other loans sit in another branch. Blank mobile numbers get
    no count (NA): unrelated customers without a mobile on file would otherwise
    merge into one phantom fleet operator. CUSTOMER_SOH is the same customer's
    total SOH across the whole upload (the "exposure" of a large customer)."""
    if "Cust Mob No" not in df.columns or "Loan No" not in df.columns:
        return df
    attrs = dict(df.attrs)
    mob = df["Cust Mob No"].astype(str).str.strip()
    has_mobile = (mob != "") & (mob.str.lower() != "nan") & df["Cust Mob No"].notna()
    counts = df[has_mobile].groupby(mob[has_mobile])["Loan No"].nunique()
    df = df.assign(**{CUSTOMER_LOAN_COUNT: mob.map(counts).where(has_mobile).astype("Int64")})
    if "SOH" in df.columns:
        soh = to_num(df, "SOH", fill=0).groupby(mob).sum()
        df[CUSTOMER_SOH] = mob.map(soh).where(has_mobile)
    df.attrs = attrs
    return df


def fleet_loan_mask(df: pd.DataFrame) -> pd.Series:
    """True for loans whose customer is a fleet operator (>= FLEET_MIN_LOANS
    loans). Uses CUSTOMER_LOAN_COUNT (whole-upload count) when present; frames
    without it (hand-built test data) are counted within themselves."""
    from config import FLEET_MIN_LOANS
    if CUSTOMER_LOAN_COUNT not in df.columns:
        df = add_customer_loan_count(df)
    if CUSTOMER_LOAN_COUNT not in df.columns:
        return pd.Series(False, index=df.index)
    return (df[CUSTOMER_LOAN_COUNT] >= FLEET_MIN_LOANS).fillna(False).astype(bool)


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["curr_bucket"] = bucket_from_arrears_emi(df["Arrears / EMI"]).to_numpy()
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

    # Pass 2  -  known aliases (never onto a name the file already has -- that
    # would leave two columns with the same name)
    df.rename(columns={k: v for k, v in COL_ALIASES.items() if k in df.columns and v not in df.columns}, inplace=True)

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


# max_entries=16: each entry is a FULL parsed raw frame (the heaviest single
# object this app caches). 16 covers a multi-file regional upload (curr + prev)
# with room to spare; older uploads' frames get LRU-evicted instead of living
# in the server process forever.
@__import__("streamlit").cache_data(show_spinner=False, max_entries=16)
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

    # Every date column (the known three plus any column named like a date,
    # e.g. "Last Received Date" / "NPA_DATE"): Excel serials, text and real
    # dates all become one date type. A wrong-but-well-formed date (a rupee
    # amount that parses as year 2170) is blanked by the plausible-year bound,
    # since e.g. Last Receipt Date feeds "paid this month" AI Query filters.
    # More than 5% unreadable values in a column -> a visible warning, so a
    # new kind of bad value never silently vanishes into the data.
    df, date_parse_warnings = parse_date_columns(df)
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

    df, data_fixes = repair_upload(df)
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
    df = add_customer_loan_count(df)

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
    df.attrs["data_fixes"] = data_fixes

    return df, []


def apply_filters(df: pd.DataFrame, region: str, branch: str, status: str, segment: tuple = (), date_from=None) -> pd.DataFrame:
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
    if date_from and "Ag_Date" in df.columns:
        ag_date = pd.to_datetime(df["Ag_Date"], errors="coerce")
        df = df[ag_date >= pd.Timestamp(date_from)]
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


def expand_to_all_columns(curated: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Swaps a curated, column-limited result for every raw column
    `source_df` actually has, for the SAME rows `curated` already selected,
    in the SAME order -- matched by Loan No (present on every loan-level
    result in this codebase). A no-op (returns `curated` unchanged) if
    either frame lacks Loan No, or `curated` is empty -- callers use this
    only once they already know `curated` is loan-level.

    A real, confirmed gap this closes: several loan-level results (the
    Investigator's own top_accounts/repossession_list/good_customers, and
    the AI Query pipeline's equivalent fast-path VIEWS -- top_delinquent_
    accounts, repossession_eligible, good_customers, all backed by the
    SAME analysis/ functions) always returned a fixed curated column
    subset with no way to ask for the raw ~90+ columns instead, even
    though a sibling step/path (non_paying_customers, customer_loan_book,
    worst_loans_by_metric, priority_accounts on the Investigator side;
    show_all_columns on the general AI Query compiler path) already had an
    "all columns" escape hatch. Shared here, once, so "give me every
    column" behaves identically and gets fixed in one place, not
    separately in each pipeline that can hit this shape of question.

    Column layout: every raw source column (in source order), then any
    curated-only COMPUTED column appended at the end (e.g. good_customers'
    "Tenure Completed %", priority "Why") -- expansion adds columns, never
    drops the derived ones that explain why a row is on the list. For a
    column present in both, the curated value wins, so the download matches
    what's on screen (curated tables carry cleaned numbers/dates).
    """
    if curated.empty or "Loan No" not in curated.columns or "Loan No" not in source_df.columns:
        return curated
    curated = curated.loc[:, ~curated.columns.duplicated()].reset_index(drop=True)
    source = source_df.loc[:, ~source_df.columns.duplicated()].drop_duplicates(subset=["Loan No"])
    # Left merge (not .loc) keeps curated's row order and tolerates a Loan No
    # the source doesn't have instead of raising KeyError.
    full = curated[["Loan No"]].merge(source, on="Loan No", how="left")
    for col in curated.columns:
        if col != "Loan No":
            full[col] = curated[col].values
    return full


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


# ── The one definition of every loan-level condition and unit metric ─────────
# Every region/branch/executive/segment table totals these same per-loan
# flags (unit_metrics), so "NPA", "delinquent", "insurance-only" etc. mean the
# same thing in every tab, and a threshold changes in config.py only.

def as_of_or_today(as_of=None) -> pd.Timestamp:
    """The report's own reporting date, or today when none is given."""
    return pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()


def segment_column(df: pd.DataFrame) -> str | None:
    """The vehicle-segment column this file uses ("SegmentName" or "Segment")."""
    return next((c for c in ("SegmentName", "Segment") if c in df.columns), None)


def insurance_only_mask(df: pd.DataFrame) -> pd.Series:
    """Delinquent only because of the insurance/expense charge: EMI fully paid
    (installment arrears <= 0), insurance/expense arrears above
    config.INSURANCE_EXP_ARREARS_MIN, and Arrears/EMI above 0. A blank
    installment/expense arrears means none recorded (0)."""
    return ((to_num(df, "ARREARS AGAINST INST", fill=0) <= 0)
            & (to_num(df, "ARREARS AGAINST EXP", fill=0) > INSURANCE_EXP_ARREARS_MIN)
            & (to_num(df, "Arrears / EMI") > 0))


def loan_flags(df: pd.DataFrame) -> pd.DataFrame:
    """One row per loan, one column per condition/amount (index-aligned)."""
    ae = to_num(df, "Arrears / EMI")
    bucket = df["curr_bucket"] if "curr_bucket" in df.columns else bucket_from_arrears_emi(ae)
    soh = to_num(df, "SOH", fill=0)
    npa = bucket == "NPA"
    strike = df["Strike"].astype(str).str.strip().str.upper() if "Strike" in df.columns else pd.Series("", index=df.index)
    f = pd.DataFrame({
        "delinquent": ae > 0,
        "sma1": bucket == "SMA-1",
        "sma2": bucket == "SMA-2",
        "npa": npa,
        "hard": ae >= HARD_BUCKET_ARREARS_EMI_MIN,
        "not_paying_3m": is_yes(df, NOT_PAYING_3M_FLAG_COL),
        "insurance_only": insurance_only_mask(df),
        "soh": soh,
        "npa_soh": soh.where(npa, 0.0),
        "pos": to_num(df, "POS", fill=0),
        "demand": to_num(df, "Net Collection Demand Inst+Exp+BC", fill=0),
        "collected": to_num(df, "Month Collection (Excluding Reserve Collection)", fill=0),
        "strike_valid": strike.isin(["Y", "N", "YES", "NO"]),
        "strike_yes": strike.isin(["Y", "YES"]),
    }, index=df.index)
    if "prev_bucket" in df.columns and "curr_bucket" in df.columns:
        curr_sc, prev_sc = df["curr_bucket"].map(BUCKET_SCORE), df["prev_bucket"].map(BUCKET_SCORE)
        valid = curr_sc.notna() & prev_sc.notna()
        f["roll_valid"] = valid
        f["roll_fwd"] = valid & (curr_sc > prev_sc)
        f["roll_bwd"] = valid & (curr_sc < prev_sc)
        f["rescued"] = f["roll_bwd"] & df["prev_bucket"].isin(["SMA-1", "SMA-2", "NPA"])
    return f


def _round_each(s: pd.Series, decimals: int) -> pd.Series:
    """Python's round() per value (NaN kept), like _safe_pct: pandas' .round()
    can differ by 0.01 on exact ties such as 5.975."""
    return s.map(lambda v: v if pd.isna(v) else round(float(v), decimals)).astype(float)


def _pct_col(num: pd.Series, den: pd.Series, decimals: int = 2) -> pd.Series:
    """x as % of y per row, 0.0 when y is 0."""
    return _round_each(num / den.where(den != 0) * 100, decimals).fillna(0.0)


def unit_metrics(df: pd.DataFrame, by: list[str], min_accounts: int = 0,
                 flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every standard metric per unit, from ONE groupby over loan_flags.

    by: grouping columns (e.g. ["RegionName"], ["MNT NAME", "Unit"]); [] gives
    one portfolio row. A blank label is left out (as every table always did).
    Units with fewer than min_accounts are dropped. Columns:
    Accounts, Delinquent, Delinquency%, SMA-1, SMA-2, SMA-2%, NPA, NPA%,
    NPA% (SOH), NPA SOH (Cr), Hard Bucket, Hard Bucket%, Not Paying 3M+, Not Paying 3M+%,
    Insurance-Only, Collection%, Strike%, SOH (Cr), POS, Demand, Collected,
    SOH and, when prev_bucket is present, Roll Fwd%/Roll Bwd% (NaN when no
    loan has both months' bucket), Rescued, Slipped.

    flags: loan_flags(df), when the caller totals the same frame several ways
    (pass it once instead of recomputing every per-loan check each time)."""
    if df.empty:
        return pd.DataFrame()
    f = (loan_flags(df) if flags is None else flags.loc[df.index]).copy()
    keys = [df[c] for c in by] if by else [pd.Series(0, index=df.index)]
    f["_id"] = df["Loan No"] if "Loan No" in df.columns else pd.Series(range(len(df)), index=df.index)
    g = f.groupby(keys, sort=True)
    s = g.sum(numeric_only=True)
    n = g["_id"].nunique()
    out = pd.DataFrame(index=s.index)
    out["Accounts"] = n.astype(int)
    for name, flag in (("Delinquent", "delinquent"), ("SMA-1", "sma1"), ("SMA-2", "sma2"), ("NPA", "npa"),
                       ("Hard Bucket", "hard"), ("Not Paying 3M+", "not_paying_3m"),
                       ("Insurance-Only", "insurance_only")):
        out[name] = s[flag].astype(int)
    out["Delinquency%"] = _pct_col(out["Delinquent"], out["Accounts"])
    out["SMA-2%"] = _pct_col(out["SMA-2"], out["Accounts"])
    out["NPA%"] = _pct_col(out["NPA"], out["Accounts"])
    out["NPA% (SOH)"] = _pct_col(s["npa_soh"], s["soh"])
    out["Hard Bucket%"] = _pct_col(out["Hard Bucket"], out["Accounts"])
    out["Not Paying 3M+%"] = _pct_col(out["Not Paying 3M+"], out["Accounts"])
    out["Collection%"] = _pct_col(s["collected"], s["demand"])
    out["Strike%"] = _pct_col(s["strike_yes"], s["strike_valid"])
    out["SOH (Cr)"] = _round_each(s["soh"] / 1e7, 2)
    out["NPA SOH (Cr)"] = _round_each(s["npa_soh"] / 1e7, 2)
    out["POS"], out["Demand"], out["Collected"], out["SOH"] = s["pos"], s["demand"], s["collected"], s["soh"]
    if "roll_valid" in s.columns:
        valid = s["roll_valid"].where(s["roll_valid"] > 0)
        out["Roll Fwd%"] = _round_each(s["roll_fwd"] / valid * 100, 1)
        out["Roll Bwd%"] = _round_each(s["roll_bwd"] / valid * 100, 1)
        out["Rescued"] = s["rescued"].astype(int)
        out["Slipped"] = s["roll_fwd"].astype(int)
        out["_roll_valid"] = s["roll_valid"].astype(int)
    out = out[out["Accounts"] >= min_accounts]
    out.index.names = by or ["_all"]
    return out.reset_index() if by else out.reset_index(drop=True)


def first_value(df: pd.DataFrame, by: list[str], col: str) -> pd.Series:
    """The first `col` value per group (e.g. a branch's Region), aligned to
    unit_metrics' index order."""
    if col not in df.columns:
        return pd.Series(dtype=object)
    return df.groupby(by, sort=True)[col].first()


def portfolio_metrics(df: pd.DataFrame) -> pd.Series | None:
    """unit_metrics for the whole frame as one row (None when empty)."""
    m = unit_metrics(df, [])
    return None if m.empty else m.iloc[0]


def compute_strike_pct(df: pd.DataFrame) -> float:
    """% of accounts current on their installment obligation (Strike=Y), among
    accounts with a valid Y/N Strike value ("N"/"NO" count against the rate).

    A view of utils.unit_metrics (the one definition every table uses). The
    AI Query path (registry/ontology.py's strike_pct METRIC) is declarative
    and can't call this, so tests/test_metric_consistency.py checks the two
    agree.
    """
    m = portfolio_metrics(df)
    return 0.0 if m is None else float(m["Strike%"])


def compute_hard_bucket_pct(df: pd.DataFrame) -> float:
    """% of accounts >= HARD_BUCKET_ARREARS_EMI_MIN EMIs overdue (view of unit_metrics)."""
    m = portfolio_metrics(df)
    return 0.0 if m is None else float(m["Hard Bucket%"])


def compute_delinquency_count(df: pd.DataFrame) -> int:
    """Accounts with any arrears: Arrears/EMI > 0 (view of unit_metrics). Matches
    registry/ontology.py's CONCEPTS["delinquent"] row filter."""
    if "Arrears / EMI" not in df.columns:
        return 0
    m = portfolio_metrics(df)
    return 0 if m is None else int(m["Delinquent"])


def compute_delinquency_pct(df: pd.DataFrame) -> float:
    """% of accounts with any arrears (view of unit_metrics). See
    compute_strike_pct re: the AI Query path."""
    if "Arrears / EMI" not in df.columns:
        return 0.0
    m = portfolio_metrics(df)
    return 0.0 if m is None else float(m["Delinquency%"])


def _unit_key(v) -> str:
    """Join key for a manually typed label ("Pune " and "PUNE" are one unit)."""
    return str(v).strip().upper()


def delinquency_by(df: pd.DataFrame, cols: list[str]) -> dict[tuple, tuple[int, int]]:
    """{normalized label tuple: (accounts, delinquent)} for each unit in `df`,
    from unit_metrics on case/space-normalized labels."""
    if df is None or df.empty or not set(cols) <= set(df.columns) or "Arrears / EMI" not in df.columns:
        return {}
    keys = [f"_k{i}" for i in range(len(cols))]
    m = unit_metrics(df.assign(**{k: df[c].map(_unit_key) for k, c in zip(keys, cols)}), keys)
    return {tuple(r[k] for k in keys): (int(r["Accounts"]), int(r["Delinquent"])) for _, r in m.iterrows()}


def match_prev(prev: dict, key: tuple, fuzzy_pos: int | None = None):
    """Last month's entry for a normalized label tuple: exact match first;
    then, if fuzzy_pos is given (a name the source system truncates at
    different lengths each month), the ONE entry whose name at that position
    is a prefix of this one (or vice versa) with every other label equal.
    None when there's no match or the prefix match is ambiguous."""
    if key in prev:
        return prev[key]
    if fuzzy_pos is None:
        return None
    hits = [pk for pk in prev
            if all(pk[i] == key[i] for i in range(len(key)) if i != fuzzy_pos)
            and (pk[fuzzy_pos].startswith(key[fuzzy_pos]) or key[fuzzy_pos].startswith(pk[fuzzy_pos]))]
    return prev[hits[0]] if len(hits) == 1 else None


def attach_prev_delinquency(
    table: pd.DataFrame, df_prev: pd.DataFrame, on: dict[str, str],
    pct_col: str = "Delinquency%", fuzzy: str | None = None, decimals: int = 2,
) -> pd.DataFrame:
    """Add last month's delinquency % and the change (in percentage points)
    right after `pct_col` in a per-region/branch/executive table.

    on: {table column: previous-file column}, e.g. {"Branch": "Unit"}. Last
    month is grouped by the previous file's OWN labels, so an executive's
    previous figure is the book they held then. fuzzy: a table column whose
    spelling gets truncated between months (MNT NAME); when there's no exact
    match, a single prefix match within the same other labels is used.
    No columns are added when there is no previous file."""
    prev = delinquency_by(df_prev, list(on.values()))
    if table.empty or not prev or pct_col not in table.columns:
        return table
    pos = list(on).index(fuzzy) if fuzzy in on else None

    prev_pct, delta = [], []
    for _, row in table.iterrows():
        hit = match_prev(prev, tuple(_unit_key(row[c]) for c in on), pos)
        p = round(_safe_pct(hit[1], hit[0]), decimals) if hit else None
        c = row[pct_col]
        prev_pct.append(p)
        delta.append(round(c - p, 2) if p is not None and c is not None and not pd.isna(c) else None)

    out = table.copy()
    at = out.columns.get_loc(pct_col) + 1
    out.insert(at, f"Prev {pct_col}", pd.Series(prev_pct, index=out.index, dtype=float))
    out.insert(at + 1, f"Δ {pct_col}", pd.Series(delta, index=out.index, dtype=float))
    return out


# Reading order for every region/branch/executive table: size of the book,
# how many are behind, the rate, last month's rate and the change, then the
# deeper buckets. Both spellings ("Delinquency%" / "Delinquency %") covered.
UNIT_TABLE_LEAD = [
    "Accounts", "Delinquent", "Delinquent Accounts",
    "Delinquency%", "Delinquency %", "Prev Delinquency%", "Prev Delinquency %",
    "Δ Delinquency%", "Δ Delinquency %",
    "SMA-2", "SMA-2%", "SMA-2 %", "Δ SMA-2%", "NPA", "NPA%", "NPA %", "NPA% (SOH)", "NPA % (SOH)", "Δ NPA%",
]


def order_unit_columns(cols, identity) -> list:
    """`cols` reordered as: identity columns, then UNIT_TABLE_LEAD, then the
    rest in their original order. Missing names are skipped."""
    cols = list(cols)
    head = [c for c in identity if c in cols] + [c for c in UNIT_TABLE_LEAD if c in cols and c not in identity]
    return head + [c for c in cols if c not in head]


def _mom_pct(curr, prev):
    if prev == 0:
        return None  # no prior-period base to compare against, not "no change"
    return round((curr - prev) / abs(prev) * 100, 2)


def compute_metrics(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    _zero = {
        "Month Demand": 0.0, "Total Collection": 0.0, "Collection %": 0.0,
        "Strike %": 0.0, "NPA %": 0.0, "NPA % (SOH)": 0.0, "Hard Bucket %": 0.0, "SMA-2 %": 0.0,
        "Delinquency %": 0.0,
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

        # The shared per-loan definitions (unit_metrics), portfolio-wide.
        m = unit_metrics(df, []).iloc[0]
        strike_pct = float(m["Strike%"])
        npa_pct = float(m["NPA%"])
        npa_soh_pct = float(m["NPA% (SOH)"])
        sma2_pct = float(m["SMA-2%"])
        hard_pct = float(m["Hard Bucket%"])
        delinq_pct = float(m["Delinquency%"])
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
        # CMD% = Total Cum Collection / Cum Coll (Inst+Exp) -- deliberately a
        # collection-vs-collection ratio, NOT collection-vs-demand despite the
        # name. It measures what fraction of the broader Total Cum Collection
        # figure (includes BC/other components -- see the LCC% note above)
        # came in through the Inst+Exp channel specifically. Confirmed
        # intentional; only the label is confusing, not the formula.
        cmd_pct = _safe_pct(cum_coll_total, cum_coll_inst_exp)

        return {
            "Month Demand": demand,
            "Total Collection": collection,
            "Collection %": _safe_pct(collection, demand),
            "Strike %": strike_pct,
            "NPA %": npa_pct,
            "NPA % (SOH)": npa_soh_pct,
            "Hard Bucket %": hard_pct,
            "Delinquency %": delinq_pct,
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
        if mom is None:
            mom_html = '<span style="color:#9ca3af;font-weight:700;">no prev data</span>'
        else:
            arrow = "&#9650;" if mom >= 0 else "&#9660;"
            color = ("#CC0000" if mom >= 0 else "#00A651") if inverse else ("#00A651" if mom >= 0 else "#CC0000")
            mom_html = f'<span style="color:{color};font-weight:700;">{arrow} {abs(mom):.2f}%</span>'
        return (
            f'<div style="background:#fff;border:1px solid #e5e7eb;border-bottom:3px solid {YELLOW};'
            f'border-radius:10px;padding:16px 14px;min-width:120px;flex:1;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px;">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:#111827;line-height:1;letter-spacing:-0.5px;">{value}{unit}</div>'
            f'<div style="font-size:11px;margin-top:8px;color:#9ca3af;">MoM {mom_html}</div>'
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
<title>CollectionIQ: Regional Collection Dashboard {month_label}</title>
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
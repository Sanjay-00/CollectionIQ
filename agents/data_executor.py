import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from config import RECENT_ADVANCES_MONTHS

# Bucket severity scores - higher = worse
_BUCKET_SCORE = {"STD": 0, "1-30 DPD": 1, "SMA-1": 2, "SMA-2": 3, "NPA": 4}

# Generic column-vs-column numeric comparison operators (e.g. "Month Receipt
# Amount col_lte Net Collection Demand Inst+Exp+BC" -- short_collection). The
# "col_" prefix makes the intent unambiguous: "value" holds a COLUMN NAME to
# compare against, never a literal, so there's no ambiguity with the plain
# ==/!=/>/>=/</<= ops (which always compare against a literal). Distinct from
# bucket_worse_than/bucket_better_than, which are bucket-SEVERITY-aware (map
# through _BUCKET_SCORE first) -- these are plain numeric column comparisons.
_COL_COMPARE_OPS = {
    "col_lt":  lambda a, b: a < b,
    "col_lte": lambda a, b: a <= b,
    "col_gt":  lambda a, b: a > b,
    "col_gte": lambda a, b: a >= b,
    "col_eq":  lambda a, b: a == b,
    "col_ne":  lambda a, b: a != b,
}


def _apply_condition(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    col = cond["column"]
    op = cond["op"]
    val = cond["value"]

    if col not in df.columns:
        return df

    # Cross-column bucket comparison operators
    if op in ("bucket_worse_than", "bucket_better_than"):
        ref_col = str(val)
        if ref_col not in df.columns:
            return df
        curr_score = df[col].map(_BUCKET_SCORE)
        prev_score = df[ref_col].map(_BUCKET_SCORE)
        # Exclude rows where either bucket is unknown/NaN (new accounts, missing prev)
        valid = curr_score.notna() & prev_score.notna()
        if op == "bucket_worse_than":
            return df[valid & (curr_score > prev_score)]
        else:
            return df[valid & (curr_score < prev_score)]

    # Generic column-vs-column numeric comparison
    if op in _COL_COMPARE_OPS:
        ref_col = str(val)
        if ref_col not in df.columns:
            return df
        a = pd.to_numeric(df[col], errors="coerce")
        b = pd.to_numeric(df[ref_col], errors="coerce")
        valid = a.notna() & b.notna()
        return df[valid & _COL_COMPARE_OPS[op](a, b)]

    series = df[col]

    # Date-aware comparison
    if pd.api.types.is_datetime64_any_dtype(series):
        # Ops like "in"/"contains" carry a list or free-text value that pd.Timestamp()
        # can't parse - fail loudly with a clear message rather than a cryptic pandas
        # internal TypeError (still a loud failure either way, never a silent no-op).
        if op not in ("==", "!=", ">", ">=", "<", "<="):
            raise ValueError(f"op '{op}' is not valid for date column '{col}'")
        val = pd.Timestamp(val)
        ops = {
            "==": series == val,
            "!=": series != val,
            ">":  series > val,
            ">=": series >= val,
            "<":  series < val,
            "<=": series <= val,
        }
        mask = ops.get(op)
        # An op not in this dict (e.g. "contains", meant for a string column) must
        # be a loud failure, not a silent full-table return -- ops.get(op) is None
        # for an unsupported op, and `if mask is not None else df` used to fall
        # through to returning df UNFILTERED, which looks like "everything
        # matched" instead of "this filter doesn't apply here". A confidently
        # wrong answer is worse than an error the repair loop can act on.
        if mask is None:
            raise ValueError(f"op '{op}' is not valid for date column '{col}'")
        return df[mask]

    # Numeric comparison
    if pd.api.types.is_numeric_dtype(series):
        try:
            val_num = float(val) if not isinstance(val, list) else val
        except (TypeError, ValueError):
            return df
        ops = {
            "==": series == val_num,
            "!=": series != val_num,
            ">":  series > val_num,
            ">=": series >= val_num,
            "<":  series < val_num,
            "<=": series <= val_num,
            "in": series.isin([float(v) for v in val]) if isinstance(val, list) else series == val_num,
        }
        mask = ops.get(op)
        # Same "loud failure, not silent unfiltered return" guard as the date branch above.
        if mask is None:
            raise ValueError(f"op '{op}' is not valid for numeric column '{col}'")
        return df[mask]

    # String / categorical comparison
    str_series = series.astype(str).str.strip().str.upper()
    if op == "==":
        return df[str_series == str(val).upper()]
    if op == "!=":
        return df[str_series != str(val).upper()]
    if op == "in":
        vals_upper = [str(v).upper() for v in (val if isinstance(val, list) else [val])]
        return df[str_series.isin(vals_upper)]
    if op == "contains":
        return df[series.astype(str).str.contains(str(val), case=False, na=False)]

    return df


def _build_mask(df: pd.DataFrame, conditions) -> pd.Series:
    """Boolean mask of rows matching ALL conditions (AND), reusing the type-aware
    _apply_condition logic. Shared by the plan engine and by aggregation-mode
    'where' counts so a multi-condition case can be counted in either engine."""
    matched = df
    for cond in conditions or []:
        matched = _apply_condition(matched, cond)
    return df.index.isin(matched.index)


QUERY_DISPLAY_COLS = [
    "Loan No", "Zone", "RegionName", "Unit", "Ag_Date", "MNT NAME",
    "Due Dt", "Tenure", "Loan Status", "Loan Amount", "Veh ID", "Cust Name",
    "Guar Name", "Cust Mob No", "Guar Mob No", "Vehicle Description",
    "prev_bucket", "curr_bucket",
    "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)", "MONTH DUE PC",
    "Net Collection Demand Inst+Exp+BC", "Month Receipt Amount", "Closing Arrears", "Arrears against Inst+Exp",
    "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
    "LCC%", "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC",
    "POS", "SOH", "Non Starter", "Strike", "Last Receipt Date", "Last Receipt Amount",
    "ParentLDueDate", "No Coll 3 Months and >6 EMI", "NACHStatus",
    "TyreFlag", "FUEL_TYPE",
]


def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to date-only for clean display."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date
    return df


def compute_result_kpis(df_full: pd.DataFrame, filtered: pd.DataFrame) -> dict:
    """Compute query-specific KPIs for the matching loans.

    Uses df_full re-filtered by Loan No so KPI columns (SOH, Demand, Collection,
    Arrears/EMI) are always present  -  even when filtered has been column-narrowed
    by a display-column select step. Falls back to filtered directly when Loan No
    is unavailable (e.g. aggregation results).
    """
    if (
        df_full is not None
        and not df_full.empty
        and "Loan No" in filtered.columns
        and "Loan No" in df_full.columns
    ):
        loan_nos = set(filtered["Loan No"])
        work = df_full[df_full["Loan No"].isin(loan_nos)].copy()
    else:
        work = filtered.copy()
    work = work.drop(columns=["Priority", "_rank"], errors="ignore")

    n = work["Loan No"].nunique() if "Loan No" in work.columns else len(work)

    def _sum(col):
        if col not in work.columns:
            return 0
        return pd.to_numeric(work[col], errors="coerce").sum()

    def _mean(col):
        if col not in work.columns:
            return 0
        vals = pd.to_numeric(work[col], errors="coerce").dropna()
        return round(vals.mean(), 2) if len(vals) > 0 else 0

    pos        = _sum("SOH")
    demand     = _sum("Net Collection Demand Inst+Exp+BC")
    collection = _sum("Month Collection (Excluding Reserve Collection)")
    coll_pct   = round(collection / demand * 100, 2) if demand > 0 else 0
    avg_arrears = _mean("Arrears / EMI")

    return {
        "Count": n,
        "Total POS": pos,
        "Avg Arrears/EMI": avg_arrears,
        "Total Demand": demand,
        "Total Collection": collection,
        "Collection %": coll_pct,
    }


def execute_priority_mode(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Run all priority rules in ranked order, combine results with a Priority column."""
    from agents.domain_expert import PRIORITY_RULES

    cutoff_1y = pd.Timestamp(date.today() - relativedelta(months=RECENT_ADVANCES_MONTHS))

    all_rows = []
    seen_loans = set()

    # Dedup below relies on processing highest-priority (lowest rank) rules
    # first -- "each loan appears only under its highest priority rule" only
    # holds if this loop visits rank order. PRIORITY_RULES is already authored
    # in rank order, but sort explicitly so that invariant can't silently break
    # if the list is ever reordered without updating ranks to match.
    for rule in sorted(PRIORITY_RULES, key=lambda r: r["rank"]):
        conditions = rule["conditions"]
        subset = df.copy()

        for cond in conditions:
            col = cond["column"]
            op  = cond["op"]
            val = cond["value"]

            if col not in subset.columns:
                subset = subset.iloc[0:0]
                break

            # Resolve dynamic cutoff placeholder
            if val == "__CUTOFF_1Y__":
                val = cutoff_1y

            subset = _apply_condition(subset, {**cond, "value": val})

        if len(subset) == 0:
            continue

        # Deduplicate  -  each loan appears only under its highest priority rule
        loan_col = "Loan No" if "Loan No" in subset.columns else subset.columns[0]
        subset = subset[~subset[loan_col].isin(seen_loans)]
        seen_loans.update(subset[loan_col].tolist())

        subset = subset.copy()
        subset.insert(0, "Priority", f"P{rule['rank']}: {rule['label']}")
        subset.insert(1, "_rank", rule["rank"])
        all_rows.append(subset)

    if not all_rows:
        return pd.DataFrame(), "No priority cases found in the current portfolio."

    result = pd.concat(all_rows, ignore_index=True)

    # Select display columns (no Why column)
    display_cols = ["Priority", "Loan No", "Cust Name", "Cust Mob No",
                    "RegionName", "Unit", "MNT NAME", "Ag_Date",
                    "curr_bucket", "Arrears / EMI", "POS",
                    "Net Collection Demand Inst+Exp+BC", "Closing Arrears"]
    display_cols = [c for c in display_cols if c in result.columns]
    out = result[display_cols] if display_cols else result
    return _format_dates(out), ""


def distribute_priority_accounts(df: pd.DataFrame, total_n: int) -> pd.DataFrame:
    """Distribute total_n accounts proportionally across priority groups."""
    if len(df) == 0:
        return df

    groups = []
    for priority, group in df.groupby("Priority", sort=False):
        groups.append((priority, group))

    n_active = len(groups)
    if n_active == 0:
        return df.iloc[0:0]

    base = total_n // n_active
    remainder = total_n % n_active

    parts = []
    for i, (priority, group) in enumerate(groups):
        alloc = base + (1 if i < remainder else 0)
        parts.append(group.head(alloc))

    return pd.concat(parts, ignore_index=True)


def compute_contextual_rankings(df_full: pd.DataFrame, filtered: pd.DataFrame) -> dict:
    """Compute top-N breakdowns: region, branch, bucket distribution.

    execute_plan resets the filtered index to 0-N, so we cannot use it to
    join back to df_full. Rejoin on Loan No (same approach as compute_result_kpis)
    so the breakdown cards reflect the actual matching accounts.
    """
    if (
        df_full is not None and not df_full.empty
        and len(filtered) > 0
        and "Loan No" in filtered.columns
        and "Loan No" in df_full.columns
    ):
        loan_nos = set(filtered["Loan No"])
        sub = df_full[df_full["Loan No"].isin(loan_nos)]
    elif len(filtered) > 0:
        sub = filtered
    else:
        sub = pd.DataFrame(columns=df_full.columns if df_full is not None else [])

    rankings = {}

    if "RegionName" in sub.columns and len(sub) > 0:
        region_counts = sub.groupby("RegionName")["Loan No"].nunique().sort_values(ascending=False)
        rankings["region_counts"] = region_counts.head(5).to_dict()

    if "MNT NAME" in sub.columns and "Unit" in sub.columns and len(sub) > 0:
        # One row per (MNT NAME, Unit)  -  sorted by account count desc, top 8 rows total
        grp_cols = sub.groupby(["MNT NAME", "Unit"])
        mnt_branch = grp_cols["Loan No"].nunique().reset_index(name="count")
        if "SOH" in sub.columns:
            mnt_branch_pos = grp_cols["SOH"].apply(lambda x: pd.to_numeric(x, errors="coerce").sum()).reset_index(name="pos")
            mnt_branch = mnt_branch.merge(mnt_branch_pos, on=["MNT NAME", "Unit"])
        else:
            mnt_branch["pos"] = 0
        mnt_branch = mnt_branch.sort_values("count", ascending=False).head(8)
        rankings["mnt_details"] = [
            {"name": r["MNT NAME"], "branch": r["Unit"], "count": int(r["count"]), "pos": float(r["pos"])}
            for _, r in mnt_branch.iterrows()
        ]

    if "Unit" in sub.columns and len(sub) > 0:
        branch_counts = sub.groupby("Unit")["Loan No"].nunique().sort_values(ascending=False)
        rankings["branch_counts"] = branch_counts.head(5).to_dict()

        if "SOH" in sub.columns:
            branch_pos = sub.groupby("Unit")["SOH"].sum().sort_values(ascending=False)
            rankings["branch_pos"] = branch_pos.head(5).to_dict()

    if "curr_bucket" in sub.columns and len(sub) > 0:
        bucket_dist = sub["curr_bucket"].value_counts()
        total = len(sub)
        rankings["bucket_dist"] = {k: round(v / total * 100, 1) for k, v in bucket_dist.items()}

    return rankings

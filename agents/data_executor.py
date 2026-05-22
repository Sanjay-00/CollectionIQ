import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta


def _apply_condition(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    col = cond["column"]
    op = cond["op"]
    val = cond["value"]

    if col not in df.columns:
        return df

    series = df[col]

    # Date-aware comparison
    if pd.api.types.is_datetime64_any_dtype(series):
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
        return df[mask] if mask is not None else df

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
        return df[mask] if mask is not None else df

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


QUERY_DISPLAY_COLS = [
    "Loan No", "Zone", "RegionName", "Unit", "Ag_Date", "MNT NAME",
    "Due Dt", "Tenure", "Loan Status", "Loan Amount", "Veh ID", "Cust Name",
    "Guar Name", "Cust Mob No", "Guar Mob No", "Vehicle Description",
    "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)", "MONTH DUE PC",
    "Month Receipt Amount", "Closing Arrears", "Arrears against Inst+Exp",
    "LCC%", "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC",
    "POS", "Non Starter", "Strike", "Last Receipt Date", "Last Receipt Amount",
    "ParentLDueDate", "No Coll 3 Months and >6 EMI", "NACHStatus",
    "TyreFlag", "FUEL_TYPE",
]


def execute_filters(df: pd.DataFrame, parsed: dict) -> tuple[pd.DataFrame, str]:
    """Apply parsed filter conditions to df. Returns (result_df, error_message)."""
    result = df.copy()

    for cond in parsed.get("conditions", []):
        try:
            result = _apply_condition(result, cond)
        except Exception as e:
            return pd.DataFrame(), f"Filter error on column '{cond.get('column')}': {e}"

    if len(result) == 0:
        return pd.DataFrame(), "No records match the given criteria."

    sort_col = parsed.get("sort_by")
    sort_asc = parsed.get("sort_asc", False)
    if sort_col and sort_col in result.columns:
        result = result.sort_values(sort_col, ascending=sort_asc)

    display_cols = [c for c in QUERY_DISPLAY_COLS if c in result.columns]
    return result[display_cols] if display_cols else result, ""


def compute_result_kpis(df_full: pd.DataFrame, filtered: pd.DataFrame) -> dict:
    """Compute query-specific KPIs from the filtered DataFrame."""
    n = filtered["Loan No"].nunique() if "Loan No" in filtered.columns else len(filtered)

    def _col_sum(col):
        return df_full.loc[df_full.index.isin(filtered.index), col].sum() if col in df_full.columns else 0

    def _col_mean(col):
        vals = df_full.loc[df_full.index.isin(filtered.index), col]
        return round(vals.mean(), 2) if col in df_full.columns and len(vals) > 0 else 0

    pos = _col_sum("POS")
    demand = _col_sum("NET Collection Demand Inst+Exp")
    collection = _col_sum("Month Receipt Amount")
    coll_pct = round(collection / demand * 100, 2) if demand > 0 else 0
    avg_arrears = _col_mean("Arrears / EMI")

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

    cutoff_1y = pd.Timestamp(date.today() - relativedelta(months=12))

    all_rows = []
    seen_loans = set()

    for rule in PRIORITY_RULES:
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

        # Deduplicate — each loan appears only under its highest priority rule
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
                    "curr_bucket", "Arrears / EMI", "POS", "ClosingPC", "Closing Arrears"]
    display_cols = [c for c in display_cols if c in result.columns]

    return result[display_cols], ""


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
    """Compute top-N breakdowns: region, branch, bucket distribution."""
    idx = filtered.index if len(filtered) > 0 else pd.Index([])
    sub = df_full.loc[df_full.index.isin(idx)] if len(idx) > 0 else pd.DataFrame(columns=df_full.columns)

    rankings = {}

    if "RegionName" in sub.columns and len(sub) > 0:
        region_counts = sub.groupby("RegionName")["Loan No"].nunique().sort_values(ascending=False)
        rankings["region_counts"] = region_counts.head(5).to_dict()

    if "Unit" in sub.columns and len(sub) > 0:
        branch_counts = sub.groupby("Unit")["Loan No"].nunique().sort_values(ascending=False)
        rankings["branch_counts"] = branch_counts.head(5).to_dict()

        if "POS" in sub.columns:
            branch_pos = sub.groupby("Unit")["POS"].sum().sort_values(ascending=False)
            rankings["branch_pos"] = branch_pos.head(5).to_dict()

    if "curr_bucket" in sub.columns and len(sub) > 0:
        bucket_dist = sub["curr_bucket"].value_counts()
        total = len(sub)
        rankings["bucket_dist"] = {k: round(v / total * 100, 1) for k, v in bucket_dist.items()}

    return rankings

import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

# Bucket severity scores - higher = worse
_BUCKET_SCORE = {"STD": 0, "1-30 DPD": 1, "SMA-1": 2, "SMA-2": 3, "NPA": 4}


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
    "prev_bucket", "curr_bucket",
    "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)", "MONTH DUE PC",
    "Month Receipt Amount", "Closing Arrears", "Arrears against Inst+Exp",
    "LCC%", "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC",
    "POS", "Non Starter", "Strike", "Last Receipt Date", "Last Receipt Amount",
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
    out = result[display_cols] if display_cols else result
    return _format_dates(out), ""


def compute_result_kpis(df_full: pd.DataFrame, filtered: pd.DataFrame) -> dict:
    """Compute query-specific KPIs from the filtered DataFrame.
    Operates directly on filtered to avoid index-alignment bugs after concat/sort.
    df_full is kept in signature for API compatibility but is not used.
    """
    work = filtered.drop(columns=["Priority", "_rank"], errors="ignore")

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

    pos        = _sum("POS")
    demand     = _sum("NET Collection Demand Inst+Exp")
    collection = _sum("Month Receipt Amount")
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
    out = result[display_cols] if display_cols else result
    return _format_dates(out), ""


def execute_aggregation(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, str]:
    """GROUP BY aggregation with ratio/metric computation.
    spec keys: group_by (str or [col1, col2]), counts, sums, metric, metric_label, sort_asc
    """
    group_by_spec = spec.get("group_by", "")
    df = df.copy()

    # Support multi-column group_by — combine into a single display column
    if isinstance(group_by_spec, list) and len(group_by_spec) >= 2:
        col1, col2 = str(group_by_spec[0]), str(group_by_spec[1])
        missing = [c for c in [col1, col2] if c not in df.columns]
        if missing:
            return pd.DataFrame(), f"Group-by columns not found: {missing}"
        group_col = f"{col1} ({col2})"
        df[group_col] = (
            df[col1].astype(str).str.strip()
            + " ("
            + df[col2].astype(str).str.strip()
            + ")"
        )
    else:
        group_col = str(group_by_spec)
        if not group_col or group_col not in df.columns:
            return pd.DataFrame(), f"Group-by column '{group_col}' not found in data."

    grouped = df.groupby(group_col, sort=False)
    agg = pd.DataFrame({group_col: list(grouped.groups.keys())}).set_index(group_col)

    # Count rows per group — supports equality AND bucket_worse_than / bucket_better_than
    for cnt in spec.get("counts", []):
        alias = cnt.get("alias", "")
        col   = cnt.get("column", "")
        op    = cnt.get("op", "==")
        val   = cnt.get("value", "")
        if not alias:
            continue

        if op in ("bucket_worse_than", "bucket_better_than", "bucket_stable"):
            ref_col = str(val)
            if col not in df.columns or ref_col not in df.columns:
                agg[alias] = 0
            else:
                curr_score = df[col].map(_BUCKET_SCORE)
                prev_score = df[ref_col].map(_BUCKET_SCORE)
                valid      = curr_score.notna() & prev_score.notna()
                if op == "bucket_worse_than":
                    mask = valid & (curr_score > prev_score)
                elif op == "bucket_better_than":
                    mask = valid & (curr_score < prev_score)
                else:  # bucket_stable
                    mask = valid & (curr_score == prev_score)
                counts = df[mask].groupby(group_col).size().reindex(agg.index, fill_value=0)
                agg[alias] = counts
        else:
            val_upper = str(val).upper()
            if col not in df.columns:
                agg[alias] = 0
            else:
                agg[alias] = grouped[col].apply(
                    lambda s, v=val_upper: int((s.astype(str).str.strip().str.upper() == v).sum())
                )

    # Sum numeric column per group
    for sm in spec.get("sums", []):
        alias = sm.get("alias", "")
        col   = sm.get("column", "")
        if not alias:
            continue
        if col not in df.columns:
            agg[alias] = 0.0
        else:
            agg[alias] = grouped[col].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())

    # Compute derived metric using pandas eval
    metric_expr  = spec.get("metric", "")
    metric_label = spec.get("metric_label", "Metric")
    if metric_expr:
        try:
            computed = agg.eval(metric_expr)
            computed = computed.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
            agg[metric_label] = computed.round(4)
        except Exception as e:
            return pd.DataFrame(), f"Metric computation failed ({metric_expr}): {e}"

    # Apply HAVING filters — post-aggregation group filtering (e.g. run_count >= 1)
    _having_ops = {">=": lambda a, v: a >= v, ">": lambda a, v: a > v,
                   "<=": lambda a, v: a <= v, "<": lambda a, v: a < v,
                   "==": lambda a, v: a == v, "!=": lambda a, v: a != v}
    for h in spec.get("having", []):
        alias = h.get("alias", "")
        op    = h.get("op", ">=")
        val   = h.get("value", 0)
        if alias in agg.columns and op in _having_ops:
            agg = agg[_having_ops[op](agg[alias], val)]

    if len(agg) == 0:
        return pd.DataFrame(), "No groups matched the having conditions."

    sort_asc = spec.get("sort_asc", True)
    if metric_label in agg.columns:
        agg = agg.sort_values(metric_label, ascending=sort_asc)

    agg = agg.reset_index()
    agg.insert(0, "Rank", range(1, len(agg) + 1))
    return agg, ""


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

    if "MNT NAME" in sub.columns and "Unit" in sub.columns and len(sub) > 0:
        # One row per (MNT NAME, Unit) — sorted by account count desc, top 8 rows total
        grp_cols = sub.groupby(["MNT NAME", "Unit"])
        mnt_branch = grp_cols["Loan No"].nunique().reset_index(name="count")
        if "POS" in sub.columns:
            mnt_branch_pos = grp_cols["POS"].apply(lambda x: pd.to_numeric(x, errors="coerce").sum()).reset_index(name="pos")
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

        if "POS" in sub.columns:
            branch_pos = sub.groupby("Unit")["POS"].sum().sort_values(ascending=False)
            rankings["branch_pos"] = branch_pos.head(5).to_dict()

    if "curr_bucket" in sub.columns and len(sub) > 0:
        bucket_dist = sub["curr_bucket"].value_counts()
        total = len(sub)
        rankings["bucket_dist"] = {k: round(v / total * 100, 1) for k, v in bucket_dist.items()}

    return rankings

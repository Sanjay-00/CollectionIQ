import re
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


def execute_filters(df: pd.DataFrame, parsed: dict) -> tuple[pd.DataFrame, str]:
    """Apply parsed filter conditions to df. Returns (result_df, error_message)."""
    result = df.copy()

    for cond in parsed.get("conditions") or []:
        try:
            result = _apply_condition(result, cond)
        except Exception as e:
            return pd.DataFrame(), f"Filter error on column '{cond.get('column')}': {e}"

    if len(result) == 0:
        return pd.DataFrame(), "No records match the given criteria."

    sort_col = parsed.get("sort_by")
    sort_asc = parsed.get("sort_asc")
    if sort_asc is None:
        sort_asc = False
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
                    "curr_bucket", "Arrears / EMI", "POS",
                    "Net Collection Demand Inst+Exp+BC", "Closing Arrears"]
    display_cols = [c for c in display_cols if c in result.columns]
    out = result[display_cols] if display_cols else result
    return _format_dates(out), ""


def execute_aggregation(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, str]:
    """GROUP BY aggregation with ratio/metric computation.
    spec keys: group_by (str or [col1, col2]), counts, sums, metric, metric_label, sort_asc
    """
    group_by_spec = spec.get("group_by") or ""
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

    # Count rows per group — supports total, equality, numeric comparison, and bucket ops
    for cnt in spec.get("counts") or []:
        alias = cnt.get("alias") or ""
        col   = cnt.get("column") or ""
        op    = cnt.get("op") or "=="
        val   = cnt.get("value")
        if not alias:
            continue

        if col == "__total__":
            agg[alias] = grouped.size()
            continue

        if not col or col not in df.columns:
            agg[alias] = 0
            continue

        if op in ("bucket_worse_than", "bucket_better_than", "bucket_stable"):
            ref_col = str(val) if val is not None else ""
            if ref_col not in df.columns:
                agg[alias] = 0
            else:
                curr_score = df[col].map(_BUCKET_SCORE)
                prev_score = df[ref_col].map(_BUCKET_SCORE)
                valid      = curr_score.notna() & prev_score.notna()
                if op == "bucket_worse_than":
                    mask = valid & (curr_score > prev_score)
                elif op == "bucket_better_than":
                    mask = valid & (curr_score < prev_score)
                else:
                    mask = valid & (curr_score == prev_score)
                counts = df[mask].groupby(group_col).size().reindex(agg.index, fill_value=0)
                agg[alias] = counts
        elif op in (">", ">=", "<", "<=", "!="):
            try:
                val_num = float(val)
            except (TypeError, ValueError):
                agg[alias] = 0
                continue
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            _num_ops = {
                ">":  lambda s, v: s > v,
                ">=": lambda s, v: s >= v,
                "<":  lambda s, v: s < v,
                "<=": lambda s, v: s <= v,
                "!=": lambda s, v: s != v,
            }
            mask = _num_ops[op](numeric_series, val_num)
            counts = df[mask].groupby(group_col).size().reindex(agg.index, fill_value=0)
            agg[alias] = counts
        elif op == "in" and isinstance(val, list):
            vals_upper = [str(v).upper() for v in val]
            mask = df[col].astype(str).str.strip().str.upper().isin(vals_upper)
            counts = df[mask].groupby(group_col).size().reindex(agg.index, fill_value=0)
            agg[alias] = counts
        else:
            val_upper = str(val if val is not None else "").upper()
            agg[alias] = grouped[col].apply(
                lambda s, v=val_upper: int((s.astype(str).str.strip().str.upper() == v).sum())
            )

    # Sum numeric column per group
    for sm in spec.get("sums") or []:
        alias = sm.get("alias") or ""
        col   = sm.get("column") or ""
        if not alias:
            continue
        if col not in df.columns:
            agg[alias] = 0.0
        else:
            agg[alias] = grouped[col].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())

    # Compute derived metrics using pandas eval
    # Supports both old single metric (metric + metric_label) and new list (metrics)
    metrics_list = spec.get("metrics") or []
    metric_expr  = spec.get("metric") or ""
    metric_label = spec.get("metric_label") or "Metric"

    # Backward compat: single metric → wrap into list
    if not metrics_list and metric_expr:
        metrics_list = [{"expr": metric_expr, "label": metric_label}]

    for m in metrics_list:
        expr  = m.get("expr") or ""
        label = m.get("label") or "Metric"
        if not expr:
            continue
        try:
            computed = agg.eval(expr)
            computed = computed.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
            agg[label] = computed.round(2)
        except Exception:
            agg[label] = 0

    # Apply HAVING filters — post-aggregation group filtering (e.g. run_count >= 1)
    _having_ops = {">=": lambda a, v: a >= v, ">": lambda a, v: a > v,
                   "<=": lambda a, v: a <= v, "<": lambda a, v: a < v,
                   "==": lambda a, v: a == v, "!=": lambda a, v: a != v}
    for h in spec.get("having") or []:
        alias = h.get("alias") or ""
        op    = h.get("op") or ">="
        val   = h.get("value", 0)
        if alias in agg.columns and op in _having_ops:
            agg = agg[_having_ops[op](agg[alias], val)]

    if len(agg) == 0:
        return pd.DataFrame(), "No groups matched the having conditions."

    sort_asc = spec.get("sort_asc")
    if sort_asc is None:
        sort_asc = True
    # Sort by the derived metric. metric_label is only populated in the legacy
    # single-metric format; for the metrics-list format fall back to the first
    # metric's label so "rank/order by <metric>" queries actually sort.
    sort_label = metric_label if metric_label in agg.columns else None
    if sort_label is None and metrics_list:
        first_label = metrics_list[0].get("label")
        if first_label in agg.columns:
            sort_label = first_label
    if sort_label:
        agg = agg.sort_values(sort_label, ascending=sort_asc)

    # Top-N limit — applied after sort so we keep the highest/lowest ranked groups.
    # Set by the planner only when the user asks for "top N" / "first N" groups.
    limit = spec.get("limit")
    try:
        limit = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit = None
    if limit and limit > 0:
        agg = agg.head(limit)

    agg = agg.reset_index()
    agg.insert(0, "Rank", range(1, len(agg) + 1))
    return agg, ""


# ── Spec validation (used by the validate-and-repair loop in graph.py) ─────────
# Pure logic, no LLM. Catches hallucinated column names / undefined metric aliases
# BEFORE execution, so a bad spec becomes a repair request instead of silently
# returning wrong (unfiltered / all-zero) results.

def validate_aggregation_spec(spec: dict, columns) -> list[str]:
    """Return a list of human-readable problems with an aggregation_spec given the
    actual DataFrame columns. Empty list = valid."""
    cols = set(columns)
    errs: list[str] = []

    gb = spec.get("group_by")
    gb_cols = gb if isinstance(gb, list) else [gb]
    for c in gb_cols:
        if not c or c not in cols:
            errs.append(f"group_by column '{c}' does not exist")

    aliases: set[str] = set()
    for cnt in spec.get("counts") or []:
        alias = cnt.get("alias")
        if alias:
            aliases.add(alias)
        col = cnt.get("column")
        op = cnt.get("op")
        if col == "__total__":
            continue
        if not col or col not in cols:
            errs.append(f"count column '{col}' does not exist")
        if op in ("bucket_worse_than", "bucket_better_than", "bucket_stable"):
            ref = cnt.get("value")
            if not ref or ref not in cols:
                errs.append(f"bucket comparison reference column '{ref}' does not exist")

    for sm in spec.get("sums") or []:
        alias = sm.get("alias")
        if alias:
            aliases.add(alias)
        col = sm.get("column")
        if not col or col not in cols:
            errs.append(f"sum column '{col}' does not exist")

    for m in spec.get("metrics") or []:
        expr = m.get("expr") or ""
        idents = set(re.findall(r"[A-Za-z_]\w*", expr))
        unknown = sorted(i for i in idents if i not in aliases)
        if unknown:
            errs.append(
                f"metric '{m.get('label')}' uses undefined names {unknown}; "
                f"defined aliases are {sorted(aliases)}"
            )

    return errs


def validate_filter_spec(parsed: dict, columns) -> list[str]:
    """Return a list of problems with a parsed filter spec given actual columns."""
    cols = set(columns)
    errs: list[str] = []
    for cond in parsed.get("conditions") or []:
        col = cond.get("column")
        op = cond.get("op")
        if not col or col not in cols:
            errs.append(f"filter column '{col}' does not exist")
        if op in ("bucket_worse_than", "bucket_better_than"):
            ref = cond.get("value")
            if not ref or ref not in cols:
                errs.append(f"bucket comparison reference column '{ref}' does not exist")
    return errs


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

"""Step-plan dataflow engine.

The Domain Expert can emit a ``plan``: an ordered list of typed steps. Each step
takes the previous step's DataFrame and returns a new one, so arbitrary-depth
analytics (nested aggregation, count-distinct, per-entity thresholds) compose
WITHOUT special-casing each shape.

Design constraints:
- Pure pandas, no LLM.
- No arbitrary code execution  -  only the whitelisted operations below run.
- Each step is validatable against the column set produced by the prior step,
  so a bad plan becomes a repair request (see validate_plan) instead of a crash
  or a silently-wrong answer.
"""

import re
import pandas as pd

from agents.data_executor import _apply_condition, _build_mask, _COL_COMPARE_OPS

_AGG_FUNCS = {"sum", "count", "nunique", "mean", "min", "max", "first"}

# Ops whose "value" is a COLUMN NAME to compare against, not a literal -- both
# the left ("column") and right ("value") sides must exist for these.
_COLUMN_VALUE_OPS = set(_COL_COMPARE_OPS) | {"bucket_worse_than", "bucket_better_than"}


def _check_condition_columns(cond: dict, cols: set, where: str) -> list[str]:
    """Validate a single {column, op, value} condition against the known column
    set. For column-vs-column ops (col_lt/col_lte/.../bucket_worse_than/...),
    "value" is itself a column reference and must also exist -- a hallucinated
    reference column would otherwise silently no-op (see _apply_condition's
    graceful `if ref_col not in df.columns: return df`) instead of failing loud."""
    errs = []
    c = cond.get("column")
    if not c or c not in cols:
        errs.append(f"{where}: column '{c}' does not exist")
    op = (cond.get("op") or "").lower()
    if op in _COLUMN_VALUE_OPS:
        ref = cond.get("value")
        if not ref or str(ref) not in cols:
            errs.append(f"{where}: op '{op}' compares against column '{ref}', which does not exist")
    return errs


# Belt-and-suspenders guard on 'derive' expressions (which reach pandas' df.eval).
# The REAL gate is validate_plan()'s identifier whitelist -- every bare name in an
# expr must already be a known column/alias, which already blocks sandbox-escape
# patterns like "(1).__class__.__bases__[0].__subclasses__()" before a plan is even
# allowed to execute (verified: pandas' own eval parser also already rejects
# arbitrary function calls like __import__/open/getattr as "not a supported
# function"). This regex is a second, independent layer here at the point of
# execution itself, so a future code path that ever skips validate_plan still
# can't reach dunder/attribute-escape tricks through this executor.
_DANGEROUS_EXPR_RE = re.compile(r"__\w+__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\bgetattr\b|\bsubprocess\b|\bos\.")


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


# ── Step handlers ─────────────────────────────────────────────────────────────

def _op_group_aggregate(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    group_by = [str(c) for c in _as_list(step.get("group_by"))]
    if not group_by:
        raise ValueError("group_aggregate requires 'group_by'")
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"group_by columns not found: {missing}")

    aggs = step.get("aggregations") or []
    if not aggs:
        raise ValueError("group_aggregate requires at least one aggregation")

    # Full set of group keys (first-appearance order)  -  every aggregation is
    # reindexed onto this so a conditional ('where') aggregation that matches no
    # rows in a group yields 0 there rather than dropping the group.
    full_index = df.groupby(group_by, sort=False).size().index

    data: dict = {}
    for a in aggs:
        alias = a.get("alias")
        func = (a.get("func") or "").lower()
        col = a.get("column")
        where = a.get("where")
        if not alias:
            raise ValueError("aggregation missing 'alias'")
        if func not in _AGG_FUNCS:
            raise ValueError(f"unknown aggregation func '{func}'")

        # 'where' restricts this aggregation to rows matching a per-row condition.
        src = df[_build_mask(df, where)] if where else df

        if func == "count":
            if len(src) == 0:
                data[alias] = pd.Series(0, index=full_index)
            else:
                data[alias] = src.groupby(group_by, sort=False).size().reindex(full_index, fill_value=0)
            continue

        if not col or col not in df.columns:
            raise ValueError(f"{func} column '{col}' not found")

        if len(src) == 0:
            data[alias] = pd.Series(0, index=full_index)
        elif func == "nunique":
            data[alias] = src.groupby(group_by, sort=False)[col].nunique().reindex(full_index, fill_value=0)
        elif func == "first":
            # Non-numeric passthrough (e.g. attaching a display label like Cust
            # Name to a group keyed on Cust Mob No) -- must NOT go through the
            # pd.to_numeric coercion below, which would blank out text values.
            data[alias] = src.groupby(group_by, sort=False)[col].first().reindex(full_index)
        else:
            numeric = pd.to_numeric(src[col], errors="coerce")
            series = numeric.groupby([src[g] for g in group_by], sort=False).agg(func)
            data[alias] = series.reindex(full_index, fill_value=0)

    out = pd.DataFrame(data, index=full_index).reset_index()
    return out


def _op_filter(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    result = df
    for cond in step.get("conditions") or []:
        result = _apply_condition(result, cond)
    return result


def _op_derive(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    col = step.get("column")
    expr = step.get("expr") or ""
    if not col:
        raise ValueError("derive requires 'column'")
    if _DANGEROUS_EXPR_RE.search(expr):
        raise ValueError(f"derive expression rejected (disallowed pattern): {expr!r}")
    df = df.copy()
    try:
        computed = df.eval(expr)
    except Exception as e:
        raise ValueError(f"derive expression failed: {e}")
    if hasattr(computed, "replace"):
        computed = computed.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
        # Optional clipping (e.g. cap a ratio like LCC% at 100). Applied before
        # rounding; absent keys leave the value untouched (backward compatible).
        clip_min, clip_max = step.get("clip_min"), step.get("clip_max")
        if (clip_min is not None or clip_max is not None) and hasattr(computed, "clip"):
            computed = computed.clip(lower=clip_min, upper=clip_max)
        try:
            computed = computed.round(2)
        except Exception:
            pass
    df[col] = computed
    return df


def _op_sort(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    by = step.get("by")
    asc = step.get("ascending")
    if asc is None:
        asc = False
    if not by:
        return df
    # Exact match first; case-insensitive fallback so alias mismatches don't silently no-op.
    col = by if by in df.columns else next(
        (c for c in df.columns if c.lower() == by.lower()), None
    )
    if col:
        return df.sort_values(col, ascending=asc)
    return df


def _op_limit(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    n = step.get("n")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return df
    return df.head(n) if n > 0 else df


def _op_select(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Keep only the listed columns. Unknown columns are silently skipped so
    missing optional display columns don't hard-fail on a schema variation."""
    cols = [c for c in (step.get("columns") or []) if c in df.columns]
    return df[cols] if cols else df


def _op_entity_semi_join(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Keep every original row belonging to an entity that satisfies the given
    having predicates -- a loan_table-shaped counterpart to group_aggregate's
    collapse-to-one-row-per-entity behavior. Used for queries like "list every
    loan for fleet owners in NPA," where the answer must stay at loan grain."""
    key_cols = [str(c) for c in _as_list(step.get("key_cols"))]
    if not key_cols:
        raise ValueError("entity_semi_join requires 'key_cols'")
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"entity_semi_join key columns not found: {missing}")

    agg_step = {"group_by": key_cols, "aggregations": step.get("aggregations") or []}
    qualifying = _op_group_aggregate(df, agg_step)
    qualifying = _op_filter(qualifying, {"conditions": step.get("having_conditions") or []})

    # A plain key-column merge (not a row-wise tuple/isin check) -- vectorized
    # regardless of whether the entity key is single- or multi-column (e.g. the
    # executive entity's ["MNT NAME", "Unit"]), and avoids Python-level
    # per-row tuple construction on a large upload.
    return df.merge(qualifying[key_cols].drop_duplicates(), on=key_cols, how="inner")


_OPS = {
    "group_aggregate":  _op_group_aggregate,
    "entity_semi_join": _op_entity_semi_join,
    "filter":          _op_filter,
    "derive":          _op_derive,
    "sort":            _op_sort,
    "limit":           _op_limit,
    "select":          _op_select,
}

# Internal scratch columns the "ratio"/"count_ratio" measure handlers generate
# (compiler/measures.py: f"{alias}__n{i}" / f"{alias}__d{i}", e.g.
# "collection_pct__n0") to hold the raw sum/count before dividing. The AI never
# sees or authors these names -- they exist purely so the derive step has
# something to divide -- so they should never reach the displayed table.
# Matched by suffix only, since the alias prefix is arbitrary/AI-chosen.
_HELPER_COL_RE = re.compile(r"__[nd]\d+$")


# ── Runner ────────────────────────────────────────────────────────────────────

def execute_plan(df: pd.DataFrame, plan: list) -> tuple[pd.DataFrame, str]:
    """Run an ordered list of steps. Returns (result_df, error_message)."""
    if not plan:
        return pd.DataFrame(), "Empty plan."
    result = df.copy()
    for i, step in enumerate(plan, start=1):
        op = (step.get("op") or "").lower()
        handler = _OPS.get(op)
        if handler is None:
            return pd.DataFrame(), f"Step {i}: unknown operation '{op}'."
        try:
            result = handler(result, step)
        except Exception as e:
            return pd.DataFrame(), f"Step {i} ({op}) failed: {e}"
        if result is None:
            return pd.DataFrame(), f"Step {i} ({op}) returned None."

    result = result.reset_index(drop=True)
    helper_cols = [c for c in result.columns if _HELPER_COL_RE.search(str(c))]
    if helper_cols:
        result = result.drop(columns=helper_cols)
    if "Rank" not in result.columns:
        result.insert(0, "Rank", range(1, len(result) + 1))
    return result, ""


# ── Validation (column-tracking, used by the validate-and-repair loop) ─────────

def validate_plan(plan: list, initial_columns) -> list[str]:
    """Validate a plan step-by-step against the column set each step produces.
    Returns a list of human-readable problems; empty list = valid."""
    if not plan:
        return ["Plan is empty."]

    cols = set(map(str, initial_columns))
    errs: list[str] = []

    for i, step in enumerate(plan, start=1):
        op = (step.get("op") or "").lower()
        if op not in _OPS:
            errs.append(f"step {i}: unknown operation '{op}'")
            continue

        if op == "group_aggregate":
            gb = [str(c) for c in _as_list(step.get("group_by"))]
            if not gb:
                errs.append(f"step {i}: group_aggregate needs 'group_by'")
            for c in gb:
                if c not in cols:
                    errs.append(f"step {i}: group_by column '{c}' does not exist")
            aggs = step.get("aggregations") or []
            if not aggs:
                errs.append(f"step {i}: group_aggregate needs at least one aggregation")
            new_cols = set(gb)
            for a in aggs:
                alias = a.get("alias")
                func = (a.get("func") or "").lower()
                col = a.get("column")
                if not alias:
                    errs.append(f"step {i}: aggregation missing 'alias'")
                    continue
                if func not in _AGG_FUNCS:
                    errs.append(f"step {i}: unknown agg func '{func}'")
                elif func != "count" and (not col or col not in cols):
                    errs.append(f"step {i}: agg column '{col}' does not exist")
                for cond in (a.get("where") or []):
                    errs.extend(_check_condition_columns(cond, cols, f"step {i}: where"))
                new_cols.add(alias)
            # A group_aggregate drops every column except the keys and new aliases.
            cols = new_cols

        elif op == "entity_semi_join":
            # Preserves every original column (a semi-join, not a collapse) --
            # 'cols' is intentionally left untouched, unlike group_aggregate above.
            key_cols = [str(c) for c in _as_list(step.get("key_cols"))]
            if not key_cols:
                errs.append(f"step {i}: entity_semi_join needs 'key_cols'")
            for c in key_cols:
                if c not in cols:
                    errs.append(f"step {i}: entity_semi_join key column '{c}' does not exist")
            for a in step.get("aggregations") or []:
                func = (a.get("func") or "").lower()
                col = a.get("column")
                if func not in _AGG_FUNCS:
                    errs.append(f"step {i}: unknown agg func '{func}'")
                elif func != "count" and (not col or col not in cols):
                    errs.append(f"step {i}: agg column '{col}' does not exist")
                for cond in (a.get("where") or []):
                    errs.extend(_check_condition_columns(cond, cols, f"step {i}: where"))
            # having_conditions reference an aggregation alias defined above
            # (e.g. "__ef0_0"), not a raw input column, so no column-existence
            # check applies to them here.

        elif op == "filter":
            for cond in step.get("conditions") or []:
                errs.extend(_check_condition_columns(cond, cols, f"step {i}: filter"))

        elif op == "derive":
            col = step.get("column")
            expr = step.get("expr") or ""
            if _DANGEROUS_EXPR_RE.search(expr):
                errs.append(f"step {i}: derive expr rejected (disallowed pattern): {expr!r}")
            idents = set(re.findall(r"[A-Za-z_]\w*", expr))
            unknown = sorted(idn for idn in idents if idn not in cols)
            if unknown:
                errs.append(f"step {i}: derive expr uses undefined names {unknown}")
            if col:
                cols.add(col)

        elif op == "sort":
            by = step.get("by")
            if by and by not in cols:
                errs.append(f"step {i}: sort column '{by}' does not exist")

        elif op == "select":
            # Unknown columns are skipped gracefully at execute time; no validation error.
            pass

        # limit: nothing to validate against columns

    return errs

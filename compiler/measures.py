"""Generic measure framework.

A measure KIND (additive, semi_additive, count, count_distinct, ratio, ...) is
lowered to physical plan pieces by a registered handler. The compiler dispatches
on `kind` and concatenates what handlers return  -  so adding a NEW measure type is
a new handler registration, never a change to compiler assembly logic.

A handler returns a `Lowered`:
- pre_derives  : row-level {column, expr[, clip_*]} computed BEFORE grouping
- aggregations : group_aggregate entries {alias, func, column?, where?}
- post_derives : {column, expr[, clip_*]} computed AFTER grouping (over aliases)
- errors       : human-readable problems (-> repair/clarification, never silent)

This module is intentionally free of any registry/concept/column knowledge: the
compiler resolves metric refs and expands concepts into a concrete measure
definition (`mdef`) first, then calls the handler. That keeps this framework a
pure, reusable lowering layer (no import cycle with compiler.core).

Measure definition (`mdef`) fields a handler may read:
  alias                      output column name (always present)
  kind                       the measure kind (dispatch key)
  agg                        aggregation func for additive/semi_additive (default sum)
  column                     source column (additive/semi_additive)
  distinct                   key column (count_distinct)
  where                      pre-expanded conditions (count)  -  concepts already resolved
  numerator / denominator    column or list-of-columns (ratio); summed then divided
  scale, cap, floor          ratio post-processing (e.g. LCC% -> *100, cap 100)
  non_additive_over          axes the measure must not be aggregated over (semi_additive)
"""
from dataclasses import dataclass, field


@dataclass
class Lowered:
    pre_derives: list = field(default_factory=list)
    aggregations: list = field(default_factory=list)
    post_derives: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class RollupLowered:
    """How a measure is computed across TWO grains in a nested aggregation: an
    intermediate pass (at the finer entity grain) and a terminal pass (at the
    output grain). Only measures whose math decomposes across a grain are nestable.
    """
    intermediate_aggs: list = field(default_factory=list)   # computed at the finer grain
    terminal_aggs: list = field(default_factory=list)        # re-aggregated at output grain
    terminal_post_derives: list = field(default_factory=list)
    errors: list = field(default_factory=list)


MEASURE_HANDLERS: dict = {}
MEASURE_ROLLUPS: dict = {}


def measure_handler(kind: str):
    """Register a single-pass handler for a measure kind. Adding a kind = adding one."""
    def deco(fn):
        MEASURE_HANDLERS[kind] = fn
        return fn
    return deco


def measure_rollup(kind: str):
    """Register a NESTED (two-grain) lowering for a measure kind. A kind without a
    rollup is simply not nestable (the compiler errors loudly). Adding a nestable
    kind = registering both a handler and a rollup  -  same extensibility contract."""
    def deco(fn):
        MEASURE_ROLLUPS[kind] = fn
        return fn
    return deco


def _as_cols(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


# ── Built-in measure kinds ────────────────────────────────────────────────────

@measure_handler("additive")
def _additive(mdef: dict, ctx: dict) -> Lowered:
    """Fully additive across every dimension (sum/mean/min/max of a column)."""
    alias, col = mdef["alias"], mdef.get("column")
    func = (mdef.get("agg") or "sum").lower()
    if not col:
        return Lowered(errors=[f"measure '{alias}': additive needs a 'column'"])
    return Lowered(aggregations=[{"alias": alias, "func": func, "column": col}])


@measure_handler("semi_additive")
def _semi_additive(mdef: dict, ctx: dict) -> Lowered:
    """Additive across some dimensions but NOT others (e.g. a stock like SOH is
    additive across loans but not across time). Refuses aggregation over a banned
    axis  -  the measure-side twin of the dropped-condition guard."""
    alias, col = mdef["alias"], mdef.get("column")
    banned = set(mdef.get("non_additive_over") or [])
    over = set(ctx.get("aggregating_over") or [])
    bad = banned & over
    if bad:
        return Lowered(errors=[
            f"measure '{alias}' is semi-additive and cannot be aggregated over "
            f"{sorted(bad)}; use a point-in-time rule (e.g. last/first)"
        ])
    func = (mdef.get("agg") or "sum").lower()
    if not col:
        return Lowered(errors=[f"measure '{alias}': semi_additive needs a 'column'"])
    return Lowered(aggregations=[{"alias": alias, "func": func, "column": col}])


@measure_handler("count")
def _count(mdef: dict, ctx: dict) -> Lowered:
    """Row count per group, optionally conditional via a pre-expanded 'where'."""
    alias = mdef["alias"]
    agg = {"alias": alias, "func": "count"}
    where = mdef.get("where")
    if where:
        agg["where"] = where
    return Lowered(aggregations=[agg])


@measure_handler("count_distinct")
def _count_distinct(mdef: dict, ctx: dict) -> Lowered:
    """Distinct count of a key column (e.g. unique customers)."""
    alias = mdef["alias"]
    col = mdef.get("distinct") or mdef.get("column")
    if not col:
        return Lowered(errors=[f"measure '{alias}': count_distinct needs a 'distinct' column"])
    return Lowered(aggregations=[{"alias": alias, "func": "nunique", "column": col}])


@measure_handler("ratio")
def _ratio(mdef: dict, ctx: dict) -> Lowered:
    """Non-additive ratio. Computed as sum(numerator)/sum(denominator) AT THE TARGET
    GRAIN  -  never as an average of per-row ratios (which is wrong). numerator and
    denominator may each be a column or a list of columns (summed). Optional
    `scale` (e.g. *100 for a percentage) and `cap`/`floor` clip the result.

    Source columns appear only as group_aggregate `column` values (df[col])  -  never
    inside an eval expression  -  so spaced/parenthesised column names are safe.
    Only the generated snake-safe aliases are used in derive expressions."""
    alias = mdef["alias"]
    num, den = _as_cols(mdef.get("numerator")), _as_cols(mdef.get("denominator"))
    if not num or not den:
        return Lowered(errors=[f"measure '{alias}': ratio needs 'numerator' and 'denominator'"])

    aggs, n_aliases, d_aliases = [], [], []
    for i, c in enumerate(num):
        a = f"{alias}__n{i}"
        aggs.append({"alias": a, "func": "sum", "column": c})
        n_aliases.append(a)
    for i, c in enumerate(den):
        a = f"{alias}__d{i}"
        aggs.append({"alias": a, "func": "sum", "column": c})
        d_aliases.append(a)

    scale = mdef.get("scale", 1)
    scale_suffix = f" * {scale}" if scale and scale != 1 else ""
    final = {"column": alias, "expr": f"({' + '.join(n_aliases)}) / ({' + '.join(d_aliases)}){scale_suffix}"}
    if mdef.get("cap") is not None:
        final["clip_max"] = mdef["cap"]
    if mdef.get("floor") is not None:
        final["clip_min"] = mdef["floor"]
    return Lowered(aggregations=aggs, post_derives=[final])


def _ratio_clip(mdef: dict, derive: dict) -> dict:
    if mdef.get("cap") is not None:
        derive["clip_max"] = mdef["cap"]
    if mdef.get("floor") is not None:
        derive["clip_min"] = mdef["floor"]
    return derive


# ── Nested (two-grain) rollups ────────────────────────────────────────────────
# Each returns how the measure is computed at the intermediate (finer) grain and
# re-aggregated at the terminal (output) grain. Only safe-to-decompose math is
# allowed; anything else is a loud error (never a silent wrong number).

@measure_rollup("additive")
def _r_additive(mdef: dict, ctx: dict) -> RollupLowered:
    alias, col = mdef["alias"], mdef.get("column")
    func = (mdef.get("agg") or "sum").lower()
    if not col:
        return RollupLowered(errors=[f"measure '{alias}': additive needs a 'column'"])
    # sum/min/max are associative across grains; mean is NOT (needs weights).
    if func not in ("sum", "min", "max"):
        return RollupLowered(errors=[
            f"measure '{alias}': '{func}' cannot be rolled up across a nested grain "
            "(only sum/min/max decompose; mean needs weighting)"
        ])
    part = f"{alias}__p"
    return RollupLowered(
        intermediate_aggs=[{"alias": part, "func": func, "column": col}],
        terminal_aggs=[{"alias": alias, "func": func, "column": part}],
    )


@measure_rollup("semi_additive")
def _r_semi_additive(mdef: dict, ctx: dict) -> RollupLowered:
    banned = set(mdef.get("non_additive_over") or [])
    over = set(ctx.get("aggregating_over") or [])
    if banned & over:
        return RollupLowered(errors=[
            f"measure '{mdef['alias']}' is semi-additive and cannot be aggregated over "
            f"{sorted(banned & over)}"
        ])
    return _r_additive(mdef, ctx)


@measure_rollup("count")
def _r_count(mdef: dict, ctx: dict) -> RollupLowered:
    # A count is additive: count per finer group, then SUM those counts.
    alias = mdef["alias"]
    part = f"{alias}__p"
    inter = {"alias": part, "func": "count"}
    if mdef.get("where"):
        inter["where"] = mdef["where"]
    return RollupLowered(
        intermediate_aggs=[inter],
        terminal_aggs=[{"alias": alias, "func": "sum", "column": part}],
    )


@measure_rollup("count_distinct")
def _r_count_distinct(mdef: dict, ctx: dict) -> RollupLowered:
    # Distinct doesn't decompose in general  -  but if the key IS one of the
    # intermediate group keys (e.g. counting the nested entity itself), it survives
    # as a column and can be counted directly at the terminal grain.
    alias = mdef["alias"]
    col = mdef.get("distinct") or mdef.get("column")
    if col not in set(ctx.get("intermediate_keys") or []):
        return RollupLowered(errors=[
            f"measure '{alias}': count_distinct of '{col}' cannot be rolled up across a "
            "nested grain unless it is one of the nested group keys"
        ])
    return RollupLowered(terminal_aggs=[{"alias": alias, "func": "nunique", "column": col}])


@measure_rollup("ratio")
def _r_ratio(mdef: dict, ctx: dict) -> RollupLowered:
    # num/den are additive even though the ratio is not: carry their sums through the
    # intermediate grain, re-sum at the terminal grain, divide once. (Averaging a
    # per-entity ratio would be wrong.)
    alias = mdef["alias"]
    num, den = _as_cols(mdef.get("numerator")), _as_cols(mdef.get("denominator"))
    if not num or not den:
        return RollupLowered(errors=[f"measure '{alias}': ratio needs 'numerator' and 'denominator'"])
    inter, n_aliases, d_aliases = [], [], []
    for i, c in enumerate(num):
        a = f"{alias}__n{i}"
        inter.append({"alias": a, "func": "sum", "column": c})
        n_aliases.append(a)
    for i, c in enumerate(den):
        a = f"{alias}__d{i}"
        inter.append({"alias": a, "func": "sum", "column": c})
        d_aliases.append(a)
    # Re-sum each partial at the terminal grain (partial column -> same alias).
    terminal = [{"alias": a, "func": "sum", "column": a} for a in (n_aliases + d_aliases)]
    scale = mdef.get("scale", 1)
    scale_suffix = f" * {scale}" if scale and scale != 1 else ""
    final = _ratio_clip(mdef, {
        "column": alias,
        "expr": f"({' + '.join(n_aliases)}) / ({' + '.join(d_aliases)}){scale_suffix}",
    })
    return RollupLowered(intermediate_aggs=inter, terminal_aggs=terminal,
                         terminal_post_derives=[final])

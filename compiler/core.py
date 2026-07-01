"""Compiler core  -  logical IR (IR-1) -> physical step-plan (IR-2).

IR-1 (what the LLM emits; flat and declarative):
    {
      "intent":     "aggregation" | "loan_table" | "single_value",  # informational
      "filters":    [ {"concept": "colending_at_risk"},             # concept ref, OR
                      {"column": "RegionName", "op": "==", "value": "PUNE"} ],  # raw
      "dimensions": ["branch"],                  # aliases (branch/region/...) or columns
      "measures":   [ {"metric": "exposure"},    # metric ref (+ optional agg/alias), OR
                      {"column": "POS", "agg": "sum", "alias": "total_pos"} ],  # raw
      "order_by":   [ {"by": "exposure", "dir": "desc"} ],
      "limit":      5
    }

The compiler NEVER lets the LLM author the dependency structure: it expands concept
references to their full condition lists from the registry, resolves dimension
aliases and metric definitions, resolves the dynamic 1-year cutoff, assembles the
plan, and validates it with the existing validate_plan. Unknown concept/metric ->
a loud error (routes to repair/clarification), never a silent wrong plan.
"""
import re
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta

from registry.ontology import CONCEPTS, METRICS, ENTITY_CONCEPTS
from registry.semantic_model import ENTITIES, entity_key, resolve_dimension, is_coarser
from agents.plan_executor import validate_plan
from compiler.measures import MEASURE_HANDLERS, MEASURE_ROLLUPS
from utils import PREV_CARRYOVER_COLS

# Key columns of loan-level "dimension" entities (branch/region/executive). A
# per-entity predicate that aggregates over one of these  -  when it is not the output
# dimension  -  is grain-ambiguous (a customer can span them), so the compiler asks
# rather than guesses.
_DIMENSION_KEY_COLS = {
    col for e in ENTITIES.values() if e["kind"] == "dimension" for col in e["key"]
}


def _snake(name: str) -> str:
    """Eval-safe alias from an arbitrary column name (matches the prev_* convention)."""
    s = re.sub(r"\W+", "_", str(name).strip()).strip("_").lower()
    return s or "value"


def _resolve_cutoff(value):
    """Resolve the dynamic __CUTOFF_1Y__ placeholder to a concrete timestamp.
    Mirrors agents.data_executor.execute_priority_mode so concept definitions that
    use the placeholder (recent_advance_high_bucket) behave identically."""
    if value == "__CUTOFF_1Y__":
        return pd.Timestamp(date.today() - relativedelta(months=12))
    return value


def _expand_filters(filters: list, errs: list) -> list:
    """Expand each filter item into concrete {column, op, value} conditions.
    Concept refs expand to their FULL condition list (the anti-dropped-condition
    guarantee). Raw conditions pass through. ANDed together downstream."""
    conditions: list = []
    for item in filters or []:
        if not isinstance(item, dict):
            errs.append(f"filter item is not an object: {item!r}")
            continue
        if "concept" in item:
            name = item["concept"]
            concept = CONCEPTS.get(name)
            if concept is None:
                errs.append(f"unknown concept '{name}'")
                continue
            for cond in concept["conditions"]:
                conditions.append({**cond, "value": _resolve_cutoff(cond["value"])})
        elif "column" in item and "op" in item:
            conditions.append({**item, "value": _resolve_cutoff(item.get("value"))})
        else:
            errs.append(f"uninterpretable filter (need 'concept' or 'column'+'op'): {item!r}")
    return conditions


# ── Count-metric fallback resolution ─────────────────────────────────────────
# Maps common short names → curr_bucket values so the model can say {metric:"npa_count"}
# instead of knowing the exact where-clause.
_BUCKET_ALIASES: dict[str, str] = {
    "npa":   "NPA",
    "sma2":  "SMA-2",
    "sma1":  "SMA-1",
    "dpd":   "1-30 DPD",
    "dpd30": "1-30 DPD",
    "std":   "STD",
}

_STATUS_ALIASES: dict[str, str] = {
    "mat": "MAT",
    "run": "RUN",
    "sns": "S&S",
}

_MOVEMENT_ALIASES: dict[str, dict] = {
    "roll_forward":   {"column": "curr_bucket", "op": "bucket_worse_than",  "value": "prev_bucket"},
    "rolled_forward": {"column": "curr_bucket", "op": "bucket_worse_than",  "value": "prev_bucket"},
    "worsened":       {"column": "curr_bucket", "op": "bucket_worse_than",  "value": "prev_bucket"},
    "roll_backward":  {"column": "curr_bucket", "op": "bucket_better_than", "value": "prev_bucket"},
    "rolled_backward":{"column": "curr_bucket", "op": "bucket_better_than", "value": "prev_bucket"},
    "improved":       {"column": "curr_bucket", "op": "bucket_better_than", "value": "prev_bucket"},
    "cured":          {"column": "curr_bucket", "op": "bucket_better_than", "value": "prev_bucket"},
    "bucket_stable":  {"column": "curr_bucket", "op": "bucket_stable",      "value": "prev_bucket"},
    "stable":         {"column": "curr_bucket", "op": "bucket_stable",      "value": "prev_bucket"},
}

_TOTAL_ALIASES = {"total_count", "total_cases", "all_count", "all_cases", "count"}


def _resolve_count_metric(name: str) -> dict | None:
    """Try to resolve an unknown metric name as a count pattern.

    Handles these classes (in order):
      1. total / all → count all rows
      2. {movement} / {movement}_count → bucket_worse/better_than where clause
      3. prev_{bucket}_count → count where prev_bucket == value
      4. {concept}_count → expand from CONCEPTS registry
      5. {bucket}_count → count where curr_bucket == value
      6. {status}_count → count where Loan Status == value

    Returns a partial measure dict (no alias  -  caller sets it), or None.
    """
    n = name.lower().strip()

    # 1. Total
    base = n[:-6] if n.endswith("_count") else n  # strip _count suffix if present
    if n in _TOTAL_ALIASES or base in {"total", "all", "total_cases"}:
        return {"agg": "count", "kind": "count"}

    # 2. Bucket movement (with or without _count suffix)
    for key, cond in _MOVEMENT_ALIASES.items():
        if n == key or n == f"{key}_count":
            return {"agg": "count", "kind": "count", "where": [cond]}

    # Only the suffix patterns below apply when name ends with _count
    if not n.endswith("_count"):
        return None

    prefix = n[:-6]  # strip _count

    # 3. prev_{bucket}_count → count where prev_bucket == bucket
    if prefix.startswith("prev_"):
        bucket_part = prefix[5:]
        bucket_val = _BUCKET_ALIASES.get(bucket_part)
        if bucket_val:
            return {"agg": "count", "kind": "count",
                    "where": [{"column": "prev_bucket", "op": "==", "value": bucket_val}]}

    # 4. {concept}_count → expand from registry
    concept = CONCEPTS.get(prefix)
    if concept:
        where = [
            {**cond, "value": _resolve_cutoff(cond["value"])}
            for cond in concept["conditions"]
        ]
        return {"agg": "count", "kind": "count", "where": where}

    # 5. {bucket}_count → count where curr_bucket == value
    bucket_val = _BUCKET_ALIASES.get(prefix)
    if bucket_val:
        return {"agg": "count", "kind": "count",
                "where": [{"column": "curr_bucket", "op": "==", "value": bucket_val}]}

    # 6. {status}_count → count where Loan Status == value
    status_val = _STATUS_ALIASES.get(prefix)
    if status_val:
        return {"agg": "count", "kind": "count",
                "where": [{"column": "Loan Status", "op": "==", "value": status_val}]}

    return None


def _default_measure_alias(m: dict) -> str:
    if m.get("concept"):
        return m["concept"]
    if m.get("distinct"):
        return _snake(m["distinct"])
    if m.get("column"):
        return _snake(m["column"])
    return "count"


def _infer_kind(d: dict):
    """Infer a measure kind from its shape (back-compat for IR-1 measures that
    carry no explicit 'kind'). 'additivity' is accepted as a synonym for 'kind'."""
    if d.get("kind"):
        return d["kind"]
    if d.get("additivity"):
        return d["additivity"]
    if "numerator" in d or "denominator" in d:
        return "ratio"
    if "distinct" in d or (d.get("agg") or "").lower() == "nunique":
        return "count_distinct"
    if (d.get("agg") or "").lower() == "count" or "concept" in d or "where" in d:
        return "count"
    if "column" in d:
        return "additive"
    return None


def _measure_def(m: dict, errs: list):
    """Resolve an IR measure into a uniform measure-definition (mdef): a metric ref
    is expanded from the registry; an inline measure is used directly. Concepts on
    a count measure are expanded to a concrete 'where' HERE (core owns concept
    knowledge  -  compiler.measures stays a pure lowering layer)."""
    if not isinstance(m, dict):
        errs.append(f"measure is not an object: {m!r}")
        return None

    if "metric" in m:
        md = METRICS.get(m["metric"])
        if md is None:
            # Unknown named metric  -  try to resolve as a count pattern before erroring.
            fallback = _resolve_count_metric(m["metric"])
            if fallback is not None:
                mdef = dict(fallback)
                mdef["alias"] = m.get("alias") or m["metric"]
                mdef["kind"] = _infer_kind(mdef)
                return mdef
            errs.append(f"unknown metric '{m['metric']}'")
            return None
        mdef = dict(md)
        mdef["alias"] = m.get("alias") or m["metric"]
        if m.get("agg"):
            mdef["agg"] = m["agg"]
    else:
        mdef = dict(m)
        mdef["alias"] = m.get("alias") or _default_measure_alias(m)

    mdef["kind"] = _infer_kind(mdef)
    if mdef["kind"] is None:
        errs.append(f"measure '{mdef.get('alias')}': cannot determine measure kind from {m!r}")
        return None

    # Expand concept/where (concepts -> full conditions) for count measures.
    if mdef["kind"] == "count":
        if "concept" in mdef:
            mdef["where"] = _expand_filters([{"concept": mdef["concept"]}], errs)
        elif "where" in mdef:
            mdef["where"] = _expand_filters(mdef["where"], errs)

    # Nested-grain measures are a Step 2 capability  -  reject loudly for now.
    if mdef.get("grain") and mdef["grain"] != "loan":
        errs.append(
            f"measure '{mdef['alias']}': grain '{mdef['grain']}' (nested aggregation) "
            "is not supported yet"
        )
    return mdef


def _resolve_measures(measures: list, ctx: dict, errs: list) -> tuple[list, list, list, list]:
    """Dispatch every measure to its kind handler and concatenate the lowered
    pieces. Returns (pre_derives, aggregations, post_derives, grains). Adding a new
    measure kind requires NO change here  -  only a new handler in compiler.measures."""
    pre: list = []
    aggs: list = []
    post: list = []
    grains: list = []
    for m in measures or []:
        mdef = _measure_def(m, errs)
        if mdef is None:
            continue
        handler = MEASURE_HANDLERS.get(mdef["kind"])
        if handler is None:
            errs.append(f"measure '{mdef.get('alias')}': unknown measure kind '{mdef['kind']}'")
            continue
        low = handler(mdef, ctx)
        pre.extend(low.pre_derives)
        aggs.extend(low.aggregations)
        post.extend(low.post_derives)
        errs.extend(low.errors)
        grains.append(mdef.get("grain", "loan"))
    return pre, aggs, post, grains


def grain_issues(measure_grains: list[str], dimensions: list[str]) -> list[str]:
    """Detect grain ambiguity: a measure defined at a grain COARSER than loan
    (e.g. a per-customer measure) grouped by a loan-level dimension it does not
    functionally determine (branch/region/executive). A customer can span several
    branches, so "this customer-grain number, by branch" has no single right
    answer  -  the compiler flags it so the pipeline can ask for clarification
    instead of returning a confident wrong number. Pure + unit-testable."""
    issues: list[str] = []
    for mg in measure_grains:
        if mg != "loan" and is_coarser(mg, "loan"):
            for d in dimensions:
                ent = ENTITIES.get(d)
                if ent and ent["kind"] == "dimension":
                    issues.append(
                        f"measure at '{mg}' grain grouped by loan-level dimension "
                        f"'{d}' is ambiguous (a {mg} can span multiple {d})"
                    )
    return issues


def _as_cols(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _resolve_time_compare(time_block: dict, ir_measures: list, errs: list) -> tuple[list, list]:
    """Expand a `time.compare` block into extra aggregations and optional derive steps.

    The LLM declares {"time": {"compare": {"type": "snapshot|change", "from": "prev",
    "to": "curr"}}} and the compiler deterministically maps that to concrete prev_*
    column aggregations from PREV_CARRYOVER_COLS  -  the LLM never sees or authors
    raw prev_* column names.

    Returns (extra_aggs, extra_post_derives):
    - extra_aggs    : additional group_aggregate entries (prev-period columns)
    - extra_derives : derive steps added AFTER the group_aggregate
                      (change = curr_alias - prev_alias, for type:"change")

    Measures whose source column has no prev_* mapping are silently skipped
    (not a fatal error  -  the result just won't have a prev column for that measure).
    Only type:"snapshot" and type:"change" with from:"prev", to:"curr" are supported;
    other period references return a loud error.
    """
    compare = time_block.get("compare") or {}
    frm = compare.get("from", "prev")
    to_ = compare.get("to", "curr")
    typ = compare.get("type", "snapshot")

    if frm != "prev" or to_ != "curr":
        errs.append(
            f"time.compare only supports from:'prev', to:'curr' "
            f"(got from:'{frm}', to:'{to_}'); multi-period windows require a data-layer change"
        )
        return [], []

    extra_aggs: list = []
    extra_derives: list = []
    local_errs: list = []  # non-fatal: only used to resolve mdef, not propagated

    for m in ir_measures:
        mdef = _measure_def(m, local_errs)
        if mdef is None:
            continue
        alias = mdef["alias"]
        kind = mdef["kind"]
        prev_alias = f"prev_{alias}"

        if kind in ("additive", "semi_additive"):
            col = mdef.get("column")
            prev_col = PREV_CARRYOVER_COLS.get(col) if col else None
            if not prev_col:
                continue  # no prev mapping for this column  -  skip gracefully
            func = (mdef.get("agg") or "sum").lower()
            extra_aggs.append({"alias": prev_alias, "func": func, "column": prev_col})
            if typ == "change":
                extra_derives.append({"column": f"{alias}_change", "expr": f"{alias} - {prev_alias}"})

        elif kind == "ratio":
            # Carry prev num/den sums through the same group_aggregate, then divide.
            num_cols = _as_cols(mdef.get("numerator"))
            den_cols = _as_cols(mdef.get("denominator"))
            prev_n, prev_d = [], []
            supported = True
            for i, c in enumerate(num_cols):
                pc = PREV_CARRYOVER_COLS.get(c)
                if not pc:
                    supported = False
                    break
                a = f"{prev_alias}__n{i}"
                extra_aggs.append({"alias": a, "func": "sum", "column": pc})
                prev_n.append(a)
            if supported:
                for i, c in enumerate(den_cols):
                    pc = PREV_CARRYOVER_COLS.get(c)
                    if not pc:
                        supported = False
                        break
                    a = f"{prev_alias}__d{i}"
                    extra_aggs.append({"alias": a, "func": "sum", "column": pc})
                    prev_d.append(a)
            if supported and prev_n and prev_d:
                scale = mdef.get("scale", 1)
                scale_s = f" * {scale}" if scale and scale != 1 else ""
                derive = {"column": prev_alias,
                          "expr": f"({' + '.join(prev_n)}) / ({' + '.join(prev_d)}){scale_s}"}
                if mdef.get("cap") is not None:
                    derive["clip_max"] = mdef["cap"]
                extra_derives.append(derive)
                if typ == "change":
                    extra_derives.append({"column": f"{alias}_change", "expr": f"{alias} - {prev_alias}"})

        # count / count_distinct: no natural prev version; skip silently

    return extra_aggs, extra_derives


def compile_logical(ir: dict, columns) -> tuple[list, list]:
    """Lower a logical IR into a physical step-plan. Returns (plan, errors).
    A non-empty errors list means the plan must NOT be executed (route to
    repair/clarification). The returned plan is also passed through the existing
    validate_plan so hallucinated columns are caught with the same gate the
    LLM-authored path uses."""
    errs: list[str] = []
    ir = ir or {}

    conditions = _expand_filters(ir.get("filters") or [], errs)
    dims_raw = ir.get("dimensions") or []
    group_by: list[str] = []
    for d in dims_raw:
        group_by.extend(resolve_dimension(d))

    entity_filters = ir.get("entity_filters") or []

    plan: list = []
    if conditions:
        plan.append({"op": "filter", "conditions": conditions})

    if entity_filters:
        # NESTED path: the compiler derives a multi-pass plan from the grain lattice.
        plan += _build_nested(ir, group_by, entity_filters, errs)
    elif group_by:
        # SINGLE-PASS path (Step 1 shapes).
        ctx = {"dimensions": dims_raw, "group_by": group_by,
               "time": ir.get("time"), "aggregating_over": set()}
        pre_derives, aggs, post_derives, measure_grains = _resolve_measures(
            ir.get("measures") or [], ctx, errs
        )
        errs.extend(grain_issues(measure_grains, dims_raw))  # hard stop, like an unknown concept

        # Time comparison: expand prev_* aggs and optional change derives.
        time_block = ir.get("time")
        if time_block and time_block.get("compare"):
            t_aggs, t_derives = _resolve_time_compare(time_block, ir.get("measures") or [], errs)
            aggs.extend(t_aggs)
            post_derives.extend(t_derives)

        for d in pre_derives:
            plan.append({"op": "derive", **d})
        if not aggs:
            aggs = [{"alias": "count", "func": "count"}]
        plan.append({"op": "group_aggregate", "group_by": group_by, "aggregations": aggs})
        for d in post_derives:
            plan.append({"op": "derive", **d})
        for d in ir.get("metrics") or []:
            expr, label = d.get("expr"), d.get("label") or d.get("alias")
            if expr and label:
                plan.append({"op": "derive", "column": label, "expr": expr})
        for h in ir.get("having") or []:
            alias = h.get("alias")
            if alias:
                plan.append({"op": "filter", "conditions": [
                    {"column": alias, "op": h.get("op") or ">=", "value": h.get("value", 0)}
                ]})
    elif not conditions:
        errs.append("empty query: no filters and no dimensions to compute")

    _append_sort_limit(ir, plan)

    # For loan_table queries, append a select step to subset visible columns.
    display_cols = ir.get("display_columns") or []
    if display_cols and ir.get("intent") in (None, "", "loan_table"):
        plan.append({"op": "select", "columns": display_cols})

    # Reuse the existing deterministic column validator as the final gate.
    if plan and not errs:
        errs.extend(validate_plan(plan, columns))

    return plan, errs


def _append_sort_limit(ir: dict, plan: list) -> None:
    for ob in ir.get("order_by") or []:
        # Accept any of the key names the model might use for the sort column.
        by = ob.get("by") or ob.get("measure") or ob.get("column") or ob.get("alias") or ob.get("field")
        if by:
            plan.append({"op": "sort", "by": by, "ascending": ob.get("dir", "desc") != "desc"})
    limit = ir.get("limit")
    if limit:
        plan.append({"op": "limit", "n": limit})


def _build_having_pred(pred: dict, alias: str, errs: list) -> tuple[dict, dict]:
    """Lower one entity HAVING predicate to (intermediate aggregation, filter cond).
    The aggregate is computed at the (output-dimension, entity) grain; the filter
    applies the threshold."""
    agg = (pred.get("agg") or "count").lower()
    a: dict = {"alias": alias}
    if agg == "count":
        a["func"] = "count"
        if "concept" in pred:
            a["where"] = _expand_filters([{"concept": pred["concept"]}], errs)
        elif "where" in pred:
            a["where"] = _expand_filters(pred["where"], errs)
    elif agg in ("nunique", "sum", "min", "max", "mean"):
        col = pred.get("distinct") or pred.get("column")
        if not col:
            errs.append(f"entity having predicate with agg '{agg}' needs a column")
        a["func"] = agg
        a["column"] = col
    else:
        errs.append(f"entity having predicate has unknown agg '{agg}'")
    cond = {"column": alias, "op": pred.get("op") or ">=", "value": pred.get("value", 1)}
    return a, cond


def _build_nested(ir: dict, group_by: list, entity_filters: list, errs: list) -> list:
    """Derive an intermediate→filter→terminal plan from the grain lattice. The LLM
    supplied only entity names + declarative predicates + measure names; ALL of the
    structure below (keys, pass order, rollup decomposition) is derived here."""
    if not group_by:
        errs.append("nested aggregation currently requires a grouping dimension")
        return []

    # Resolve entity-concept references to concrete {entity, having} predicates.
    resolved = []
    for ef in entity_filters:
        if "concept" in ef:
            ec = ENTITY_CONCEPTS.get(ef["concept"])
            if ec is None:
                errs.append(f"unknown entity concept '{ef['concept']}'")
                continue
            resolved.append(ec)
        else:
            resolved.append(ef)
    if not resolved:
        return []

    entities = {ef.get("entity") for ef in resolved}
    if len(entities) > 1:
        errs.append(f"nested aggregation over multiple entities {sorted(entities)} is not supported yet")
        return []
    g = next(iter(entities))
    if g not in ENTITIES:
        errs.append(f"unknown entity '{g}' in entity_filters")
        return []

    ekey = entity_key(g)
    inter_keys = group_by + ekey
    ctx = {"group_by": group_by, "intermediate_keys": set(inter_keys), "aggregating_over": set()}

    # Ambiguity: a predicate aggregating over a loan-level dimension key that is not
    # the output dimension is ambiguous (the entity can span it) -> clarify, not guess.
    for ef in resolved:
        for pred in ef.get("having") or []:
            pcol = pred.get("distinct") or pred.get("column")
            if pcol in _DIMENSION_KEY_COLS and pcol not in group_by:
                errs.append(
                    f"grain ambiguity: the per-{g} predicate aggregates over '{pcol}', "
                    f"which crosses the output dimension {group_by}; clarify whether "
                    "'{pcol}' should be counted globally per entity or within each group"
                )

    # Intermediate aggregations: predicate aggregates + measure rollup partials.
    inter_aggs: list = []
    pred_filters: list = []
    for i, ef in enumerate(resolved):
        for j, pred in enumerate(ef.get("having") or []):
            a, cond = _build_having_pred(pred, f"__ef{i}_{j}", errs)
            inter_aggs.append(a)
            pred_filters.append(cond)

    term_aggs: list = []
    term_post: list = []
    measures = ir.get("measures") or []
    for m in measures:
        mdef = _measure_def(m, errs)
        if mdef is None:
            continue
        rollup = MEASURE_ROLLUPS.get(mdef["kind"])
        if rollup is None:
            errs.append(f"measure '{mdef.get('alias')}': kind '{mdef['kind']}' is not nestable")
            continue
        r = rollup(mdef, ctx)
        inter_aggs.extend(r.intermediate_aggs)
        term_aggs.extend(r.terminal_aggs)
        term_post.extend(r.terminal_post_derives)
        errs.extend(r.errors)

    # No explicit measure -> count the distinct surviving entities per group.
    if not term_aggs and not measures:
        if len(ekey) == 1:
            term_aggs = [{"alias": "count", "func": "nunique", "column": ekey[0]}]
        else:
            errs.append(f"nested entity '{g}' has a composite key; an explicit measure is required")

    steps: list = [
        {"op": "group_aggregate", "group_by": inter_keys, "aggregations": inter_aggs},
        {"op": "filter", "conditions": pred_filters},
        {"op": "group_aggregate", "group_by": group_by, "aggregations": term_aggs},
    ]
    for d in term_post:
        steps.append({"op": "derive", **d})
    for d in ir.get("metrics") or []:
        expr, label = d.get("expr"), d.get("label") or d.get("alias")
        if expr and label:
            steps.append({"op": "derive", "column": label, "expr": expr})
    return steps

"""Registry foundation tests (Phase 0 of the v2 architecture).

These lock in two things:
1. ZERO BEHAVIOR CHANGE  -  migrating PRIORITY_RULES into registry/ontology.py must
   not alter the generated priority prompt section or the executor's rules.
2. REGISTRY INTEGRITY  -  every concept/metric/entity references a column that
   actually exists after load+bucketing, and the priority framework stays
   consistent with the concept ontology so they can't silently drift apart while
   both exist during the transition.
"""
import importlib
import re

from utils import CRITICAL_COLS, REQUIRED_COLS, PREV_CARRYOVER_COLS
from registry import (
    CONCEPTS, METRICS, PRIORITY_RULES, ENTITIES, DIMENSIONS, ENTITY_CONCEPTS, VIEWS,
    entity_key, is_coarser, resolve_dimension,
)

# Columns that exist only AFTER load + assign_buckets() / the prev-file merge,
# so they are legitimately not in the raw REQUIRED_COLS list.
_DERIVED_COLS = {
    "curr_bucket", "curr_score", "SOH", "prev_bucket",
    "Overdue", "MonthDemandExclPC", "OverdueCollected", "DemandCollected",
    "Overdue Collection %", "Month Demand Collection %",
} | set(PREV_CARRYOVER_COLS.values())
# The dynamic cutoff placeholder is resolved at execution time, not a real value.
_DYNAMIC_VALUES = {"__CUTOFF_1Y__"}

_KNOWN_COLS = set(CRITICAL_COLS) | set(REQUIRED_COLS) | _DERIVED_COLS

# Which concept each priority tier corresponds to (consistency guard).
_PRIORITY_TO_CONCEPT = {
    "Non Starters":                  "non_starter",
    "Easy Settlements":              "easy_settlement",
    "Recent Advances - High Bucket": "recent_advance_high_bucket",
    "Insurance-Driven Delinquency":  "insurance_driven_delinquency",
    "Co-lending at Risk":            "colending_at_risk",
    "No Collection 3 Months":        "no_collection_3m",
    "NPA Accounts":                  "npa",
}


class TestZeroBehaviorChange:
    def test_priority_rules_reexported_identically(self):
        # The migrated registry list IS what agents.domain_expert exposes  -  no copy,
        # no drift. execute_priority_mode imports it lazily from domain_expert.
        from agents.domain_expert import PRIORITY_RULES as DE_RULES
        assert DE_RULES is PRIORITY_RULES

    def test_priority_text_unchanged(self):
        # The generated prompt section must still list all 7 tiers in rank order
        # with their exact labels  -  this is the text the LLM sees.
        from agents.domain_expert import _build_priority_text
        text = _build_priority_text()
        for r in PRIORITY_RULES:
            assert f"Priority {r['rank']} - {r['label']}" in text
        assert text.count("Priority ") == len(PRIORITY_RULES) == 7

    def test_priority_mode_still_imports(self):
        # Guard the lazy import path used by the executor at runtime.
        from agents.data_executor import execute_priority_mode  # noqa: F401


class TestPriorityConceptConsistency:
    def test_each_priority_tier_matches_its_concept(self):
        # While PRIORITY_RULES and CONCEPTS both exist, their shared definitions
        # must be identical so the two can't drift during the v2 transition.
        for label, concept_name in _PRIORITY_TO_CONCEPT.items():
            rule = next(r for r in PRIORITY_RULES if r["label"] == label)
            assert concept_name in CONCEPTS, f"missing concept {concept_name}"
            assert rule["conditions"] == CONCEPTS[concept_name]["conditions"], (
                f"priority tier '{label}' drifted from concept '{concept_name}'"
            )


class TestConceptIntegrity:
    def test_concept_conditions_reference_real_columns(self):
        for name, c in CONCEPTS.items():
            assert c.get("conditions"), f"concept {name} has no conditions"
            for cond in c["conditions"]:
                col = cond["column"]
                assert col in _KNOWN_COLS, f"concept {name}: unknown column '{col}'"
                assert cond.get("op"), f"concept {name}: condition missing op"

    def test_concept_required_fields(self):
        for name, c in CONCEPTS.items():
            assert c.get("label"), f"concept {name} missing label"
            assert c.get("description"), f"concept {name} missing description"


class TestMetricIntegrity:
    def test_metric_columns_exist(self):
        # A metric is column-based (additive/semi-additive), numerator/denominator
        # COLUMNS (ratio), or numerator/denominator WHERE-CLAUSES (count_ratio --
        # a count-of-rows-matching-a-condition ratio, e.g. hard_bucket_pct/
        # strike_pct, distinct from "ratio"'s sum-of-a-column shape). Validate
        # whichever columns each shape references.
        for name, m in METRICS.items():
            cols = []
            if "column" in m:
                cols.append(m["column"])
            for key in ("numerator", "denominator"):
                v = m.get(key)
                if isinstance(v, (list, tuple)):
                    cols.extend(v)
                elif v:
                    cols.append(v)
            for key in ("numerator_where", "denominator_where"):
                for cond in m.get(key) or []:
                    cols.append(cond["column"])
            if m.get("kind") == "count_ratio":
                # denominator_where may legitimately be [] ("count all rows") --
                # only require at least the numerator side to reference something.
                assert m.get("numerator_where"), f"metric {name}: count_ratio needs numerator_where"
            else:
                assert cols, f"metric {name}: no column / numerator / denominator"
            for c in cols:
                assert c in _KNOWN_COLS, f"metric {name}: unknown column '{c}'"

    def test_metric_grain_is_known_entity(self):
        for name, m in METRICS.items():
            assert m["grain"] in ENTITIES, f"metric {name}: grain '{m['grain']}' is not an entity"

    def test_every_metric_has_a_kind(self):
        for name, m in METRICS.items():
            assert m.get("kind"), f"metric {name} missing 'kind'"


class TestSemanticModel:
    def test_entity_keys_are_real_columns(self):
        for name in ENTITIES:
            for col in entity_key(name):
                assert col in _KNOWN_COLS, f"entity {name}: key column '{col}' unknown"

    def test_dimension_aliases_resolve_to_real_columns(self):
        for alias in DIMENSIONS:
            for col in resolve_dimension(alias):
                assert col in _KNOWN_COLS, f"dimension {alias}: column '{col}' unknown"

    def test_customer_is_coarser_than_loan(self):
        # The grain fact that lets the compiler derive nested rollups.
        assert is_coarser("customer", "loan")
        assert not is_coarser("loan", "customer")

    def test_branch_alias_maps_to_unit(self):
        assert resolve_dimension("branch") == ["Unit"]
        assert resolve_dimension("executive") == ["MNT NAME", "Unit"]


class TestEntityConceptIntegrity:
    """ENTITY_CONCEPTS (per-entity rollup predicates, e.g. fleet_operator) --
    previously untested; the CONCEPTS/METRICS suites above don't cover this dict
    at all, so a typo'd column/entity here would only surface at query time."""

    def test_entity_concepts_reference_known_entity(self):
        for name, ec in ENTITY_CONCEPTS.items():
            assert ec.get("entity") in ENTITIES, f"entity concept {name}: unknown entity '{ec.get('entity')}'"

    def test_entity_concepts_having_reference_real_columns(self):
        for name, ec in ENTITY_CONCEPTS.items():
            having = ec.get("having")
            assert having, f"entity concept {name} has no having predicates"
            for pred in having:
                col = pred.get("distinct") or pred.get("column")
                assert col in _KNOWN_COLS, f"entity concept {name}: unknown column '{col}'"
                assert pred.get("op"), f"entity concept {name}: predicate missing op"

    def test_entity_concepts_required_fields(self):
        for name, ec in ENTITY_CONCEPTS.items():
            assert ec.get("label"), f"entity concept {name} missing label"
            assert ec.get("description"), f"entity concept {name} missing description"


class TestCrossRegistryNameCollisions:
    """The planner picks a name from one of FOUR parallel vocabularies (CONCEPTS,
    METRICS, ENTITY_CONCEPTS, VIEWS) based on context (filters vs measures vs
    entity_filters vs the top-level "view" field). A name reused across two of
    these is a real confusion risk we've hit live (fleet_operator misrouted
    between CONCEPTS/ENTITY_CONCEPTS; "pulse_kpis" -- a VIEW name -- referenced
    as if it were a METRIC). This guards against ever reintroducing that class
    of ambiguity as the vocabularies keep growing."""

    def test_no_name_shared_across_registries(self):
        registries = {
            "CONCEPTS": set(CONCEPTS), "METRICS": set(METRICS),
            "ENTITY_CONCEPTS": set(ENTITY_CONCEPTS), "VIEWS": set(VIEWS),
        }
        names = list(registries.items())
        for i, (name_a, keys_a) in enumerate(names):
            for name_b, keys_b in names[i + 1:]:
                overlap = keys_a & keys_b
                assert not overlap, f"name(s) {overlap} exist in both {name_a} and {name_b}"


class TestViewsIntegrity:
    """VIEWS entries are only ever exercised at runtime by whichever query happens
    to match them -- a typo'd 'fn' dotted path or 'subkey' would sit undetected
    until a real user hits that exact view. Resolve every one at test time."""

    def test_every_view_fn_resolves(self):
        from registry.views import resolve_view_fn
        for name in VIEWS:
            fn = resolve_view_fn(name)
            assert callable(fn), f"view {name}: 'fn' did not resolve to a callable"

    def test_every_view_has_required_fields(self):
        _VALID_OUTPUTS = {"df", "df_dict_tuple", "dict_with_top_df", "matrix_tuple",
                           "tuple_df_fig", "list_of_dicts", "dict_subkey_df", "good_bad_dict"}
        _VALID_INPUTS = {"df_curr", "df_prev", "rr_meta"}
        _VALID_GRAINS = {"loan", "customer", "region", "branch", "executive",
                          "segment", "signal", "matrix", "portfolio"}
        for name, spec in VIEWS.items():
            assert spec.get("label"), f"view {name} missing label"
            assert spec.get("description"), f"view {name} missing description"
            assert spec.get("fn"), f"view {name} missing fn"
            assert spec.get("output") in _VALID_OUTPUTS, f"view {name}: unknown output kind '{spec.get('output')}'"
            assert spec.get("grain") in _VALID_GRAINS, f"view {name}: missing or unknown 'grain' '{spec.get('grain')}'"
            for inp in spec.get("inputs") or []:
                assert inp in _VALID_INPUTS, f"view {name}: unknown input '{inp}'"
            if spec.get("output") == "dict_subkey_df":
                assert spec.get("subkey"), f"view {name}: dict_subkey_df output requires a 'subkey'"
            for req in spec.get("requires") or []:
                assert req in _VALID_INPUTS, f"view {name}: unknown 'requires' entry '{req}'"

    def test_highlightable_metrics_have_known_direction_and_label_col(self):
        # Fix B: a view declaring "metrics" (highlight_metrics targets) must also
        # declare "label_col" (view_node needs it to build a card caption), and
        # every metric column must have a known good/bad direction -- otherwise
        # view_node would silently drop it and the planner's request would go
        # nowhere, which is safe but should never happen for a column we ourselves
        # declared highlightable.
        from registry.views import _METRIC_DIRECTION
        for name, spec in VIEWS.items():
            metrics = spec.get("metrics")
            if not metrics:
                continue
            assert spec.get("label_col"), f"view {name}: has 'metrics' but no 'label_col'"
            for col in metrics:
                assert col in _METRIC_DIRECTION, f"view {name}: metric '{col}' has no direction in _METRIC_DIRECTION"

    def test_highlightable_metrics_have_known_aggregation_kind(self):
        # Portfolio KPI rollup: every declared metric column must know whether
        # a portfolio-wide summary should mean() or sum() it -- otherwise
        # _build_portfolio_kpis silently drops it (safe, but should never
        # happen for a column we ourselves declared highlightable).
        from registry.views import _METRIC_AGG
        for name, spec in VIEWS.items():
            for col in spec.get("metrics") or []:
                assert col in _METRIC_AGG, f"view {name}: metric '{col}' has no aggregation kind in _METRIC_AGG"
                assert _METRIC_AGG[col] in ("mean", "sum")

    def test_view_module_paths_resolve_without_importing_agents_or_compiler(self):
        # analysis/ must stay a leaf dependency (no import cycle risk) -- every
        # VIEWS 'fn' should live under the analysis package.
        for name, spec in VIEWS.items():
            assert spec["fn"].startswith("analysis."), f"view {name}: fn '{spec['fn']}' is not under analysis."
            module_path = spec["fn"].rsplit(".", 1)[0]
            importlib.import_module(module_path)  # raises if it doesn't exist

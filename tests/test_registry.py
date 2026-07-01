"""Registry foundation tests (Phase 0 of the v2 architecture).

These lock in two things:
1. ZERO BEHAVIOR CHANGE  -  migrating PRIORITY_RULES into registry/ontology.py must
   not alter the generated priority prompt section or the executor's rules.
2. REGISTRY INTEGRITY  -  every concept/metric/entity references a column that
   actually exists after load+bucketing, and the priority framework stays
   consistent with the concept ontology so they can't silently drift apart while
   both exist during the transition.
"""
import re

from utils import CRITICAL_COLS, REQUIRED_COLS, PREV_CARRYOVER_COLS
from registry import (
    CONCEPTS, METRICS, PRIORITY_RULES, ENTITIES, DIMENSIONS,
    entity_key, is_coarser, resolve_dimension,
)

# Columns that exist only AFTER load + assign_buckets() / the prev-file merge,
# so they are legitimately not in the raw REQUIRED_COLS list.
_DERIVED_COLS = {"curr_bucket", "curr_score", "SOH", "prev_bucket"} | set(PREV_CARRYOVER_COLS.values())
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
        # A metric is either column-based (additive/semi-additive) or defined by a
        # numerator/denominator (ratio). Validate whichever columns it references.
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

"""ui/tabs/investigator.py::_execute_step wiring, for EVERY registered step.

Found via audit (assistant_feature.md): the dispatcher was only ever checked
by a one-off script, so a newly-added step with no dispatch branch -- or a
branch passing a kwarg the step function doesn't accept -- surfaced only when
someone clicked through the live app. Each STEP_REGISTRY function is replaced
by a stub that binds the dispatcher's actual call against the REAL function's
signature, so argument drift fails here without needing realistic data.
"""
import inspect

import pandas as pd
import pytest

from investigator.state import EntityMemory
from investigator.steps import STEP_REGISTRY
from ui.tabs.investigator import _execute_step

# Minimal params per step -- the smallest shape parse_turn can legally emit.
_MIN_PARAMS = {
    "dimension_breakdown":       {"dimension": "branch"},
    "roll_rate_summary":         {},
    "roll_rate_by_dimension":    {"dimension": "branch"},
    "vintage_summary":           {},
    "demand_summary":            {"dimension": "branch"},
    "new_advances_summary":      {},
    "new_advances_by_dimension": {"dimension": "branch"},
    "new_advances_trend":        {},
    "product_analysis":          {"axis": "segment"},
    "top_accounts":              {"all_columns": True},
    "fleet_exposure":            {},
    "repossession_list":         {"all_columns": True},
    "good_customers":            {"all_columns": True},
    "entity_summary":            {"entity_type": "branch", "entity_value": "MAHAD"},
    "underperformer_quartile":   {"metric": "Collection %"},
    "bottom_n_per_group":        {"metric": "Collection %", "group_col": "branch"},
    "executive_roster":          {},
    "intersect_entities":        {"source_entities": []},
    "non_paying_customers":      {"source_entities": [], "all_columns": True},
    "customer_loan_book":        {"cust_mob_no": "9876543210"},
    "concept_filter":            {"concept": "npa"},
    "concept_breakdown":         {"source_entities": [], "dimension": "branch"},
    "priority_accounts":         {"all_columns": True},
    "priority_menu":             {},
    "priority_by_executive":     {"source_entities": []},
    "top_closing_arrears":       {},
    "high_arrears_at_risk":      {},
    "fleet_defaulters":          {},
    "worst_loans_by_metric":     {"metric": "NPA %", "all_columns": True},
}


def test_min_params_table_covers_exactly_the_registry():
    # A new step added to STEP_REGISTRY must be added here too, forcing a
    # dispatch test for it instead of silently skipping it.
    assert set(_MIN_PARAMS) == set(STEP_REGISTRY)


@pytest.mark.parametrize("step_type", sorted(STEP_REGISTRY))
def test_every_step_dispatches_with_a_signature_compatible_call(step_type, monkeypatch):
    real_fn = STEP_REGISTRY[step_type]
    seen = {}

    def stub(*args, **kwargs):
        # TypeError here = the dispatcher passes an argument the real step
        # function doesn't accept (or omits a required one).
        seen["bound"] = inspect.signature(real_fn).bind(*args, **kwargs)
        return pd.DataFrame()

    monkeypatch.setitem(STEP_REGISTRY, step_type, stub)
    out = _execute_step(
        step_type, dict(_MIN_PARAMS[step_type]), pd.DataFrame({"Loan No": ["L1"]}),
        pd.DataFrame(), EntityMemory(), as_of="2026-03",
    )
    assert isinstance(out, pd.DataFrame)
    assert "bound" in seen, f"{step_type}: dispatcher never called the step function"
    # all_columns must actually reach the steps that support it.
    if _MIN_PARAMS[step_type].get("all_columns"):
        assert seen["bound"].arguments.get("all_columns") is True


def test_unknown_step_type_raises_clearly():
    with pytest.raises(KeyError):
        _execute_step("not_a_step", {}, pd.DataFrame(), pd.DataFrame(), EntityMemory())

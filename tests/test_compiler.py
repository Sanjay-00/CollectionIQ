"""Compiler tests (Phase 1 of the v2 architecture).

Central guarantee: a logical IR that *references* a multi-condition concept by
name compiles to a physical plan containing the concept's COMPLETE condition set  - 
so the silent dropped-condition bug (which prompted this redesign) cannot recur on
the compiler path. Co-lending-at-risk is the canonical case.
"""
import pandas as pd

from compiler import compile_logical, grain_issues
from agents.plan_executor import execute_plan


def _df():
    """Branch A has one true co-lending-at-risk loan (L1, SOH 100); L2 is
    co-lending but NOT at-risk (no arrears); L3 is at-risk but NOT co-lending.
    So the correct A exposure is 100  -  and it differs from the number you'd get
    if EITHER condition were dropped (150 if arrears dropped, 300 if co-lending
    dropped). Branch B's single at-risk loan is L4 (SOH 300)."""
    return pd.DataFrame({
        "Loan No":         [1, 2, 3, 4, 5],
        "Unit":            ["A", "A", "A", "B", "B"],
        "MNT NAME":        ["x", "x", "y", "z", "z"],
        "CoLending_Loans": ["Y", "Y", "N", "Y", "N"],
        "Arrears / EMI":   [2, 0, 5, 3, 0],
        "SOH":             [100.0, 50.0, 200.0, 300.0, 10.0],
        "RegionName":      ["PUNE", "PUNE", "PUNE", "MUM", "MUM"],
    })


class TestConceptExpansion:
    def test_colending_at_risk_expands_to_both_conditions(self):
        plan, errs = compile_logical(
            {"filters": [{"concept": "colending_at_risk"}]}, _df().columns
        )
        assert errs == []
        conds = plan[0]["conditions"]
        cols = {c["column"] for c in conds}
        # BOTH legs must be present  -  this is the anti-dropped-condition guarantee.
        assert cols == {"CoLending_Loans", "Arrears / EMI"}

    def test_unknown_concept_is_loud_error(self):
        plan, errs = compile_logical({"filters": [{"concept": "nope"}]}, _df().columns)
        assert any("unknown concept 'nope'" in e for e in errs)

    def test_raw_condition_passes_through(self):
        plan, errs = compile_logical(
            {"filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}]},
            _df().columns,
        )
        assert errs == []
        assert plan[0]["conditions"][0]["column"] == "RegionName"


class TestEndToEndAgainstDroppedCondition:
    def test_per_branch_exposure_is_correct_not_dropped(self):
        ir = {
            "intent": "aggregation",
            "filters": [{"concept": "colending_at_risk"}],
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure"}],
            "order_by": [{"by": "exposure", "dir": "desc"}],
        }
        df = _df()
        plan, errs = compile_logical(ir, df.columns)
        assert errs == []
        out, err = execute_plan(df, plan)
        assert err == ""
        a = out[out["Unit"] == "A"].iloc[0]["exposure"]
        b = out[out["Unit"] == "B"].iloc[0]["exposure"]
        # Correct, full-condition answer:
        assert a == 100.0
        assert b == 300.0
        # Explicitly NOT the dropped-condition wrong answers (150 / 300):
        assert a != 150.0  # would be L1+L2 if arrears leg dropped
        assert a != 300.0  # would be L1+L3 if co-lending leg dropped


class TestResolution:
    def test_dimension_aliases_resolve(self):
        plan, errs = compile_logical(
            {"dimensions": ["branch"], "measures": [{"metric": "exposure"}]}, _df().columns
        )
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        assert ga["group_by"] == ["Unit"]

    def test_executive_dimension_is_name_plus_unit(self):
        plan, errs = compile_logical(
            {"dimensions": ["executive"], "measures": [{"metric": "exposure"}]}, _df().columns
        )
        ga = next(s for s in plan if s["op"] == "group_aggregate")
        assert ga["group_by"] == ["MNT NAME", "Unit"]

    def test_metric_resolves_to_column_and_default_agg(self):
        plan, errs = compile_logical(
            {"dimensions": ["branch"], "measures": [{"metric": "exposure"}]}, _df().columns
        )
        agg = next(s for s in plan if s["op"] == "group_aggregate")["aggregations"][0]
        assert agg == {"alias": "exposure", "func": "sum", "column": "SOH"}

    def test_dimension_only_defaults_to_count(self):
        plan, errs = compile_logical({"dimensions": ["branch"]}, _df().columns)
        agg = next(s for s in plan if s["op"] == "group_aggregate")["aggregations"][0]
        assert agg["func"] == "count"


class TestValidatorGate:
    def test_missing_metric_column_is_caught_by_validate_plan(self):
        # exposure -> SOH; if SOH isn't in the data, the existing validate_plan
        # gate must flag it (same gate the LLM-authored path uses).
        cols = ["Loan No", "Unit", "CoLending_Loans", "Arrears / EMI"]  # no SOH
        plan, errs = compile_logical(
            {"dimensions": ["branch"], "measures": [{"metric": "exposure"}]}, cols
        )
        assert any("SOH" in e for e in errs)

    def test_empty_query_is_rejected(self):
        plan, errs = compile_logical({}, _df().columns)
        assert any("empty query" in e for e in errs)


class TestGrainAmbiguity:
    def test_coarse_measure_by_loan_dimension_is_flagged(self):
        # A customer-grain measure grouped by branch is ambiguous (customer spans branches).
        assert grain_issues(["customer"], ["branch"]) != []

    def test_loan_measure_by_branch_is_fine(self):
        assert grain_issues(["loan"], ["branch"]) == []

    def test_coarse_measure_flows_through_compile_as_error(self):
        # Integration: a measure whose metric is defined at customer grain, grouped
        # by branch, must surface a grain-ambiguity error FROM compile_logical (not
        # just the standalone detector). Inject a temporary coarse metric.
        from registry.ontology import METRICS
        METRICS["cust_exposure"] = {
            "label": "Customer Exposure", "column": "SOH",
            "default_agg": "sum", "grain": "customer", "description": "test-only",
        }
        try:
            plan, errs = compile_logical(
                {"dimensions": ["branch"], "measures": [{"metric": "cust_exposure"}]},
                _df().columns,
            )
            assert any("ambiguous" in e for e in errs)
        finally:
            del METRICS["cust_exposure"]

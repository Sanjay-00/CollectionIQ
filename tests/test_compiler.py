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


class TestShowAllColumns:
    """Regression: a real "give me all cases ... with all columns including X
    and Y" query got a hand-built ~28-column list instead of every column,
    because the Logical Planner's OWN "include named evidence columns" rule
    fired instead of its "leave display_columns empty for all-columns
    requests" rule -- a prompt-only instruction isn't a guarantee. show_all_
    columns is a deterministic IR-1 field the compiler enforces regardless of
    whatever the model also put in display_columns."""

    def test_show_all_columns_true_skips_select_even_with_display_columns_set(self):
        ir1 = {
            "intent": "loan_table",
            "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}],
            "display_columns": ["Loan No", "SOH"],  # model built a list anyway
            "show_all_columns": True,
        }
        plan, errs = compile_logical(ir1, list(_df().columns))
        assert errs == []
        assert not any(step.get("op") == "select" for step in plan)

        out, exec_err = execute_plan(_df(), plan)
        assert exec_err == ""
        # Every original column present, not just the model's hand-built list.
        assert set(_df().columns) <= set(out.columns)

    def test_show_all_columns_false_still_applies_display_columns(self):
        ir1 = {
            "intent": "loan_table",
            "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}],
            "display_columns": ["Loan No", "SOH"],
            "show_all_columns": False,
        }
        plan, errs = compile_logical(ir1, list(_df().columns))
        assert errs == []
        select_steps = [s for s in plan if s.get("op") == "select"]
        assert len(select_steps) == 1
        assert select_steps[0]["columns"] == ["Loan No", "SOH"]

    def test_show_all_columns_defaults_to_false_when_absent(self):
        # Older/hand-built IR-1 dicts without the field at all must behave
        # exactly as before -- display_columns still applies normally.
        ir1 = {"intent": "loan_table", "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}], "display_columns": ["Loan No"]}
        plan, errs = compile_logical(ir1, list(_df().columns))
        assert errs == []
        assert any(step.get("op") == "select" and step["columns"] == ["Loan No"] for step in plan)


class TestLoanTableIgnoresDimensions:
    """Regression: a real query ("...regionwise and branchwise... give all
    cases individually and give all columns") pattern-matched BOTH the
    Logical Planner's aggregation routing rule ("by/per/across branch|
    region") and its loan_table one ("give all cases individually"). The
    model kept intent="loan_table" but still populated dimensions/measures --
    contradicting its own prompt, which says loan_table needs neither.
    Without this guard, the stray dimensions list forced the SINGLE-PASS/
    group_aggregate branch, collapsing the per-loan frame to one row per
    region/branch; a later order_by on a raw per-loan column ("Closing
    Arrears") then failed to compile because that column no longer existed
    post-aggregation. loan_table must mean individual rows regardless of
    what dimensions/measures the model also attached."""

    def test_dimensions_and_measures_ignored_when_intent_is_loan_table(self):
        ir1 = {
            "intent": "loan_table",
            "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}],
            "dimensions": ["region", "branch"],
            "measures": [{"column": "SOH", "agg": "sum", "alias": "total_soh"}],
            "order_by": [{"by": "SOH", "dir": "desc"}],
            "show_all_columns": True,
        }
        plan, errs = compile_logical(ir1, list(_df().columns))
        assert errs == []
        assert not any(step.get("op") == "group_aggregate" for step in plan)
        assert any(step.get("op") == "sort" and step["by"] == "SOH" for step in plan)

        out, exec_err = execute_plan(_df(), plan)
        assert exec_err == ""
        # Per-loan rows survive (not collapsed to one row per region/branch).
        assert len(out) == len(_df()[_df()["RegionName"] == "PUNE"])
        assert "SOH" in out.columns

    def test_dimensions_still_drive_group_by_for_aggregation_intent(self):
        # Guard is intent-specific -- a genuine aggregation query must keep
        # working exactly as before.
        ir1 = {
            "intent": "aggregation",
            "dimensions": ["region"],
            "measures": [{"column": "SOH", "agg": "sum", "alias": "total_soh"}],
        }
        plan, errs = compile_logical(ir1, list(_df().columns))
        assert errs == []
        assert any(step.get("op") == "group_aggregate" for step in plan)


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

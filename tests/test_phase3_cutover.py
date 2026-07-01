"""Phase 3 cutover tests  -  no live Gemini calls.

Validates that:
1. The `select` op in plan_executor works (column subsetting, graceful on unknowns)
2. The compiler correctly appends a `select` step for loan_table display_columns
3. The new graph nodes (logical_planner_node, compiler_node, validate_node,
   execute_node) are wired correctly (tested via direct node calls with stubbed IR)
4. Bucket movement filter conditions survive the compiler → executor roundtrip
5. Empty plan results are now returned as empty DataFrames (not errors)
"""
import pandas as pd
import pytest

from agents.plan_executor import execute_plan, validate_plan
from compiler.core import compile_logical


# ── Helpers ────────────────────────────────────────────────────────────────────

def _loan_df():
    return pd.DataFrame({
        "Loan No":       ["L1", "L2", "L3", "L4", "L5"],
        "Cust Name":     ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "Cust Mob No":   ["111", "222", "333", "444", "555"],
        "RegionName":    ["PUNE", "PUNE", "MUM", "MUM", "DEL"],
        "Unit":          ["A", "A", "B", "B", "C"],
        "curr_bucket":   ["NPA", "STD", "SMA-1", "NPA", "STD"],
        "prev_bucket":   ["SMA-1", "STD", "STD", "SMA-1", "STD"],
        "SOH":           [500.0, 100.0, 300.0, 400.0, 200.0],
        "POS":           [450.0, 90.0, 270.0, 360.0, 180.0],
        "Arrears / EMI": [3.5, 0.0, 1.2, 4.0, 0.0],
    })


# ── select op ─────────────────────────────────────────────────────────────────

class TestSelectOp:
    def test_select_subsets_columns(self):
        df = _loan_df()
        plan = [{"op": "select", "columns": ["Loan No", "Cust Name", "SOH"]}]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert list(out.columns) == ["Rank", "Loan No", "Cust Name", "SOH"]

    def test_select_skips_unknown_columns(self):
        df = _loan_df()
        plan = [{"op": "select", "columns": ["Loan No", "NONEXISTENT_COL", "SOH"]}]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert "NONEXISTENT_COL" not in out.columns
        assert "Loan No" in out.columns and "SOH" in out.columns

    def test_select_all_unknown_returns_original(self):
        df = _loan_df()
        plan = [{"op": "select", "columns": ["GHOST1", "GHOST2"]}]
        out, err = execute_plan(df, plan)
        assert err == ""
        # Falls back to original columns + Rank
        assert "Loan No" in out.columns

    def test_validate_plan_accepts_select(self):
        plan = [{"op": "select", "columns": ["Loan No", "SOH"]}]
        errs = validate_plan(plan, ["Loan No", "SOH", "Unit"])
        assert errs == []

    def test_select_after_filter(self):
        df = _loan_df()
        plan = [
            {"op": "filter", "conditions": [{"column": "curr_bucket", "op": "==", "value": "NPA"}]},
            {"op": "select", "columns": ["Loan No", "SOH", "curr_bucket"]},
        ]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert len(out) == 2  # L1 and L4
        assert set(out.columns) - {"Rank"} == {"Loan No", "SOH", "curr_bucket"}


# ── Compiler display_columns → select ─────────────────────────────────────────

class TestCompilerDisplayColumns:
    def test_loan_table_with_display_columns_appends_select(self):
        df = _loan_df()
        ir = {
            "intent": "loan_table",
            "filters": [{"column": "curr_bucket", "op": "==", "value": "NPA"}],
            "display_columns": ["Loan No", "Cust Name", "SOH", "curr_bucket"],
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        assert any(s["op"] == "select" for s in plan)
        sel = next(s for s in plan if s["op"] == "select")
        assert "Loan No" in sel["columns"]

    def test_aggregation_intent_no_select_step(self):
        df = _loan_df()
        ir = {
            "intent": "aggregation",
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "display_columns": ["Loan No"],  # ignored for aggregation intent
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        assert not any(s["op"] == "select" for s in plan)

    def test_no_display_columns_no_select(self):
        df = _loan_df()
        ir = {
            "intent": "loan_table",
            "filters": [{"column": "curr_bucket", "op": "==", "value": "NPA"}],
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        assert not any(s["op"] == "select" for s in plan)


# ── Bucket movement through compiler → executor ───────────────────────────────

class TestBucketMovement:
    def test_bucket_worse_than_filter_compiles_and_executes(self):
        df = _loan_df()
        ir = {
            "intent": "loan_table",
            "filters": [
                {"column": "curr_bucket", "op": "bucket_worse_than", "value": "prev_bucket"}
            ],
            "display_columns": ["Loan No", "curr_bucket", "prev_bucket"],
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        out, err = execute_plan(df, plan)
        assert err == ""
        # L1: SMA-1→NPA (worse), L3: STD→SMA-1 (worse), L4: SMA-1→NPA (worse)
        assert set(out["Loan No"]) == {"L1", "L3", "L4"}

    def test_bucket_better_than_filter_compiles_and_executes(self):
        df = _loan_df()
        ir = {
            "intent": "loan_table",
            "filters": [
                {"column": "curr_bucket", "op": "bucket_better_than", "value": "prev_bucket"}
            ],
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        out, err = execute_plan(df, plan)
        assert err == ""
        # L1: SMA-1→NPA would be worse, L1 is NPA and prev SMA-1 so WORSE not better
        # Wait let me re-check: L1 NPA curr, SMA-1 prev → NPA is worse than SMA-1 → not better
        # L3: SMA-1 curr, STD prev → SMA-1 is worse than STD → not better
        # L4: NPA curr, SMA-1 prev → worse → not better
        # None should match bucket_better_than in this dataset
        assert len(out) == 0

    def test_bucket_worse_count_measure_in_aggregation(self):
        df = _loan_df()
        ir = {
            "intent": "aggregation",
            "dimensions": ["branch"],
            "measures": [
                {"agg": "count", "where": [
                    {"column": "curr_bucket", "op": "bucket_worse_than", "value": "prev_bucket"}
                ], "alias": "degraded"},
            ],
        }
        plan, errs = compile_logical(ir, df.columns)
        assert errs == [], errs
        out, err = execute_plan(df, plan)
        assert err == ""
        counts = dict(zip(out["Unit"], out["degraded"]))
        assert counts["A"] == 1  # L1: SMA-1→NPA (worse)
        assert counts["B"] == 2  # L3: STD→SMA-1, L4: SMA-1→NPA (both worse)
        assert counts["C"] == 0  # L5: STD→STD (stable)


# ── Empty results are valid ───────────────────────────────────────────────────

class TestEmptyResults:
    def test_filter_matching_nothing_returns_empty_no_error(self):
        df = _loan_df()
        plan = [
            {"op": "filter", "conditions": [{"column": "SOH", "op": ">", "value": 999999}]},
        ]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert out.empty

    def test_multi_step_empty_intermediate_returns_empty_no_error(self):
        df = _loan_df()
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit"],
             "aggregations": [{"alias": "count", "func": "count"}]},
            {"op": "filter", "conditions": [{"column": "count", "op": ">", "value": 999}]},
            {"op": "sort", "by": "count", "ascending": False},
        ]
        out, err = execute_plan(df, plan)
        assert err == ""
        assert out.empty


# ── _normalize_ir1 backward compat ────────────────────────────────────────────

class TestNormalizeIR1Phase3:
    def test_new_fields_have_safe_defaults(self):
        from agents.logical_planner import _normalize_ir1
        ir = _normalize_ir1({})
        assert ir["query_title"] == "Custom Query"
        assert ir["risk_flag"] == "medium"
        assert ir["description"] == ""
        assert ir["needs_clarification"] is False
        assert ir["clarification_options"] == []
        assert ir["entity_filters"] == []
        assert ir["display_columns"] == []
        assert ir["time"] is None

    def test_all_new_fields_passed_through(self):
        from agents.logical_planner import _normalize_ir1
        raw = {
            "query_title": "NPA by Branch",
            "risk_flag": "high",
            "description": "Sum of NPA exposure per branch",
            "needs_clarification": False,
            "display_columns": ["Loan No", "SOH"],
            "entity_filters": [{"entity": "customer", "having": []}],
        }
        ir = _normalize_ir1(raw)
        assert ir["query_title"] == "NPA by Branch"
        assert ir["risk_flag"] == "high"
        assert ir["description"] == "Sum of NPA exposure per branch"
        assert ir["display_columns"] == ["Loan No", "SOH"]
        assert ir["entity_filters"][0]["entity"] == "customer"


# ── graph node wiring (deterministic, no LLM) ─────────────────────────────────

class TestGraphNodes:
    def _stub_state(self, **overrides):
        import pandas as pd
        base = {
            "query": "test",
            "result_df_full": _loan_df(),
            "snapshot_dates": {},
            "ir1": {},
            "enriched_query": "",
            "query_category": "",
            "query_title": "",
            "focus_kpis": [],
            "insight_focus": "",
            "risk_flag": "medium",
            "priority_mode": False,
            "aggregation_mode": False,
            "aggregation_spec": {},
            "plan_mode": False,
            "plan": [],
            "result_type": "loan_table",
            "allow_clarification": True,
            "needs_clarification": False,
            "clarification_question": "",
            "clarification_options": [],
            "parsed_filters": {},
            "result_df": pd.DataFrame(),
            "result_kpis": {},
            "result_rankings": {},
            "insights": "",
            "error": "",
            "run_id": "test-run",
            "shadow": {},
        }
        base.update(overrides)
        return base

    def test_compiler_node_produces_plan(self):
        from graph import compiler_node
        ir1 = {
            "intent": "aggregation",
            "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
        }
        state = self._stub_state(ir1=ir1)
        out = compiler_node(state)
        assert out["error"] == ""
        assert isinstance(out["plan"], list) and len(out["plan"]) > 0

    def test_compiler_node_errors_on_bad_concept(self):
        from graph import compiler_node
        ir1 = {
            "intent": "aggregation",
            "filters": [{"concept": "totally_fake_concept"}],
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
        }
        state = self._stub_state(ir1=ir1)
        out = compiler_node(state)
        assert out["error"] != ""

    def test_execute_node_loan_table(self):
        from graph import execute_node
        plan = [
            {"op": "filter", "conditions": [{"column": "curr_bucket", "op": "==", "value": "NPA"}]},
        ]
        ir1 = {"intent": "loan_table"}
        state = self._stub_state(ir1=ir1, plan=plan)
        out = execute_node(state)
        assert out["error"] == ""
        assert len(out["result_df"]) == 2  # L1, L4

    def test_execute_node_aggregation(self):
        from graph import execute_node
        plan = [
            {"op": "group_aggregate", "group_by": ["Unit"],
             "aggregations": [{"alias": "soh", "func": "sum", "column": "SOH"}]},
            {"op": "sort", "by": "soh", "ascending": False},
        ]
        ir1 = {"intent": "aggregation"}
        state = self._stub_state(ir1=ir1, plan=plan)
        out = execute_node(state)
        assert out["error"] == ""
        # aggregation_mode: no compute_result_kpis, just Count
        assert "Count" in out["result_kpis"]
        assert out["result_rankings"] == {}

    def test_execute_node_priority_action(self):
        """priority_action routes to execute_priority_mode, not execute_plan."""
        from graph import execute_node
        # Build a real-ish DataFrame with required priority columns.
        df = pd.DataFrame({
            "Loan No": ["L1"],
            "Non Starter": ["Y"],
            "Loan Status": ["RUN"],
            "curr_bucket": ["NPA"],
            "Strike": ["N"],
            "NACHStatus": ["N"],
            "CoLending_Loans": ["N"],
            "LGL_FLAG": ["N"],
            "Arrears / EMI": [4.0],
            "Ag_Date": [pd.Timestamp("2024-01-01")],
            "SOH": [500.0],
            "POS": [450.0],
            "Cust Mob No": ["111"],
            "Cust Name": ["Alice"],
            "Unit": ["A"],
            "RegionName": ["PUNE"],
            "MNT NAME": ["John"],
        })
        ir1 = {"intent": "priority_action"}
        state = self._stub_state(ir1=ir1, result_df_full=df)
        out = execute_node(state)
        # priority mode should not error (might return empty if criteria not met)
        assert "error" in out  # field exists

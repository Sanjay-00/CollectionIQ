"""Golden-query regression suite  -  no live Gemini calls.

Every case here pins a REAL bug found via live manual testing during the v2
fast-path/hardening work (see conversation history / commit messages for the
exact user queries that first surfaced each one). Manual live-testing is how
all of these were originally discovered; nothing protected any of them from
silently regressing until this file existed. Where a real fix touched the
planner prompt (untestable without a live model), this file pins the
DOWNSTREAM contract instead: given the kind of IR-1 the model is now
documented/expected to emit, does the compiler/executor/view layer handle it
correctly.
"""
import pandas as pd
import pytest

from compiler.core import compile_logical
from agents.plan_executor import execute_plan, validate_plan


def _loan_df():
    return pd.DataFrame({
        "Loan No":       ["L1", "L2", "L3", "L4", "L5"],
        "Cust Mob No":   ["111", "222", "333", "444", "555"],
        "RegionName":    ["PUNE", "PUNE", "MUM", "MUM", "DEL"],
        "Unit":          ["A", "A", "B", "B", "C"],
        "MNT NAME":      ["Raj", "Raj", "Sam", "Sam", "Kim"],
        "curr_bucket":   ["NPA", "STD", "SMA-1", "NPA", "STD"],
        "prev_bucket":   ["SMA-1", "STD", "STD", "SMA-1", "STD"],
        "SOH":           [500.0, 100.0, 300.0, 400.0, 200.0],
        "prev_SOH":      [480.0, 90.0, 270.0, 360.0, 180.0],
        "POS":           [450.0, 90.0, 270.0, 360.0, 180.0],
        "Strike":        ["Y", "N", "Y", "N", "Y"],
        "Month Collection (Excluding Reserve Collection)": [9000, 5000, 7000, 4000, 6000],
        "Net Collection Demand Inst+Exp+BC":                [10000, 4000, 8000, 5000, 6000],
        "prev_Month_Collection":       [8000, 4500, 6500, 3800, 5800],
        "prev_Net_Collection_Demand":  [9500, 4800, 7800, 4900, 5900],
        # Month Receipt Amount deliberately differs from Month Collection above,
        # so no_collection/short_collection tests can't accidentally pass by
        # reusing the wrong column. Mix of zero/over/partial/exact-match payers:
        # L1 zero, L2 over-collected, L3 partial, L4 exactly at demand, L5 over.
        "Month Receipt Amount":       [0, 5000, 6000, 5000, 9000],
    })


# ── HAVING alias/column key fallback ────────────────────────────────────────
# Bug: "Give me all executive with previous NPA count... having previous npa
# count at least 10" silently dropped the threshold entirely, no error, no
# filter step -- because compile_logical's having loop only recognized the key
# "alias", but the planner naturally emits "column" (matching the FILTER
# format used everywhere else in the schema).

class TestHavingKeyFallback:
    def test_having_with_column_key_still_filters(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "npa_count", "alias": "npa"}],
            "metrics": [], "order_by": [],
            "having": [{"column": "npa", "op": ">=", "value": 1}],  # old buggy shape
            "limit": None, "display_columns": [], "time": None,
        }
        cols = list(_loan_df().columns)
        plan, errs = compile_logical(ir1, cols)
        assert errs == []
        filter_steps = [s for s in plan if s["op"] == "filter" and s is not plan[0]]
        assert any(
            c["column"] == "npa" and c["op"] == ">=" and c["value"] == 1
            for s in plan if s["op"] == "filter"
            for c in s["conditions"]
        ), f"having threshold missing from plan: {plan}"

    def test_having_with_alias_key_still_works(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "npa_count", "alias": "npa"}],
            "metrics": [], "order_by": [],
            "having": [{"alias": "npa", "op": ">=", "value": 1}],  # documented shape
            "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(_loan_df().columns))
        assert errs == []
        assert any(s["op"] == "filter" and s["conditions"][0]["column"] == "npa" for s in plan[1:])

    def test_having_with_no_resolvable_key_errors_loudly(self):
        # Never silently drop a threshold: if neither alias/column/measure/field
        # is present, this must be a compile error, not a no-op.
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "npa_count", "alias": "npa"}],
            "metrics": [], "order_by": [],
            "having": [{"op": ">=", "value": 1}],
            "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(_loan_df().columns))
        assert any("having" in e for e in errs)


# ── Dangerous derive-expression guardrail ───────────────────────────────────
# Security hardening: a 'derive'/'metrics' expr ultimately reaches
# pandas.eval(). Two independent layers reject dunder/sandbox-escape patterns:
# validate_plan (compile-time) and _op_derive (execute-time, belt-and-suspenders).

class TestDangerousExprGuardrail:
    PAYLOAD = "(1).__class__.__bases__[0].__subclasses__()"

    def test_compile_time_rejects_dunder_payload(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "exposure", "alias": "soh"}],
            "metrics": [{"alias": "pwned", "expr": self.PAYLOAD, "label": "pwned"}],
            "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(_loan_df().columns))
        assert any("disallowed pattern" in e for e in errs)
        # Belt-and-suspenders: the identifier whitelist independently also catches it.
        assert any("undefined names" in e for e in errs)

    def test_execute_time_rejects_dunder_payload_even_if_validation_skipped(self):
        plan = [{"op": "derive", "column": "pwned", "expr": self.PAYLOAD}]
        result, err = execute_plan(_loan_df(), plan)
        assert err != ""
        assert "disallowed pattern" in err

    def test_legitimate_internal_ratio_alias_not_flagged(self):
        # Regression guard for the false positive the guardrail itself introduced
        # on first pass: ratio measures generate internal aliases like "lcc__n0"
        # (double underscore in the MIDDLE, not wrapping the whole identifier) --
        # must never be confused with a dunder escape ("__class__").
        plan = [{"op": "derive", "column": "lcc_pct", "expr": "lcc__n0 / lcc__d0 * 100"}]
        df = _loan_df()
        df["lcc__n0"] = 50.0
        df["lcc__d0"] = 100.0
        result, err = execute_plan(df, plan)
        assert err == ""
        assert (result["lcc_pct"] == 50.0).all()


# ── Standalone prev_{metric} resolution (no time.compare needed) ───────────
# Bug: "branch-wise portfolio health vs last month" tried prev_total_count,
# prev_exposure, prev_collection_pct as bare metric names (not inside a
# time.compare block) and none of them resolved -- only prev_{bucket}_count
# had this standalone support; count/additive/ratio metrics didn't.

class TestStandalonePrevMetric:
    def _cols(self):
        return list(_loan_df().columns)

    def test_prev_total_count(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "prev_total_count", "alias": "prev_accounts"}],
            "metrics": [], "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        agg = plan[0]["aggregations"][0]
        assert agg["alias"] == "prev_accounts"
        assert agg["where"][0]["column"] == "prev_bucket"

    def test_prev_exposure_additive_metric(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "prev_exposure", "alias": "prev_soh"}],
            "metrics": [], "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        agg = plan[0]["aggregations"][0]
        assert agg == {"alias": "prev_soh", "func": "sum", "column": "prev_SOH"}

    def test_prev_collection_pct_ratio_metric(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [
                {"metric": "collection_pct", "alias": "curr_coll"},
                {"metric": "prev_collection_pct", "alias": "prev_coll"},
            ],
            "metrics": [{"alias": "coll_change", "expr": "curr_coll - prev_coll"}],
            "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        result, err = execute_plan(_loan_df(), plan)
        assert err == ""
        assert "prev_coll" in result.columns and "coll_change" in result.columns

    def test_prev_metric_without_carryover_mapping_is_unresolved(self):
        # lcc_pct's numerator/denominator columns (Cum Coll/Cum Due-*) have no
        # PREV_CARRYOVER_COLS mapping -- must fail loudly (unknown metric),
        # never silently return a wrong "0" for a metric it can't actually build.
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [],
            "dimensions": ["branch"],
            "measures": [{"metric": "prev_lcc_pct", "alias": "x"}],
            "metrics": [], "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert any("unknown metric" in e for e in errs)


# ── View routing / fallback contract ────────────────────────────────────────
# Bug class: an unmatched/unusable view must clear ir1["view"] and fall
# through to the general compiler path, never error outright and never
# silently execute against the wrong data.

class TestViewNodeFallback:
    def _state(self, **overrides):
        base = {
            "query": "test", "result_df_full": _loan_df(), "df_prev": pd.DataFrame(),
            "precomputed_views": {}, "rr_meta": {}, "ir1": {},
        }
        base.update(overrides)
        return base

    def test_unknown_view_name_falls_through(self):
        from graph import view_node
        state = self._state(ir1={"view": {"name": "not_a_real_view", "params": {}, "filters": []}})
        out = view_node(state)
        assert out["ir1"]["view"] is None

    def test_missing_required_df_prev_falls_through(self):
        from graph import view_node
        state = self._state(ir1={"view": {"name": "roll_rate_matrix", "params": {}, "filters": []}})
        out = view_node(state)
        assert out["ir1"]["view"] is None  # roll_rate_matrix requires df_prev

    def test_filter_on_non_filterable_view_falls_through(self):
        from graph import view_node
        state = self._state(ir1={"view": {
            "name": "pulse_kpis", "params": {},
            "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}],
        }})
        out = view_node(state)
        assert out["ir1"]["view"] is None  # pulse_kpis is filterable: False

    def test_successful_view_populates_agg_rows_for_insight_generator(self):
        # Regression guard: a view result must always carry "_agg_rows" so
        # analyze_node describes REAL data, not defaulted-zero loan-table fields
        # (the "8 accounts with zero POS" fabricated-narrative bug).
        from graph import view_node
        state = self._state(ir1={"view": {"name": "top_delinquent_accounts", "params": {"n": 2}, "filters": []}})
        out = view_node(state)
        assert out["ir1"]["view"] is not None
        assert out["error"] == ""
        assert "_agg_rows" in out["result_kpis"]
        assert len(out["result_kpis"]["_agg_rows"]) > 0

    def test_filter_applies_consistently_to_both_curr_and_prev(self):
        # Regression guard for the confirmed df_prev-not-filtered bug: a view
        # that consumes both df_curr and df_prev must compare like-for-like,
        # never "filtered current" against "unfiltered previous".
        df_prev = _loan_df().copy()
        df_prev["curr_bucket"] = ["NPA", "NPA", "STD", "STD", "STD"]  # deliberately different from df_curr
        state = self._state(
            df_prev=df_prev,
            ir1={"view": {
                "name": "region_scorecard", "params": {},
                "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}],
            }},
        )
        from graph import view_node
        out = view_node(state)
        assert out["ir1"]["view"] is not None
        assert out["error"] == ""
        # Only PUNE rows (L1, L2) should have fed the "previous" computation.
        # df_prev PUNE rows are both "NPA" -> NPA%(Prev) for Pune must be 100,
        # not blended with MUM/DEL's mostly-STD prev rows.
        rdf = out["result_df"]
        assert len(rdf) == 1
        assert rdf.iloc[0]["NPA% (Prev)"] == 100.0


# ── Out-of-scope / guardrail contract ───────────────────────────────────────

class TestOutOfScopeGuardrails:
    def test_out_of_scope_ir_shape(self):
        from agents.logical_planner import _out_of_scope_ir
        ir1 = _out_of_scope_ir("I can only answer portfolio questions.")
        assert ir1["needs_clarification"] is True
        assert ir1["clarification_options"] == []
        assert ir1["view"] is None
        assert ir1["filters"] == [] and ir1["dimensions"] == [] and ir1["measures"] == []

    def test_oversized_query_short_circuits_before_any_api_call(self, monkeypatch):
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")

        def _fail_if_called(*a, **k):
            raise AssertionError("must not call Gemini for an oversized query")
        monkeypatch.setattr(lp, "_call_gemini_with_retry", _fail_if_called)

        ir1 = lp.plan_logical("x" * (lp.MAX_QUERY_CHARS + 1))
        assert ir1["needs_clarification"] is True
        assert "too long" in ir1["clarification_question"]

    def test_broken_json_response_becomes_out_of_scope_not_a_raw_exception(self, monkeypatch):
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")

        class _FakeResponse:
            text = "I'm sorry, I can't help with that."  # not JSON -- model broke format

        monkeypatch.setattr(lp, "_call_gemini_with_retry", lambda *a, **k: _FakeResponse())
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)

        ir1 = lp.plan_logical("some query that confused the model")
        assert ir1["needs_clarification"] is True
        assert ir1["view"] is None


# ── Priority-mode result must keep its "Priority" column ────────────────────
# Bug: "Show those accounts that need immediate action" crashed the UI with
# KeyError: 'Priority'. execute_priority_mode() already returns its own
# curated display columns led by "Priority" -- but execute_node's generic
# "apply default curated columns" step ran a SECOND time afterward using
# QUERY_DISPLAY_COLS (which has no concept of "Priority"), silently dropping
# it before distribute_priority_accounts()'s groupby("Priority", ...) ever ran.

class TestPriorityModeKeepsPriorityColumn:
    def _priority_df(self):
        return pd.DataFrame({
            "Loan No":     ["L1", "L2", "L3"],
            "Cust Name":   ["Alice", "Bob", "Carol"],
            "Cust Mob No": ["111", "222", "333"],
            "RegionName":  ["PUNE", "MUM", "DEL"],
            "Unit":        ["A", "B", "C"],
            "MNT NAME":    ["Raj", "Sam", "Kim"],
            "Ag_Date":     pd.Timestamp("2024-01-01"),
            "curr_bucket": ["NPA", "NPA", "STD"],
            "Arrears / EMI": [3.5, 3.5, 0.0],
            "POS":         [100.0, 200.0, 300.0],
            "Closing Arrears": [1000.0, 2000.0, 0.0],
            "Net Collection Demand Inst+Exp+BC": [5000.0, 6000.0, 7000.0],
            "Non Starter": ["N", "N", "N"],
        })

    def test_execute_node_priority_action_keeps_priority_column(self):
        from graph import execute_node
        state = {
            "result_df_full": self._priority_df(),
            "ir1": {"intent": "priority_action", "display_columns": []},
            "plan": [],
        }
        out = execute_node(state)
        assert out["error"] == ""
        assert "Priority" in out["result_df"].columns

    def test_distribute_priority_accounts_does_not_raise(self):
        # The exact downstream call the UI makes with execute_node's output --
        # this is what actually crashed ("KeyError: 'Priority'").
        from graph import execute_node
        from agents.data_executor import distribute_priority_accounts
        state = {
            "result_df_full": self._priority_df(),
            "ir1": {"intent": "priority_action", "display_columns": []},
            "plan": [],
        }
        out = execute_node(state)
        distributed = distribute_priority_accounts(out["result_df"], 30)
        assert "Priority" in distributed.columns


# ── Column-vs-column comparison (no_collection / short_collection) ─────────
# Built on request: "no collection" = Month Receipt Amount <= 0. "short
# collection" = Month Receipt Amount <= Net Collection Demand Inst+Exp+BC
# (column-vs-column, previously unsupported -- the compiler could only compare
# a column against a fixed literal, never against another column).

class TestColumnVsColumnComparison:
    def test_col_lte_direct(self):
        from agents.data_executor import _apply_condition
        result = _apply_condition(_loan_df(), {
            "column": "Month Receipt Amount", "op": "col_lte",
            "value": "Net Collection Demand Inst+Exp+BC",
        })
        assert set(result["Loan No"]) == {"L1", "L3", "L4"}

    def test_col_compare_missing_ref_column_is_graceful_noop(self):
        from agents.data_executor import _apply_condition
        result = _apply_condition(_loan_df(), {
            "column": "Month Receipt Amount", "op": "col_lte", "value": "Nonexistent Col",
        })
        assert len(result) == len(_loan_df())  # unchanged, not an empty/crashed result

    def test_no_collection_concept(self):
        ir1 = {
            "intent": "loan_table", "view": None,
            "filters": [{"concept": "no_collection"}],
            "dimensions": [], "measures": [], "metrics": [], "having": [],
            "order_by": [], "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(_loan_df().columns))
        assert errs == []
        result, err = execute_plan(_loan_df(), plan)
        assert err == ""
        assert set(result["Loan No"]) == {"L1"}

    def test_short_collection_concept(self):
        ir1 = {
            "intent": "loan_table", "view": None,
            "filters": [{"concept": "short_collection"}],
            "dimensions": [], "measures": [], "metrics": [], "having": [],
            "order_by": [], "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(_loan_df().columns))
        assert errs == []
        result, err = execute_plan(_loan_df(), plan)
        assert err == ""
        assert set(result["Loan No"]) == {"L1", "L3", "L4"}

    def test_col_compare_op_with_hallucinated_ref_column_fails_loud_at_compile_time(self):
        # Never let a bad reference column silently no-op through to execution --
        # validate_plan must catch it, same as an ordinary hallucinated column.
        ir1 = {
            "intent": "loan_table", "view": None,
            "filters": [{"column": "Month Receipt Amount", "op": "col_lte", "value": "Not A Real Column"}],
            "dimensions": [], "measures": [], "metrics": [], "having": [],
            "order_by": [], "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(_loan_df().columns))
        assert any("does not exist" in e for e in errs)

    def test_bucket_worse_than_missing_prev_bucket_now_fails_loud_too(self):
        # Same validation now also covers bucket_worse_than/bucket_better_than's
        # reference column (previously unchecked -- a missing prev_bucket would
        # have silently no-opped to "return everyone" instead of erroring).
        cols = [c for c in _loan_df().columns if c != "prev_bucket"]
        ir1 = {
            "intent": "loan_table", "view": None,
            "filters": [{"column": "curr_bucket", "op": "bucket_worse_than", "value": "prev_bucket"}],
            "dimensions": [], "measures": [], "metrics": [], "having": [],
            "order_by": [], "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, cols)
        assert any("prev_bucket" in e for e in errs)

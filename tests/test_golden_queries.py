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

    def test_limit_shrinks_result_df_and_count_reflects_it_not_the_full_set(self):
        # Regression: caught live, not by a unit test -- {"Count": len(result_df),
        # **result_kpis} let the NORMALIZER's own pre-limit "Count" (every
        # normalizer sets one) silently overwrite the post-limit count back to
        # the full, un-capped size. "top 2 regions" was correctly capped to 2
        # rows in result_df, but result_kpis["Count"] still read 3 (all regions)
        # until the merge order was fixed to spread result_kpis FIRST.
        state = self._state(ir1={
            "view": {"name": "region_scorecard", "params": {}, "filters": [], "sort_by": None},
            "limit": 2,
        })
        from graph import view_node
        out = view_node(state)
        assert out["ir1"]["view"] is not None
        assert out["error"] == ""
        assert len(out["result_df"]) == 2
        assert out["result_kpis"]["Count"] == 2

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
        # df_prev PUNE rows are both "NPA" -> NPA%(Prev) for Pune is 100, vs
        # curr NPA% of 50 (L1 NPA, L2 STD) -> Δ NPA% must be -50, not blended
        # with MUM/DEL's mostly-STD prev rows.
        rdf = out["result_df"]
        assert len(rdf) == 1
        assert rdf.iloc[0]["Δ NPA%"] == -50.0


class TestViewNodeNoCompilerFallback:
    """Regression: caught live via a real screenshot -- a "worst/best
    executive business wise" clarification click matched new_advances_by_executive
    (a compiler_fallback=False view, see registry/views.py's schema docstring),
    but the Logical Planner ALSO attached view-scoped "measures" (e.g.
    "new_advances_by_executive.accounts_this_month") despite its own prompt
    saying to leave measures empty whenever a view matches -- confirmed live,
    the model does this on every single run for this query shape, not a rare
    fluke. Those measure names were never real registry METRICS, so if
    view_node ever falls through for ANY reason (a filter attached to this
    non-filterable view, an analysis-fn exception, ...), handing the SAME
    poisoned ir1 to the general compiler crashed with a confusing "unknown
    metric" error instead of the old graceful "just try the compiler instead"
    fallthrough this design relies on elsewhere. Worse alternative if measures
    had simply been empty: the compiler defaults a dimension-only aggregation
    with zero measures to a bare per-group row COUNT -- a different, silently
    WRONG answer, no error at all. Neither is acceptable for a view whose
    whole point (this month's own originations) has no general-compiler
    equivalent, so these three views now route straight to an honest error
    instead of ever reaching the compiler."""

    def _state(self, **overrides):
        base = {
            "query": "test", "result_df_full": _loan_df(), "df_prev": pd.DataFrame(),
            "precomputed_views": {}, "rr_meta": {}, "ir1": {},
        }
        base.update(overrides)
        return base

    def test_fallthrough_on_a_no_compiler_fallback_view_sets_a_clean_error(self):
        from graph import view_node
        state = self._state(ir1={
            "view": {
                "name": "new_advances_by_executive", "params": {},
                # new_advances_by_executive is filterable: False -- attaching
                # a filter is the same trigger the real bug reproduced with.
                "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}],
            },
            "measures": [{"metric": "new_advances_by_executive.accounts_this_month",
                           "alias": "Accounts This Month"}],
        })
        out = view_node(state)
        assert out["ir1"]["view"] is None
        assert out["error"] != ""
        assert "new advances" in out["error"].lower() or "executive" in out["error"].lower()
        # Must never leak the raw hallucinated metric name into the message --
        # that's the exact confusing text this fix exists to replace.
        assert "unknown metric" not in out["error"].lower()

    def test_regular_compiler_fallback_view_still_falls_through_silently(self):
        # A view with no compiler_fallback flag (default True, e.g. pulse_kpis)
        # must be completely unaffected by this change -- same old behavior,
        # no error set, plain fallthrough to the compiler.
        from graph import view_node
        state = self._state(ir1={"view": {
            "name": "pulse_kpis", "params": {},
            "filters": [{"column": "RegionName", "op": "==", "value": "PUNE"}],
        }})
        out = view_node(state)
        assert out["ir1"]["view"] is None
        assert out.get("error", "") == ""

    def test_all_three_new_advances_views_are_flagged(self):
        from registry.views import VIEWS
        for name in ("new_advances_by_region", "new_advances_by_branch", "new_advances_by_executive"):
            assert VIEWS[name]["compiler_fallback"] is False, name


# ── Clarification follow-up context (query-text-scoped, not flag-scoped) ────

class TestDeterministicBusinessAmbiguity:
    """_resolve_business_ambiguity_deterministically is pure Python -- no
    Gemini call, no monkeypatching needed, fully deterministic by design.
    Replaces LLM judgment with a keyword+registry-dictionary lookup for the
    one case that's fully specifiable (see the function's own docstring for
    why "risk" is deliberately NOT handled here)."""

    def _resolve(self, query):
        from agents.logical_planner import _resolve_business_ambiguity_deterministically
        return _resolve_business_ambiguity_deterministically(query)

    @pytest.mark.parametrize("query,grain", [
        ("give me my best branch top 5 wrt business", "branches"),
        ("which region has best business", "regions"),
        ("give me my best executive top 5 wrt business", "executives"),
        ("show me business details for units", "branches"),
    ])
    def test_resolves_with_grain_aware_options(self, query, grain):
        ir1 = self._resolve(query)
        assert ir1 is not None
        assert ir1["needs_clarification"] is True
        assert len(ir1["clarification_options"]) == 2
        assert grain in ir1["clarification_question"]

    def test_branch_options_never_mention_executive_only_metrics(self):
        ir1 = self._resolve("give me my best branch top 5 wrt business")
        opts_text = " ".join(ir1["clarification_options"])
        assert "Net Recovery" not in opts_text  # executive_recovery-only metric

    def test_executive_options_never_mention_branch_only_metrics(self):
        # Regression: this exact case previously hallucinated "Concern Score"
        # for an executive query -- executive_recovery doesn't have that
        # metric, only branch_quadrant does. A dictionary lookup can't make
        # this mistake; this test only needs to exist to prove it never can.
        ir1 = self._resolve("give me my best executive top 5 wrt business")
        opts_text = " ".join(ir1["clarification_options"])
        assert "Concern Score" not in opts_text

    @pytest.mark.parametrize("query", [
        "top 5 branches by new advances this month",       # already says "new advances"
        "which branch originated the most loans",            # already says "originated"
        "which branch has the best NPA%",                     # already says "NPA%"
        "which branch has the best collection performance",   # already says "collection"
    ])
    def test_already_disambiguated_query_falls_through_to_llm(self, query):
        # None means "don't resolve deterministically, let the LLM proceed
        # normally" -- these queries already say which reading they mean, so
        # forcing a clarification here would be a regression, not a fix.
        assert self._resolve(query) is None

    def test_no_grain_named_falls_through(self):
        # "business" is present but WHO the question is about is unclear --
        # can't confidently build grain-correct options, so don't guess.
        assert self._resolve("show me the business details") is None

    def test_no_business_term_at_all_is_a_no_op(self):
        assert self._resolve("show me the top delinquent accounts") is None

    def test_case_insensitive(self):
        assert self._resolve("GIVE ME BEST BRANCH BUSINESS DETAILS") is not None

    def test_does_not_false_positive_on_substring_of_another_word(self):
        # "businesslike" should not match \bbusiness\b as a whole word -- this
        # is a contrived case, real queries won't say this, but the word
        # boundary regex should still behave correctly if they did.
        assert self._resolve("show me businesslike branches") is None

    def test_clicked_interpretation_text_falls_through_not_reresolved(self):
        # The augmented query a clarification click produces (query + "
        # (interpretation: New advances...)") still contains "business" from
        # the original text, but ALSO now contains "new advances"/"funded" --
        # must fall through via the same already-disambiguated check above,
        # not re-trigger a second round of the same clarification.
        augmented = (
            "give me my best branch top 5 wrt business "
            "(interpretation: New advances (accounts/funded this period))"
        )
        assert self._resolve(augmented) is None


class TestPlannerTemperature:
    """The Logical Planner does structured classification/extraction, not
    open-ended writing -- the SDK default temperature (~1.0) was the confirmed,
    reproducible cause of run-to-run inconsistency observed live (identical
    query, different clarification behavior across runs). Locks in that a low,
    non-default temperature actually reaches the Gemini call config."""

    def test_temperature_is_set_low_not_left_at_sdk_default(self, monkeypatch):
        import agents.logical_planner as lp
        from config import PLANNER_TEMPERATURE
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")
        sent = {}

        class _FakeResponse:
            text = '{"intent": "loan_table", "filters": []}'

        def _capture(client, model, contents, config):
            sent["config"] = config
            return _FakeResponse()

        monkeypatch.setattr(lp, "_call_gemini_with_retry", _capture)
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)

        lp.plan_logical("show me the top branches")
        assert sent["config"]["temperature"] == PLANNER_TEMPERATURE
        assert PLANNER_TEMPERATURE < 0.5  # meaningfully below the SDK's open-ended-writing default


class TestClarificationFollowupContext:
    """Regression guard for a real conflict caught before it shipped: the
    "use the clarified metric for sort_by/highlight_metrics" instruction is
    keyed off the query TEXT containing "(interpretation:" (the marker
    ui/tabs/ai_query.py appends when a clarification option is clicked), NOT
    off allow_clarification=False -- that flag is ALSO set False by the
    unrelated compiler repair loop (graph.py's one-shot retry on a validation
    error), which must never receive clarification-specific instructions
    since no interpretation exists there. These tests capture the literal
    text sent to Gemini (never a live call) and check the marker only
    appears exactly when the query text itself contains it."""

    def _capture_prompt_text(self, monkeypatch):
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")
        sent = {}

        class _FakeResponse:
            text = '{"intent": "loan_table", "filters": []}'

        def _capture(client, model, contents, config):
            sent["contents"] = contents
            return _FakeResponse()

        monkeypatch.setattr(lp, "_call_gemini_with_retry", _capture)
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)
        return lp, sent

    def test_plain_query_gets_no_clarification_followup_context(self, monkeypatch):
        lp, sent = self._capture_prompt_text(monkeypatch)
        lp.plan_logical("show me the top branches")
        assert "(interpretation:" not in sent["contents"]
        assert "CLARIFICATION RESOLVED" not in sent["contents"]

    def test_clarification_followup_query_gets_the_context(self, monkeypatch):
        lp, sent = self._capture_prompt_text(monkeypatch)
        lp.plan_logical(
            "give me my best branches (interpretation: Collection Efficiency)",
            allow_clarification=False,
        )
        assert "CLARIFICATION RESOLVED" in sent["contents"]

    def test_unrelated_compiler_repair_retry_does_not_get_the_context(self, monkeypatch):
        # allow_clarification=False here for the SAME reason the real repair
        # loop sets it (graph.py:691) -- but the query itself has no
        # interpretation marker, so this must NOT get clarification-specific
        # instructions injected into an unrelated repair retry.
        lp, sent = self._capture_prompt_text(monkeypatch)
        lp.plan_logical(
            "show me the top branches",
            repair_feedback="unknown column 'Foo'",
            allow_clarification=False,
        )
        assert "CLARIFICATION RESOLVED" not in sent["contents"]
        assert "[REPAIR" in sent["contents"]

    def test_clarification_followups_own_repair_retry_still_gets_the_context(self, monkeypatch):
        # A clarification follow-up that ALSO needs a repair pass (query text
        # still carries the marker on retry) should still get both directives
        # -- this is a real, valid combination, not a case to suppress.
        lp, sent = self._capture_prompt_text(monkeypatch)
        lp.plan_logical(
            "give me my best branches (interpretation: Collection Efficiency)",
            repair_feedback="unknown column 'Foo'",
            allow_clarification=False,
        )
        assert "CLARIFICATION RESOLVED" in sent["contents"]
        assert "[REPAIR" in sent["contents"]


# ── Out-of-scope / guardrail contract ───────────────────────────────────────

class TestOutOfScopeGuardrails:
    def test_out_of_scope_ir_shape(self):
        # Regression: clarification_options used to default to [] here, which
        # renders NO clickable buttons in the UI -- a genuine dead end (unlike
        # a real ambiguity, which always gives the user something to click).
        # Now defaults to a fixed, safe example pair so there's always a way
        # forward without retyping from scratch.
        from agents.logical_planner import _out_of_scope_ir, _OUT_OF_SCOPE_EXAMPLE_OPTIONS
        ir1 = _out_of_scope_ir("I can only answer portfolio questions.")
        assert ir1["needs_clarification"] is True
        assert ir1["clarification_options"] == _OUT_OF_SCOPE_EXAMPLE_OPTIONS
        assert len(ir1["clarification_options"]) > 0
        assert ir1["view"] is None
        assert ir1["filters"] == [] and ir1["dimensions"] == [] and ir1["measures"] == []

    def test_out_of_scope_ir_accepts_explicit_options_override(self):
        from agents.logical_planner import _out_of_scope_ir
        ir1 = _out_of_scope_ir("msg", options=["Custom option"])
        assert ir1["clarification_options"] == ["Custom option"]


class TestDeterministicBusinessFastPathWiring:
    """plan_logical must actually short-circuit BEFORE any Gemini call when
    the deterministic resolver matches -- this is the whole point (zero
    latency/cost for this case, not just more reliable than the LLM path)."""

    def test_matching_query_never_calls_gemini(self, monkeypatch):
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")

        def _fail_if_called(*a, **k):
            raise AssertionError("must not call Gemini when the deterministic resolver matches")
        monkeypatch.setattr(lp, "_call_gemini_with_retry", _fail_if_called)

        ir1 = lp.plan_logical("give me my best branch top 5 wrt business")
        assert ir1["needs_clarification"] is True
        assert len(ir1["clarification_options"]) == 2

    def test_post_clarification_retry_is_not_intercepted(self, monkeypatch):
        # allow_clarification=False (the clarification-followup re-run) must
        # skip the deterministic check entirely and reach the normal LLM path
        # -- verified by confirming Gemini DOES get called here, the opposite
        # assertion from the test above.
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")
        called = {"yes": False}

        class _FakeResponse:
            text = '{"intent": "loan_table", "filters": []}'

        def _capture(*a, **k):
            called["yes"] = True
            return _FakeResponse()

        monkeypatch.setattr(lp, "_call_gemini_with_retry", _capture)
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)

        lp.plan_logical(
            "give me my best branch top 5 wrt business (interpretation: New advances)",
            allow_clarification=False,
        )
        assert called["yes"] is True

    def test_repair_retry_is_not_intercepted(self, monkeypatch):
        # Same gate, same reasoning as the clarification-followup case above --
        # a repair retry (allow_clarification=False) must reach the LLM too.
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")
        called = {"yes": False}

        class _FakeResponse:
            text = '{"intent": "loan_table", "filters": []}'

        def _capture(*a, **k):
            called["yes"] = True
            return _FakeResponse()

        monkeypatch.setattr(lp, "_call_gemini_with_retry", _capture)
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)

        lp.plan_logical(
            "give me my best branch top 5 wrt business",
            repair_feedback="unknown column 'Foo'",
            allow_clarification=False,
        )
        assert called["yes"] is True

    def test_model_classified_out_of_scope_with_empty_options_gets_the_fallback(self, monkeypatch):
        # The model's OWN live out-of-scope classification (per the prompt's
        # explicit "clarification_options=[]" instruction) must ALSO get the
        # same safe fallback, not just the Python-side guardrail paths above.
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")

        class _FakeResponse:
            text = (
                '{"intent": "loan_table", "needs_clarification": true, '
                '"clarification_question": "I can only answer portfolio questions.", '
                '"clarification_options": []}'
            )

        monkeypatch.setattr(lp, "_call_gemini_with_retry", lambda *a, **k: _FakeResponse())
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)

        ir1 = lp.plan_logical("asdkjhasdkjh")
        assert ir1["needs_clarification"] is True
        assert len(ir1["clarification_options"]) > 0

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

    def test_all_columns_phrasing_forces_show_all_columns_even_if_model_omits_it(self, monkeypatch):
        # Regression: observed in practice -- two live calls with the EXACT
        # same real-world query text ("...with all columns including overdue
        # collection % and month collection%...") produced DIFFERENT IR-1s;
        # one correctly set show_all_columns, the other didn't and also built
        # a display_columns list missing the very columns the user named. The
        # model's own judgment on this field is not a guarantee -- this keyword
        # backstop makes the common, explicit phrasing deterministic regardless
        # of what the model decides to set.
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")

        class _FakeResponse:
            text = '{"intent": "loan_table", "filters": [], "display_columns": ["Loan No", "SOH"], "show_all_columns": false}'

        monkeypatch.setattr(lp, "_call_gemini_with_retry", lambda *a, **k: _FakeResponse())
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)

        ir1 = lp.plan_logical("give me all cases with all columns including overdue collection %")
        assert ir1["show_all_columns"] is True

    def test_query_without_all_columns_phrasing_leaves_model_choice_alone(self, monkeypatch):
        import agents.logical_planner as lp
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")

        class _FakeResponse:
            text = '{"intent": "loan_table", "filters": [], "display_columns": ["Loan No", "SOH"], "show_all_columns": false}'

        monkeypatch.setattr(lp, "_call_gemini_with_retry", lambda *a, **k: _FakeResponse())
        monkeypatch.setattr(lp, "_add_token_usage", lambda *a, **k: None)

        ir1 = lp.plan_logical("show me Loan No and SOH only")
        assert ir1["show_all_columns"] is False


class TestQueryRequestsAllColumns:
    """Unit coverage for the deterministic keyword backstop itself."""

    @pytest.mark.parametrize("query", [
        "give me all columns",
        "show every column",
        "all the columns please",
        "GIVE ME ALL COLUMNS",
        "with all columns including overdue collection % and month collection% case wise",
    ])
    def test_matches_common_phrasings(self, query):
        from agents.logical_planner import _query_requests_all_columns
        assert _query_requests_all_columns(query) is True

    @pytest.mark.parametrize("query", [
        "show me all the accounts in Akola",
        "give me every executive's collection %",
        "show me Loan No and SOH only",
    ])
    def test_does_not_false_positive_on_unrelated_all_every(self, query):
        from agents.logical_planner import _query_requests_all_columns
        assert _query_requests_all_columns(query) is False


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
            "VehEMI Accrued": [1, 1, 1],
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

    def test_dedup_respects_rank_even_if_rules_list_is_out_of_order(self, monkeypatch):
        # "Each loan appears only under its highest priority rule" (agents/
        # data_executor.py::execute_priority_mode) depends on visiting rules in
        # ascending rank order, since a loan gets claimed by whichever rule
        # reaches it first. PRIORITY_RULES happens to be authored in rank order
        # today, but nothing enforced that -- scramble the list here and confirm
        # a loan matching both a rank-1 and a rank-7 rule still lands under
        # rank 1 (its true highest priority), not whichever rule came first
        # in an out-of-order list.
        import agents.domain_expert as domain_expert
        rules = list(domain_expert.PRIORITY_RULES)
        rank1 = next(r for r in rules if r["rank"] == 1)
        rank7 = next(r for r in rules if r["rank"] == 7)
        scrambled = [rank7, rank1] + [r for r in rules if r["rank"] not in (1, 7)]
        monkeypatch.setattr(domain_expert, "PRIORITY_RULES", scrambled)

        # A Non Starter (rank 1: Non Starters) that's also NPA (rank 7: NPA
        # Accounts) -- matches both tiers, must be claimed by rank 1.
        df = self._priority_df()
        df["Non Starter"] = ["Y", "N", "N"]  # L1 matches both rank-1 and rank-7 rules

        from agents.data_executor import execute_priority_mode
        out, err = execute_priority_mode(df)
        assert err == ""
        l1_row = out[out["Loan No"] == "L1"].iloc[0]
        assert l1_row["Priority"].startswith("P1:")


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


# ── count_ratio measure kind (hard_bucket_pct / strike_pct) ────────────────
# "Collection %"-style metrics sum two COLUMNS; Hard Bucket %/Strike % are a
# different shape entirely -- count rows matching a condition, divide by
# another count. Registered as real named metrics (not left as a manual
# count+derive the AI had to reconstruct every time) via a new "count_ratio"
# measure kind, following the same handler-registration pattern as "ratio".

class TestCountRatioMetrics:
    def _cols(self):
        return ["Loan No", "Unit", "Arrears / EMI", "prev_Arrears_EMI", "Strike", "prev_Strike"]

    def _df(self):
        return pd.DataFrame({
            "Loan No": [f"L{i}" for i in range(10)],
            "Unit": ["A"] * 5 + ["B"] * 5,
            "Arrears / EMI": [7, 8, 0, 1, 2, 6, 6, 0, 0, 0],
            "Strike": ["Y", "Y", "Y", "N", "N", "Y", "N", "N", "N", "N"],
        })

    def test_hard_bucket_pct_current_period(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [], "dimensions": ["branch"],
            "measures": [{"metric": "hard_bucket_pct", "alias": "hb"}],
            "metrics": [], "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        result, err = execute_plan(self._df(), plan)
        assert err == ""
        by_unit = dict(zip(result["Unit"], result["hb"]))
        assert by_unit == {"A": 40.0, "B": 40.0}  # 2/5 each, matching the manual test earlier

    def test_strike_pct_current_period(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [], "dimensions": ["branch"],
            "measures": [{"metric": "strike_pct", "alias": "sp"}],
            "metrics": [], "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        result, err = execute_plan(self._df(), plan)
        assert err == ""
        by_unit = dict(zip(result["Unit"], result["sp"]))
        assert by_unit == {"A": 60.0, "B": 20.0}

    def test_strike_pct_time_compare_produces_prev_and_change(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [], "dimensions": ["branch"],
            "measures": [{"metric": "strike_pct", "alias": "curr_sp"}],
            "metrics": [], "having": [], "order_by": [], "limit": None, "display_columns": [],
            "time": {"grain": "month", "compare": {"type": "change", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        df = self._df()
        df["prev_Strike"] = ["Y", "N", "N", "N", "N", "Y", "Y", "N", "N", "N"]
        result, err = execute_plan(df, plan)
        assert err == ""
        row_a = result[result["Unit"] == "A"].iloc[0]
        assert row_a["curr_sp"] == 60.0
        assert row_a["prev_curr_sp"] == 20.0
        assert row_a["curr_sp_change"] == 40.0

    def test_standalone_prev_strike_pct_without_time_compare(self):
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [], "dimensions": ["branch"],
            "measures": [{"metric": "prev_strike_pct", "alias": "prev_sp"}],
            "metrics": [], "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        df = self._df()
        df["prev_Strike"] = ["Y", "N", "N", "N", "N", "Y", "Y", "N", "N", "N"]
        result, err = execute_plan(df, plan)
        assert err == ""
        by_unit = dict(zip(result["Unit"], result["prev_sp"]))
        assert by_unit == {"A": 20.0, "B": 40.0}

    def test_helper_columns_never_reach_the_display(self):
        # ratio/count_ratio measures compute via internal "{alias}__n0"/"{alias}__d0"
        # scratch columns before dividing -- these must never leak into the table
        # a lead actually sees.
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [], "dimensions": ["branch"],
            "measures": [{"metric": "strike_pct", "alias": "curr_sp"}],
            "metrics": [], "having": [], "order_by": [], "limit": None, "display_columns": [],
            "time": {"grain": "month", "compare": {"type": "change", "from": "prev", "to": "curr"}},
        }
        plan, errs = compile_logical(ir1, self._cols())
        assert errs == []
        df = self._df()
        df["prev_Strike"] = ["Y", "N", "N", "N", "N", "Y", "Y", "N", "N", "N"]
        result, err = execute_plan(df, plan)
        assert err == ""
        assert not any(c.endswith(("__n0", "__d0")) for c in result.columns)
        assert set(result.columns) == {"Rank", "Unit", "curr_sp", "prev_curr_sp", "curr_sp_change"}


# ── Deep-audit fixes: grain-ambiguity message + time.compare column threading ──

class TestDeepAuditFixes:
    def test_grain_ambiguity_message_interpolates_the_column_name(self):
        # Bug: the error ended in the literal text "'{pcol}'" instead of the real
        # column name, because the third string segment of the f-string
        # concatenation was missing its "f" prefix.
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [], "dimensions": ["region"],
            "entity_filters": [{"entity": "customer", "having": [
                {"agg": "nunique", "column": "Unit", "op": ">=", "value": 2}
            ]}],
            "measures": [], "metrics": [], "having": [], "order_by": [], "limit": None,
            "display_columns": [], "time": None,
        }
        cols = ["Loan No", "Unit", "RegionName", "Cust Mob No"]
        plan, errs = compile_logical(ir1, cols)
        assert any("'Unit' should be counted" in e for e in errs)
        assert not any("{pcol}" in e for e in errs)

    def test_time_compare_resolves_raw_column_guess_metrics(self):
        # Bug: _resolve_time_compare called _measure_def without 'columns', so the
        # last-resort raw-column-guess fallback (e.g. "soh" -> "SOH") couldn't
        # resolve inside time.compare specifically -- silently producing the
        # current-period value with NO prev/change columns, no error either.
        ir1 = {
            "intent": "aggregation", "view": None, "filters": [], "dimensions": ["branch"],
            "measures": [{"metric": "soh", "alias": "curr_soh"}],
            "metrics": [], "having": [], "order_by": [], "limit": None, "display_columns": [],
            "time": {"grain": "month", "compare": {"type": "change", "from": "prev", "to": "curr"}},
        }
        cols = ["Loan No", "Unit", "SOH", "prev_SOH"]
        plan, errs = compile_logical(ir1, cols)
        assert errs == []
        df = pd.DataFrame({
            "Loan No": [f"L{i}" for i in range(6)], "Unit": ["A"] * 3 + ["B"] * 3,
            "SOH": [100, 200, 300, 50, 60, 70], "prev_SOH": [90, 180, 270, 40, 55, 65],
        })
        result, err = execute_plan(df, plan)
        assert err == ""
        row_a = result[result["Unit"] == "A"].iloc[0]
        assert row_a["curr_soh"] == 600
        assert row_a["prev_curr_soh"] == 540
        assert row_a["curr_soh_change"] == 60


# ── Person-name filters use "contains", not "==" ────────────────────────────
# Bug: "what's the month demand for executive named yash bhagoji deve" returned
# 0 matching accounts for a real executive because the planner filtered MNT NAME
# with an exact "==" match. MNT NAME/Cust Name/Guar Name are free text typed once
# at loan origination (not a controlled vocabulary like RegionName), so any
# spelling/spacing variance in how the user types the name silently returns zero
# rows for someone who is actually in the data. Fixed by instructing the planner
# to use "contains" (case-insensitive substring) for these fields instead - this
# pins the downstream contract (the fix itself is a prompt change, untestable
# without a live model).

class TestPersonNameContainsFilter:
    def _df(self):
        return pd.DataFrame({
            "Loan No":  ["L1", "L2", "L3"],
            "MNT NAME": ["Yash Bhagoji Deve", "Sunil Kumar Patil", "Rajesh Sharma"],
            "SOH":      [1000.0, 2000.0, 1500.0],
        })

    def test_contains_matches_exact_name_case_insensitively(self):
        ir1 = {
            "intent": "loan_table", "view": None,
            "filters": [{"column": "MNT NAME", "op": "contains", "value": "yash bhagoji deve"}],
            "dimensions": [], "measures": [], "metrics": [], "having": [],
            "order_by": [], "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(self._df().columns))
        assert errs == []
        result, err = execute_plan(self._df(), plan)
        assert err == ""
        assert len(result) == 1
        assert result.iloc[0]["Loan No"] == "L1"

    def test_contains_matches_partial_name(self):
        # A user typing just part of the name (or a slight variant) still finds
        # the account - the failure mode "==" had.
        ir1 = {
            "intent": "loan_table", "view": None,
            "filters": [{"column": "MNT NAME", "op": "contains", "value": "bhagoji"}],
            "dimensions": [], "measures": [], "metrics": [], "having": [],
            "order_by": [], "limit": None, "display_columns": [], "time": None,
        }
        plan, errs = compile_logical(ir1, list(self._df().columns))
        assert errs == []
        result, err = execute_plan(self._df(), plan)
        assert err == ""
        assert len(result) == 1
        assert result.iloc[0]["Loan No"] == "L1"

    def test_contains_misapplied_to_numeric_column_errors_not_unfiltered(self):
        # Guard against the model ever emitting "contains" against a numeric/date
        # column (nothing scopes the op to a column TYPE, only the prompt's naming
        # guidance) -- must surface as a query error, never silently return the
        # full unfiltered table (a "confidently wrong answer").
        ir1 = {
            "intent": "loan_table", "view": None,
            "filters": [{"column": "SOH", "op": "contains", "value": "100"}],
            "dimensions": [], "measures": [], "metrics": [], "having": [],
            "order_by": [], "limit": None, "display_columns": [], "time": None,
        }
        cols = ["Loan No", "MNT NAME", "SOH"]
        plan, errs = compile_logical(ir1, cols)
        assert errs == []
        df = pd.DataFrame({"Loan No": ["L1", "L2"], "MNT NAME": ["A", "B"], "SOH": [100.0, 200.0]})
        result, err = execute_plan(df, plan)
        assert err != ""
        assert result.empty

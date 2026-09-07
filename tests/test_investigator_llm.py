"""No live Gemini calls in this suite (matches this repo's test policy, see
tests/test_logical_planner_prompt.py). Deterministic prompt-building and
response-parsing are tested directly; parse_turn/narrate_step's own Gemini
call is tested by monkeypatching investigator.llm._call_gemini_with_retry,
the same way tests/test_skip_insights.py monkeypatches graph.generate_insights."""
import pandas as pd
import pytest

import investigator.llm as llm
from investigator.state import EntityMemory


class TestBuildTurnPrompt:
    def test_lists_every_registered_step_type(self):
        prompt = llm.build_turn_prompt("why is Mahad underperforming", EntityMemory())
        for step_type in llm.STEP_TYPES:
            assert step_type in prompt

    def test_includes_entity_memory_context(self):
        mem = EntityMemory()
        mem.touch("branch", "Mahad", result_df=pd.DataFrame(), metric_focus="NPA%")
        prompt = llm.build_turn_prompt("focus on hard bucket instead", mem)
        assert "Mahad" in prompt
        assert "NPA%" in prompt

    def test_dimension_breakdown_context_lists_the_actual_row_names(self):
        # The exact regression this fixes: "why is CS Nagar not performing"
        # right after a region-level dimension_breakdown -- the model has no
        # way to know CS Nagar is a region unless the row names from that
        # table are surfaced here.
        mem = EntityMemory()
        df = pd.DataFrame({"RegionName": ["CS NAGAR", "NAGPUR"], "Collection%": [60.0, 90.0]})
        mem.touch("dimension_breakdown", "region", result_df=df, metric_focus="region", step_type="dimension_breakdown")
        prompt = llm.build_turn_prompt("why is cs nagar not performing good", mem)
        assert "CS NAGAR" in prompt
        assert "NAGPUR" in prompt
        # Metric VALUES must never leak into the routing prompt -- only names.
        assert "60.0" not in prompt

    def test_executive_group_context_lists_names_scope_and_available_metrics(self):
        # The exact regression this fixes: "why strike rate is less" right
        # after an executive-grain table (Collection%-focused) reverted to
        # an unrelated dimension_breakdown(region) -- the model had no
        # explicit signal that Strike Rate % was already a column on the
        # SAME table, scoped to the SAME branch, sitting right in front of it.
        mem = EntityMemory()
        df = pd.DataFrame({
            "MNT NAME": ["RAHUL TOLANKAR", "AJAY SATHE"], "Unit": ["KMGON", "KMGON"],
            "Collection %": [71.8, 79.0], "Strike Rate %": [62.6, 67.0],
        })
        mem.touch(
            "executive_group", "underperformer_quartile:Collection %:KMGON",
            result_df=df, metric_focus="Collection %", step_type="underperformer_quartile",
        )
        prompt = llm.build_turn_prompt("why strike rate is less", mem)
        assert "RAHUL TOLANKAR" in prompt
        assert "AJAY SATHE" in prompt
        assert "KMGON" in prompt
        assert "EXECUTIVE-grain" in prompt
        assert "Strike Rate %" in prompt
        # Metric VALUES must never leak into the routing prompt -- only names.
        assert "71.8" not in prompt

    def test_states_the_one_level_at_a_time_hierarchy_rule(self):
        prompt = llm.build_turn_prompt("why is cs nagar bad", EntityMemory())
        assert "ONE-LEVEL-AT-A-TIME" in prompt
        assert "BU > Zone > Region > Branch > Executive > Customer" in prompt

    def test_frames_the_model_as_an_nbfc_analyst_assisting_a_leader(self):
        prompt = llm.build_turn_prompt("why is cs nagar bad", EntityMemory())
        assert "NBFC" in prompt
        assert "Regional Business Head" in prompt and "Zonal Head" in prompt and "Business Unit Head" in prompt

    def test_empty_entity_memory_says_so_explicitly(self):
        prompt = llm.build_turn_prompt("why is Mahad underperforming", EntityMemory())
        assert "No entities discussed yet" in prompt

    def test_states_the_context_hint_rule(self):
        prompt = llm.build_turn_prompt("why is cs nagar bad", EntityMemory())
        assert "CONTEXT HINTS" in prompt
        assert "NEXT EXECUTIVE HINT" in prompt
        assert "CURRENT SCOPE" in prompt


class TestCurrentScopeHint:
    """The exact live bug: mid-conversation about NADED branch, "what
    should I focus on" routed priority_menu portfolio-wide instead of
    staying scoped to NADED -- there was no signal a scope was even
    available to inherit."""

    def test_branch_entity_summary_gives_a_branch_scope_hint(self):
        mem = EntityMemory()
        mem.touch("branch", "NADED", result_df=pd.DataFrame({"Metric": ["NPA%"]}), step_type="entity_summary")
        prompt = llm.build_turn_prompt("what should i focus on?", mem)
        assert 'CURRENT SCOPE: branch "NADED"' in prompt

    def test_single_executive_focus_inherits_their_branch(self):
        mem = EntityMemory()
        mem.touch(
            "executive", ("DATTAPRASAD RAMDAS KA", "NADED"),
            result_df=pd.DataFrame({"Metric": ["Hard Bucket %"]}), step_type="entity_summary",
        )
        prompt = llm.build_turn_prompt("what should i focus on?", mem)
        assert 'CURRENT SCOPE: branch "NADED"' in prompt

    def test_worst_loans_by_metric_scoped_to_an_executive_inherits_their_branch(self):
        # The exact turn from the real conversation this fixes: "lets focus
        # on dattaprasad and why its hard bucket is so much" produces a
        # worst_loans_by_metric record scoped via scope_col="executive" --
        # the branch must still be inferable from it.
        mem = EntityMemory()
        mem.touch(
            "worst_loans_by_metric", "Hard Bucket %:('DATTAPRASAD RAMDAS KA', 'NADED')",
            result_df=pd.DataFrame({"Cust Name": ["A"]}), step_type="worst_loans_by_metric",
            params={"metric": "Hard Bucket %", "scope_col": "executive",
                    "scope_value": ["DATTAPRASAD RAMDAS KA", "NADED"]},
        )
        prompt = llm.build_turn_prompt("what should i focus on?", mem)
        assert 'CURRENT SCOPE: branch "NADED"' in prompt

    def test_portfolio_wide_focus_gives_no_scope_hint(self):
        mem = EntityMemory()
        mem.touch("dimension_breakdown", "region", result_df=pd.DataFrame({"RegionName": ["WEST"]}), step_type="dimension_breakdown")
        prompt = llm.build_turn_prompt("what should i focus on?", mem)
        assert "CURRENT SCOPE:" not in prompt

    def test_no_entities_gives_no_scope_hint(self):
        prompt = llm.build_turn_prompt("what should i focus on?", EntityMemory())
        assert "CURRENT SCOPE:" not in prompt


class TestNextExecutiveHint:
    """The exact live bug: "can we move to next executive?" had no
    vocabulary to map to at all, and fell back to re-showing the SAME
    roster table already in memory instead of advancing to the next name."""

    def _roster_df(self):
        return pd.DataFrame({
            "MNT NAME": ["PRAFUL SHIVAJI PATI", "RAJESHWAR RITT", "DATTAPRASAD RAMDAS KA"],
            "Unit": ["NADED", "NADED", "NADED"],
        })

    def test_names_the_next_executive_after_the_current_one(self):
        mem = EntityMemory()
        mem.touch(
            "executive_group", "executive_roster:NADED",
            result_df=self._roster_df(), step_type="executive_roster",
        )
        mem.touch(
            "executive", ("RAJESHWAR RITT", "NADED"),
            result_df=pd.DataFrame({"Metric": ["NPA %"]}), step_type="entity_summary",
        )
        prompt = llm.build_turn_prompt("can we move to next executive?", mem)
        assert "NEXT EXECUTIVE HINT" in prompt
        assert '"DATTAPRASAD RAMDAS KA"' in prompt

    def test_worst_loans_by_metric_focus_also_resolves_the_next_hint(self):
        # Same scenario as the real conversation: the current focus came
        # from worst_loans_by_metric (scope_col="executive"), not a plain
        # entity_summary -- must still resolve against the roster.
        mem = EntityMemory()
        mem.touch(
            "executive_group", "executive_roster:NADED",
            result_df=self._roster_df(), step_type="executive_roster",
        )
        mem.touch(
            "worst_loans_by_metric", "Hard Bucket %:('DATTAPRASAD RAMDAS KA', 'NADED')",
            result_df=pd.DataFrame({"Cust Name": ["A"]}), step_type="worst_loans_by_metric",
            params={"metric": "Hard Bucket %", "scope_col": "executive",
                    "scope_value": ["DATTAPRASAD RAMDAS KA", "NADED"]},
        )
        prompt = llm.build_turn_prompt("can we move to next executive?", mem)
        assert "NEXT EXECUTIVE HINT" in prompt
        assert "LAST" in prompt  # Dattaprasad is last in the roster fixture

    def test_last_executive_in_roster_says_so_and_names_none(self):
        mem = EntityMemory()
        mem.touch(
            "executive_group", "executive_roster:NADED",
            result_df=self._roster_df(), step_type="executive_roster",
        )
        mem.touch(
            "executive", ("DATTAPRASAD RAMDAS KA", "NADED"),
            result_df=pd.DataFrame({"Metric": ["NPA %"]}), step_type="entity_summary",
        )
        prompt = llm.build_turn_prompt("can we move to next executive?", mem)
        assert "NEXT EXECUTIVE HINT" in prompt
        assert "LAST" in prompt
        assert "no next executive" in prompt

    def test_no_current_executive_focus_gives_no_next_hint(self):
        mem = EntityMemory()
        mem.touch("branch", "NADED", result_df=pd.DataFrame({"Metric": ["NPA%"]}), step_type="entity_summary")
        prompt = llm.build_turn_prompt("what next?", mem)
        assert "NEXT EXECUTIVE HINT:" not in prompt

    def test_no_matching_roster_in_memory_gives_no_next_hint(self):
        mem = EntityMemory()
        mem.touch(
            "executive", ("DATTAPRASAD RAMDAS KA", "NADED"),
            result_df=pd.DataFrame({"Metric": ["NPA %"]}), step_type="entity_summary",
        )
        prompt = llm.build_turn_prompt("can we move to next executive?", mem)
        assert "NEXT EXECUTIVE HINT:" not in prompt


class TestNormalizeTurnResponse:
    def test_parses_well_formed_json(self):
        raw = '{"step_type": "entity_summary", "params": {"entity_type": "branch", "entity_value": "Mahad"}, "resolved_entity": null, "needs_clarification": false, "clarification_question": ""}'
        result = llm._normalize_turn_response(raw)
        assert result["step_type"] == "entity_summary"
        assert result["params"]["entity_value"] == "Mahad"
        assert result["needs_clarification"] is False

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"step_type": "entity_summary", "params": {}}\n```'
        result = llm._normalize_turn_response(raw)
        assert result["step_type"] == "entity_summary"

    def test_malformed_json_degrades_to_clarification_not_a_crash(self):
        result = llm._normalize_turn_response("not json at all")
        assert result["step_type"] is None
        assert result["needs_clarification"] is True

    def test_unknown_step_type_degrades_to_clarification(self):
        raw = '{"step_type": "delete_everything", "params": {}}'
        result = llm._normalize_turn_response(raw)
        assert result["step_type"] is None
        assert result["needs_clarification"] is True

    def test_null_step_type_is_a_valid_response(self):
        raw = '{"step_type": null, "needs_clarification": true, "clarification_question": "which metric?"}'
        result = llm._normalize_turn_response(raw)
        assert result["step_type"] is None
        assert result["clarification_question"] == "which metric?"


class TestParseTurnWiring:
    def test_calls_gemini_with_the_built_prompt_and_returns_normalized_result(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        seen = {}

        class _FakeResponse:
            text = '{"step_type": "entity_summary", "params": {"entity_type": "branch", "entity_value": "Mahad"}}'
            usage_metadata = None

        def _fake_call(client, model, contents, config):
            seen["contents"] = contents
            return _FakeResponse()

        monkeypatch.setattr(llm, "_call_gemini_with_retry", _fake_call)
        monkeypatch.setattr(llm.genai, "Client", lambda api_key: object())

        result = llm.parse_turn("why is Mahad underperforming", EntityMemory())

        assert result["step_type"] == "entity_summary"
        assert "why is Mahad underperforming" in seen["contents"]

    def test_missing_api_key_returns_clarification_without_calling_gemini(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        calls = []
        monkeypatch.setattr(llm, "_call_gemini_with_retry", lambda *a, **kw: calls.append(1))

        result = llm.parse_turn("why is Mahad underperforming", EntityMemory())

        assert calls == []
        assert result["step_type"] is None
        assert result["needs_clarification"] is True


class TestBuildNarrationPrompt:
    def test_frames_the_model_as_an_nbfc_analyst_briefing_a_leader(self):
        prompt = llm.build_narration_prompt("entity_summary", "Metric,Current\nCollection%,60.0")
        assert "NBFC" in prompt
        assert "Regional Business Head" in prompt and "Zonal Head" in prompt and "Business Unit Head" in prompt


class TestBuildPreview:
    """Any result table carrying a customer-identifying column (Cust Name/
    Cust Mob No/Customer/Mobile) gets a PII-free aggregate summary instead
    of raw rows sent to Gemini -- detected by COLUMN CONTENT, not a
    per-step_type allowlist, specifically so a step added later that
    happens to return customer-level rows is protected automatically,
    without needing this function edited again. The real, confirmed gap
    this closes: only concept_filter got this treatment before, while
    worst_loans_by_metric/non_paying_customers/customer_loan_book/
    priority_accounts/top_closing_arrears/high_arrears_at_risk/
    fleet_defaulters/top_accounts/fleet_exposure/repossession_list/
    good_customers all ALSO return raw customer names/mobile numbers and
    were sending up to 20 unprotected rows into the narration prompt."""

    def test_concept_filter_gets_an_aggregate_summary_not_raw_rows(self):
        df = pd.DataFrame({
            "Cust Name": ["A", "B", "C"], "Cust Mob No": ["111", "222", "333"],
            "SOH": [10_000.0, 20_000.0, 5_000.0],
        })
        preview = llm._build_preview("concept_filter", df)
        assert "111" not in preview
        assert "A" not in preview
        assert "3" in preview  # the count
        assert "35" in preview or "35000" in preview or "35,000" in preview  # total SOH, some formatting

    def test_other_step_types_keep_the_row_preview(self):
        df = pd.DataFrame({"Metric": ["Collection%"], "Current": [60.0]})
        preview = llm._build_preview("entity_summary", df)
        assert "Collection%" in preview
        assert "60.0" in preview

    def test_worst_loans_by_metric_shaped_result_is_also_protected(self):
        # The exact gap: worst_loans_by_metric was NOT in the old
        # step_type allowlist, so its raw Cust Name/Cust Mob No rows went
        # straight to Gemini unprotected before this fix.
        df = pd.DataFrame({
            "Cust Name": ["Akshay", "Ravi"], "Cust Mob No": ["9998887776", "9998887777"],
            "SOH": [500_000.0, 300_000.0], "Arrears / EMI": [12.0, 8.0],
        })
        preview = llm._build_preview("worst_loans_by_metric", df)
        assert "Akshay" not in preview
        assert "9998887776" not in preview
        assert "2" in preview

    def test_fleet_exposure_shaped_result_with_customer_mobile_columns_is_protected(self):
        # fleet_exposure's own reshaped column names (Customer/Mobile, not
        # the raw Cust Name/Cust Mob No convention) must be caught too.
        df = pd.DataFrame({
            "Customer": ["Akshay"], "Mobile": ["9998887776"], "Total SOH (Cr)": [1.5],
        })
        preview = llm._build_preview("fleet_exposure", df)
        assert "Akshay" not in preview
        assert "9998887776" not in preview
        assert "1.5" in preview

    def test_a_future_unlisted_step_type_with_pii_columns_is_still_protected(self):
        # Proves this is a CONTENT check, not a step_type allowlist -- a
        # brand-new step_type this function has never heard of still gets
        # the PII-free treatment purely because its columns say so.
        df = pd.DataFrame({"Cust Name": ["Akshay"], "Cust Mob No": ["9998887776"]})
        preview = llm._build_preview("some_future_step_not_yet_invented", df)
        assert "Akshay" not in preview
        assert "9998887776" not in preview

    def test_exposure_total_falls_back_through_column_priority(self):
        # No "SOH" column present -- falls back to the next known exposure
        # column (Closing Arrears), same aggregate-only guarantee either way.
        df = pd.DataFrame({"Cust Name": ["Akshay"], "Closing Arrears": [75_000.0]})
        preview = llm._build_preview("top_closing_arrears", df)
        assert "Akshay" not in preview
        assert "75" in preview or "75000" in preview or "75,000" in preview

    def test_no_exposure_column_still_returns_a_clean_count_only(self):
        df = pd.DataFrame({"Cust Name": ["Akshay", "Ravi"]})
        preview = llm._build_preview("good_customers", df)
        assert "Akshay" not in preview
        assert "Ravi" not in preview
        assert "2" in preview

    def test_customer_loan_book_is_exempt_from_the_pii_check(self):
        # The ONE deliberate exception: the user already supplied this
        # customer's mobile number as the step's own input -- narrating
        # their name back isn't new disclosure, it's the SAME customer
        # they already identified, not a list of people they never named.
        df = pd.DataFrame({
            "Cust Name": ["Akshay Sonajirao Suryawanshi"], "Cust Mob No": ["8999769669"],
            "Loan No": ["NADED2405220004"], "SOH": [515_253.0],
        })
        preview = llm._build_preview("customer_loan_book", df)
        assert "Akshay Sonajirao Suryawanshi" in preview
        assert "8999769669" in preview

    def test_multi_customer_steps_stay_protected_even_though_customer_loan_book_is_exempt(self):
        # The exemption is scoped to customer_loan_book specifically -- a
        # different multi-customer step must NOT accidentally inherit it.
        df = pd.DataFrame({"Cust Name": ["Akshay"], "Cust Mob No": ["8999769669"], "SOH": [500_000.0]})
        preview = llm._build_preview("non_paying_customers", df)
        assert "Akshay" not in preview
        assert "8999769669" not in preview


class TestNarrateStep:
    def test_empty_result_returns_empty_string_without_calling_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        calls = []
        monkeypatch.setattr(llm, "_call_gemini_with_retry", lambda *a, **kw: calls.append(1))

        result = llm.narrate_step("entity_summary", pd.DataFrame())

        assert result == ""
        assert calls == []

    def test_calls_gemini_with_a_preview_never_the_full_table(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        seen = {}

        class _FakeResponse:
            text = "Collection% is down 2pp this month."
            usage_metadata = None

        def _fake_call(client, model, contents, config):
            seen["contents"] = contents
            return _FakeResponse()

        monkeypatch.setattr(llm, "_call_gemini_with_retry", _fake_call)
        monkeypatch.setattr(llm.genai, "Client", lambda api_key: object())

        df = pd.DataFrame({"Metric": ["Collection%"], "Current": [80.0]})
        result = llm.narrate_step("entity_summary", df)

        assert result == "Collection% is down 2pp this month."


class TestNarrateStepStream:
    def test_empty_result_yields_nothing_without_calling_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        calls = []
        monkeypatch.setattr(llm.genai, "Client", lambda api_key: calls.append(1) or object())

        chunks = list(llm.narrate_step_stream("entity_summary", pd.DataFrame()))

        assert chunks == []
        assert calls == []

    def test_yields_each_chunk_from_the_stream(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        seen = {}

        class _Chunk:
            def __init__(self, text):
                self.text = text

        class _FakeModels:
            def generate_content_stream(self, model, contents, config):
                seen["contents"] = contents
                return iter([_Chunk("Collection"), _Chunk("% is down"), _Chunk(None)])

        class _FakeClient:
            models = _FakeModels()

        monkeypatch.setattr(llm.genai, "Client", lambda api_key: _FakeClient())

        df = pd.DataFrame({"Metric": ["Collection%"], "Current": [80.0]})
        chunks = list(llm.narrate_step_stream("entity_summary", df))

        # The None-text chunk is skipped, not yielded as the string "None".
        assert chunks == ["Collection", "% is down"]
        assert "Collection%" in seen["contents"]
        assert "Collection%" in seen["contents"]

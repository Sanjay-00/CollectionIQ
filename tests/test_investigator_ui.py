"""Regression coverage for ui/tabs/investigator.py's pure (non-Streamlit)
helper functions. _derive_entity_key exists specifically because a live
Gemini check caught a real bug: trusting the LLM's "resolved_entity" blindly
for customer_loan_book overwrote an unrelated executive's stored result
instead of creating a new "customer" entry."""
import pandas as pd
import streamlit as st

from ui.tabs.investigator import (
    _derive_entity_key,
    _get_active_thread_id,
    _get_memory,
    _get_threads,
    _get_turns,
    _maybe_set_thread_title,
    _new_thread,
    _top_active_alerts,
    _turn_dl_key,
    _worst_metric,
)


def _clear_thread_state():
    st.session_state.pop("investigator_threads", None)
    st.session_state.pop("investigator_active_thread", None)


class TestThreads:
    """st.session_state works as a plain in-process dict outside a real
    Streamlit script run (confirmed directly) -- explicit clearing is
    needed since pytest runs every test file in one process and
    session_state is a global singleton."""

    def setup_method(self):
        _clear_thread_state()

    def teardown_method(self):
        _clear_thread_state()

    def test_no_threads_yet_auto_creates_one(self):
        assert _get_threads() == {}
        active = _get_active_thread_id()
        assert active in _get_threads()
        assert _get_threads()[active]["title"] == "New chat"

    def test_new_thread_gets_its_own_empty_memory_and_turns(self):
        first = _get_active_thread_id()
        _get_turns().append({"role": "user", "text": "why is Mahad underperforming"})

        second = _new_thread()
        st.session_state["investigator_active_thread"] = second

        assert _get_turns() == []
        assert _get_memory() is not _get_threads()[first]["memory"]

    def test_switching_back_to_first_thread_preserves_its_turns(self):
        first = _get_active_thread_id()
        _get_turns().append({"role": "user", "text": "why is Mahad underperforming"})

        second = _new_thread()
        st.session_state["investigator_active_thread"] = second
        _get_turns().append({"role": "user", "text": "how are my regions performing"})

        st.session_state["investigator_active_thread"] = first
        assert _get_turns() == [{"role": "user", "text": "why is Mahad underperforming"}]

    def test_title_is_set_once_from_the_first_question_only(self):
        _maybe_set_thread_title("why is Mahad branch underperforming")
        active = _get_active_thread_id()
        assert _get_threads()[active]["title"] == "why is Mahad branch underperforming"

        _maybe_set_thread_title("a completely different second question")
        assert _get_threads()[active]["title"] == "why is Mahad branch underperforming"

    def test_long_question_title_is_truncated(self):
        long_q = "why is the collection percentage so much lower than last month across every single region"
        _maybe_set_thread_title(long_q)
        active = _get_active_thread_id()
        title = _get_threads()[active]["title"]
        assert len(title) <= 43  # 40 chars + "..."
        assert title.endswith("...")


class TestTurnDlKey:
    def test_same_entity_referenced_twice_gets_distinct_keys(self):
        # The exact production crash: a vague follow-up ("whats the
        # issue?") resolved back to an entity already asked about earlier
        # in the SAME conversation -- two chat bubbles both keyed purely on
        # entity identity collided ("multiple elements with the same key").
        # turns is append-only and never reorders, so combining its index
        # with entity identity is both unique and stable.
        key_at_turn_1 = _turn_dl_key(1, "region", "CS Nagar")
        key_at_turn_5 = _turn_dl_key(5, "region", "CS Nagar")
        assert key_at_turn_1 != key_at_turn_5

    def test_same_turn_index_and_entity_is_deterministic(self):
        assert _turn_dl_key(2, "branch", "Mahad") == _turn_dl_key(2, "branch", "Mahad")


class TestDeriveEntityKey:
    def test_entity_summary_uses_its_own_params(self):
        key = _derive_entity_key(
            "entity_summary", {"entity_type": "branch", "entity_value": "Mahad"}, None,
        )
        assert key == ("branch", "Mahad")

    def test_entity_summary_executive_value_becomes_a_tuple(self):
        key = _derive_entity_key(
            "entity_summary", {"entity_type": "executive", "entity_value": ["Rahul", "Pune Unit"]}, None,
        )
        assert key == ("executive", ("Rahul", "Pune Unit"))

    def test_customer_loan_book_never_trusts_a_stale_resolved_entity(self):
        # The exact regression: resolved_entity points at a leftover
        # executive reference from earlier conversational context.
        key = _derive_entity_key(
            "customer_loan_book", {"cust_mob_no": "9876543210"},
            {"entity_type": "executive", "entity_value": ["Rahul", "Pune Unit"]},
        )
        assert key == ("customer", "9876543210")

    def test_underperformer_quartile_keys_on_metric_and_scope(self):
        key = _derive_entity_key(
            "underperformer_quartile", {"metric": "NPA %", "scope_col": "region", "scope_value": "WEST"}, None,
        )
        assert key == ("executive_group", "underperformer_quartile:NPA %:WEST")

    def test_non_paying_customers_keys_on_its_sources(self):
        key = _derive_entity_key(
            "non_paying_customers",
            {"source_entities": [{"entity_type": "branch_executives", "entity_value": "Mahad:Collection %"}]},
            None,
        )
        assert key == ("non_paying_customers", "branch_executives:Mahad:Collection %")


    def test_dimension_breakdown_keys_on_the_dimension_name(self):
        # The exact bug this step type fixes: "how are my regions
        # performing" has no single entity_value at all -- the key must not
        # depend on one existing.
        key = _derive_entity_key("dimension_breakdown", {"dimension": "region"}, None)
        assert key == ("dimension_breakdown", "region")

    def test_roll_rate_summary_keys_on_its_scope(self):
        key = _derive_entity_key("roll_rate_summary", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("roll_rate_summary", "Mahad")

    def test_roll_rate_summary_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("roll_rate_summary", {}, None)
        assert key == ("roll_rate_summary", "portfolio")

    def test_vintage_summary_keys_on_its_scope(self):
        key = _derive_entity_key("vintage_summary", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("vintage_summary", "Mahad")

    def test_roll_rate_by_dimension_keys_on_its_dimension(self):
        key = _derive_entity_key("roll_rate_by_dimension", {"dimension": "branch"}, None)
        assert key == ("roll_rate_by_dimension", "branch")

    def test_high_arrears_at_risk_keys_on_its_scope(self):
        key = _derive_entity_key("high_arrears_at_risk", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("high_arrears_at_risk", "Mahad")

    def test_fleet_defaulters_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("fleet_defaulters", {}, None)
        assert key == ("fleet_defaulters", "portfolio")

    def test_priority_menu_keys_on_its_scope(self):
        key = _derive_entity_key("priority_menu", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("priority_menu", "Mahad")

    def test_top_closing_arrears_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("top_closing_arrears", {}, None)
        assert key == ("top_closing_arrears", "portfolio")

    def test_new_advances_summary_keys_on_its_scope(self):
        key = _derive_entity_key("new_advances_summary", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("new_advances_summary", "Mahad")

    def test_new_advances_summary_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("new_advances_summary", {}, None)
        assert key == ("new_advances_summary", "portfolio")

    def test_new_advances_by_dimension_keys_on_its_dimension_and_scope(self):
        key = _derive_entity_key(
            "new_advances_by_dimension",
            {"dimension": "branch", "scope_col": "region", "scope_value": "WEST"},
            None,
        )
        assert key == ("new_advances_by_dimension", "branch:WEST")

    def test_new_advances_trend_keys_on_granularity_and_scope(self):
        key = _derive_entity_key(
            "new_advances_trend", {"granularity": "Quarterly", "scope_col": "branch", "scope_value": "Mahad"}, None,
        )
        assert key == ("new_advances_trend", "Quarterly:Mahad")

    def test_new_advances_trend_unscoped_defaults_to_monthly_portfolio(self):
        key = _derive_entity_key("new_advances_trend", {}, None)
        assert key == ("new_advances_trend", "Monthly:portfolio")

    def test_product_analysis_keys_on_its_axis_and_scope(self):
        key = _derive_entity_key("product_analysis", {"axis": "segment", "scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("product_analysis", "segment:Mahad")

    def test_top_accounts_keys_on_its_scope(self):
        key = _derive_entity_key("top_accounts", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("top_accounts", "Mahad")

    def test_top_accounts_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("top_accounts", {}, None)
        assert key == ("top_accounts", "portfolio")

    def test_fleet_exposure_keys_on_its_scope(self):
        key = _derive_entity_key("fleet_exposure", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("fleet_exposure", "Mahad")

    def test_fleet_exposure_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("fleet_exposure", {}, None)
        assert key == ("fleet_exposure", "portfolio")

    def test_repossession_list_keys_on_its_scope(self):
        key = _derive_entity_key("repossession_list", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("repossession_list", "Mahad")

    def test_good_customers_keys_on_its_scope(self):
        key = _derive_entity_key("good_customers", {"scope_col": "branch", "scope_value": "Mahad"}, None)
        assert key == ("good_customers", "Mahad")

    def test_good_customers_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("good_customers", {}, None)
        assert key == ("good_customers", "portfolio")

    def test_worst_loans_by_metric_keys_on_metric_and_scope(self):
        key = _derive_entity_key(
            "worst_loans_by_metric",
            {"metric": "NPA %", "scope_col": "executive", "scope_value": ["RAHUL", "PUNE"]},
            None,
        )
        assert key == ("worst_loans_by_metric", "NPA %:('RAHUL', 'PUNE')")

    def test_worst_loans_by_metric_unscoped_keys_on_portfolio(self):
        key = _derive_entity_key("worst_loans_by_metric", {"metric": "Collection %"}, None)
        assert key == ("worst_loans_by_metric", "Collection %:portfolio")

    def test_demand_summary_keys_on_its_dimension_and_scope(self):
        key = _derive_entity_key(
            "demand_summary",
            {"dimension": "executive", "scope_col": "branch", "scope_value": "Mahad"},
            None,
        )
        assert key == ("demand_summary", "executive:Mahad")

    def test_concept_breakdown_keys_on_its_sources_and_dimension(self):
        key = _derive_entity_key(
            "concept_breakdown",
            {
                "source_entities": [{"entity_type": "concept_filter", "entity_value": "non_starter"}],
                "dimension": "branch",
            },
            None,
        )
        assert key == ("concept_breakdown", "concept_filter:non_starter:branch")


class TestTopActiveAlerts:
    """The proactive opening message's own filter/sort -- mirrors
    report_agent/sections/risk_flags.py::compute_risk_flags's severity
    ranking exactly (critical > high > medium, then count descending),
    but operates on an ALREADY-COMPUTED alerts list (app.py's own cached
    run_all_alerts call, reused rather than re-run) instead of recomputing
    from a raw DataFrame."""

    def _alert(self, title, severity, count):
        return {"title": title, "severity": severity, "count": count}

    def test_zero_count_alerts_are_excluded(self):
        alerts = [self._alert("A", "critical", 0), self._alert("B", "high", 5)]
        result = _top_active_alerts(alerts)
        assert [a["title"] for a in result] == ["B"]

    def test_sorted_by_severity_then_count_descending(self):
        alerts = [
            self._alert("Medium1", "medium", 100),
            self._alert("Critical1", "critical", 1),
            self._alert("High1", "high", 50),
            self._alert("High2", "high", 200),
        ]
        result = _top_active_alerts(alerts)
        assert [a["title"] for a in result] == ["Critical1", "High2", "High1"]

    def test_limited_to_top_n(self):
        alerts = [self._alert(f"A{i}", "critical", i + 1) for i in range(5)]
        result = _top_active_alerts(alerts, limit=3)
        assert len(result) == 3

    def test_empty_or_none_returns_empty(self):
        assert _top_active_alerts([]) == []
        assert _top_active_alerts(None) == []


class TestWorstMetric:
    def test_none_when_no_deltas_present(self):
        df = pd.DataFrame({
            "Metric": ["Collection%", "NPA%"], "Current": [80.0, 5.0],
            "Previous": [None, None], "Delta": [None, None],
            "Higher Is Better": [True, False],
        })
        assert _worst_metric(df) is None

    def test_picks_the_metric_that_moved_the_wrong_direction(self):
        df = pd.DataFrame({
            "Metric": ["Collection%", "NPA%"], "Current": [80.0, 10.0],
            "Previous": [85.0, 5.0], "Delta": [-5.0, 5.0],
            "Higher Is Better": [True, False],
        })
        # Collection% fell 5pp (bad, higher_is_better) and NPA% rose 5pp
        # (bad, lower_is_better) -- tie in magnitude, either is a defensible
        # pick, but the function must return one of them, not None/crash.
        assert _worst_metric(df) in ("Collection%", "NPA%")

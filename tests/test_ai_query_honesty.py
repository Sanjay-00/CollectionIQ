"""AI Query must never answer a different question quietly (audit, 2026-10-16).

Each class is one way it used to: a filter it couldn't apply returned every
loan; "last month" missed loans that had since left the book; "X vs last
month" dropped the comparison; an "in" filter on a number column crashed;
AI observations quoted numbers that weren't in the result; the summary cards
averaged each unit's rate.
"""
import pandas as pd
import pytest

from agents.data_executor import _apply_condition
from agents.plan_executor import execute_plan
from compiler.core import compile_logical
from helpers import make_df


def _df():
    return pd.DataFrame({
        "Loan No": ["L1", "L2", "L3", "L4", "L5"],
        "Due Dt": [5, 10, 15, 20, 5],
        "Arrears / EMI": [0.0, 1.5, 3.0, 0.2, 7.0],
        "RegionName": ["PUNE", "PUNE", "PUNE", "NASIK", "NASIK"],
    })


class TestFilterThatCannotApplyRefuses:
    @pytest.mark.parametrize("cond", [
        {"column": "Arrears / EMI", "op": ">", "value": "high"},     # text on a number column
        {"column": "RegionName", "op": ">", "value": "M"},           # order on a text column
        {"column": "RegionName", "op": "startswith", "value": "P"},  # unknown op
        {"column": "Zone", "op": "==", "value": "WEST"},             # column not in the file
    ], ids=["text_on_number", "order_on_text", "unknown_op", "missing_column"])
    def test_raises_instead_of_returning_every_loan(self, cond):
        with pytest.raises(ValueError):
            _apply_condition(_df(), cond)

    def test_in_on_a_number_column(self):
        # Used to crash: every comparison was built up front, including series == [5, 10].
        assert list(_apply_condition(_df(), {"column": "Due Dt", "op": "in", "value": [5, 10]})["Loan No"]) \
            == ["L1", "L2", "L5"]

    def test_value_with_stray_spaces_matches(self):
        assert len(_apply_condition(_df(), {"column": "RegionName", "op": "==", "value": " pune "})) == 3
        assert len(_apply_condition(_df(), {"column": "RegionName", "op": "in", "value": [" nasik"]})) == 2


class TestViewFilterOnMissingColumn:
    def test_falls_through_instead_of_serving_the_whole_portfolio(self):
        from graph import view_node
        state = {"query": "t", "result_df_full": make_df([{"RegionName": "PUNE"}, {"RegionName": "NASIK"}]),
                 "df_prev": pd.DataFrame(), "precomputed_views": {}, "rr_meta": {},
                 "ir1": {"view": {"name": "region_scorecard", "params": {},
                                  "filters": [{"column": "Zone", "op": "==", "value": "WEST"}]}}}
        out = view_node(state)
        assert out["ir1"]["view"] is None          # handed to the compiler, which will say Zone is missing


# ── Last month = last month's whole book ─────────────────────────────────────

def _months():
    """This month: L1-L3. Last month: L1-L3 plus L4 (NPA), which has since closed."""
    prev = make_df([
        {"Loan No": "L1", "RegionName": "PUNE", "curr_bucket": "NPA", "SOH": 100.0},
        {"Loan No": "L2", "RegionName": "PUNE", "curr_bucket": "STD", "SOH": 100.0},
        {"Loan No": "L3", "RegionName": "NASIK", "curr_bucket": "STD", "SOH": 100.0},
        {"Loan No": "L4", "RegionName": "NASIK", "curr_bucket": "NPA", "SOH": 500.0},
    ])
    curr = make_df([
        {"Loan No": "L1", "RegionName": "PUNE", "curr_bucket": "NPA", "SOH": 90.0},
        {"Loan No": "L2", "RegionName": "PUNE", "curr_bucket": "NPA", "SOH": 90.0},
        {"Loan No": "L3", "RegionName": "NASIK", "curr_bucket": "STD", "SOH": 90.0},
    ])
    carried = prev[["Loan No", "curr_bucket", "SOH"]].rename(columns={"curr_bucket": "prev_bucket", "SOH": "prev_SOH"})
    return curr.merge(carried, on="Loan No", how="left"), prev


def _run(ir, curr, prev):
    base = {"intent": "aggregation", "view": None, "filters": [], "dimensions": ["region"], "measures": [],
            "metrics": [], "having": [], "order_by": [], "limit": None, "display_columns": [], "time": None}
    plan, errs = compile_logical({**base, **ir}, list(curr.columns))
    assert errs == [], errs
    result, err = execute_plan(curr, plan, df_prev=prev)
    assert err == "", err
    return result.set_index("RegionName")


class TestLastMonthIsLastMonthsWholeBook:
    def test_prev_count_includes_a_loan_that_has_since_left(self):
        curr, prev = _months()
        r = _run({"measures": [{"metric": "prev_npa_count", "alias": "prev_npa"},
                               {"metric": "npa_count", "alias": "npa"}]}, curr, prev)
        assert r.loc["NASIK", "prev_npa"] == 1          # L4, closed since: still last month's NPA
        assert r.loc["PUNE", "prev_npa"] == 1 and r.loc["PUNE", "npa"] == 2

    def test_prev_amount_matches_last_months_file(self):
        curr, prev = _months()
        r = _run({"measures": [{"metric": "prev_exposure", "alias": "prev_soh"}]}, curr, prev)
        assert r["prev_soh"].sum() == prev["SOH"].sum() == 800.0

    def test_without_last_months_file_the_carried_values_stand(self):
        curr, _ = _months()
        r = _run({"measures": [{"metric": "prev_npa_count", "alias": "prev_npa"}]}, curr, None)
        assert r["prev_npa"].sum() == 1                 # only L1 is visible on this month's loans

    def test_a_filter_on_this_months_state_keeps_the_same_loans(self):
        # "Loans NPA now: how many were NPA last month" is about THIS month's NPA loans.
        curr, prev = _months()
        r = _run({"filters": [{"concept": "npa"}],
                  "measures": [{"metric": "prev_npa_count", "alias": "prev_npa"}]}, curr, prev)
        assert r["prev_npa"].sum() == 1                 # L1 only, not last month's L4

    def test_a_region_that_only_existed_last_month_gets_a_row(self):
        curr, prev = _months()
        prev = pd.concat([prev, make_df([{"Loan No": "L9", "RegionName": "THANE", "curr_bucket": "NPA"}])])
        r = _run({"measures": [{"metric": "prev_npa_count", "alias": "prev_npa"},
                               {"metric": "npa_count", "alias": "npa"}]}, curr, prev)
        assert r.loc["THANE", "prev_npa"] == 1 and r.loc["THANE", "npa"] == 0


class TestVsLastMonthNeverDropsTheComparison:
    def test_count_measure_gets_its_last_month_column(self):
        curr, prev = _months()
        r = _run({"measures": [{"metric": "npa_count", "alias": "npa"}],
                  "time": {"compare": {"type": "change", "from": "prev", "to": "curr"}}}, curr, prev)
        assert {"npa", "prev_npa", "npa_change"} <= set(r.columns)
        assert r.loc["NASIK", "npa_change"] == -1

    def test_explicit_last_month_measure_plus_a_time_block(self):
        # The planner's real IR for the "biggest NPA reduction" example chip
        # (traced 2026-10-24): it names prev_npa_count AND adds time.compare.
        # prev_npa is already last month's side; it used to fail as "no
        # last-month value for prev_bucket", every time, repair included.
        curr, prev = _months()
        ir = {"measures": [{"metric": "prev_npa_count", "alias": "prev_npa"},
                           {"metric": "npa_count", "alias": "curr_npa"}],
              "metrics": [{"alias": "npa_reduction", "expr": "prev_npa - curr_npa"}],
              "order_by": [{"by": "npa_reduction", "dir": "desc"}]}
        with_time = _run({**ir, "time": {"grain": "month", "compare": {"type": "snapshot", "from": "prev",
                                                                       "to": "curr"}}}, curr, prev)
        without = _run(ir, curr, prev)
        pd.testing.assert_frame_equal(with_time, without)   # no duplicate last-month column either
        assert with_time.loc["NASIK", "prev_npa"] == 1 and with_time.loc["PUNE", "npa_reduction"] == -1

    def test_change_reuses_the_explicit_last_month_measure(self):
        # "npa" vs last month would be named prev_npa: the same figure the IR
        # already has under that name, so it is reused, not added twice.
        curr, prev = _months()
        r = _run({"measures": [{"metric": "prev_npa_count", "alias": "prev_npa"},
                               {"metric": "npa_count", "alias": "npa"}],
                  "time": {"compare": {"type": "change", "from": "prev", "to": "curr"}}}, curr, prev)
        assert list(r.columns) == ["Rank", "prev_npa", "npa", "npa_change"]
        assert r.loc["PUNE", "npa_change"] == 1 and r.loc["NASIK", "npa_change"] == -1

    @pytest.mark.parametrize("prev_loaded", [True, False])
    def test_upload_tip_only_when_last_month_is_missing(self, monkeypatch, prev_loaded):
        # The tip said "upload a previous-period file" with one uploaded.
        import graph
        ir = {"intent": "aggregation", "dimensions": ["region"],
              # NPA last month AND in arrears now: has no "month before" version.
              "measures": [{"agg": "count", "alias": "x", "where": [
                  {"column": "prev_bucket", "op": "==", "value": "NPA"},
                  {"column": "Arrears / EMI", "op": ">", "value": 0}]}],
              "time": {"compare": {"type": "change", "from": "prev", "to": "curr"}}}
        monkeypatch.setattr(graph, "plan_logical", lambda *a, **k: ir)   # the repair returns the same IR
        curr, _ = _months()
        dates = {"curr": "2026-09-30", **({"prev": "2026-08-31"} if prev_loaded else {})}
        out = graph.compile_and_validate_node({"query": "t", "ir1": ir, "result_df_full": curr,
                                               "snapshot_dates": dates})
        assert "Could not compile" in out["error"]
        assert ("upload a previous-period file" in out["error"]) is (not prev_loaded)

    @pytest.mark.parametrize("compare", ["snapshot", "change"])
    def test_a_roll_rate_is_already_the_comparison(self, compare):
        """A roll rate IS last month vs this month; there is no "last month's
        roll rate" in a two-file upload. The planner adds a time block to
        "roll forward rate by branch" anyway, and that failed the question
        (live, 2026-10-28). It's shown as it is."""
        curr, prev = _months()
        ir = {"measures": [{"metric": "roll_forward_pct", "alias": "roll_fwd"}]}
        with_time = _run({**ir, "time": {"grain": "month", "compare": {"type": compare, "from": "prev",
                                                                       "to": "curr"}}}, curr, prev)
        pd.testing.assert_frame_equal(with_time, _run(ir, curr, prev))
        assert with_time.loc["PUNE", "roll_fwd"] == 50.0           # L2 went STD -> NPA, L1 stayed NPA

    def test_a_measure_with_no_last_month_value_is_an_error(self):
        curr, _ = _months()
        ir = {"intent": "aggregation", "dimensions": ["region"],
              "measures": [{"column": "Month Receipt Amount", "agg": "sum", "alias": "rcpt"}],
              "time": {"compare": {"type": "change", "from": "prev", "to": "curr"}}}
        _, errs = compile_logical(ir, list(curr.columns))
        assert any("last month" in e for e in errs)


class TestRateColumnIsNotAddedUp:
    def test_metric_naming_a_rate_column_is_refused(self):
        cols = ["Unit", "LCC%"]
        _, errs = compile_logical({"intent": "aggregation", "dimensions": ["branch"],
                                   "measures": [{"metric": "lcc%", "alias": "x"}]}, cols)
        assert any("can't be added up" in e for e in errs)


class TestInsightNumbersAreChecked:
    def test_a_bullet_with_a_number_not_in_the_result_is_removed(self):
        from agents.insight_generator import check_numbers
        facts = "RESULT TABLE: PUNE, NPA%: 12.5, SOH (Cr): 345.61"
        out = check_numbers("• PUNE NPA% is 12.50\n• SOH is ₹350 Cr\n• Focus on the top 3 branches", facts)
        assert "12.50" in out and "top 3" in out
        assert "350" not in out and "1 observation(s) removed" in out

    def test_all_bullets_unverified(self):
        from agents.insight_generator import check_numbers
        assert "removed" in check_numbers("• NPA rose 44.4%", "NPA%: 12.5")


class TestSummaryCardIsThePortfolioRate:
    def test_rate_card_is_not_an_average_of_unit_rates(self):
        from graph import _build_portfolio_kpis
        # PUNE: 1 loan, NPA. NASIK: 9 loans, none NPA. Average of the two rates
        # would be 50%; the portfolio's real NPA% is 10%.
        loans = make_df([{"RegionName": "PUNE", "curr_bucket": "NPA"}]
                        + [{"RegionName": "NASIK", "curr_bucket": "STD"}] * 9)
        table = pd.DataFrame({"Region": ["PUNE", "NASIK"], "NPA%": [100.0, 0.0]})
        cards = _build_portfolio_kpis({"metrics": ["NPA%"]}, table, loans)
        assert cards == [{"label": "Portfolio NPA%", "value": "10.00%"}]


# ── Pre-commit review findings (2026-10-16) ──────────────────────────────────

class TestReviewFindings:
    def test_roll_forward_vs_last_month_is_refused_not_zero(self):
        # Last month's roll forward needs the month before; it used to become
        # "prev_bucket worse than prev_bucket" -- always 0.
        from compiler.core import _resolve_time_compare
        errs: list = []
        aggs, _ = _resolve_time_compare({"compare": {"type": "change"}},
                                        [{"metric": "roll_forward_count", "alias": "rf"}], errs)
        assert aggs == [] and any("last month" in e for e in errs)

    def test_same_month_column_comparison_maps_both_sides(self):
        from compiler.core import _map_where_to_prev
        got = _map_where_to_prev([{"column": "Closing Arrears", "op": "col_gt", "value": "ClosingPC"}])
        assert got[0] == {"column": "prev_Closing_Arrears", "op": "col_gt", "value": "prev_ClosingPC"}

    def test_bucket_stable_runs(self):
        df = make_df([{"curr_bucket": "NPA", "prev_bucket": "NPA"}, {"curr_bucket": "STD", "prev_bucket": "NPA"},
                      {"curr_bucket": "STD", "prev_bucket": None}])
        assert len(_apply_condition(df, {"column": "curr_bucket", "op": "bucket_stable", "value": "prev_bucket"})) == 1

    def test_all_blank_flag_column_matches_nothing(self):
        # A file with no co-lending loans reads CoLending_Loans as float NaN.
        df = make_df([{"Loan No": "A"}, {"Loan No": "B"}]).assign(CoLending_Loans=float("nan"))
        assert _apply_condition(df, {"column": "CoLending_Loans", "op": "in", "value": ["Y", "YES"]}).empty

    def test_int_and_float_keys_are_one_group_and_counts_stay_whole(self):
        curr = make_df([{"Loan No": "L1", "Zone": 5, "curr_bucket": "NPA", "prev_bucket": "NPA"}])
        prev = make_df([{"Loan No": "L1", "Zone": 5.0, "curr_bucket": "NPA"},
                        {"Loan No": "L9", "Zone": None, "curr_bucket": "STD"}])
        plan, errs = compile_logical({"intent": "aggregation", "dimensions": ["zone"],
                                      "measures": [{"metric": "npa_count", "alias": "npa"}],
                                      "time": {"compare": {"type": "snapshot"}}}, list(curr.columns))
        r, err = execute_plan(curr, plan, df_prev=prev)
        assert err == "" and errs == []
        five = r[r["Zone"] == 5.0]
        assert len(five) == 1 and five["npa"].iloc[0] == 1 and five["prev_npa"].iloc[0] == 1
        assert str(r["npa"].dtype).startswith("int")

    def test_rupee_amount_with_a_slash_in_its_name_adds_up(self):
        cols = ["Unit", "Uncleared Cheque/Amount Not remitted by RE"]
        _, errs = compile_logical({"intent": "aggregation", "dimensions": ["branch"],
                                   "measures": [{"metric": "uncleared_cheque/amount_not_remitted_by_re"}]}, cols)
        assert errs == []

    def test_narration_quoting_a_rounded_value_is_kept(self):
        from investigator.llm import _build_preview, _checked
        preview = _build_preview("dimension_breakdown", pd.DataFrame({"Collection%": [87.23456]}))
        assert _checked("Collection is 87.23%.", preview) == "Collection is 87.23%."

    def test_pulse_card_shows_no_change_when_last_month_cant_compute_it(self):
        from analysis.portfolio_intelligence import compute_pulse_kpis
        curr = make_df([{"ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": 5_000.0, "Arrears / EMI": 0.5}])
        prev = make_df([{"Arrears / EMI": 0.5}]).drop(columns=["ARREARS AGAINST EXP"])
        card = next(k for k in compute_pulse_kpis(curr, prev) if k["label"] == "Insurance Debit Cases")
        assert card["delta"] is None

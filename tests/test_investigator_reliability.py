"""Investigator reliability (audit 2026-10-23, real Sept closing): an answer
is either right or a clear "can't", never a false zero, never the wrong
measure, never a "why did it change" read from one month alone."""
import pandas as pd
import pytest

from investigator.guardrails import validate_and_correct
from investigator.state import EntityMemory
from investigator.steps import (
    dimension_breakdown, executive_roster, non_paying_customers, underperformer_quartile,
)
from ui.tabs.investigator import _execute_step, _resolve_source_tables, _result_view
from utils import unit_metrics

_FLAG = "No Coll 3 Months and >6 EMI"


def _book():
    """Two branches; AKLUJ has 2 loans not paying, PUNE has 1."""
    rows = [
        ("AKLUJ", "SOLAPUR", "RAJ KUMAR", 8.0, "Y"), ("AKLUJ", "SOLAPUR", "RAJ KUMAR", 7.0, "Y"),
        ("AKLUJ", "SOLAPUR", "AMIT", 1.0, "N"), ("AKLUJ", "SOLAPUR", "AMIT", 0.0, "N"),
        ("PUNE", "PUNE", "SEEMA", 9.0, "Y"), ("PUNE", "PUNE", "SEEMA", 0.0, "N"),
    ]
    df = pd.DataFrame(rows, columns=["Unit", "RegionName", "MNT NAME", "Arrears / EMI", _FLAG])
    return df.assign(**{"Loan No": [f"L{i}" for i in range(len(df))], "Cust Name": "C", "Cust Mob No": "9",
                        "SOH": 100000.0, "POS": 100000.0, "Loan Status": "RUN"})


class TestNoFalseZero:
    def test_not_paying_for_a_named_branch_needs_no_earlier_result(self):
        out = non_paying_customers(_book(), scope_col="branch", scope_value="akluj")
        assert len(out) == 2

    def test_a_branch_that_does_not_exist_is_refused(self):
        with pytest.raises(ValueError, match="no branch named 'NOWHERE'"):
            non_paying_customers(_book(), scope_col="branch", scope_value="NOWHERE")

    def test_missing_earlier_result_is_redirected_to_the_branch_it_names(self):
        # Live: "customers not paying in Akluj" as a first question pointed at a
        # "branch_executives" table that was never built, and showed 0 (real: 27).
        turn = {"step_type": "non_paying_customers", "needs_clarification": False,
                "params": {"source_entities": [{"entity_type": "branch_executives",
                                                "entity_value": "Akluj:Collection %"}]}}
        fixed = validate_and_correct(turn, EntityMemory(), _book())
        assert fixed["params"] == {"scope_col": "branch", "scope_value": "AKLUJ"}
        out = _execute_step(fixed["step_type"], fixed["params"], _book(), pd.DataFrame(), EntityMemory())
        assert len(out) == 2

    def test_missing_earlier_result_with_no_place_named_is_refused(self):
        params = {"source_entities": [{"entity_type": "executive_group", "entity_value": "something old"}]}
        with pytest.raises(ValueError, match="earlier answer that isn't in this chat"):
            _execute_step("non_paying_customers", params, _book(), pd.DataFrame(), EntityMemory())

    def test_typed_executive_name_is_matched_to_the_files_spelling(self):
        params = {"source_entities": [{"entity_type": "executive", "entity_value": ["raj", "akluj"]}]}
        (table,) = _resolve_source_tables(params, EntityMemory(), _book())
        assert table.iloc[0].to_dict() == {"MNT NAME": "RAJ KUMAR", "Unit": "AKLUJ"}
        out = _execute_step("non_paying_customers", params, _book(), pd.DataFrame(), EntityMemory())
        assert len(out) == 2

    def test_an_earlier_result_that_really_had_no_rows_stays_empty(self):
        mem = EntityMemory()
        mem.touch("executive_group", "none", result_df=pd.DataFrame(), step_type="underperformer_quartile")
        params = {"source_entities": [{"entity_type": "executive_group", "entity_value": "none"}]}
        assert _execute_step("non_paying_customers", params, _book(), pd.DataFrame(), mem).empty

    def test_executive_ranking_accepts_either_spelling_and_refuses_unknown(self):
        from config import SCORECARD_MIN_ACCOUNTS
        df = pd.concat([_book()] * SCORECARD_MIN_ACCOUNTS, ignore_index=True)   # every executive ranked
        df["Loan No"] = [f"L{i}" for i in range(len(df))]
        ranked = underperformer_quartile(df, "NPA%")
        assert not ranked.empty and ranked.equals(underperformer_quartile(df, "NPA %"))
        with pytest.raises(ValueError, match="can't rank executives by 'SOH'"):
            executive_roster(df, n=2, metric="SOH")
        assert list(executive_roster(df, n=1, metric="Delinquency%", worst_first=True)["MNT NAME"]) == ["RAJ KUMAR"]


def test_executive_with_no_emi_due_is_not_ranked_weakest_on_collection():
    # Real Sept file: 2 of 766 executives had no EMI due; the shared engine
    # shows their Collection % as 0.0, which put them first among the "weakest".
    from investigator.steps import _rankable
    sc = pd.DataFrame({"Executive (Branch)": ["NONE DUE", "LOW"], "Collection %": [0.0, 40.0],
                       "Demand (L)": [0.0, 2.0], "NPA %": [0.0, 5.0]})
    assert list(_rankable(sc, "Collection %")["Executive (Branch)"]) == ["LOW"]
    assert len(_rankable(sc, "NPA %")) == 2                     # every other measure ranks everyone


class TestDelinquentMeansDelinquent:
    def _turn(self, metric="NPA %"):
        return {"step_type": "underperformer_quartile", "params": {"metric": metric}, "needs_clarification": False}

    def test_most_delinquent_loans_ranks_by_delinquency_not_npa(self):
        out = validate_and_correct(self._turn(), EntityMemory(), question="which executives have the most delinquent loans")
        assert out["params"]["metric"] == "Delinquency %"

    def test_a_question_that_says_npa_keeps_npa(self):
        out = validate_and_correct(self._turn(), EntityMemory(), question="which delinquent executives have high NPA")
        assert out["params"]["metric"] == "NPA %"

    def test_branch_and_region_tables_show_delinquency_from_the_shared_engine(self):
        out = dimension_breakdown(_book(), "branch").set_index("Unit")
        engine = unit_metrics(_book(), ["Unit"]).set_index("Unit")
        assert (out["Delinquency%"] == engine.loc[out.index, "Delinquency%"]).all()


def test_typed_place_names_take_the_files_spelling():
    from ui.tabs.investigator import _canonical_params
    out = _canonical_params({"scope_col": "region", "scope_value": "solapur ", "entity_type": "branch",
                             "entity_value": "akluj"}, _book())
    assert out["scope_value"] == "SOLAPUR" and out["entity_value"] == "AKLUJ"
    assert _canonical_params({"scope_col": "branch", "scope_value": "nowhere"}, _book())["scope_value"] == "nowhere"


class TestOneMonthOnly:
    def test_change_columns_left_out_and_said_when_no_previous_month(self):
        summary = pd.DataFrame({"Metric": ["NPA%"], "Current": [8.4], "Previous": [None], "Delta": [None]})
        view, note = _result_view("entity_summary", summary, pd.DataFrame())
        assert list(view.columns) == ["Metric", "Current"] and "Only this month's file" in note

    def test_with_previous_month_nothing_is_hidden(self):
        summary = pd.DataFrame({"Metric": ["NPA%"], "Current": [8.4], "Previous": [8.0], "Delta": [0.4]})
        view, note = _result_view("entity_summary", summary, pd.DataFrame({"x": [1]}))
        assert "Delta" in view.columns and note is None

"""Investigator answers in plain words (2026-10-23): a headline read off the
step's own table, the Root Cause split for one unit, plain step names and a
table without internal columns."""
import pandas as pd

from investigator.headline import STEP_LABELS, display_table, headline, why_split
from investigator.steps import STEP_REGISTRY


def test_every_step_has_a_plain_name():
    assert set(STEP_REGISTRY) <= set(STEP_LABELS)
    assert not any("_" in label for label in STEP_LABELS.values())


class TestHeadline:
    def test_ranking_names_the_weakest_and_strongest_with_their_own_figures(self):
        t = pd.DataFrame({"Unit": ["AKLUJ", "PUNE", "BARSHI"], "Collection%": [71.5, 88.0, 93.25]})
        h = headline("dimension_breakdown", t, {"dimension": "branch", "scope_value": "SOLAPUR"})
        assert h == ("3 branches in SOLAPUR. Weakest on Collection%: AKLUJ at 71.50%. "
                     "Strongest: BARSHI at 93.25%.")

    def test_ranking_by_the_metric_asked_in_either_spelling(self):
        t = pd.DataFrame({"Unit": ["A", "B"], "Collection%": [80.0, 90.0], "NPA%": [9.0, 2.0]})
        assert "Weakest on NPA%: A at 9.00%" in headline("dimension_breakdown", t, {"dimension": "branch",
                                                                                     "metric": "NPA %"})

    def test_summary_states_changes_without_judging_them(self):
        t = pd.DataFrame({"Metric": ["Collection%", "NPA %", "Delinquency%"], "Current": [87.76, 8.36, 38.6],
                          "Previous": [88.0, 8.0, 38.6], "Delta": [-0.24, 0.36, 0.0]})
        h = headline("entity_summary", t, {}, entity_value="SOLAPUR")
        assert h == ("SOLAPUR: Collection% 87.76%, NPA% 8.36%, Delinquency% 38.60%. "
                     "Collection% down 0.24 % points MoM; NPA% up 0.36 % points MoM.")
        assert not any(w in h.lower() for w in ("worse", "better", "improv", "deteriorat"))

    def test_summary_for_an_executive_names_the_branch_too(self):
        t = pd.DataFrame({"Metric": ["Collection %"], "Current": [70.0], "Delta": [None]})
        assert headline("entity_summary", t, {}, entity_value=("RAJ", "AKLUJ")) == "RAJ (AKLUJ): Collection% 70.00%."

    def test_loan_list_counts_loans_soh_and_where_most_are(self):
        t = pd.DataFrame({"Unit": ["AKLUJ", "AKLUJ", "PUNE"], "SOH": [1e7, 5e6, 5e6]})
        assert headline("non_paying_customers", t, {"scope_value": "SOLAPUR"}) == (
            "3 loans not paying for 3 months in SOLAPUR, ₹2.00 Cr SOH. Most in AKLUJ (2).")

    def test_rule_lists_use_the_rules_own_name(self):
        assert headline("concept_filter", pd.DataFrame({"Loan No": ["L1"]}), {"concept": "non_starter"}) == \
            "1 loan: Non Starter."

    def test_an_odd_table_gives_no_headline_never_an_error(self):
        assert headline("dimension_breakdown", pd.DataFrame({"x": [1]}), {}) is None
        assert headline("entity_summary", pd.DataFrame({"Metric": ["SOH (Cr)"], "Current": [None]}), {}) is None
        assert headline("roll_rate_summary", pd.DataFrame({"a": [1]}), {}) is None


def test_table_shows_plain_columns_and_hides_internal_ones():
    t = pd.DataFrame({"Executive (Branch)": ["RAJ (AKLUJ)"], "MNT NAME": ["RAJ"], "Unit": ["AKLUJ"],
                      "curr_bucket": ["NPA"], "curr_score": [4], "Higher Is Better": [True],
                      "NPA %": [9.0], "SaleType": [None]})
    assert list(display_table("underperformer_quartile", t).columns) == ["Executive (Branch)", "NPA%"]
    loans = pd.DataFrame({"Cust Name": ["A"], "Unit": ["AKLUJ"], "curr_bucket": ["NPA"],
                          "Not Paying 3M (receipts)": ["Y"]})
    assert list(display_table("non_paying_customers", loans).columns) == ["Customer", "Branch", "Bucket"]
    s = pd.DataFrame({"Metric": ["Collection %", "Total SOH (L)"], "Current": [70.0, 12.0]})
    assert list(display_table("entity_summary", s)["Metric"]) == ["Collection%", "Total SOH (L)"]


def test_why_split_is_the_dashboards_own_split_for_that_unit():
    from analysis.root_cause import (
        EARLY_DELINQUENCY, GROUP_LABELS, HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, root_cause_snapshot,
    )
    book = pd.DataFrame({
        "Loan No": [f"L{i}" for i in range(6)], "Unit": ["AKLUJ"] * 6, "RegionName": ["SOLAPUR"] * 6,
        "Arrears / EMI": [7.0, 7.0, 2.0, 0.0, 1.0, 0.5], "No Coll 3 Months and >6 EMI": ["Y", "N", "Y", "N", "N", "N"],
        "ARREARS AGAINST INST": [9, 9, 3, 0, 1, 0], "ARREARS AGAINST EXP": [0, 0, 0, 0, 0, 5000],
    })
    why = why_split(book)
    snap = root_cause_snapshot(book)
    assert why["delinquent"] == snap["delinquent"] == 5
    order = [g for g in (HARD_NOT_PAYING, HARD_STILL_PAYING, WAS_SILENT, EARLY_DELINQUENCY, "Insurance-Only")
             if g in snap["groups"]]
    assert why["rows"] == [(GROUP_LABELS[g], *snap["groups"][g]) for g in order]
    assert sum(n for label, n, _ in why["rows"] if label != GROUP_LABELS["Insurance-Only"]) == 5

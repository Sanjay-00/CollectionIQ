"""A loan belongs to the executive who holds it NOW (user decision 2026-10-23:
"from now he will handle that loan, so accountable"). An executive's "last
month" figure is the loans they hold now, as they stood last month, so a book
handed over is never read as a jump. Regions and branches keep last month's
file as it is (the official totals).

Real Aug -> Sept Mumbai files: 328 of 747 executives had a different last-month
NPA count under the old "last month's own book" rule; 70 by 5 or more loans."""
from analysis.action_center import attention
from analysis.executive_scorecard import compute_executive_scorecard
from analysis.summary import unit_table
from agents.plan_executor import execute_plan
from tests.helpers import make_df
from utils import prev_held_now


def _handover(n=20):
    """RAVI took over SURESH's NPA loans; nothing about the loans changed."""
    prev = make_df([{"MNT NAME": "SURESH", "Unit": "PUNE", "curr_bucket": "NPA", "Arrears / EMI": 4.0}] * n
                   + [{"MNT NAME": "RAVI", "Unit": "PUNE", "curr_bucket": "STD", "Arrears / EMI": 0.0}] * n)
    curr = prev.assign(**{"MNT NAME": "RAVI"})                       # same loans, same buckets, one owner now
    return curr, prev


class TestTheRule:
    def test_last_month_rows_carry_this_months_executive(self):
        curr, prev = _handover()
        held = prev_held_now(curr, prev)
        assert set(held["MNT NAME"]) == {"RAVI"} and len(held) == len(prev)
        assert held.set_index("Loan No")["curr_bucket"].equals(prev.set_index("Loan No")["curr_bucket"])

    def test_a_loan_that_left_the_book_belongs_to_no_one(self):
        curr, prev = _handover()
        held = prev_held_now(curr.iloc[5:], prev)                     # 5 loans left the running book
        assert len(held) == len(prev) - 5 and set(held["Loan No"]) == set(curr.iloc[5:]["Loan No"])

    def test_a_file_without_loan_numbers_is_left_as_it_is(self):
        curr, prev = _handover()
        assert prev_held_now(curr.drop(columns="Loan No"), prev) is prev
        assert prev_held_now(curr, None) is None


class TestEveryExecutiveComparison:
    def test_scorecard_shows_no_jump_for_a_handed_over_book(self):
        curr, prev = _handover()
        ravi = compute_executive_scorecard(curr, min_accounts=1, df_prev=prev).set_index("MNT NAME").loc["RAVI"]
        assert ravi["Prev Delinquency %"] == ravi["Delinquency %"] == 50.0 and ravi["Δ Delinquency %"] == 0

    def test_dashboard_unit_table_shows_no_jump(self):
        curr, prev = _handover()
        row = unit_table(curr, prev, "Executive").iloc[0]
        assert row["Prev Delinquency%"] == row["Delinquency%"] and row["Δ Delinquency%"] == 0

    def test_needs_attention_does_not_flag_a_handover_as_rising(self):
        curr, prev = _handover(n=40)
        flagged = [u for u in attention(curr, prev, "executive")["worst"]
                   if any("delinquency up" in r for r in u["reasons"])]
        assert flagged == []

    def test_regions_and_branches_keep_last_months_file(self):
        curr, prev = _handover()
        branch = unit_table(curr.iloc[5:], prev, "Branch").iloc[0]
        # 5 loans left the book: the branch's last month still has all 40 (the official total).
        assert branch["Prev Delinquency%"] == 50.0


def test_ai_query_last_month_by_executive_is_the_loans_held_now():
    curr, prev = _handover()
    npa = [{"column": "curr_bucket", "op": "==", "value": "NPA"}]
    step = {"op": "group_aggregate", "group_by": ["MNT NAME", "Unit"],
            "aggregations": [{"alias": "npa_now", "func": "count", "column": "Loan No", "where": npa},
                             {"alias": "npa_last_month", "func": "count", "column": "Loan No", "where": npa}],
            "prev_file": {"conditions": [], "aggregations": [
                {"alias": "npa_last_month", "func": "count", "column": "Loan No", "where": npa}]}}
    out, err = execute_plan(curr, [step], df_prev=prev)
    assert err == ""
    assert out.set_index("MNT NAME").loc["RAVI", ["npa_now", "npa_last_month"]].tolist() == [20, 20]
    assert "SURESH" not in set(out["MNT NAME"])          # holds nothing now: no row of last-month-only figures

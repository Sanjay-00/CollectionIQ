"""Roll / flow analysis by count and by last month's SOH (analysis/roll_flow.py)."""
import pandas as pd
import pytest

from analysis import roll_flow as rf
from config import ROLL_STEPS


def _loans(rows):
    # rows: (loan, region, branch, exec, last month bucket, now bucket, last month SOH)
    return pd.DataFrame(rows, columns=["Loan No", "RegionName", "Unit", "MNT NAME", "prev_bucket",
                                       "curr_bucket", "prev_SOH"]).assign(SOH=lambda d: d["prev_SOH"] * 1.1)


BOOK = _loans([
    ("A", "WEST", "PEN", "RAHUL", "STD", "1-30 DPD", 1_000_000),     # new defaulter (big)
    ("B", "WEST", "PEN", "RAHUL", "STD", "STD", 100_000),
    ("C", "WEST", "PEN", "RAHUL", "STD", "STD", 100_000),
    ("D", "WEST", "PEN", "RAHUL", "1-30 DPD", "SMA-1", 200_000),     # slipped further
    ("E", "WEST", "PEN", "RAHUL", "1-30 DPD", "STD", 300_000),       # cured
    ("F", "EAST", "KARAD", "RAHUL", "SMA-2", "NPA", 500_000),        # new NPA, other RAHUL
    ("G", "EAST", "KARAD", "RAHUL", "NPA", "NPA", 400_000),
    ("H", "EAST", "KARAD", "RAHUL", None, "STD", None),              # new loan: not matched
])


def test_steps_by_count_and_by_last_months_soh():
    s = rf.roll_steps_summary(BOOK)
    new = s["STD → Behind"]
    assert (new["base"], new["n"]) == (3, 1) and new["pct"] == pytest.approx(33.33)
    assert new["soh_pct"] == pytest.approx(83.33)               # 10L of 12L: the big loan slipped
    assert s["1-30 → SMA-1+"]["n"] == 1 and s["1-30 → SMA-1+"]["pct"] == 50.0
    assert s["SMA-2 → NPA"]["n"] == 1 and s["SMA-2 → NPA"]["pct"] == 100.0
    cure = s["Back to STD"]
    assert (cure["base"], cure["n"]) == (4, 1)                  # was behind: D, E, F, G; E recovered


def test_unmatched_loans_and_no_previous_month():
    assert rf.roll_steps_summary(BOOK)["STD → Behind"]["base"] == 3          # H (new) left out
    assert rf.roll_steps_summary(BOOK.drop(columns=["prev_bucket"])) is None
    assert rf.roll_steps_by(BOOK.drop(columns=["prev_bucket"]), "branch").empty


def test_soh_falls_back_to_this_month_when_last_month_has_none():
    s = rf.roll_steps_summary(BOOK.drop(columns=["prev_SOH"]))
    assert s["STD → Behind"]["soh_pct"] == pytest.approx(83.33)     # same proportions on current SOH


@pytest.mark.parametrize("measure", ["count_pct", "soh_pct"])
def test_every_matrix_row_adds_to_100(measure):
    mat = rf.migration_matrix(BOOK, measure)
    for bucket, row in mat.iterrows():
        assert row.sum() == pytest.approx(100, abs=0.2) or row.sum() == 0, bucket


def test_count_matrix_matches_the_existing_roll_rate_matrix():
    from analysis.roll_rate import compute_roll_rate_matrix
    prev = BOOK.dropna(subset=["prev_bucket"]).assign(curr_bucket=lambda d: d["prev_bucket"])
    old, _ = compute_roll_rate_matrix(BOOK, prev)
    assert (rf.migration_matrix(BOOK, "count").values == old.values).all()


def test_soh_matrix_is_last_months_crore():
    mat = rf.migration_matrix(BOOK, "soh")
    assert mat.loc["STD", "1-30 DPD"] == 0.1 and mat.loc["SMA-2", "NPA"] == 0.05


def test_executive_is_name_plus_branch_and_carries_region(monkeypatch):
    monkeypatch.setitem(rf._GRAINS, "executive", (["MNT NAME", "Unit"], {"MNT NAME": "Executive", "Unit": "Branch"}, 1))
    ex = rf.roll_steps_by(BOOK, "executive")
    assert sorted(zip(ex["Executive"], ex["Branch"])) == [("RAHUL", "KARAD"), ("RAHUL", "PEN")]
    assert dict(zip(ex["Branch"], ex["Region"])) == {"PEN": "WEST", "KARAD": "EAST"}
    assert ex.iloc[0]["Branch"] == "PEN"                         # most new defaulters (by %) first


def test_call_list_is_the_loans_behind_the_step_biggest_first():
    calls = rf.loans_that_rolled(BOOK, "STD")
    assert calls["Loan No"].tolist() == ["A"]
    assert {"Executive", "Branch", "Last Month", "Now"} <= set(calls.columns)


def test_steps_come_from_config_with_the_early_slips_first():
    assert list(ROLL_STEPS.values())[:2] == ["STD", "1-30 DPD"]
    assert rf.step_labels()[-1] == "Back to STD"


def test_where_table_total_row_is_sum_over_sum():
    from ui.tabs.migration import _with_total
    df = pd.DataFrame({"Branch": ["A", "B"], "Matched Accounts": [10, 490], "Matched SOH (Cr)": [1.0, 9.0]})
    for lb in rf.step_labels():
        df[f"{lb} | Base"], df[f"{lb} | Base SOH (Cr)"] = [10, 490], [1.0, 9.0]
        df[f"{lb} | Accounts"], df[f"{lb} | SOH (Cr)"] = [8, 2], [0.5, 0.5]
        df[f"{lb} | %"], df[f"{lb} | SOH %"] = [80.0, 0.41], [50.0, 5.56]
    total = _with_total(df, ["Branch"]).iloc[-1]
    assert total["Branch"] == "Total" and total["STD → Behind | %"] == 2.0    # 10 of 500, not the 40% average
    assert total["STD → Behind | SOH %"] == 10.0


def test_one_roll_engine_counts_a_transferred_loan_where_it_is_now():
    """A loan that moved from branch A to B since last month: in B's view it
    rolled (STD -> SMA-1) under B, the branch that owns it today. The headline
    matrix (compute_roll_rate_matrix) and the roll steps agree on that."""
    from analysis.roll_rate import compute_roll_rate_matrix
    prev = pd.DataFrame({"Loan No": ["T1", "B1"], "Unit": ["A", "B"], "curr_bucket": ["STD", "STD"]})
    curr = pd.DataFrame({"Loan No": ["T1", "B1", "N1"], "Unit": ["B", "B", "B"],
                         "curr_bucket": ["SMA-1", "STD", "STD"], "SOH": [1.0, 1.0, 1.0]})
    curr = curr.merge(prev[["Loan No", "curr_bucket"]].rename(columns={"curr_bucket": "prev_bucket"}),
                      on="Loan No", how="left")            # what app.py attaches (whole previous file)
    view_c, view_p = curr[curr["Unit"] == "B"], prev[prev["Unit"] == "B"]
    matrix, meta = compute_roll_rate_matrix(view_c, view_p)
    assert meta["matched_count"] == 2 and matrix.loc["STD", "SMA-1"] == 1
    assert meta["new_entries"] == 1                       # N1 only: T1 existed last month (in branch A)
    assert rf.roll_steps_summary(view_c)["STD → Behind"]["n"] == 1

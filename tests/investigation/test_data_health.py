"""data_health: check the two files before explaining anything (spec 7.1)."""
from investigation.evidence import EvidenceStore
from investigation.panel import make_pair
from investigation.tools import data_health
from tests.investigation.books import make_book


def _health(prev, curr, prev_month="2026-08"):
    return data_health(EvidenceStore(), make_pair(curr, prev, "2026-09", prev_month)).summary


def _finding(summary, check):
    return next(f for f in summary["findings"] if f["check"] == check)


def test_clean_book_passes():
    s = _health(*make_book(3000, seed=10))
    assert s["status"] == "pass"
    assert {f["status"] for f in s["findings"]} <= {"pass", "info"}


def test_large_branch_missing_this_month_halts():
    s = _health(*make_book(6000, seed=11, drop_branch="BR3"))
    assert s["status"] == "halt"
    f = _finding(s, "units_missing")
    assert f["status"] == "halt" and "BR3" in f["detail"]


def test_small_branch_missing_warns():
    s = _health(*make_book(6000, seed=12, drop_branch="TINYBR"))
    f = _finding(s, "units_missing")
    assert f["status"] == "warn" and "TINYBR" in f["detail"] and s["status"] == "warn"


def test_population_jump_halts():
    s = _health(*make_book(3000, seed=13, grow=0.2))
    assert _finding(s, "population_change")["status"] == "halt" and s["status"] == "halt"


def test_days_disagreement_warns_above_threshold():
    assert _finding(_health(*make_book(6000, seed=14, days_off=0.03)), "days_column_agreement")["status"] == "warn"
    assert _finding(_health(*make_book(6000, seed=14, days_off=0.01)), "days_column_agreement")["status"] == "pass"


def test_missing_days_column_is_noted_not_failed():
    prev, curr = make_book(2000, seed=15)
    f = _finding(_health(prev.drop(columns="DelinquencyDays"), curr.drop(columns="DelinquencyDays")),
                 "days_column_agreement")
    assert f["status"] == "info" and "not checked" in f["detail"]


def test_header_renames_and_duplicates_are_reported():
    prev, curr = make_book(2000, seed=16)
    curr.attrs.update(header_renames={"CoLending_Loan s": "CoLending_Loans", "MNT Name": "MNT NAME"},
                      dropped_duplicate_loans=611)
    s = _health(prev, curr)
    renames, dups = _finding(s, "header_renames"), _finding(s, "duplicates_dropped")
    assert renames["status"] == "warn" and "CoLending_Loan s" in renames["detail"] and "MNT Name" not in renames["detail"]
    assert dups["status"] == "warn" and "611" in dups["detail"]


def test_reassigned_loans_are_counted():
    f = _finding(_health(*make_book(2000, seed=17, reassign=7)), "executives_changed")
    assert f["status"] == "info" and "7" in f["detail"]


def test_months_not_next_to_each_other_warn():
    f = _finding(_health(*make_book(2000, seed=18), prev_month="2026-06"), "months_in_order")
    assert f["status"] == "warn"


def test_labels_changed_for_the_same_loans_warns():
    # Real Sept 2026: CHANNEL held the source file name for 31% of the same loans.
    f = _finding(_health(*make_book(3000, seed=21, relabel=0.3)), "labels_changed")
    assert f["status"] == "warn" and "Channel" in f["detail"] and "30.0%" in f["detail"]
    assert _finding(_health(*make_book(3000, seed=21)), "labels_changed")["status"] == "pass"


def test_branch_missing_from_last_months_file_halts():
    # Real Sept 2026: HINGL and WASHM (1,657 loans back to 2022) were only in
    # Sept's file: 1.8% of the book not comparable (like for like +0.71, not +0.78).
    s = _health(*make_book(6000, seed=23, add_branch=("HINGL", 0.02, "2024-05-01")))
    f = _finding(s, "units_new")
    assert f["status"] == "halt" and "HINGL" in f["detail"] and "missing from last month's file" in f["detail"]
    assert s["status"] == "halt"


def test_small_branch_missing_from_last_months_file_warns():
    f = _finding(_health(*make_book(6000, seed=24, add_branch=("HINGL", 0.004, "2024-05-01"))), "units_new")
    assert f["status"] == "warn" and "missing from last month's file" in f["detail"]


def test_genuinely_new_branch_is_information():
    f = _finding(_health(*make_book(6000, seed=25, add_branch=("NEWBR", 0.02, "2026-09-10"))), "units_new")
    assert f["status"] == "info" and "NEWBR" in f["detail"]


def test_new_branch_is_information():
    prev, curr = make_book(3000, seed=19)
    curr.loc[curr.index[:40], "Unit"] = "HINGL"
    f = _finding(_health(prev, curr), "units_new")
    assert f["status"] == "info" and "HINGL" in f["detail"]

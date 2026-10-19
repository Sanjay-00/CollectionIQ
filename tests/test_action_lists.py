"""One home for every loan-level list: each list once, in four groups,
with the duplicates (same loans under two names) left out."""
from analysis.action_lists import GROUPS, RAW_NAMES, build_lists, distinct_loans
from smart_alerts import run_all_alerts
from tests.test_action_center import _book, _prev


def _lists():
    c = _book()
    return build_lists(c, "2026-08", run_all_alerts(c, as_of="2026-08"), run_all_alerts(_prev(c), as_of="2026-07"))


def test_groups_in_order_and_no_duplicate_lists():
    lists = _lists()
    order = [GROUPS.index(e["group"]) for e in lists]
    assert order == sorted(order)
    names = [e["name"] for e in lists]
    assert len(names) == len(set(names))
    # Same loans under another name: shown once, in "Call this week".
    assert "Insurance-Driven Delinquency" not in names and "Non Starters" not in names
    assert "STD → Behind" not in names                      # that is "New defaulters"


def test_every_list_counts_its_loans_and_soh_the_same_way():
    for e in _lists():
        loans = e["loans"]
        assert e["count"] == loans["Loan No"].nunique() > 0
        assert e["soh_cr"] == round(float(loans["SOH"].sum()) / 1e7, 2)
        assert not {"RegionName", "Unit", "MNT NAME"} & set(loans.columns)     # one naming everywhere


def test_sort_order():
    for e in _lists():
        soh = e["loans"]["SOH"]
        if e["name"] == "Good customers":
            assert soh.is_monotonic_increasing                # easiest to refinance first
        else:
            assert soh.is_monotonic_decreasing


def test_distinct_loans_counts_a_loan_once():
    lists = _lists()
    every = set().union(*[set(e["loans"]["Loan No"]) for e in lists])
    assert distinct_loans(lists) == len(every)
    assert distinct_loans(lists, "Call this week") <= distinct_loans(lists)


def test_download_names_map_back_to_the_upload():
    assert RAW_NAMES["Region"] == "RegionName" and RAW_NAMES["Branch"] == "Unit" and RAW_NAMES["Executive"] == "MNT NAME"


def test_change_vs_last_month_uses_the_same_rule_both_months():
    # A bucket list (new defaulters left out) can't be rebuilt for last month
    # without the month before last, so it compares the whole bucket instead;
    # new defaulters and the slips stay blank rather than mix two rules.
    c = _book()
    p = _prev(c).drop(columns=["prev_bucket"], errors="ignore")    # two files uploaded: no month before last
    lists = {e["name"]: e for e in build_lists(c, "2026-08", run_all_alerts(c, as_of="2026-08"),
                                               run_all_alerts(p, as_of="2026-07"), df_prev=p, prev_as_of="2026-07")}
    bucket = lambda df, b: df.loc[df["curr_bucket"] == b, "Loan No"].nunique()   # noqa: E731
    for name, b in (("SMA-2: last chance before NPA", "SMA-2"), ("SMA-1: stop the slide", "SMA-1"),
                    ("1-30 DPD: bring them current", "1-30 DPD")):
        if name in lists:
            assert lists[name]["change"] == bucket(c, b) - bucket(p, b) and lists[name]["change_basis"] == "bucket"
    if "New defaulters" in lists:
        assert lists["New defaulters"]["change"] is None
    for e in lists.values():
        if e["change_basis"] == "list":
            assert e["change"] == e["count"] - e["prev"]

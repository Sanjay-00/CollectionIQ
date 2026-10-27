"""PeriodPair: two closings, one join on Loan No, scoping by each month's labels."""
import pandas as pd
import pytest

from investigation.evidence import ToolError
from investigation.panel import make_pair


def _frame(rows):
    df = pd.DataFrame(rows)
    for col, default in (("Loan Status", "RUN"), ("Unit", "B1"), ("MNT NAME", "RAJ"), ("RegionName", "R1")):
        if col not in df.columns:
            df[col] = default
    return df


def test_join_marks_matched_new_and_left_loans():
    prev = _frame([{"Loan No": "L1", "curr_bucket": "NPA"}, {"Loan No": "L2", "curr_bucket": "STD"}])
    curr = _frame([{"Loan No": "L2", "curr_bucket": "SMA-1"}, {"Loan No": "L3", "curr_bucket": "NA"}])
    j = make_pair(curr, prev, "2026-09", "2026-08").joined().set_index("Loan No")
    assert bool(j.loc["L1", "left_book"]) and not bool(j.loc["L1", "in_curr"])
    assert bool(j.loc["L2", "in_prev"]) and bool(j.loc["L2", "in_curr"])
    assert (j.loc["L2", "prev_bucket"], j.loc["L2", "curr_bucket"]) == ("STD", "SMA-1")
    assert j.loc["L3", "prev_bucket"] is None and not bool(j.loc["L3", "in_prev"])
    assert j.loc["L1", "curr_bucket"] is None


def test_loan_numbers_typed_differently_still_match():
    prev = _frame([{"Loan No": 12345.0, "curr_bucket": "STD"}])
    curr = _frame([{"Loan No": " 12345", "curr_bucket": "STD"}])
    j = make_pair(curr, prev, "2026-09", "2026-08").joined()
    assert len(j) == 1 and bool(j.iloc[0]["in_prev"]) and bool(j.iloc[0]["in_curr"])


def test_months_must_be_in_order():
    df = _frame([{"Loan No": "L1", "curr_bucket": "STD"}])
    with pytest.raises(ToolError):
        make_pair(df, df, "2026-08", "2026-09")
    with pytest.raises(ToolError):
        make_pair(df, df, "2026-09", "2026-09")


def test_empty_or_unbucketed_month_is_refused():
    df = _frame([{"Loan No": "L1", "curr_bucket": "STD"}])
    with pytest.raises(ToolError):
        make_pair(df.iloc[0:0], df, "2026-09", "2026-08")
    with pytest.raises(ToolError):
        make_pair(df.drop(columns="curr_bucket"), df, "2026-09", "2026-08")


def test_executive_last_month_is_the_loans_held_now():
    prev = _frame([{"Loan No": "L1", "curr_bucket": "STD", "MNT NAME": "RAJ"},
                   {"Loan No": "L2", "curr_bucket": "NPA", "MNT NAME": "RAJ"}])
    curr = _frame([{"Loan No": "L1", "curr_bucket": "STD", "MNT NAME": "AMIT"}])
    pair = make_pair(curr, prev, "2026-09", "2026-08")
    labels = pair.labels("executive", "prev")
    assert list(labels) == ["AMIT (B1)", "Left the running book"]
    assert list(pair.labels("branch", "prev")) == ["B1", "B1"]       # other dimensions: last month's own labels


def test_scope_matching_nothing_is_refused():
    df = _frame([{"Loan No": "L1", "curr_bucket": "STD"}])
    with pytest.raises(ToolError, match="NOWHERE"):
        make_pair(df, df, "2026-09", "2026-08").scoped({"branch": "NOWHERE"})


def test_label_drift_is_the_share_of_the_same_loans_relabelled():
    from tests.investigation.books import make_book
    prev, curr = make_book(2000, seed=20, relabel=0.25)
    pair = make_pair(curr, prev, "2026-09", "2026-08")
    assert pair.label_drift_pct("channel") == 25.0
    assert pair.label_drift_pct("product") == 0.0


def test_scope_on_a_missing_column_is_refused_plainly():
    df = _frame([{"Loan No": "L1", "curr_bucket": "STD"}])
    with pytest.raises(ToolError, match="no Make column"):
        make_pair(df, df, "2026-09", "2026-08").scoped({"make": "TATA"})


def test_a_loan_that_moved_belongs_to_its_branch_now_in_both_months():
    # Review finding: scoping each month by its own label made L1 "leave" AKLUJ.
    prev = _frame([{"Loan No": "L1", "curr_bucket": "NPA", "Unit": "AKLUJ"},
                   {"Loan No": "L2", "curr_bucket": "STD", "Unit": "AKLUJ"},
                   {"Loan No": "L3", "curr_bucket": "NPA", "Unit": "AKLUJ"}])     # L3 left the book
    curr = _frame([{"Loan No": "L1", "curr_bucket": "NPA", "Unit": "PUNE"},
                   {"Loan No": "L2", "curr_bucket": "STD", "Unit": "AKLUJ"}])
    pair = make_pair(curr, prev, "2026-09", "2026-08")
    akluj, pune = pair.scoped({"branch": "AKLUJ"}), pair.scoped({"branch": "PUNE"})
    assert sorted(akluj.prev["Loan No"]) == ["L2", "L3"] and list(akluj.curr["Loan No"]) == ["L2"]
    assert list(pune.prev["Loan No"]) == ["L1"] and list(pune.curr["Loan No"]) == ["L1"]


def test_scope_on_a_relabelled_dimension_is_refused():
    from tests.investigation.books import make_book
    prev, curr = make_book(2000, seed=20, relabel=0.25)
    with pytest.raises(ToolError, match="labels changed"):
        make_pair(curr, prev, "2026-09", "2026-08").scoped({"channel": "DIRECT"})


def test_no_scope_is_the_same_pair():
    df = _frame([{"Loan No": "L1", "curr_bucket": "STD"}])
    pair = make_pair(df, df, "2026-09", "2026-08")
    assert pair.scoped(None) is pair and pair.scoped({}) is pair


class TestNarrowedPairSharesWorkWithoutChangingAnswers:
    """A narrowed pair (scoped, without_units) takes row-wise labels, metric
    masks and both-months positions from the pair it came from (speed work,
    2026-10-25: 169 s -> 2.7 s for a 90k-loan review). Everything must equal
    a pair built from scratch on the same rows."""

    def _pairs(self, narrow):
        from investigation.dimensions import DIMENSIONS
        from investigation.metrics import get_metric
        from investigation.panel import PeriodPair
        from investigation.synthetic import make_book
        prev, curr = make_book(3000, seed=4, reassign=200, rate_up=("TRACTOR", 4.0))
        full = make_pair(curr, prev, "2026-09", "2026-08")
        for dim in DIMENSIONS:                      # fill the origin's caches first, as a real run does
            if full.unavailable(dim) is None:
                full.labels(dim, "curr"), full.labels(dim, "prev")
        small = narrow(full)
        fresh = PeriodPair(small.prev, small.curr, small.prev_month, small.curr_month)
        return small, fresh, get_metric("thirty_plus_rate"), DIMENSIONS

    @pytest.mark.parametrize("narrow", [lambda p: p.scoped({"region": "R1"}),
                                        lambda p: p.scoped({"product": "TRACTOR"}).scoped({"branch": "BR4"}),
                                        lambda p: p.without_units(["BR2", "BR7"])],
                             ids=["scoped", "scoped_twice", "without_units"])
    def test_every_cached_answer_matches_a_fresh_pair(self, narrow):
        small, fresh, metric, dims = self._pairs(narrow)
        for dim in dims:
            assert small.unavailable(dim) == fresh.unavailable(dim), dim
            if fresh.unavailable(dim) is not None:
                continue
            for month in ("prev", "curr"):
                pd.testing.assert_series_equal(small.labels(dim, month), fresh.labels(dim, month),
                                               check_names=False, obj=f"{dim} {month}")
            pd.testing.assert_series_equal(small.loan_label(dim).sort_index(), fresh.loan_label(dim).sort_index(),
                                           check_names=False, obj=f"{dim} loan_label")
            assert small.label_drift_pct(dim) == fresh.label_drift_pct(dim), dim
        for month in ("prev", "curr"):
            for a, b in zip(small.masks(metric)[month], fresh.masks(metric)[month]):
                assert (a == b).all()
        sp, sc = small._both_months()
        fp, fc = fresh._both_months()
        assert sorted(zip(sp, sc)) == sorted(zip(fp, fc))

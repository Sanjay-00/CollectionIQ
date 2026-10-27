"""Each plant in investigation.synthetic does what it says, measured directly."""
import pandas as pd

from investigation.metrics import THIRTY_PLUS_BUCKETS
from investigation.synthetic import make_book


def _rate(df, mask=None):
    df = df if mask is None else df[mask]
    elig = df[df["curr_bucket"] != "NA"]
    return 100 * elig["curr_bucket"].isin(THIRTY_PLUS_BUCKETS).mean()


def test_vintage_rate_up_hits_only_that_window():
    prev, curr = make_book(20000, seed=50, vintage_rate_up=("2024-07-01", "2024-09-30", 10))
    win = lambda d: pd.to_datetime(d["Ag_Date"]).between("2024-07-01", "2024-09-30")
    assert _rate(curr, win(curr)) - _rate(prev, win(prev)) >= 9
    assert abs(_rate(curr, ~win(curr)) - _rate(prev, ~win(prev))) < 1e-9


def test_product_by_vintage_puts_the_product_in_recent_loans_only():
    prev, _ = make_book(20000, seed=51, product_by_vintage=("TRACTOR", "2025-04-01"))
    ag = pd.to_datetime(prev.loc[prev["SegmentName"] == "TRACTOR", "Ag_Date"])
    assert ag.min() >= pd.Timestamp("2025-04-01")


def test_exec_rate_up_makes_one_executive_worse():
    prev, curr = make_book(20000, seed=52, exec_rate_up=("BR3", "EXEC7", 40, 300))
    mine = lambda d: (d["Unit"] == "BR3") & (d["MNT NAME"] == "EXEC7")
    assert mine(prev).sum() >= 300 and mine(curr).sum() == mine(prev).sum()
    assert _rate(curr, mine(curr)) - _rate(prev, mine(prev)) >= 35


def test_cure_block_keeps_loans_that_would_have_cured():
    p0, c0 = make_book(20000, seed=53, churn=0.2)
    p1, c1 = make_book(20000, seed=53, churn=0.2, cure_block=1.0)
    cures = lambda p, c: ((p["curr_bucket"].isin(THIRTY_PLUS_BUCKETS)) & (c["curr_bucket"].isin(["STD", "1-30 DPD"]))).sum()
    assert cures(p0, c0) > 100 and cures(p1, c1) == 0


def test_base_grow_moves_no_emi_loans_into_the_base_only():
    prev, curr = make_book(20000, seed=54, base_grow=400)
    assert (curr["curr_bucket"] == "NA").sum() == (prev["curr_bucket"] == "NA").sum() - 400
    assert curr["curr_bucket"].isin(THIRTY_PLUS_BUCKETS).sum() == prev["curr_bucket"].isin(THIRTY_PLUS_BUCKETS).sum()


def test_segment_swing_moves_a_small_segment_only():
    prev, curr = make_book(20000, seed=55, segment_swing=("Make", "RAREMAKE", 250, 30))
    rare = lambda d: d["Make"] == "RAREMAKE"
    assert rare(prev).sum() == 250 and _rate(curr, rare(curr)) - _rate(prev, rare(prev)) >= 25


def test_plants_pick_random_loans_not_the_first_rows():
    # Review: every plant took the lowest-index rows, so two plants overlapped
    # (56% of a product plant landed in the 10% of loans also reassigned).
    prev, curr = make_book(30000, seed=57, rate_up=("TRACTOR", 5), reassign=3000)
    planted = (prev["curr_bucket"] == "STD") & (curr["curr_bucket"] == "SMA-1")
    reassigned = curr["MNT NAME"] == "NEWEXEC"
    assert 0.03 < (planted & reassigned).sum() / planted.sum() < 0.2      # about its 10% share


def test_duplicates_and_attrs_land_on_this_month():
    prev, curr = make_book(3000, seed=56, duplicates=12, attrs={"dropped_duplicate_loans": 12})
    assert curr["Loan No"].duplicated().sum() == 12 and curr.attrs["dropped_duplicate_loans"] == 12
    assert not prev["Loan No"].duplicated().any()

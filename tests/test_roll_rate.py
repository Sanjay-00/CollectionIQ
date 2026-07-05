import pandas as pd
import pytest

from analysis.roll_rate import (
    VALID_BUCKETS,
    compute_roll_rate_matrix,
    compute_roll_rate_kpis,
    compute_bucket_roll_summary,
    build_roll_rate_heatmap,
)
from helpers import make_df


def _make_matrix(data: dict) -> pd.DataFrame:
    """Build a migration matrix from {(prev_bucket, curr_bucket): count}."""
    matrix = pd.DataFrame(0, index=VALID_BUCKETS, columns=VALID_BUCKETS)
    for (r, c), v in data.items():
        matrix.loc[r, c] = v
    return matrix


class TestComputeRollRateMatrix:
    def test_empty_prev_returns_zero_meta(self):
        curr = make_df([{"Loan No": "L001", "curr_bucket": "STD"}])
        _, meta = compute_roll_rate_matrix(curr, make_df([]))
        assert meta["matched_count"] == 0
        assert meta["roll_forward_rate"] == 0.0
        assert meta["roll_backward_rate"] == 0.0
        assert meta["npa_formation_rate"] == 0.0

    def test_all_stable_zero_roll_rates(self):
        """All accounts stay in the same bucket → no forward or backward movement."""
        curr = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L002", "curr_bucket": "SMA-1"},
        ])
        prev = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L002", "curr_bucket": "SMA-1"},
        ])
        _, meta = compute_roll_rate_matrix(curr, prev)
        assert meta["roll_forward_rate"] == 0.0
        assert meta["roll_backward_rate"] == 0.0
        assert meta["matched_count"] == 2

    def test_all_roll_forward_100pct(self):
        """Every STD account moves to NPA."""
        curr = make_df([
            {"Loan No": "L001", "curr_bucket": "NPA"},
            {"Loan No": "L002", "curr_bucket": "NPA"},
        ])
        prev = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L002", "curr_bucket": "STD"},
        ])
        _, meta = compute_roll_rate_matrix(curr, prev)
        assert meta["roll_forward_rate"] == 100.0
        assert meta["roll_backward_rate"] == 0.0

    def test_all_cured_100pct_backward(self):
        """Every NPA account recovers to STD."""
        curr = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L002", "curr_bucket": "STD"},
        ])
        prev = make_df([
            {"Loan No": "L001", "curr_bucket": "NPA"},
            {"Loan No": "L002", "curr_bucket": "NPA"},
        ])
        _, meta = compute_roll_rate_matrix(curr, prev)
        assert meta["roll_backward_rate"] == 100.0
        assert meta["roll_forward_rate"] == 0.0

    def test_npa_formation_rate(self):
        """2 of 4 non-NPA accounts became NPA → 50 % formation rate."""
        curr = make_df([
            {"Loan No": "L001", "curr_bucket": "NPA"},
            {"Loan No": "L002", "curr_bucket": "NPA"},
            {"Loan No": "L003", "curr_bucket": "STD"},
            {"Loan No": "L004", "curr_bucket": "STD"},
        ])
        prev = make_df([
            {"Loan No": "L001", "curr_bucket": "SMA-2"},
            {"Loan No": "L002", "curr_bucket": "SMA-1"},
            {"Loan No": "L003", "curr_bucket": "STD"},
            {"Loan No": "L004", "curr_bucket": "STD"},
        ])
        _, meta = compute_roll_rate_matrix(curr, prev)
        assert meta["npa_formation_rate"] == 50.0

    def test_new_entries_and_exits_counted(self):
        curr = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L002", "curr_bucket": "STD"},   # new  -  not in prev
        ])
        prev = make_df([
            {"Loan No": "L001", "curr_bucket": "STD"},
            {"Loan No": "L999", "curr_bucket": "STD"},   # exited  -  not in curr
        ])
        _, meta = compute_roll_rate_matrix(curr, prev)
        assert meta["new_entries"] == 1
        assert meta["exits"] == 1
        assert meta["matched_count"] == 1

    def test_matrix_shape_is_valid_buckets_x_valid_buckets(self):
        curr = make_df([{"Loan No": "L001", "curr_bucket": "STD"}])
        prev = make_df([{"Loan No": "L001", "curr_bucket": "STD"}])
        matrix, _ = compute_roll_rate_matrix(curr, prev)
        assert list(matrix.index)   == VALID_BUCKETS
        assert list(matrix.columns) == VALID_BUCKETS


class TestComputeRollRateKpis:
    def test_stable_matrix_all_zero_rates(self):
        matrix = _make_matrix({("STD", "STD"): 10, ("NPA", "NPA"): 5})
        kpis = compute_roll_rate_kpis(matrix)
        assert kpis["roll_forward_rate"] == 0.0
        assert kpis["roll_backward_rate"] == 0.0
        assert kpis["npa_formation_rate"] == 0.0

    def test_partial_roll_forward(self):
        # 5 moved STD→NPA, 5 stayed STD → 50 % roll forward
        matrix = _make_matrix({("STD", "NPA"): 5, ("STD", "STD"): 5})
        kpis = compute_roll_rate_kpis(matrix)
        assert kpis["roll_forward_rate"] == 50.0
        assert kpis["roll_backward_rate"] == 0.0

    def test_partial_roll_backward(self):
        # 4 cured NPA→STD, 4 stayed NPA → 50 % backward
        matrix = _make_matrix({("NPA", "STD"): 4, ("NPA", "NPA"): 4})
        kpis = compute_roll_rate_kpis(matrix)
        assert kpis["roll_backward_rate"] == 50.0
        assert kpis["roll_forward_rate"] == 0.0

    def test_mixed_movement(self):
        # 2 forward, 2 backward, 6 stable → 20% each
        matrix = _make_matrix({
            ("STD",   "NPA"): 2,
            ("NPA",   "STD"): 2,
            ("SMA-1", "SMA-1"): 6,
        })
        kpis = compute_roll_rate_kpis(matrix)
        assert kpis["roll_forward_rate"]  == pytest.approx(20.0, abs=0.01)
        assert kpis["roll_backward_rate"] == pytest.approx(20.0, abs=0.01)

    def test_npa_formation_from_sma(self):
        # 3 SMA-2 became NPA, 7 stayed SMA-2 → 30 % NPA formation
        matrix = _make_matrix({("SMA-2", "NPA"): 3, ("SMA-2", "SMA-2"): 7})
        kpis = compute_roll_rate_kpis(matrix)
        assert kpis["npa_formation_rate"] == 30.0

    def test_empty_matrix_returns_zero(self):
        matrix = pd.DataFrame(0, index=VALID_BUCKETS, columns=VALID_BUCKETS)
        kpis = compute_roll_rate_kpis(matrix)
        assert kpis["roll_forward_rate"] == 0.0
        assert kpis["roll_backward_rate"] == 0.0
        assert kpis["npa_formation_rate"] == 0.0


class TestComputeBucketRollSummary:
    def test_all_stable_bucket(self):
        # 10 STD accounts, all stayed STD -> 0% fwd, 100% stable, 0% bwd
        matrix = _make_matrix({("STD", "STD"): 10, ("NPA", "NPA"): 5})
        summary = compute_bucket_roll_summary(matrix)
        std_row = summary[summary["Bucket"] == "STD"].iloc[0]
        assert std_row["Accounts"] == 10
        assert std_row["Roll Fwd%"] == 0.0
        assert std_row["Stable%"] == 100.0
        assert std_row["Roll Bwd%"] == 0.0

    def test_mixed_bucket_percentages_sum_to_100(self):
        # SMA-1: 2 rolled fwd to SMA-2, 2 cured to STD, 6 stayed SMA-1 (10 total)
        matrix = _make_matrix({
            ("SMA-1", "SMA-2"): 2,
            ("SMA-1", "STD"):   2,
            ("SMA-1", "SMA-1"): 6,
        })
        summary = compute_bucket_roll_summary(matrix)
        row = summary[summary["Bucket"] == "SMA-1"].iloc[0]
        assert row["Accounts"] == 10
        assert row["Roll Fwd%"] == 20.0
        assert row["Stable%"] == 60.0
        assert row["Roll Bwd%"] == 20.0
        assert row["Roll Fwd%"] + row["Stable%"] + row["Roll Bwd%"] == pytest.approx(100.0)

    def test_std_bucket_can_never_roll_backward(self):
        # STD is the best bucket -- nothing can be "better", so Roll Bwd% must be 0
        # regardless of matrix contents (score comparison, not a special case).
        matrix = _make_matrix({("STD", "STD"): 5, ("STD", "NPA"): 5})
        row = compute_bucket_roll_summary(matrix)
        std_row = row[row["Bucket"] == "STD"].iloc[0]
        assert std_row["Roll Bwd%"] == 0.0
        assert std_row["Roll Fwd%"] == 50.0

    def test_percentages_always_sum_to_exactly_100(self):
        # Regression: independently rounding Fwd%/Stable%/Bwd% can each round
        # correctly yet sum to 99.99 or 100.01 (e.g. a 1/1/1 split of 3 accounts:
        # 33.33+33.33+33.33=99.99). Stable% must be derived as the remainder,
        # not rounded independently, to make the "sums to 100" docstring claim
        # actually hold for every possible account count.
        matrix = _make_matrix({("SMA-1", "SMA-2"): 1, ("SMA-1", "STD"): 1, ("SMA-1", "SMA-1"): 1})
        row = compute_bucket_roll_summary(matrix)
        sma1_row = row[row["Bucket"] == "SMA-1"].iloc[0]
        assert sma1_row["Roll Fwd%"] + sma1_row["Stable%"] + sma1_row["Roll Bwd%"] == 100.0

    def test_npa_bucket_can_never_roll_forward(self):
        # NPA is the worst bucket -- nothing can be "worse".
        matrix = _make_matrix({("NPA", "NPA"): 5, ("NPA", "STD"): 5})
        row = compute_bucket_roll_summary(matrix)
        npa_row = row[row["Bucket"] == "NPA"].iloc[0]
        assert npa_row["Roll Fwd%"] == 0.0
        assert npa_row["Roll Bwd%"] == 50.0

    def test_zero_accounts_in_a_bucket_is_all_zero_not_divide_error(self):
        matrix = pd.DataFrame(0, index=VALID_BUCKETS, columns=VALID_BUCKETS)
        summary = compute_bucket_roll_summary(matrix)
        assert (summary["Accounts"] == 0).all()
        assert (summary["Roll Fwd%"] == 0.0).all()
        assert (summary["Stable%"] == 0.0).all()
        assert (summary["Roll Bwd%"] == 0.0).all()

    def test_returns_one_row_per_valid_bucket(self):
        matrix = _make_matrix({("STD", "STD"): 1})
        summary = compute_bucket_roll_summary(matrix)
        assert summary["Bucket"].tolist() == VALID_BUCKETS


class TestHeatmapColoring:
    def _z(self, matrix):
        fig = build_roll_rate_heatmap(matrix)
        return fig, fig.data[0]

    def test_zmid_zero_anchors_white(self):
        matrix = _make_matrix({("STD", "NPA"): 3, ("NPA", "STD"): 1})
        _, hm = self._z(matrix)
        assert hm.zmid == 0   # no-change is pinned to the white midpoint

    def test_same_bucket_cell_is_zero(self):
        # SMA-1 -> SMA-1 (the diagonal) must be exactly 0 -> renders white
        matrix = _make_matrix({("SMA-1", "SMA-1"): 50, ("STD", "NPA"): 5})
        _, hm = self._z(matrix)
        i = VALID_BUCKETS.index("SMA-1")
        assert hm.z[i][i] == 0

    def test_direction_sign_worse_red_better_green(self):
        matrix = _make_matrix({("STD", "NPA"): 1, ("NPA", "STD"): 1})
        _, hm = self._z(matrix)
        i_std, i_npa = VALID_BUCKETS.index("STD"), VALID_BUCKETS.index("NPA")
        assert hm.z[i_std][i_npa] > 0    # STD -> NPA = worsened = red side (positive)
        assert hm.z[i_npa][i_std] < 0    # NPA -> STD = improved = green side (negative)

    def test_intensity_is_severity_weighted(self):
        # Equal counts, different jump distance: STD->NPA (jump 4) vs STD->1-30 (jump 1)
        matrix = _make_matrix({("STD", "NPA"): 1, ("STD", "1-30 DPD"): 1})
        _, hm = self._z(matrix)
        i_std = VALID_BUCKETS.index("STD")
        big = hm.z[i_std][VALID_BUCKETS.index("NPA")]
        small = hm.z[i_std][VALID_BUCKETS.index("1-30 DPD")]
        assert big == pytest.approx(4 * small)   # weighted by jump distance, not count alone

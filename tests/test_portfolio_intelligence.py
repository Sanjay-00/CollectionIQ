import pandas as pd
import pytest

from analysis.portfolio_intelligence import (
    compute_pulse_kpis,
    compute_region_scorecard,
    compute_npa_sma2_comparison,
    compute_branch_quadrant,
    compute_executive_recovery,
    compute_good_bad,
    compute_risk_flag_comparison,
    compute_product_analysis,
    compute_risk_indicators,
    compute_fleet_exposure,
    compute_top_accounts,
    compute_repossession_list,
    compute_good_customers,
)
from helpers import make_df


def _kpi(kpis: list[dict], label: str) -> dict:
    return next(k for k in kpis if k["label"] == label)


# ── compute_pulse_kpis ───────────────────────────────────────────────────────

class TestComputePulseKpis:
    def _curr(self):
        return make_df([
            {"curr_bucket": "STD",   "Arrears / EMI": 0.0, "Strike": "Y"},
            {"curr_bucket": "SMA-2", "Arrears / EMI": 2.5, "Strike": "Y"},
            {"curr_bucket": "SMA-2", "Arrears / EMI": 2.1, "Strike": "Y"},
            {"curr_bucket": "NPA",   "Arrears / EMI": 5.0, "Strike": "N"},
        ])

    def test_no_prev_month_all_deltas_none(self):
        kpis = compute_pulse_kpis(self._curr(), make_df([]))
        assert all(k["delta"] is None for k in kpis)

    def test_counts_and_percentages(self):
        kpis = compute_pulse_kpis(self._curr(), make_df([]))
        assert _kpi(kpis, "Total Accounts")["value"] == "4"
        assert _kpi(kpis, "SMA-2 Accounts")["value"] == "2"
        assert _kpi(kpis, "SMA-2 %")["value"] == "50.00%"
        assert _kpi(kpis, "NPA Accounts")["value"] == "1"
        assert _kpi(kpis, "NPA %")["value"] == "25.00%"
        assert _kpi(kpis, "Collection %")["value"] == "100.00%"
        assert _kpi(kpis, "Strike %")["value"] == "75.00%"

    def test_soh_converted_to_crores(self):
        # 4 accounts x default POS 1,00,000 -> 4,00,000 -> 0.04 Cr
        kpis = compute_pulse_kpis(self._curr(), make_df([]))
        assert _kpi(kpis, "Total SOH")["value"] == "₹0.04Cr"

    def test_delta_and_inverse_direction(self):
        prev = make_df([
            {"curr_bucket": "STD", "Arrears / EMI": 0.0},
            {"curr_bucket": "NPA", "Arrears / EMI": 5.0},
        ])
        kpis = compute_pulse_kpis(self._curr(), prev)
        # curr NPA% 25.0, prev NPA% 50.0 -> raw delta -25.0, inverse=True -> +25.0
        assert _kpi(kpis, "NPA %")["delta"] == 25.0
        # curr accounts 4, prev 2 -> not inverted -> +2
        assert _kpi(kpis, "Total Accounts")["delta"] == 2
        # curr coll% 100, prev coll% 100 -> 0, not inverted
        assert _kpi(kpis, "Collection %")["delta"] == 0.0


# ── compute_region_scorecard ─────────────────────────────────────────────────

class TestComputeRegionScorecard:
    def _dfs(self):
        curr = make_df([
            *[{"RegionName": "WEST", "curr_bucket": "STD"} for _ in range(3)],
            *[{"RegionName": "EAST", "curr_bucket": b} for b in ["NPA", "NPA", "STD"]],
            *[{"RegionName": "NORTH", "curr_bucket": "NPA"} for _ in range(3)],
        ])
        prev = make_df([
            *[{"RegionName": "WEST", "curr_bucket": b} for b in ["NPA", "STD", "STD"]],
            *[{"RegionName": "EAST", "curr_bucket": b} for b in ["NPA", "STD", "STD"]],
            # NORTH absent from prev entirely
        ])
        return curr, prev

    def test_empty_or_missing_region_column_returns_empty(self):
        assert compute_region_scorecard(make_df([]), make_df([])).empty
        df = make_df([{"Loan No": "L1"}]).drop(columns=["RegionName"])
        assert compute_region_scorecard(df, make_df([])).empty

    def test_npa_pct_per_region(self):
        curr, prev = self._dfs()
        out = compute_region_scorecard(curr, prev)
        rows = out.set_index("Region")
        assert rows.loc["WEST", "NPA% (Curr)"] == 0.0
        assert rows.loc["EAST", "NPA% (Curr)"] == pytest.approx(66.67, abs=0.01)
        assert rows.loc["NORTH", "NPA% (Curr)"] == 100.0

    def test_sorted_descending_by_curr_npa(self):
        curr, prev = self._dfs()
        out = compute_region_scorecard(curr, prev)
        assert out["Region"].tolist() == ["NORTH", "EAST", "WEST"]

    def test_status_labels_from_delta_threshold(self):
        curr, prev = self._dfs()
        out = compute_region_scorecard(curr, prev).set_index("Region")
        assert out.loc["WEST", "Status"] == "Improving"   # NPA% fell 33.3 -> -1.0
        assert out.loc["EAST", "Status"] == "Worsening"    # NPA% rose 33.3 -> +1.0
        assert out.loc["NORTH", "Status"] == "-"           # no matching prev region

    def test_region_absent_from_prev_has_null_prev_and_delta(self):
        curr, prev = self._dfs()
        out = compute_region_scorecard(curr, prev).set_index("Region")
        assert pd.isna(out.loc["NORTH", "NPA% (Prev)"])
        assert pd.isna(out.loc["NORTH", "Δ NPA%"])


# ── compute_npa_sma2_comparison ──────────────────────────────────────────────

class TestComputeNpaSma2Comparison:
    def _dfs(self):
        curr = make_df([
            *[{"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": b}
              for b in ["NPA", "NPA", "SMA-2", "STD", "STD"]],
            # EAST/PUNE has only 2 accounts -> below the min_n=3 threshold, must be dropped
            *[{"RegionName": "EAST", "Unit": "PUNE", "curr_bucket": b} for b in ["NPA", "STD"]],
        ])
        prev = make_df([
            *[{"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": b}
              for b in ["NPA", "STD", "STD", "STD"]],
        ])
        return curr, prev

    def test_small_groups_excluded_by_min_n(self):
        curr, prev = self._dfs()
        out = compute_npa_sma2_comparison(curr, prev)
        assert "EAST" not in out["region"]["RegionName"].tolist()
        assert "PUNE" not in out["branch"]["Unit"].tolist()

    def test_delta_and_delta_pct_vs_prev(self):
        curr, prev = self._dfs()
        out = compute_npa_sma2_comparison(curr, prev)
        row = out["region"].set_index("RegionName").loc["WEST"]
        assert row["NPA (Curr)"] == 2
        assert row["NPA (Prev)"] == 1
        assert row["NPA Δ"] == 1
        assert row["NPA Δ%"] == 100.0
        # SMA-2 went from 0 (prev) to 1 (curr): special-cased 100% growth from zero
        assert row["SMA-2 (Curr)"] == 1
        assert row["SMA-2 (Prev)"] == 0
        assert row["SMA-2 Δ%"] == 100.0

    def test_no_executive_key_when_mnt_name_missing(self):
        curr, prev = self._dfs()
        out = compute_npa_sma2_comparison(curr, prev)
        assert "executive" not in out

    def test_executive_key_present_when_mnt_name_available(self):
        curr = make_df([
            {"MNT NAME": "RAHUL", "curr_bucket": b} for b in ["NPA", "NPA", "STD"]
        ])
        out = compute_npa_sma2_comparison(curr, make_df([]))
        assert "executive" in out
        assert out["executive"]["NPA (Curr)"].iloc[0] == 2


# ── compute_branch_quadrant ──────────────────────────────────────────────────

class TestComputeBranchQuadrant:
    def _df(self):
        bad = [{
            "Unit": "BADBR", "curr_bucket": b, "Arrears / EMI": arr,
            "No Coll 3 Months and >6 EMI": chronic,
        } for b, arr, chronic in [
            ("NPA", 5.0, "Y"), ("NPA", 5.0, "Y"), ("SMA-2", 2.5, "N"),
            ("SMA-2", 2.5, "N"), ("STD", 0.0, "N"),
        ]]
        good = [{
            "Unit": "GOODBR", "curr_bucket": "STD", "Arrears / EMI": 0.0,
            "No Coll 3 Months and >6 EMI": "N",
        } for _ in range(5)]
        return make_df(bad + good)

    def test_branches_below_min_n_excluded(self):
        df = make_df([{"Unit": "TINY", "curr_bucket": "STD"} for _ in range(2)])
        out, _ = compute_branch_quadrant(df)
        assert out.empty

    def test_worse_branch_has_higher_concern_score_and_rank_1(self):
        out, fig = compute_branch_quadrant(self._df())
        rows = out.set_index("Branch")
        assert rows.loc["BADBR", "Concern Score"] > rows.loc["GOODBR", "Concern Score"]
        assert out.iloc[0]["Branch"] == "BADBR"
        assert out.iloc[0]["Rank"] == 1
        assert len(fig.data) == 1   # single scatter trace with both branches as points

    def test_chronic_and_npa_counts_correct(self):
        out, _ = compute_branch_quadrant(self._df())
        rows = out.set_index("Branch")
        assert rows.loc["BADBR", "Chronic (3M+)"] == 2
        assert rows.loc["BADBR", "NPA%"] == 40.0   # 2 of 5
        assert rows.loc["GOODBR", "NPA%"] == 0.0


# ── compute_executive_recovery ───────────────────────────────────────────────

class TestComputeExecutiveRecovery:
    def _df(self):
        exec1 = [
            {"MNT NAME": "EXEC1", "prev_bucket": "NPA",   "curr_bucket": "SMA-1"},  # rescued
            {"MNT NAME": "EXEC1", "prev_bucket": "SMA-2", "curr_bucket": "STD"},    # rescued
            {"MNT NAME": "EXEC1", "prev_bucket": "SMA-1", "curr_bucket": "SMA-2"},  # slipped
            {"MNT NAME": "EXEC1", "prev_bucket": "STD",   "curr_bucket": "NPA"},    # slipped
            {"MNT NAME": "EXEC1", "prev_bucket": "STD",   "curr_bucket": "STD"},    # stable
        ]
        exec2 = [
            {"MNT NAME": "EXEC2", "prev_bucket": "NPA", "curr_bucket": "STD"} for _ in range(3)
        ]
        return make_df(exec1 + exec2)

    def test_missing_bucket_columns_returns_empty(self):
        df = make_df([{"MNT NAME": "EXEC1"}])
        assert compute_executive_recovery(df).empty

    def test_rescued_slipped_and_net_recovery(self):
        out = compute_executive_recovery(self._df())
        rows = out.set_index(out["Executive"].str.split(" (", regex=False).str[0])
        assert rows.loc["EXEC1", "Rescued"] == 2
        assert rows.loc["EXEC1", "Slipped"] == 2
        assert rows.loc["EXEC1", "Net Recovery"] == 0
        assert rows.loc["EXEC2", "Rescued"] == 3
        assert rows.loc["EXEC2", "Slipped"] == 0
        assert rows.loc["EXEC2", "Net Recovery"] == 3

    def test_sorted_descending_by_net_recovery(self):
        out = compute_executive_recovery(self._df())
        assert out.iloc[0]["Executive"].startswith("EXEC2")


# ── compute_good_bad ─────────────────────────────────────────────────────────
# Regression coverage for the itertuples/"_4" column-mismatch bug: Δ NPA% and
# SMA-2% are deliberately given very different magnitudes so a regression
# (reading the wrong positional column) would fail these assertions.

class TestComputeGoodBad:
    def _region_df(self):
        return pd.DataFrame([
            {  # should surface as "good" - and must quote the NPA delta, not SMA-2%
                "Region": "IMPROVED", "Accounts": 10, "SMA-2": 1, "SMA-2%": 99.9,
                "NPA% (Curr)": 2.0, "NPA% (Prev)": 7.0, "Δ NPA%": -5.0,
                "Collection%": 95.0, "Hard Bucket%": 1.0, "SOH (Cr)": 1.0,
                "Roll Fwd%": 5.0, "Roll Bwd%": 10.0, "Status": "Improving",
            },
            {  # should surface as "bad"
                "Region": "WORSENED", "Accounts": 10, "SMA-2": 2, "SMA-2%": 1.1,
                "NPA% (Curr)": 12.0, "NPA% (Prev)": 4.8, "Δ NPA%": 7.2,
                "Collection%": 60.0, "Hard Bucket%": 9.0, "SOH (Cr)": 2.0,
                "Roll Fwd%": 20.0, "Roll Bwd%": 2.0, "Status": "Worsening",
            },
        ])

    def test_good_bullet_quotes_npa_delta_not_sma2_pct(self):
        out = compute_good_bad(self._region_df(), pd.DataFrame(), [], pd.DataFrame(), has_prev=True)
        good_text = " ".join(out["good"])
        assert "5.0pp" in good_text
        assert "99.9" not in good_text

    def test_bad_bullet_quotes_npa_delta_not_sma2_pct(self):
        out = compute_good_bad(self._region_df(), pd.DataFrame(), [], pd.DataFrame(), has_prev=True)
        bad_text = " ".join(out["bad"])
        assert "7.2pp" in bad_text
        assert "1.1" not in bad_text

    def test_no_prev_skips_region_deltas_entirely(self):
        out = compute_good_bad(self._region_df(), pd.DataFrame(), [], pd.DataFrame(), has_prev=False)
        assert not any("pp" in g for g in out["good"])
        assert not any("pp" in b for b in out["bad"])

    def test_branch_extremes_flagged_by_concern_score(self):
        branch_df = pd.DataFrame([
            {"Branch": "WORST", "Concern Score": 80, "SMA-2%": 20.0, "NPA%": 15.0, "Collection%": 70.0},
            {"Branch": "MID",   "Concern Score": 50, "SMA-2%": 8.0,  "NPA%": 5.0,  "Collection%": 90.0},
            {"Branch": "BEST",  "Concern Score": 20, "SMA-2%": 1.0,  "NPA%": 0.5,  "Collection%": 105.0},
        ])
        out = compute_good_bad(pd.DataFrame(), branch_df, [], pd.DataFrame(), has_prev=False)
        assert any("WORST" in b for b in out["bad"])
        assert any("BEST" in g for g in out["good"])
        assert not any("MID" in x for x in out["good"] + out["bad"])

    def test_risk_indicator_direction_and_threshold(self):
        indicators = [
            {"Signal": "NPA Pool", "Δ": "+2.0%", "Note": "n/a", "_delta": 2.0, "_direction": "Worsening", "_is_count": False},
            {"Signal": "SMA-1 Pool", "Δ": "-1.0%", "Note": "n/a", "_delta": -1.0, "_direction": "Improving", "_is_count": False},
            {"Signal": "Noise Signal", "Δ": "+0.05%", "Note": "n/a", "_delta": 0.05, "_direction": "Worsening", "_is_count": False},
        ]
        out = compute_good_bad(pd.DataFrame(), pd.DataFrame(), indicators, pd.DataFrame(), has_prev=False)
        assert any("NPA Pool" in b for b in out["bad"])
        assert any("SMA-1 Pool" in g for g in out["good"])
        # below the 0.3pp threshold -> neither list
        assert not any("Noise Signal" in x for x in out["good"] + out["bad"])


# ── compute_risk_flag_comparison ─────────────────────────────────────────────

class TestComputeRiskFlagComparison:
    def test_merges_curr_and_prev_by_title(self):
        curr = [{"title": "Non Starters", "count": 12, "pos": 1_00_00_000, "severity": "high", "action": "Call"}]
        prev = [{"title": "Non Starters", "count": 8, "pos": 0, "severity": "high", "action": "Call"}]
        out = compute_risk_flag_comparison(curr, prev)
        row = out.iloc[0]
        assert row["Last Month"] == 8
        assert row["Δ"] == 4

    def test_new_risk_type_has_null_prev(self):
        curr = [{"title": "Fresh Risk", "count": 5, "pos": 0, "severity": "medium", "action": "Watch"}]
        out = compute_risk_flag_comparison(curr, [])
        assert pd.isna(out.iloc[0]["Last Month"])
        assert pd.isna(out.iloc[0]["Δ"])

    def test_empty_curr_returns_empty(self):
        assert compute_risk_flag_comparison([], [{"title": "X", "count": 1}]).empty


# ── compute_product_analysis ─────────────────────────────────────────────────

class TestComputeProductAnalysis:
    def test_segment_below_min_n_excluded(self):
        curr = make_df([{"SegmentName": "TINY", "curr_bucket": "STD"} for _ in range(4)])
        out = compute_product_analysis(curr)
        assert "segment" not in out or "TINY" not in out.get("segment", pd.DataFrame()).get("Segment", [])

    def test_segment_metrics_when_above_threshold(self):
        curr = make_df([
            *[{"SegmentName": "RETAIL", "curr_bucket": b} for b in ["NPA"] * 2 + ["STD"] * 4],
        ])
        out = compute_product_analysis(curr)
        row = out["segment"].set_index("Segment").loc["RETAIL"]
        assert row["Accounts"] == 6
        assert row["NPA%"] == pytest.approx(33.33, abs=0.01)

    def test_vintage_excludes_future_dated_cohorts(self):
        future = pd.Timestamp.today() + pd.DateOffset(months=2)
        past = pd.Timestamp.today() - pd.DateOffset(months=3)
        curr = make_df([
            *[{"Ag_Date": past, "curr_bucket": "STD"} for _ in range(10)],
            *[{"Ag_Date": future, "curr_bucket": "STD"} for _ in range(10)],
        ])
        out = compute_product_analysis(curr)
        cohorts = out["vintage"]["Disbursement Month"].tolist()
        assert str(future.to_period("M")) not in cohorts
        assert str(past.to_period("M")) in cohorts

    def test_vintage_cohort_below_min_n_excluded(self):
        past = pd.Timestamp.today() - pd.DateOffset(months=1)
        curr = make_df([{"Ag_Date": past, "curr_bucket": "STD"} for _ in range(5)])  # < 10
        out = compute_product_analysis(curr)
        assert "vintage" not in out


# ── compute_risk_indicators ───────────────────────────────────────────────────

class TestComputeRiskIndicators:
    def test_pct_signal_worsening_when_bucket_share_rises(self):
        curr = make_df([{"curr_bucket": "NPA"} for _ in range(5)] + [{"curr_bucket": "STD"} for _ in range(5)])
        prev = make_df([{"curr_bucket": "NPA"} for _ in range(1)] + [{"curr_bucket": "STD"} for _ in range(9)])
        out = compute_risk_indicators(curr, prev, None)
        npa_signal = next(i for i in out if i["Signal"] == "NPA Pool")
        assert npa_signal["_direction"] == "Worsening"

    def test_pct_signal_stable_within_threshold(self):
        curr = make_df([{"curr_bucket": "NPA"} for _ in range(10)])
        prev = make_df([{"curr_bucket": "NPA"} for _ in range(10)])
        out = compute_risk_indicators(curr, prev, None)
        npa_signal = next(i for i in out if i["Signal"] == "NPA Pool")
        assert npa_signal["_direction"] == "Stable"

    def test_count_signal_improving_when_count_drops(self):
        curr = make_df([{"Non Starter": "N"} for _ in range(10)])
        prev = make_df([{"Non Starter": "Y"} for _ in range(3)] + [{"Non Starter": "N"} for _ in range(7)])
        out = compute_risk_indicators(curr, prev, None)
        ns_signal = next(i for i in out if i["Signal"] == "Non-Starters")
        assert ns_signal["_direction"] == "Improving"
        assert ns_signal["_delta"] == -3.0

    def test_fresh_npa_formation_included_only_when_matched(self):
        curr = make_df([{"curr_bucket": "STD"}])
        out_with = compute_risk_indicators(curr, make_df([]), {"matched_count": 10, "npa_formation_rate": 4.5})
        out_without = compute_risk_indicators(curr, make_df([]), {"matched_count": 0})
        assert any(i["Signal"] == "Fresh NPA Formation" for i in out_with)
        assert not any(i["Signal"] == "Fresh NPA Formation" for i in out_without)


# ── compute_fleet_exposure ────────────────────────────────────────────────────

class TestComputeFleetExposure:
    def test_three_plus_loans_counts_as_fleet(self):
        curr = make_df([
            *[{"Cust Mob No": "9990001111", "Cust Name": "FLEET OP", "curr_bucket": "STD"} for _ in range(3)],
            *[{"Cust Mob No": "8880002222", "Cust Name": "SOLO", "curr_bucket": "STD"} for _ in range(2)],
        ])
        result = compute_fleet_exposure(curr)
        assert result["count"] == 1
        assert result["top_df"].iloc[0]["Customer"] == "FLEET OP"
        assert result["top_df"].iloc[0]["Loans"] == 3

    def test_npa_operator_counted_with_single_npa_loan(self):
        curr = make_df([
            {"Cust Mob No": "111", "curr_bucket": "NPA"},
            {"Cust Mob No": "111", "curr_bucket": "STD"},
            {"Cust Mob No": "111", "curr_bucket": "STD"},
        ])
        result = compute_fleet_exposure(curr)
        assert result["npa_operators"] == 1

    def test_no_fleet_customers_returns_zeroed_result(self):
        curr = make_df([{"Cust Mob No": "111"}, {"Cust Mob No": "222"}])
        result = compute_fleet_exposure(curr)
        assert result["count"] == 0
        assert result["top_df"].empty


# ── compute_top_accounts ──────────────────────────────────────────────────────

class TestComputeTopAccounts:
    def test_sorted_descending_by_soh_and_limited_to_n(self):
        curr = make_df([
            {"SOH": soh, "curr_bucket": "SMA-1"} for soh in [50_000, 500_000, 10_000, 250_000]
        ])
        out, _ = compute_top_accounts(curr, n=2)
        assert out["SOH"].tolist() == [500_000, 250_000]

    def test_empty_when_no_soh_column(self):
        out, summary = compute_top_accounts(make_df([{"Loan No": "L1"}]).drop(columns=["SOH"]))
        assert out.empty
        assert summary == {"total_soh_cr": 0.0, "pct_of_portfolio": 0.0, "npa_count": 0}

    def test_healthy_std_and_unknown_na_accounts_excluded_even_if_huge(self):
        curr = make_df([
            {"SOH": 10_000_000, "curr_bucket": "STD"},   # huge but healthy - excluded
            {"SOH": 5_000_000, "curr_bucket": "NA"},      # huge but unknown status - excluded
            {"SOH": 100_000, "curr_bucket": "SMA-2"},     # small but delinquent - included
        ])
        out, _ = compute_top_accounts(curr)
        assert len(out) == 1
        assert out.iloc[0]["SOH"] == 100_000

    def test_includes_segment_column_when_present(self):
        curr = make_df([{"SOH": 1_000_000, "SegmentName": "RETAIL", "curr_bucket": "NPA"}])
        out, _ = compute_top_accounts(curr)
        assert "SegmentName" in out.columns
        assert out.iloc[0]["SegmentName"] == "RETAIL"

    def test_summary_concentration_and_npa_count(self):
        # 2 big NPA accounts (in top N) + 1 mid SMA-1 (excluded by n=2) + 8 small STD (excluded entirely)
        curr = make_df(
            [{"SOH": 900_000, "curr_bucket": "NPA"} for _ in range(2)]
            + [{"SOH": 400_000, "curr_bucket": "SMA-1"}]
            + [{"SOH": 10_000, "curr_bucket": "STD"} for _ in range(8)]
        )
        out, summary = compute_top_accounts(curr, n=2)
        assert len(out) == 2
        assert summary["npa_count"] == 2
        total_soh = 2 * 900_000 + 400_000 + 8 * 10_000
        assert summary["pct_of_portfolio"] == round(2 * 900_000 / total_soh * 100, 1)
        assert summary["total_soh_cr"] == round(2 * 900_000 / 1e7, 2)

    def test_no_delinquent_accounts_returns_empty(self):
        curr = make_df([{"SOH": 500_000, "curr_bucket": "STD"} for _ in range(5)])
        out, summary = compute_top_accounts(curr)
        assert out.empty
        assert summary["npa_count"] == 0


# ── compute_repossession_list ─────────────────────────────────────────────────

class TestComputeRepossessionList:
    def test_eligible_bucket_and_recent_agreement(self):
        recent = pd.Timestamp.today() - pd.DateOffset(months=6)
        curr = make_df([
            {"curr_bucket": "NPA", "Ag_Date": recent},
            {"curr_bucket": "SMA-2", "Ag_Date": recent},
            {"curr_bucket": "STD", "Ag_Date": recent},   # wrong bucket
        ])
        out = compute_repossession_list(curr)
        assert len(out) == 2
        assert set(out["curr_bucket"]) == {"NPA", "SMA-2"}

    def test_old_agreement_excluded_even_if_delinquent(self):
        old = pd.Timestamp.today() - pd.DateOffset(months=24)
        curr = make_df([{"curr_bucket": "NPA", "Ag_Date": old}])
        out = compute_repossession_list(curr)
        assert out.empty


# ── compute_good_customers ────────────────────────────────────────────────────

class TestComputeGoodCustomers:
    def test_requires_both_tenure_and_lcc_thresholds(self):
        curr = make_df([
            {"VehEMI Accrued": 80, "Tenure": 100, "LCC%": 100.0},  # qualifies: 80% tenure, LCC 100
            {"VehEMI Accrued": 50, "Tenure": 100, "LCC%": 100.0},  # fails tenure (50%)
            {"VehEMI Accrued": 90, "Tenure": 100, "LCC%": 95.0},   # fails LCC
        ])
        out = compute_good_customers(curr)
        assert len(out) == 1
        assert out.iloc[0]["Tenure Completed %"] == 80.0

    def test_sorted_ascending_by_soh(self):
        curr = make_df([
            {"VehEMI Accrued": 90, "Tenure": 100, "LCC%": 100.0, "SOH": 200_000},
            {"VehEMI Accrued": 90, "Tenure": 100, "LCC%": 100.0, "SOH": 50_000},
        ])
        out = compute_good_customers(curr)
        assert out["SOH"].tolist() == [50_000, 200_000]

    def test_missing_tenure_columns_returns_empty(self):
        curr = make_df([{"LCC%": 100.0}]).drop(columns=["Loan No"], errors="ignore")
        out = compute_good_customers(make_df([{"LCC%": 100.0}]))
        assert out.empty

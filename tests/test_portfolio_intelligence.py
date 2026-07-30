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
    build_vintage_chart,
    compute_overdue_demand_scorecard,
    compute_new_advances,
    compute_new_advances_by_dimension,
    compute_new_advances_trend,
    roll_new_advances_trend,
    compute_new_advances_trend_chart,
)
import analysis.portfolio_intelligence as pi
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

    def test_strike_pct_rising_is_good_not_inverse(self):
        # Strike% = % of accounts current on their installment obligation --
        # rising is favorable, so it must NOT be treated as an inverse
        # metric (previously was, backwards relative to its own definition
        # and to the Dashboard tab's own Strike% card convention).
        kpis = compute_pulse_kpis(self._curr(), make_df([]))
        assert _kpi(kpis, "Strike %")["inverse"] is False

    def test_insurance_debit_cases_matches_alert_definition(self):
        # Same shape as smart_alerts.py's alert_insurance_delinquency:
        # EMI-current (inst arrears <= 0) but expense arrears above the
        # threshold, with some overall arrears -- else not counted.
        curr = make_df([
            {"ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": 6_000.0, "Arrears / EMI": 1.0},  # counts
            {"ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": 4_000.0, "Arrears / EMI": 1.0},  # below threshold
            {"ARREARS AGAINST INST": 500.0, "ARREARS AGAINST EXP": 6_000.0, "Arrears / EMI": 1.0},  # real inst arrears too -- excluded
            {"ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": 6_000.0, "Arrears / EMI": 0.0},  # no overall arrears -- excluded
        ])
        kpis = compute_pulse_kpis(curr, make_df([]))
        assert _kpi(kpis, "Insurance Debit Cases")["value"] == "1"
        assert _kpi(kpis, "Insurance Debit Cases")["inverse"] is True

    def test_nov25_onward_delinquency_cohort(self):
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2025-11-01"), "Arrears / EMI": 1.0},  # exactly at cohort start, delinquent -- counts
            {"Ag_Date": pd.Timestamp("2026-01-15"), "Arrears / EMI": 2.0},  # after cohort start, delinquent -- counts
            {"Ag_Date": pd.Timestamp("2025-10-31"), "Arrears / EMI": 1.0},  # before cohort start -- excluded
            {"Ag_Date": pd.Timestamp("2026-01-15"), "Arrears / EMI": 0.0},  # in cohort but not delinquent -- excluded
        ])
        kpis = compute_pulse_kpis(curr, make_df([]))
        assert _kpi(kpis, "NOV'25 Onward Delinquency")["value"] == "2"
        assert _kpi(kpis, "NOV'25 Onward Delinquency")["inverse"] is True

    def test_soh_converted_to_crores(self):
        # 4 accounts x default POS 1,00,000 -> 4,00,000 -> 0.04 Cr
        kpis = compute_pulse_kpis(self._curr(), make_df([]))
        assert _kpi(kpis, "Total SOH")["value"] == "₹0.04Cr"

    def test_soh_good_override_none_when_no_prev(self):
        kpis = compute_pulse_kpis(self._curr(), make_df([]))
        assert _kpi(kpis, "Total SOH")["good_override"] is None

    def test_soh_falling_is_always_good_regardless_of_driver(self):
        # SOH falls even though arrears rose, because POS fell more --
        # falling SOH stays "good" unconditionally, same as before.
        curr = make_df([{"POS": 50_000.0, "Closing Arrears": 20_000.0, "SOH": 70_000.0}])
        prev = make_df([{"POS": 100_000.0, "Closing Arrears": 10_000.0, "SOH": 110_000.0}])
        kpis = compute_pulse_kpis(curr, prev)
        assert _kpi(kpis, "Total SOH")["good_override"] is True

    def test_soh_rising_driven_by_pos_growth_is_good(self):
        curr = make_df([{"POS": 150_000.0, "Closing Arrears": 10_000.0, "SOH": 160_000.0}])
        prev = make_df([{"POS": 100_000.0, "Closing Arrears": 10_000.0, "SOH": 110_000.0}])
        kpis = compute_pulse_kpis(curr, prev)
        assert _kpi(kpis, "Total SOH")["delta"] > 0
        assert _kpi(kpis, "Total SOH")["good_override"] is True

    def test_soh_rising_driven_by_arrears_growth_is_bad(self):
        curr = make_df([{"POS": 100_000.0, "Closing Arrears": 60_000.0, "SOH": 160_000.0}])
        prev = make_df([{"POS": 100_000.0, "Closing Arrears": 10_000.0, "SOH": 110_000.0}])
        kpis = compute_pulse_kpis(curr, prev)
        assert _kpi(kpis, "Total SOH")["delta"] > 0
        assert _kpi(kpis, "Total SOH")["good_override"] is False

    def test_delta_and_inverse_direction(self):
        # delta is always the RAW (curr - prev) movement, regardless of
        # `inverse` -- inverse only controls how _kpi_card_html colors it,
        # not the number itself (see analysis/portfolio_intelligence.py's
        # _delta docstring for the double-sign-flip regression this guards).
        prev = make_df([
            {"curr_bucket": "STD", "Arrears / EMI": 0.0},
            {"curr_bucket": "NPA", "Arrears / EMI": 5.0},
        ])
        kpis = compute_pulse_kpis(self._curr(), prev)
        # curr NPA% 25.0, prev NPA% 50.0 -> raw delta -25.0 (NPA% improved)
        assert _kpi(kpis, "NPA %")["delta"] == -25.0
        assert _kpi(kpis, "NPA %")["inverse"] is True
        # curr accounts 4, prev 2 -> +2
        assert _kpi(kpis, "Total Accounts")["delta"] == 2
        # curr coll% 100, prev coll% 100 -> 0
        assert _kpi(kpis, "Collection %")["delta"] == 0.0


# ── compute_region_scorecard ─────────────────────────────────────────────────

class TestRollRatesNaBucketNotComparable:
    """Regression: analysis.portfolio_intelligence._roll_rates() -- shared by the
    region scorecard, branch quadrant, and NPA/SMA-2 comparison -- used to treat
    "NA" (Arrears/EMI missing/unparseable that period) as a real, ordered bucket
    scoring -1, lower than every real bucket. A loan moving from "NA" to STD (the
    healthiest real bucket) was miscounted as "rolled forward" purely from that
    score gap. Confirmed on real production data."""

    def test_na_to_std_is_not_counted_as_roll_forward(self):
        grp = make_df([
            {"prev_bucket": "NA", "curr_bucket": "STD"},
            {"prev_bucket": "NA", "curr_bucket": "STD"},
        ])
        assert pi._roll_rates(grp) == (None, None)

    def test_mixed_na_and_real_transitions(self):
        grp = make_df([
            {"prev_bucket": "NA", "curr_bucket": "STD"},     # not comparable, excluded
            {"prev_bucket": "SMA-2", "curr_bucket": "NPA"},  # real roll-forward
            {"prev_bucket": "NPA", "curr_bucket": "STD"},    # real roll-backward
        ])
        fwd, bwd = pi._roll_rates(grp)
        # Denominator must be 2 (the 2 real comparisons), not 3.
        assert fwd == 50.0
        assert bwd == 50.0


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
        assert rows.loc["WEST", "NPA%"] == 0.0
        assert rows.loc["EAST", "NPA%"] == pytest.approx(66.67, abs=0.01)
        assert rows.loc["NORTH", "NPA%"] == 100.0

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

    def test_region_absent_from_prev_has_null_delta(self):
        curr, prev = self._dfs()
        out = compute_region_scorecard(curr, prev).set_index("Region")
        assert pd.isna(out.loc["NORTH", "Δ NPA%"])
        assert pd.isna(out.loc["NORTH", "Δ SMA-2%"])


# ── compute_overdue_demand_scorecard ─────────────────────────────────────────

class TestComputeOverdueDemandScorecard:
    """make_df sets 'Overdue'/'MonthDemandExclPC'/'OverdueCollected'/'DemandCollected'
    directly (bypassing assign_buckets), the same pattern it already uses for SOH/
    curr_bucket -- utils.compute_overdue_demand_pct's own formula is covered
    end-to-end (through the real assign_buckets path) in test_utils.py."""

    def _dfs(self):
        # EXEC1/EXEC2 each need >= MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE (11)
        # accounts to survive the executive-grain filter -- replicated 10x/11x
        # rather than a single row per pattern, which preserves every ratio
        # (sums scale linearly) so the existing % assertions stay valid.
        return make_df(
            [{"RegionName": "WEST", "Unit": "A", "MNT NAME": "EXEC1",
              "Overdue": 10_000, "OverdueCollected": 10_000,
              "MonthDemandExclPC": 20_000, "DemandCollected": 10_000}] * 10
            + [{"RegionName": "WEST", "Unit": "A", "MNT NAME": "EXEC1",
                "Overdue": 0, "OverdueCollected": 0,
                "MonthDemandExclPC": 0, "DemandCollected": 0}]
            + [{"RegionName": "EAST", "Unit": "B", "MNT NAME": "EXEC2",
                "Overdue": 40_000, "OverdueCollected": 10_000,
                "MonthDemandExclPC": 10_000, "DemandCollected": 0}] * 11
        )

    def test_region_grain_sums_not_averages(self):
        out = compute_overdue_demand_scorecard(self._dfs())["region"]
        west = out.set_index("Region").loc["WEST"]
        # 2 loans: overdue 10k (fully collected) + overdue 0 (100% by the zero rule)
        # must NOT average to (100+100)/2 by coincidence here -- assert via the
        # underlying totals instead, since both loans happen to be 100%.
        assert west["Overdue Collection %"] == 100.0

    def test_region_with_partial_collection(self):
        out = compute_overdue_demand_scorecard(self._dfs())["region"]
        east = out.set_index("Region").loc["EAST"]
        assert east["Overdue Collection %"] == 25.0
        assert east["Month Demand Collection %"] == 0.0

    def test_branch_and_executive_grains_present(self):
        result = compute_overdue_demand_scorecard(self._dfs())
        assert set(result.keys()) == {"region", "branch", "executive"}
        assert not result["branch"].empty
        assert not result["executive"].empty

    def test_branch_rows_carry_region(self):
        out = compute_overdue_demand_scorecard(self._dfs())["branch"]
        assert "Region" in out.columns
        assert out.set_index("Branch").loc["B", "Region"] == "EAST"

    def test_executive_rows_carry_branch_and_region(self):
        out = compute_overdue_demand_scorecard(self._dfs())["executive"]
        assert {"Branch", "Region"} <= set(out.columns)
        row = out.set_index("Executive").loc["EXEC2"]
        assert row["Branch"] == "B"
        assert row["Region"] == "EAST"

    def test_overall_collection_matches_existing_collection_pct_formula(self):
        # Overall Collection % must be the SAME formula as the dashboard's existing
        # Collection % KPI (Month Collection (Excl Reserve) / Net Collection Demand
        # Inst+Exp+BC) -- NOT a third, independently-derived ratio built from the
        # new Overdue/MonthDemandExclPC columns (those two totals are NOT the same
        # thing as Net Collection Demand Inst+Exp+BC on real data).
        df = make_df([
            {"RegionName": "WEST",
             "Month Collection (Excluding Reserve Collection)": 30_000.0,
             "Net Collection Demand Inst+Exp+BC": 100_000.0,
             # Overdue/demand deliberately set to totally different numbers, so a
             # test that accidentally used them instead would fail loudly.
             "Overdue": 999_000, "OverdueCollected": 999_000,
             "MonthDemandExclPC": 1, "DemandCollected": 1},
        ])
        out = compute_overdue_demand_scorecard(df)["region"]
        west = out.set_index("Region").loc["WEST"]
        assert west["Overall Collection %"] == 30.0
        assert west["Overall Collection (Cr)"] == round(30_000 / 1e7, 2)

    def test_amount_columns_present_alongside_percentages(self):
        out = compute_overdue_demand_scorecard(self._dfs())["region"]
        for c in ["Overdue Collection (Cr)", "Month Demand Collection (Cr)", "Overall Collection (Cr)", "Overall Collection %"]:
            assert c in out.columns

    def test_empty_df_returns_empty_dict_of_empty_frames(self):
        result = compute_overdue_demand_scorecard(make_df([]))
        assert all(df.empty for df in result.values())

    def test_executive_below_min_accounts_excluded(self):
        # MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE = 11 -- an executive with only
        # a handful of loans can swing to 0%/100% on a single account, not a
        # meaningful signal in a league table. Region/Branch grains aren't
        # filtered this way.
        curr = make_df([
            {"RegionName": "WEST", "Unit": "A", "MNT NAME": "TINY_EXEC",
             "Overdue": 1_000, "OverdueCollected": 1_000,
             "MonthDemandExclPC": 1_000, "DemandCollected": 1_000},
        ] * 5)  # only 5 accounts -- below the threshold
        out = compute_overdue_demand_scorecard(curr)
        assert "TINY_EXEC" not in out["executive"].get("Executive", pd.Series(dtype=object)).values
        # But the SAME loans still count at the region/branch grain.
        assert not out["region"].empty
        assert not out["branch"].empty

    def test_executive_at_min_accounts_included(self):
        curr = make_df([
            {"RegionName": "WEST", "Unit": "A", "MNT NAME": "JUST_ENOUGH",
             "Overdue": 1_000, "OverdueCollected": 1_000,
             "MonthDemandExclPC": 1_000, "DemandCollected": 1_000},
        ] * 11)  # exactly the threshold
        out = compute_overdue_demand_scorecard(curr)
        assert "JUST_ENOUGH" in out["executive"]["Executive"].values


# ── compute_new_advances ─────────────────────────────────────────────────────

class TestComputeNewAdvances:
    """New business is identified by Ag_Date's own month == the report's
    reporting month (as_of), never gated by curr_bucket. No df_prev upload
    needed -- last month's own advances are sourced from THIS SAME df_curr's
    Ag_Date history (curr_ym - 1), since a single LCC extract already
    carries every still-open loan regardless of origination month."""

    def test_filters_by_ag_date_month_not_bucket(self):
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-15"), "Loan Amount": 200_000.0, "curr_bucket": "NPA"},
            {"Ag_Date": pd.Timestamp("2026-06-20"), "Loan Amount": 300_000.0, "curr_bucket": "STD"},
            {"Ag_Date": pd.Timestamp("2026-05-01"), "Loan Amount": 999_000.0, "curr_bucket": "STD"},
        ])
        out = compute_new_advances(curr, as_of="2026-06-30")
        assert out["accounts"] == 2
        assert out["funded_cr"] == round(500_000 / 1e7, 2)

    def test_no_prior_month_rows_has_prev_false_and_mom_none(self):
        curr = make_df([{"Ag_Date": pd.Timestamp("2026-06-10"), "Loan Amount": 100_000.0}])
        out = compute_new_advances(curr, as_of="2026-06-30")
        assert out["has_prev"] is False
        assert out["accounts_mom_pct"] is None
        assert out["funded_mom_pct"] is None

    def test_mom_comparison_from_same_upload_own_prior_month_rows(self):
        curr = make_df(
            [{"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 200_000.0}] * 2
            + [{"Ag_Date": pd.Timestamp("2026-05-05"), "Loan Amount": 100_000.0}]
        )
        out = compute_new_advances(curr, as_of="2026-06-30")
        assert out["has_prev"] is True
        assert out["prev_accounts"] == 1
        assert out["accounts_mom_pct"] == 100.0
        assert out["funded_mom_pct"] == 300.0

    def test_rows_outside_prior_month_excluded(self):
        # Only ONE calendar month back counts as "prior month" -- not any
        # older Ag_Date row in the same upload.
        curr = make_df(
            [{"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0}]
            + [{"Ag_Date": pd.Timestamp("2026-04-01"), "Loan Amount": 500_000.0}]
        )
        out = compute_new_advances(curr, as_of="2026-06-30")
        assert out["prev_accounts"] == 0
        assert out["accounts_mom_pct"] is None

    def test_segment_breakdown_gated_by_min_accounts(self):
        # Threshold-relative: builds one segment at the threshold and one just
        # below it, so retuning MIN_ACCOUNTS_PRODUCT_SEGMENT doesn't break this.
        from config import MIN_ACCOUNTS_PRODUCT_SEGMENT
        curr = make_df(
            [{"Ag_Date": pd.Timestamp("2026-06-01"), "Loan Amount": 100_000.0, "SegmentName": "CV"}]
            * MIN_ACCOUNTS_PRODUCT_SEGMENT
            + [{"Ag_Date": pd.Timestamp("2026-06-02"), "Loan Amount": 100_000.0, "SegmentName": "TINY"}]
            * (MIN_ACCOUNTS_PRODUCT_SEGMENT - 1)
        )
        out = compute_new_advances(curr, as_of="2026-06-30")
        segs = out["segment"]["Segment"].tolist()
        assert "CV" in segs
        if MIN_ACCOUNTS_PRODUCT_SEGMENT > 1:
            assert "TINY" not in segs

    def test_empty_df_returns_zeroed_result(self):
        out = compute_new_advances(make_df([]), as_of="2026-06-30")
        assert out["accounts"] == 0
        assert out["segment"].empty

    def test_missing_ag_date_column_returns_zeroed_result(self):
        df = make_df([{"Loan Amount": 100_000.0}]).drop(columns=["Ag_Date"])
        out = compute_new_advances(df, as_of="2026-06-30")
        assert out["accounts"] == 0


# ── compute_new_advances_by_dimension ─────────────────────────────────────────

class TestComputeNewAdvancesByDimension:
    """Business/origination view -- deliberately NO materiality floor on the
    executive grain (unlike the Overdue vs Month Demand league table), so
    even a single new advance shows up."""

    def test_executive_shows_up_with_a_single_account(self):
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "SOLO_EXEC"},
        ])
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        assert "SOLO_EXEC" in out["executive"]["Executive"].values

    def test_branch_carries_region_executive_carries_branch_and_region(self):
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0,
             "RegionName": "EAST", "Unit": "BR1", "MNT NAME": "EXEC1"},
        ])
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        branch_row = out["branch"].set_index("Branch").loc["BR1"]
        assert branch_row["Region"] == "EAST"
        exec_row = out["executive"].set_index("Executive").loc["EXEC1"]
        assert exec_row["Branch"] == "BR1" and exec_row["Region"] == "EAST"

    def test_mom_columns_sourced_from_same_upload(self):
        curr = make_df(
            [{"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 200_000.0,
              "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1"}] * 2
            + [{"Ag_Date": pd.Timestamp("2026-05-05"), "Loan Amount": 100_000.0,
                "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1"}]
        )
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        row = out["region"].set_index("Region").loc["WEST"]
        assert row["Prev Accounts"] == 1
        assert row["Accounts MoM %"] == 100.0

    def test_empty_df_returns_empty_frames(self):
        out = compute_new_advances_by_dimension(make_df([]), as_of="2026-06-30")
        assert all(df.empty for df in out.values())

    def test_total_accounts_is_whole_book_not_just_this_month(self):
        # EXEC1 has 3 loans total in df_curr (any Ag_Date), but only 1 was
        # originated this reporting month -- Total Accounts must reflect the
        # whole book, Accounts This Month only the fresh originations.
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1"},
            {"Ag_Date": pd.Timestamp("2024-01-01"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1"},
            {"Ag_Date": pd.Timestamp("2023-01-01"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1"},
        ])
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        row = out["executive"].set_index("Executive").loc["EXEC1"]
        assert row["Total Accounts"] == 3
        assert row["Accounts This Month"] == 1

    def test_sorted_by_accounts_this_month_descending(self):
        curr = make_df(
            [{"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 50_000.0,
              "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "BIG_EXEC"}] * 5
            + [{"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 999_000.0,
                "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "SMALL_EXEC_HIGH_AMT"}] * 1
        )
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        exec_df = out["executive"]
        # SMALL_EXEC_HIGH_AMT has more Funded (Cr) but fewer accounts -- sort
        # must be by Accounts This Month, not Funded (Cr).
        assert exec_df.iloc[0]["Executive"] == "BIG_EXEC"
        assert exec_df["Accounts This Month"].tolist() == sorted(exec_df["Accounts This Month"].tolist(), reverse=True)

    def test_unique_customers_counts_distinct_cust_mob_no_not_loans(self):
        # Same customer (by Cust Mob No) takes 2 fresh loans this month, plus
        # a second, different customer takes 1 -- Accounts This Month must
        # count all 3 loans, but Unique Customers must count only 2 people,
        # the exact gap this column exists to surface (see compute_fleet_exposure
        # for the same Cust Mob No customer-identity convention).
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1", "Cust Mob No": "9000000001"},
            {"Ag_Date": pd.Timestamp("2026-06-10"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1", "Cust Mob No": "9000000001"},
            {"Ag_Date": pd.Timestamp("2026-06-15"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1", "Cust Mob No": "9000000002"},
        ])
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        row = out["executive"].set_index("Executive").loc["EXEC1"]
        assert row["Accounts This Month"] == 3
        assert row["Unique Customers"] == 2

    def test_unique_customers_excludes_blank_mobile_numbers(self):
        # Two loans with no Cust Mob No must not collapse into "1 customer"
        # via a naive nunique on blank strings.
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1", "Cust Mob No": ""},
            {"Ag_Date": pd.Timestamp("2026-06-10"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1", "Cust Mob No": ""},
        ])
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        row = out["executive"].set_index("Executive").loc["EXEC1"]
        assert row["Accounts This Month"] == 2
        assert row["Unique Customers"] == 0

    def test_unique_customers_falls_back_to_account_count_without_cust_mob_col(self):
        # Cust Mob No isn't in this test file at all -- must degrade to the
        # account count rather than raising or silently returning 0/None.
        curr = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0,
             "RegionName": "WEST", "Unit": "MAHAD", "MNT NAME": "EXEC1"},
        ])
        assert "Cust Mob No" not in curr.columns
        out = compute_new_advances_by_dimension(curr, as_of="2026-06-30")
        row = out["executive"].set_index("Executive").loc["EXEC1"]
        assert row["Unique Customers"] == row["Accounts This Month"] == 1


# ── compute_new_advances_trend / roll_new_advances_trend ──────────────────────

class TestComputeNewAdvancesTrend:
    def _spread_df(self):
        # One loan per month from Jan 2025 through Jun 2026 (18 months).
        rows = []
        for y, m in [(2025, mo) for mo in range(1, 13)] + [(2026, mo) for mo in range(1, 7)]:
            rows.append({"Ag_Date": pd.Timestamp(y, m, 15), "Loan Amount": 100_000.0})
        return make_df(rows)

    def test_default_window_uses_config_constant(self):
        import config
        df = self._spread_df()
        out = compute_new_advances_trend(df, as_of="2026-06-30")
        # Only 18 months of data exist -- fewer than the 24-month default cap,
        # so every month present should show up (cap is a ceiling, not a floor).
        assert len(out) == 18
        assert config.NEW_ADVANCES_TREND_DEFAULT_MONTHS == 24

    def test_months_none_returns_full_history(self):
        df = self._spread_df()
        out = compute_new_advances_trend(df, as_of="2026-06-30", months=None)
        assert len(out) == 18

    def test_months_caps_window(self):
        df = self._spread_df()
        out = compute_new_advances_trend(df, as_of="2026-06-30", months=6)
        assert len(out) == 6
        assert out["Month"].min() == "2026-01"
        assert out["Month"].max() == "2026-06"

    def test_excludes_post_dated_cohorts(self):
        df = make_df([
            {"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0},
            {"Ag_Date": pd.Timestamp("2026-12-05"), "Loan Amount": 999_000.0},
        ])
        out = compute_new_advances_trend(df, as_of="2026-06-30")
        assert "2026-12" not in out["Month"].values

    def test_empty_returns_empty(self):
        assert compute_new_advances_trend(make_df([]), as_of="2026-06-30").empty


class TestRollNewAdvancesTrend:
    def _trend_df(self):
        return pd.DataFrame([
            {"Month": "2025-04", "Accounts": 2, "Funded (Cr)": 0.20, "Avg Ticket (L)": 10.0},
            {"Month": "2025-05", "Accounts": 3, "Funded (Cr)": 0.30, "Avg Ticket (L)": 10.0},
            {"Month": "2026-01", "Accounts": 1, "Funded (Cr)": 0.10, "Avg Ticket (L)": 10.0},
            {"Month": "2026-04", "Accounts": 4, "Funded (Cr)": 0.40, "Avg Ticket (L)": 10.0},
        ])

    def test_monthly_is_a_passthrough(self):
        df = self._trend_df()
        out = roll_new_advances_trend(df, "Monthly")
        pd.testing.assert_frame_equal(out, df)

    def test_quarterly_sums_within_calendar_quarter(self):
        out = roll_new_advances_trend(self._trend_df(), "Quarterly")
        q2_2025 = out.set_index("Month").loc["Q2-2025"]
        assert q2_2025["Accounts"] == 5  # Apr + May 2025
        assert q2_2025["Funded (Cr)"] == pytest.approx(0.50)

    def test_financial_year_groups_apr_to_mar_together(self):
        # FY25-26 = Apr 2025 - Mar 2026 -- must include BOTH the 2025-04/05
        # rows AND the 2026-01 row, and NOT the 2026-04 row (that's FY26-27).
        out = roll_new_advances_trend(self._trend_df(), "Financial Year")
        fy = out.set_index("Month").loc["FY25-26"]
        assert fy["Accounts"] == 2 + 3 + 1
        assert "FY26-27" in out["Month"].values
        fy_next = out.set_index("Month").loc["FY26-27"]
        assert fy_next["Accounts"] == 4

    def test_financial_year_sorted_chronologically(self):
        out = roll_new_advances_trend(self._trend_df(), "Financial Year")
        assert out["Month"].tolist() == ["FY25-26", "FY26-27"]

    def test_avg_ticket_recomputed_not_averaged(self):
        # 2 accounts @10L + 3 accounts @10L should still be 10L avg, not
        # some other artifact of summing the per-month averages directly.
        out = roll_new_advances_trend(self._trend_df(), "Quarterly")
        q2_2025 = out.set_index("Month").loc["Q2-2025"]
        assert q2_2025["Avg Ticket (L)"] == pytest.approx(10.0)

    def test_empty_df_returns_empty(self):
        assert roll_new_advances_trend(pd.DataFrame(), "Quarterly").empty


class TestComputeNewAdvancesTrendChart:
    def test_monthly_adds_quarter_end_markers(self):
        df = pd.DataFrame([
            {"Month": "2026-01", "Accounts": 1, "Funded (Cr)": 0.1, "Avg Ticket (L)": 10.0},
            {"Month": "2026-02", "Accounts": 1, "Funded (Cr)": 0.1, "Avg Ticket (L)": 10.0},
            {"Month": "2026-03", "Accounts": 1, "Funded (Cr)": 0.1, "Avg Ticket (L)": 10.0},
        ])
        fig = compute_new_advances_trend_chart(df, granularity="Monthly")
        assert len(fig.layout.shapes) == 1  # only March is a quarter-end month

    def test_non_monthly_granularity_has_no_markers(self):
        df = pd.DataFrame([
            {"Month": "Q1-2026", "Accounts": 3, "Funded (Cr)": 0.3, "Avg Ticket (L)": 10.0},
        ])
        fig = compute_new_advances_trend_chart(df, granularity="Quarterly")
        assert len(fig.layout.shapes) == 0

    def test_empty_df_returns_placeholder_figure(self):
        fig = compute_new_advances_trend_chart(pd.DataFrame())
        assert fig.layout.title.text == "No data available"


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

    def test_branch_rows_include_region_and_roll_rates(self):
        curr, prev = self._dfs()
        out = compute_npa_sma2_comparison(curr, prev)
        row = out["branch"].set_index("Unit").loc["MAHAD"]
        assert row["Region"] == "WEST"
        assert "Roll Fwd%" in out["branch"].columns
        assert "Roll Bwd%" in out["branch"].columns

    def test_region_rows_have_roll_rates_but_no_unit_or_region_column(self):
        curr, prev = self._dfs()
        out = compute_npa_sma2_comparison(curr, prev)
        assert "Roll Fwd%" in out["region"].columns
        assert "Roll Bwd%" in out["region"].columns
        assert "Unit" not in out["region"].columns
        assert "Region" not in out["region"].columns

    def test_no_executive_key_when_mnt_name_missing(self):
        curr, prev = self._dfs()
        out = compute_npa_sma2_comparison(curr, prev)
        assert "executive" not in out

    def test_executive_rows_include_unit_and_region_context(self):
        curr = make_df([
            {"MNT NAME": "RAHUL", "Unit": "PUNE1", "RegionName": "WEST", "curr_bucket": b}
            for b in ["NPA", "NPA", "STD"]
        ])
        out = compute_npa_sma2_comparison(curr, make_df([]))
        row = out["executive"].iloc[0]
        assert row["Unit"] == "PUNE1"
        assert row["Region"] == "WEST"

    def test_executive_prev_match_is_case_and_whitespace_insensitive(self):
        # MNT NAME is manually retyped each month (not a controlled vocabulary), so
        # "Sunil Waghmare" this month vs "SUNIL WAGHMARE " last month must still join.
        curr = make_df([
            {"MNT NAME": "Sunil Waghmare", "curr_bucket": b} for b in ["NPA", "NPA", "STD"]
        ])
        prev = make_df([
            {"MNT NAME": "SUNIL WAGHMARE ", "curr_bucket": b} for b in ["NPA", "STD", "STD"]
        ])
        out = compute_npa_sma2_comparison(curr, prev)
        row = out["executive"].iloc[0]
        assert row["NPA (Prev)"] == 1
        assert row["NPA Δ"] == 1

    def test_executive_key_present_when_mnt_name_available(self):
        curr = make_df([
            {"MNT NAME": "RAHUL", "curr_bucket": b} for b in ["NPA", "NPA", "STD"]
        ])
        out = compute_npa_sma2_comparison(curr, make_df([]))
        assert "executive" in out
        assert out["executive"]["NPA (Curr)"].iloc[0] == 2

    def test_same_name_different_branch_kept_as_separate_rows(self):
        # Two different people can share a name across branches (e.g. two "Rahul Sharma"s) --
        # they must never be merged into one row just because MNT NAME matches.
        curr = make_df(
            [{"MNT NAME": "Rahul Sharma", "Unit": "X", "curr_bucket": "NPA"} for _ in range(3)]
            + [{"MNT NAME": "Rahul Sharma", "Unit": "Y", "curr_bucket": "STD"} for _ in range(3)]
        )
        out = compute_npa_sma2_comparison(curr, make_df([]))
        exec_df = out["executive"]
        assert len(exec_df) == 2
        names = set(exec_df["MNT NAME"])
        assert names == {"Rahul Sharma (X)", "Rahul Sharma (Y)"}
        row_x = exec_df.set_index("MNT NAME").loc["Rahul Sharma (X)"]
        row_y = exec_df.set_index("MNT NAME").loc["Rahul Sharma (Y)"]
        assert row_x["Unit"] == "X" and row_x["NPA (Curr)"] == 3
        assert row_y["Unit"] == "Y" and row_y["NPA (Curr)"] == 0

    def test_prev_match_scoped_to_same_unit_not_cross_branch(self):
        # A same-named executive in a different branch last month must not be treated
        # as this month's match -- prev lookup (including the prefix fallback) is
        # scoped to the same Unit.
        curr = make_df([{"MNT NAME": "Rahul Sharma", "Unit": "X", "curr_bucket": "NPA"} for _ in range(3)])
        prev = make_df([{"MNT NAME": "Rahul Sharma", "Unit": "Y", "curr_bucket": "NPA"} for _ in range(3)])
        out = compute_npa_sma2_comparison(curr, prev)
        row = out["executive"].iloc[0]
        assert row["NPA (Prev)"] is None

    def test_executive_prev_match_falls_back_to_unambiguous_prefix(self):
        # Source export truncates MNT NAME to a different length between months
        # (e.g. "...DNYANESHWAR F" vs "...DNYANESHWAR FA") -- an exact/normalized
        # match misses this, but there's exactly one prefix candidate so it's safe
        # to treat as the same executive.
        curr = make_df([
            {"MNT NAME": "SHUBHAM DNYANESHWAR F", "curr_bucket": b} for b in ["NPA", "NPA", "STD"]
        ])
        prev = make_df([
            {"MNT NAME": "SHUBHAM DNYANESHWAR FA", "curr_bucket": b} for b in ["NPA", "STD", "STD"]
        ])
        out = compute_npa_sma2_comparison(curr, prev)
        row = out["executive"].iloc[0]
        assert row["NPA (Prev)"] == 1
        assert row["NPA Δ"] == 1

    def test_executive_prev_match_skips_ambiguous_prefix_candidates(self):
        # Two different previous-period names both prefix-match the current name --
        # must not guess, stays unmatched (None) rather than merging two people.
        curr = make_df([
            {"MNT NAME": "SANTOSH", "curr_bucket": b} for b in ["NPA", "NPA", "STD"]
        ])
        prev = make_df(
            [{"MNT NAME": "SANTOSH KAPSE", "curr_bucket": "NPA"} for _ in range(3)]
            + [{"MNT NAME": "SANTOSH PATIL", "curr_bucket": "NPA"} for _ in range(3)]
        )
        out = compute_npa_sma2_comparison(curr, prev)
        row = out["executive"].iloc[0]
        assert row["NPA (Prev)"] is None
        assert row["NPA Δ"] is None


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

    def test_region_and_strike_pct_present(self):
        # Region and Strike% were added on request -- Region so the dashboard's
        # branch table and the report's "Highest Concern Branches" table can
        # both show which region a branch belongs to; Strike% specifically for
        # the report table (the dashboard table deliberately doesn't display it).
        curr = make_df([
            {"Unit": "BR1", "RegionName": "AKOLA", "Strike": "Y"} for _ in range(3)
        ] + [
            {"Unit": "BR1", "RegionName": "AKOLA", "Strike": "N"} for _ in range(2)
        ])
        out, _ = compute_branch_quadrant(curr)
        row = out.set_index("Branch").loc["BR1"]
        assert row["Region"] == "AKOLA"
        assert row["Strike%"] == 60.0  # 3 of 5


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

    def test_na_prev_bucket_never_counts_as_slipped(self):
        # Regression: "NA" (Arrears/EMI missing/unparseable last period, not a
        # real delinquency state) used to score as -1 in BUCKET_SCORE, lower
        # than every real bucket -- so a loan moving from "NA" to STD (the
        # HEALTHIEST real bucket) was miscounted as "slipped" (worsened) purely
        # from the score gap, with zero real deterioration. Confirmed on real
        # production data (an executive with 0% NPA/SMA-2 showing 100% Roll
        # Fwd%, driven entirely by "NA"-origin loans).
        df = make_df([
            {"MNT NAME": "EXEC3", "prev_bucket": "NA", "curr_bucket": "STD"},
            {"MNT NAME": "EXEC3", "prev_bucket": "NA", "curr_bucket": "STD"},
            {"MNT NAME": "EXEC3", "prev_bucket": "NA", "curr_bucket": "STD"},
        ])
        out = compute_executive_recovery(df)
        # With every comparison excluded (no real prior bucket to compare against),
        # the executive is either dropped entirely (no valid comparisons at all) or
        # kept with Slipped/Rescued both 0 -- never counted as having slipped.
        assert out.empty or (
            out.loc[out["Executive"].str.startswith("EXEC3"), "Slipped"] == 0
        ).all()


# ── compute_good_bad ─────────────────────────────────────────────────────────
# Regression coverage for the itertuples/"_4" column-mismatch bug: Δ NPA% and
# SMA-2% are deliberately given very different magnitudes so a regression
# (reading the wrong positional column) would fail these assertions.

class TestComputeGoodBad:
    def _region_df(self):
        return pd.DataFrame([
            {  # should surface as "good" - and must quote the NPA delta, not SMA-2%
                "Region": "IMPROVED", "SMA-2": 1, "SMA-2%": 99.9, "NPA": 2,
                "NPA%": 2.0, "Δ SMA-2%": -1.0, "Δ NPA%": -5.0,
                "Collection%": 95.0, "Strike%": 1.0, "SOH (Cr)": 1.0,
                "Roll Fwd%": 5.0, "Roll Bwd%": 10.0, "Status": "Improving",
            },
            {  # should surface as "bad"
                "Region": "WORSENED", "SMA-2": 2, "SMA-2%": 1.1, "NPA": 12,
                "NPA%": 12.0, "Δ SMA-2%": 0.5, "Δ NPA%": 7.2,
                "Collection%": 60.0, "Strike%": 9.0, "SOH (Cr)": 2.0,
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
    @pytest.mark.parametrize("prev,expected_last_month,expected_delta", [
        ([{"title": "Non Starters", "count": 8, "pos": 0, "severity": "high", "action": "Call"}], 8, 4),
        ([], None, None),  # risk type didn't exist last month -> null prev/delta, not zero
    ], ids=["matched_by_title", "new_risk_type"])
    def test_merges_curr_and_prev_by_title(self, prev, expected_last_month, expected_delta):
        curr = [{"title": "Non Starters", "count": 12, "pos": 1_00_00_000, "severity": "high", "action": "Call"}]
        row = compute_risk_flag_comparison(curr, prev).iloc[0]
        if expected_last_month is None:
            assert pd.isna(row["Last Month"]) and pd.isna(row["Δ"])
        else:
            assert row["Last Month"] == expected_last_month
            assert row["Δ"] == expected_delta

    def test_empty_curr_returns_empty(self):
        assert compute_risk_flag_comparison([], [{"title": "X", "count": 1}]).empty


# ── compute_product_analysis ─────────────────────────────────────────────────

class TestComputeProductAnalysis:
    # Threshold-relative (reads MIN_ACCOUNTS_PRODUCT_SEGMENT from config)
    # rather than hardcoding the value, so retuning the constant doesn't
    # break these tests -- only the behavior they assert.

    def test_segment_below_min_n_excluded(self):
        from config import MIN_ACCOUNTS_PRODUCT_SEGMENT
        if MIN_ACCOUNTS_PRODUCT_SEGMENT <= 1:
            pytest.skip("threshold is 1 -- no below-threshold group can exist")
        curr = make_df([{"SegmentName": "TINY", "curr_bucket": "STD"}
                        for _ in range(MIN_ACCOUNTS_PRODUCT_SEGMENT - 1)])
        out = compute_product_analysis(curr)
        assert "segment" not in out or "TINY" not in out.get("segment", pd.DataFrame()).get("Segment", [])

    def test_segment_at_threshold_included(self):
        from config import MIN_ACCOUNTS_PRODUCT_SEGMENT
        curr = make_df([{"SegmentName": "TINY", "curr_bucket": "STD"}
                        for _ in range(MIN_ACCOUNTS_PRODUCT_SEGMENT)])
        out = compute_product_analysis(curr)
        assert "TINY" in out["segment"]["Segment"].tolist()

    def test_segment_metrics_when_above_threshold(self):
        # 12 total (4 NPA + 8 STD) clears any historical threshold value
        # (was 11, now 1) while keeping the same 33.33% ratio.
        curr = make_df([
            *[{"SegmentName": "RETAIL", "curr_bucket": b} for b in ["NPA"] * 4 + ["STD"] * 8],
        ])
        out = compute_product_analysis(curr)
        row = out["segment"].set_index("Segment").loc["RETAIL"]
        assert row["Accounts"] == 12
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

    def test_as_of_anchors_future_exclusion_to_reporting_month_not_wall_clock(self):
        # Regression: "exclude post-dated agreement dates" used to anchor to
        # the real wall-clock date (pd.Period.now), not the report's OWN
        # reporting month. Retroactively analyzing an old file (e.g. a March
        # extract opened today, months later) would silently exclude/include
        # cohorts relative to TODAY instead of March.
        report_month = pd.Timestamp("2026-03-15")
        # A cohort that is in the FUTURE relative to the March report month,
        # but in the PAST relative to the real wall-clock date this test runs
        # on -- the only way to prove as_of is actually driving the cutoff,
        # not incidentally agreeing with wall-clock "now".
        cohort_after_report_month = pd.Timestamp.today() - pd.DateOffset(months=1)
        assert cohort_after_report_month > report_month  # sanity: test premise holds
        cohort_before_report_month = pd.Timestamp("2025-12-01")

        curr = make_df([
            *[{"Ag_Date": cohort_before_report_month, "curr_bucket": "STD"} for _ in range(10)],
            *[{"Ag_Date": cohort_after_report_month, "curr_bucket": "STD"} for _ in range(10)],
        ])

        out_with_as_of = compute_product_analysis(curr, as_of=report_month)
        cohorts_with = out_with_as_of["vintage"]["Disbursement Month"].tolist()
        assert str(cohort_after_report_month.to_period("M")) not in cohorts_with
        assert str(cohort_before_report_month.to_period("M")) in cohorts_with

        # Without as_of (defaults to wall-clock now), the same "future" cohort
        # is NOT excluded, since it's actually in the past relative to today --
        # demonstrating the old, buggy behavior this default preserves only for
        # callers with no reporting date to hand.
        out_without_as_of = compute_product_analysis(curr)
        cohorts_without = out_without_as_of["vintage"]["Disbursement Month"].tolist()
        assert str(cohort_after_report_month.to_period("M")) in cohorts_without


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

    def test_fresh_npa_formation_never_reports_a_fabricated_worsening_trend(self):
        # Fresh NPA Formation has no genuine prior-period value to compare against
        # (it's a same-period roll-rate figure, not a MoM stat) -- it must render
        # as a standalone reading ("-" direction, no Delta), never as "Worsening"
        # just because it was compared against a hardcoded 0. Use a REAL, non-empty
        # df_prev (unlike the "included_only_when_matched" test above) so has_prev
        # is actually True and the bug's code path is exercised.
        curr = make_df([{"curr_bucket": "STD"} for _ in range(10)])
        prev = make_df([{"curr_bucket": "STD"} for _ in range(10)])
        out = compute_risk_indicators(curr, prev, {"matched_count": 10, "npa_formation_rate": 4.5})
        signal = next(i for i in out if i["Signal"] == "Fresh NPA Formation")
        assert signal["_direction"] == " - "
        assert signal["Δ"] == " - "
        assert signal["Last Month"] == " - "
        assert signal["This Month"] == "4.5%"

    def test_fresh_npa_formation_excluded_from_good_bad_narrative(self):
        # A "Worsening" Fresh NPA Formation signal used to always qualify for the
        # Good/Bad verdict's bad-news list, regardless of actual trend. With no
        # real direction, it must never be picked up there.
        curr = make_df([{"curr_bucket": "STD"} for _ in range(10)])
        prev = make_df([{"curr_bucket": "STD"} for _ in range(10)])
        indicators = compute_risk_indicators(curr, prev, {"matched_count": 10, "npa_formation_rate": 4.5})
        result = compute_good_bad(pd.DataFrame(), pd.DataFrame(), indicators, pd.DataFrame(), has_prev=True)
        assert not any("Fresh NPA Formation" in b for b in result["bad"])
        assert not any("Fresh NPA Formation" in g for g in result["good"])


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

    def test_blank_mobile_loans_excluded_not_merged_into_phantom_fleet(self):
        # Regression: utils.py::clean_mobile normalizes a missing Cust Mob No
        # to "" (not NaN). Grouping by Cust Mob No without excluding "" used to
        # merge every loan with no mobile number on file into one fictitious
        # "fleet operator" combining unrelated customers' exposure -- on a real
        # production file, 175 such loans collapsed into a single phantom
        # operator with ~47.9 Cr combined SOH, the largest entry in the table.
        curr = make_df([
            *[{"Cust Mob No": "", "Cust Name": f"UNRELATED {i}", "SOH": 100.0} for i in range(5)],
            *[{"Cust Mob No": "9990001111", "Cust Name": "REAL FLEET OP", "SOH": 50.0} for _ in range(3)],
        ])
        result = compute_fleet_exposure(curr)
        # Only the real 3-loan customer counts as a fleet operator -- the 5
        # blank-mobile loans (each belonging to a different, unrelated
        # customer) must not be merged into a second "operator".
        assert result["count"] == 1
        assert result["top_df"].iloc[0]["Customer"] == "REAL FLEET OP"
        assert "" not in result["top_df"]["Mobile"].values
        assert result["excluded_blank_mobile_loans"] == 5

    def test_excluded_blank_mobile_loans_key_present_even_with_no_fleet(self):
        curr = make_df([{"Cust Mob No": "", "Cust Name": "A"}, {"Cust Mob No": "111", "Cust Name": "B"}])
        result = compute_fleet_exposure(curr)
        assert result["count"] == 0
        assert result["excluded_blank_mobile_loans"] == 1

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

    def test_top_df_includes_region_and_unit(self):
        curr = make_df([
            *[{"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "AKOLA", "Unit": "UNIT_1"} for _ in range(3)],
        ])
        result = compute_fleet_exposure(curr)
        row = result["top_df"].iloc[0]
        assert row["Region"] == "AKOLA"
        assert row["Unit"] == "UNIT_1"

    def test_region_and_unit_use_most_common_when_customer_spans_branches(self):
        # Cust Mob No isn't guaranteed unique per branch -- a fleet customer's
        # loans can legitimately span more than one region/branch. The most
        # common (mode) value is shown as a single representative, not every
        # branch this customer touches.
        curr = make_df([
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "AKOLA", "Unit": "UNIT_1"},
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "AKOLA", "Unit": "UNIT_1"},
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "LATUR", "Unit": "UNIT_2"},
        ])
        result = compute_fleet_exposure(curr)
        row = result["top_df"].iloc[0]
        assert row["Region"] == "AKOLA"
        assert row["Unit"] == "UNIT_1"

    def test_region_mode_tie_broken_alphabetically_ascending(self):
        # Regression: the vectorized _grouped_mode helper (replacing a
        # per-customer Series.mode() call) must reproduce mode()'s own
        # tie-break exactly -- pandas' mode() returns every tied value
        # sorted ascending, and the original code took .iat[0] (the
        # alphabetically-first). An exact 2-2 tie between two regions is
        # the case most likely to expose a tie-break mismatch.
        curr = make_df([
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "ZONE_B", "Unit": "UNIT_1"},
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "ZONE_B", "Unit": "UNIT_1"},
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "ZONE_A", "Unit": "UNIT_2"},
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "ZONE_A", "Unit": "UNIT_2"},
        ])
        expected = curr["RegionName"].mode().iat[0]
        result = compute_fleet_exposure(curr)
        row = result["top_df"].iloc[0]
        assert row["Region"] == expected == "ZONE_A"

    def test_blank_first_row_cust_name_preserved_not_overwritten(self):
        # Regression: the vectorized customer-name lookup uses .head(1), NOT
        # .first() (which silently skips NaN) -- must match the original
        # .iloc[0]'s behavior of showing whatever the first row actually has,
        # blank or not, rather than substituting a later non-blank name.
        curr = make_df([
            {"Cust Mob No": "999", "Cust Name": None, "RegionName": "AKOLA", "Unit": "UNIT_1"},
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "AKOLA", "Unit": "UNIT_1"},
            {"Cust Mob No": "999", "Cust Name": "FLEET OP", "RegionName": "AKOLA", "Unit": "UNIT_1"},
        ])
        result = compute_fleet_exposure(curr)
        row = result["top_df"].iloc[0]
        assert row["Customer"] == "nan"  # str(float("nan")) -- matches the original str(cust_name) exactly


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
            {"curr_bucket": "STD", "Ag_Date": recent},   # wrong bucket -> excluded
        ])
        out = compute_repossession_list(curr)
        assert len(out) == 2
        assert set(out["curr_bucket"]) == {"NPA", "SMA-2"}

    def test_old_agreement_excluded_even_if_delinquent(self):
        # Eligibility requires BOTH conditions -- deep delinquency alone isn't
        # enough once the loan is past the collateral-value window.
        old = pd.Timestamp.today() - pd.DateOffset(months=24)
        curr = make_df([{"curr_bucket": "NPA", "Ag_Date": old}])
        assert compute_repossession_list(curr).empty

    def test_as_of_anchors_window_to_reporting_month_not_wall_clock(self):
        # Regression: the "still within the repossession window" cutoff used
        # to measure backward from the real wall-clock date, not the report's
        # OWN reporting month -- wrong for any retroactive/historical analysis
        # (e.g. re-opening an old file long after its actual reporting month).
        report_date = pd.Timestamp("2020-01-01")
        within_window_of_report = report_date - pd.DateOffset(months=6)  # well within 18mo of Jan 2020
        curr = make_df([{"curr_bucket": "NPA", "Ag_Date": within_window_of_report}])

        out_with_as_of = compute_repossession_list(curr, as_of=report_date)
        assert len(out_with_as_of) == 1

        # Without as_of (defaults to wall-clock today), the same loan is years
        # outside the 18-month window and must NOT appear -- the old, buggy
        # behavior this default preserves only for callers with no reporting
        # date to hand.
        out_without_as_of = compute_repossession_list(curr)
        assert out_without_as_of.empty


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


# ── build_vintage_chart ──────────────────────────────────────────────────────
# Regression: the "Critical"/"Watch" marker-color and reference-band cutoffs
# used to be hardcoded (10, 5) directly in this function, duplicating semantics
# that live nowhere in config.py - contradicting the "every threshold lives in
# config.py" rule the rest of the analysis layer follows. Now sourced from
# config.VINTAGE_CHART_CRITICAL_PCT / VINTAGE_CHART_WATCH_PCT.

class TestBuildVintageChartThresholds:
    def _vintage_df(self):
        return pd.DataFrame({
            "Disbursement Month": ["2025-01", "2025-02", "2025-03"],
            "NPA%":  [3.0, 6.0, 11.0],   # below watch / between / above critical
            "SMA-2%": [1.0, 2.0, 3.0],
        })

    def test_empty_df_returns_empty_figure(self):
        fig = build_vintage_chart(pd.DataFrame())
        assert fig.data == ()

    def test_marker_colors_follow_config_thresholds(self):
        fig = build_vintage_chart(self._vintage_df())
        npa_trace = next(t for t in fig.data if t.name == "NPA %")
        assert list(npa_trace.marker.color) == ["#16a34a", "#f97316", "#991b1b"]

    def test_marker_colors_follow_a_retuned_threshold(self, monkeypatch):
        # Tightening the critical threshold to 5 should reclassify the 6.0%
        # cohort (previously "watch") as critical - proving the chart reads
        # the live config value rather than a baked-in 10/5.
        monkeypatch.setattr(pi, "VINTAGE_CHART_CRITICAL_PCT", 5)
        fig = build_vintage_chart(self._vintage_df())
        npa_trace = next(t for t in fig.data if t.name == "NPA %")
        assert list(npa_trace.marker.color) == ["#16a34a", "#991b1b", "#991b1b"]

    def test_annotation_text_reflects_configured_thresholds(self, monkeypatch):
        monkeypatch.setattr(pi, "VINTAGE_CHART_CRITICAL_PCT", 15)
        monkeypatch.setattr(pi, "VINTAGE_CHART_WATCH_PCT", 8)
        fig = build_vintage_chart(self._vintage_df())
        annotations = [a.text for a in fig.layout.annotations]
        assert any("15%" in a for a in annotations)
        assert any("8%" in a for a in annotations)

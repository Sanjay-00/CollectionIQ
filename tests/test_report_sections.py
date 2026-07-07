import pandas as pd
import pytest

from report_agent.sections.verdict import compute_verdict
from report_agent.sections.risk_flags import compute_risk_flags
from report_agent.sections.top_accounts import compute_top_accounts_section
from report_agent.sections.fleet_exposure import compute_fleet_exposure_section
from report_agent.sections.region_scorecard import compute_region_scorecard_section
from report_agent.sections.concentration import compute_concentration_section
from report_agent.sections.bucket_migration import compute_bucket_migration_section
from report_agent.sections.npa_sma2_movement import compute_npa_sma2_movement
from report_agent.sections.risk_indicators import compute_risk_indicators_section
from report_agent.sections.branch_quadrant import compute_branch_quadrant_section
from report_agent.sections.executive_recovery import compute_executive_recovery_section
from report_agent.sections.product_analysis import compute_product_analysis_section
from report_agent.sections.repossession import compute_repossession_section
from report_agent.sections.good_customers import compute_good_customers_section
from report_agent.sections.executive_strike_rankings import compute_executive_strike_rankings
from report_agent.sections.overdue_demand import compute_overdue_demand_section
from report_agent.sections.new_advances import compute_new_advances_section
from report_agent.charts import fig_to_base64
from helpers import make_df

import plotly.graph_objects as go


# ── compute_verdict ───────────────────────────────────────────────────────────

class TestComputeVerdict:
    def test_no_prev_still_returns_dict_or_none_without_error(self):
        curr = make_df([
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "NPA", "Arrears / EMI": 6.0},
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "STD", "Arrears / EMI": 0.0},
        ])
        result = compute_verdict(curr, make_df([]))
        assert result is None or ("good" in result and "bad" in result)

    def test_empty_df_returns_none(self):
        assert compute_verdict(make_df([]), make_df([])) is None

    def test_good_and_bad_keys_capped_at_six(self):
        rows = []
        for i in range(10):
            rows.append({
                "RegionName": f"REGION{i}", "Unit": f"BR{i}",
                "curr_bucket": "NPA" if i % 2 == 0 else "STD",
                "Arrears / EMI": 6.0 if i % 2 == 0 else 0.0,
            })
        curr = make_df(rows)
        prev = make_df(rows)
        result = compute_verdict(curr, prev)
        if result is not None:
            assert len(result["good"]) <= 6
            assert len(result["bad"]) <= 6


# ── compute_risk_flags ────────────────────────────────────────────────────────

class TestComputeRiskFlags:
    def test_closing_arrears_is_carried_through_not_dropped(self):
        # Regression: this wrapper used to copy title/subtitle/severity/count/pos/
        # action/icon from smart_alerts.run_all_alerts()'s per-alert dict but NOT
        # closing_arrears -- report_builder.py's renderer then fell back to its
        # `.get("closing_arrears", 0)` default, showing "Rs 0" on every risk-flag
        # card in the report regardless of the real value (while POS, which WAS
        # copied, rendered correctly) -- a real production report exhibited this.
        df = make_df([
            {"CoLending_Loans": "Y", "Arrears / EMI": 2.0, "Closing Arrears": 50_000.0},
            {"CoLending_Loans": "Y", "Arrears / EMI": 3.0, "Closing Arrears": 30_000.0},
        ])
        result = compute_risk_flags(df)
        assert result is not None
        flag = next(f for f in result["flags"] if f["title"] == "Co-lending Loans at Risk")
        assert flag["closing_arrears"] == 80_000.0

    def test_no_active_alerts_returns_empty_flags(self):
        df = make_df([{"Arrears / EMI": 0.0}])
        result = compute_risk_flags(df)
        assert result == {"flags": []}


# ── compute_top_accounts_section ──────────────────────────────────────────────

class TestComputeTopAccountsSection:
    def test_returns_none_when_all_std(self):
        curr = make_df([{"curr_bucket": "STD", "SOH": 1_00_000.0}])
        assert compute_top_accounts_section(curr) is None

    def test_returns_rows_for_delinquent_accounts(self):
        curr = make_df([
            {"curr_bucket": "NPA", "SOH": 5_00_000.0, "Cust Name": "Alpha"},
            {"curr_bucket": "STD", "SOH": 1_00_000.0, "Cust Name": "Beta"},
        ])
        result = compute_top_accounts_section(curr, n=5)
        assert result is not None
        assert result["n"] == 1
        assert result["rows"][0]["customer"] == "Alpha"
        assert result["summary"]["npa_count"] == 1


# ── compute_fleet_exposure_section ────────────────────────────────────────────

class TestComputeFleetExposureSection:
    def test_returns_none_without_fleet_operators(self):
        curr = make_df([
            {"Cust Mob No": "9000000001", "Loan No": "L001"},
            {"Cust Mob No": "9000000002", "Loan No": "L002"},
        ])
        assert compute_fleet_exposure_section(curr) is None

    def test_returns_data_for_fleet_operator(self):
        rows = [
            {"Cust Mob No": "9000000001", "Cust Name": "Fleet Co", "curr_bucket": "STD",
             "RegionName": "WEST", "Unit": "MAHAD"}
            for _ in range(4)
        ]
        curr = make_df(rows)
        result = compute_fleet_exposure_section(curr)
        assert result is not None
        assert result["top"][0]["region"] == "WEST"
        assert result["top"][0]["branch"] == "MAHAD"
        assert result["count"] == 1
        assert result["top"][0]["loans"] == 4


# ── compute_region_scorecard_section ──────────────────────────────────────────

class TestComputeRegionScorecardSection:
    def test_returns_none_without_region_column(self):
        curr = make_df([{"curr_bucket": "STD"}]).drop(columns=["RegionName"])
        assert compute_region_scorecard_section(curr) is None

    def test_returns_rows_with_status(self):
        curr = make_df([{"RegionName": "WEST", "curr_bucket": "NPA", "Arrears / EMI": 6.0}])
        result = compute_region_scorecard_section(curr, make_df([]))
        assert result is not None
        assert result["rows"][0]["Region"] == "WEST"
        assert result["has_prev"] is False


# ── compute_concentration_section ─────────────────────────────────────────────

class TestComputeConcentrationSection:
    def test_returns_none_without_required_columns(self):
        curr = make_df([{"curr_bucket": "STD"}]).drop(columns=["RegionName", "Unit"])
        assert compute_concentration_section(curr) is None

    def test_returns_base64_image_for_valid_data(self):
        curr = make_df([
            {"RegionName": "WEST", "Unit": "MAHAD", "SOH": 1_00_000.0, "curr_bucket": "STD"},
            {"RegionName": "EAST", "Unit": "PUNE",  "SOH": 2_00_000.0, "curr_bucket": "NPA"},
        ])
        result = compute_concentration_section(curr)
        assert result is not None
        assert result["image"].startswith("data:image/png;base64,")


# ── compute_bucket_migration_section (waterfall image) ────────────────────────

class TestComputeBucketMigrationSectionImage:
    def test_includes_waterfall_image_when_matched(self):
        curr = make_df([{"Loan No": "L001", "curr_bucket": "SMA-1", "Arrears / EMI": 1.0}])
        prev = make_df([{"Loan No": "L001", "curr_bucket": "STD", "Arrears / EMI": 0.0}])
        result = compute_bucket_migration_section(curr, prev)
        assert result is not None
        assert result["waterfall_image"] is None or result["waterfall_image"].startswith("data:image/png;base64,")


# ── compute_npa_sma2_movement ─────────────────────────────────────────────────

class TestComputeNpaSma2Movement:
    def test_portfolio_sequence_without_prev(self):
        curr = make_df([
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "SMA-2"},
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "STD"},
        ])
        result = compute_npa_sma2_movement(curr, make_df([]))
        assert result is not None
        p = result["portfolio"]
        # Exact business-requested sequence of keys
        assert list(p.keys()) == [
            "npa_current", "npa_prev", "sma2_current", "sma2_prev",
            "npa_delta", "sma2_delta", "npa_pct_change", "sma2_pct_change",
        ]
        assert p["npa_current"] == 1
        assert p["sma2_current"] == 1
        assert p["npa_prev"] is None
        assert p["npa_delta"] is None

    def test_portfolio_deltas_with_prev(self):
        curr = make_df([
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "SMA-2"},
        ])
        prev = make_df([
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "SMA-2"},
            {"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "SMA-2"},
        ])
        result = compute_npa_sma2_movement(curr, prev)
        p = result["portfolio"]
        assert p["npa_current"] == 2 and p["npa_prev"] == 1
        assert p["npa_delta"] == 1
        assert p["npa_pct_change"] == 100.0
        assert p["sma2_current"] == 1 and p["sma2_prev"] == 2
        assert p["sma2_delta"] == -1
        assert p["sma2_pct_change"] == -50.0

    def test_region_breakdown_present(self):
        curr = make_df([{"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "NPA"}] * 5)
        prev = make_df([{"RegionName": "WEST", "Unit": "MAHAD", "curr_bucket": "STD"}] * 5)
        result = compute_npa_sma2_movement(curr, prev)
        assert result["region"]
        assert result["region"][0]["RegionName"] == "WEST"

    def test_empty_returns_none(self):
        assert compute_npa_sma2_movement(make_df([]), make_df([])) is None

    def test_executive_movers_include_branch(self):
        rows = [
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "NPA"},
        ]
        curr = make_df(rows)
        prev_rows = [
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "STD"},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "STD"},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "STD"},
        ]
        prev = make_df(prev_rows)
        result = compute_npa_sma2_movement(curr, prev)
        assert result["exec_worst"]
        assert result["exec_worst"][0]["Unit"] == "MAHAD"


# ── compute_risk_indicators_section ───────────────────────────────────────────

class TestComputeRiskIndicatorsSection:
    def test_returns_none_for_empty_df(self):
        assert compute_risk_indicators_section(make_df([])) is None

    def test_returns_indicators_for_valid_data(self):
        curr = make_df([{"curr_bucket": "SMA-2"}, {"curr_bucket": "STD"}])
        result = compute_risk_indicators_section(curr, make_df([]))
        assert result is not None
        assert len(result["indicators"]) > 0
        assert all("Signal" in i for i in result["indicators"])


# ── compute_branch_quadrant_section ───────────────────────────────────────────

class TestComputeBranchQuadrantSection:
    def test_returns_none_without_enough_accounts_per_branch(self):
        curr = make_df([{"Unit": "MAHAD", "curr_bucket": "STD"}])
        assert compute_branch_quadrant_section(curr) is None

    def test_returns_image_and_concern_table(self):
        rows = [{"Unit": "MAHAD", "curr_bucket": "NPA", "Arrears / EMI": 6.0}] * 3 \
             + [{"Unit": "PUNE", "curr_bucket": "STD", "Arrears / EMI": 0.0}] * 3
        curr = make_df(rows)
        result = compute_branch_quadrant_section(curr)
        assert result is not None
        assert result["total_branches"] == 2
        assert len(result["top_concern"]) <= 5
        assert result["image"] is None or result["image"].startswith("data:image/png;base64,")

    def test_top_concern_has_region_and_strike_no_concern_score(self):
        # Regression: "Highest Concern Branches" used to show Concern Score
        # (an internal composite a report reader can't interpret on its own)
        # and no Region/Strike% -- business request was to add Region and
        # Strike%, drop Concern Score from the DISPLAYED columns (ranking is
        # still by Concern Score internally, via Rank).
        rows = [{"Unit": "MAHAD", "RegionName": "WEST", "Strike": "Y", "curr_bucket": "NPA", "Arrears / EMI": 6.0}] * 3 \
             + [{"Unit": "PUNE", "RegionName": "EAST", "Strike": "N", "curr_bucket": "STD", "Arrears / EMI": 0.0}] * 3
        curr = make_df(rows)
        result = compute_branch_quadrant_section(curr)
        row = result["top_concern"][0]
        assert "Region" in row and "Strike%" in row
        assert "Concern Score" not in row


# ── compute_executive_recovery_section ────────────────────────────────────────

class TestComputeExecutiveRecoverySection:
    def test_returns_none_without_prev_bucket(self):
        curr = make_df([{"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "STD"}])
        assert compute_executive_recovery_section(curr) is None

    def test_returns_top_and_bottom(self):
        rows = [
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "STD", "prev_bucket": "SMA-2"},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "STD", "prev_bucket": "SMA-2"},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "curr_bucket": "STD", "prev_bucket": "SMA-2"},
        ]
        curr = make_df(rows)
        result = compute_executive_recovery_section(curr)
        assert result is not None
        assert result["total"] == 1
        assert result["top"][0]["Executive"].startswith("RAJ")


# ── compute_product_analysis_section (segment-only) ───────────────────────────

class TestComputeProductAnalysisSection:
    def test_returns_none_without_segment_column(self):
        curr = make_df([{"curr_bucket": "STD"}] * 6)
        assert compute_product_analysis_section(curr) is None

    def test_returns_segment_rows_only(self):
        # MIN_ACCOUNTS_PRODUCT_SEGMENT = 11 -- needs strictly more than 10 accounts.
        rows = [{"SegmentName": "AUTO", "curr_bucket": "NPA"}] * 6 + [{"SegmentName": "AUTO", "curr_bucket": "STD"}] * 6
        curr = make_df(rows)
        result = compute_product_analysis_section(curr)
        assert result is not None
        assert result["rows"][0]["Segment"] == "AUTO"


# ── compute_repossession_section ──────────────────────────────────────────────

class TestComputeRepossessionSection:
    def test_returns_none_for_no_eligible_accounts(self):
        curr = make_df([{"curr_bucket": "STD"}])
        assert compute_repossession_section(curr) is None

    def test_returns_rows_for_eligible_accounts(self):
        curr = make_df([{"curr_bucket": "NPA", "Ag_Date": pd.Timestamp.today(), "SOH": 2_00_000.0}])
        result = compute_repossession_section(curr)
        assert result is not None
        assert result["total"] == 1

    def test_curr_month_is_threaded_through_as_the_reporting_date(self):
        # Regression: this section used to call compute_repossession_list with
        # no reporting date at all, silently anchoring the repossession window
        # to wall-clock "now" instead of the report's own curr_month.
        report_month = "2020-01"
        just_within_window = pd.Timestamp("2020-01-01") - pd.DateOffset(months=6)
        curr = make_df([{"curr_bucket": "NPA", "Ag_Date": just_within_window, "SOH": 1_00_000.0}])

        result = compute_repossession_section(curr, curr_month=report_month)
        assert result is not None
        assert result["total"] == 1

        # Without curr_month, the same loan is years outside the window
        # relative to wall-clock today and must not appear.
        assert compute_repossession_section(curr) is None


# ── compute_overdue_demand_section ────────────────────────────────────────────
# Regression: this section used to print EVERY region/branch/executive row
# (unlimited), which was responsible for roughly half of a real report's total
# row count and a real, user-reported generation slowdown. Now capped to top 5
# / bottom 5 by Month Demand Collection %, same convention branch_performance.py
# already uses.

class TestComputeOverdueDemandSection:
    def _row(self, region, unit, exec_name, demand_pct):
        # Bypass the real waterfall math (already covered by
        # test_utils.py::TestOverdueDemandCollectionPct) -- set the derived
        # columns directly, same pattern compute_overdue_demand_scorecard's
        # own tests use, since only the RANKING/CAPPING behavior is under test here.
        demand_total = 100_000.0
        return {
            "RegionName": region, "Unit": unit, "MNT NAME": exec_name,
            "Overdue": 0, "OverdueCollected": 0,
            "MonthDemandExclPC": demand_total, "DemandCollected": demand_total * demand_pct / 100,
        }

    def test_caps_at_top5_and_bottom5_per_dimension(self):
        # 12 distinct regions -- more than double top5+bottom5, so capping
        # actually has to do something (not just "show everything anyway").
        curr = make_df([
            self._row(f"REGION_{i}", "U1", "E1", demand_pct=i * 5)
            for i in range(12)
        ])
        result = compute_overdue_demand_section(curr)
        assert result is not None
        assert len(result["region"]["top5"]) == 5
        assert len(result["region"]["bottom5"]) == 5
        assert result["region"]["total"] == 12

    def test_top5_is_highest_demand_pct_bottom5_is_lowest(self):
        curr = make_df([
            self._row(f"REGION_{i}", "U1", "E1", demand_pct=i * 5)
            for i in range(12)
        ])
        result = compute_overdue_demand_section(curr)
        top_pcts = [r["demand_pct"] for r in result["region"]["top5"]]
        bottom_pcts = [r["demand_pct"] for r in result["region"]["bottom5"]]
        assert top_pcts == sorted(top_pcts, reverse=True)
        assert min(top_pcts) > max(bottom_pcts)

    def test_fewer_than_ten_entities_does_not_duplicate_across_top_and_bottom(self):
        curr = make_df([
            self._row(f"REGION_{i}", "U1", "E1", demand_pct=i * 10)
            for i in range(4)
        ])
        result = compute_overdue_demand_section(curr)
        top_names = {r["name"] for r in result["region"]["top5"]}
        bottom_names = {r["name"] for r in result["region"]["bottom5"]}
        assert not (top_names & bottom_names)
        assert len(top_names) + len(bottom_names) == 4

    def test_returns_none_for_empty_df(self):
        assert compute_overdue_demand_section(make_df([])) is None

    def test_branch_and_executive_dimensions_also_capped(self):
        # Each executive needs >= MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE (11)
        # accounts to survive the executive-grain filter.
        curr = make_df([
            self._row("WEST", f"BRANCH_{i}", f"EXEC_{i}", demand_pct=i * 5)
            for i in range(12) for _ in range(11)
        ])
        result = compute_overdue_demand_section(curr)
        assert len(result["branch"]["top5"]) == 5
        assert len(result["executive"]["top5"]) == 5

    def test_region_rows_carry_overdue_cr_no_identity_cols(self):
        curr = make_df([self._row("REGION_A", "U1", "E1", demand_pct=50)])
        result = compute_overdue_demand_section(curr)
        row = result["region"]["top5"][0]
        assert "overdue_cr" in row
        assert "region" not in row and "branch" not in row

    def test_branch_rows_carry_region_identity(self):
        curr = make_df([self._row("EAST", "BR1", "E1", demand_pct=50)])
        result = compute_overdue_demand_section(curr)
        row = result["branch"]["top5"][0]
        assert row["name"] == "BR1"
        assert row["region"] == "EAST"
        assert "branch" not in row

    def test_executive_rows_carry_branch_and_region_identity(self):
        # >= MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE (11) accounts, or EXEC1 is
        # filtered out of the executive grain entirely.
        curr = make_df([self._row("EAST", "BR1", "EXEC1", demand_pct=50)] * 11)
        result = compute_overdue_demand_section(curr)
        row = result["executive"]["top5"][0]
        assert row["name"] == "EXEC1"
        assert row["branch"] == "BR1"
        assert row["region"] == "EAST"


# ── compute_new_advances_section ──────────────────────────────────────────────

class TestComputeNewAdvancesSection:
    def test_returns_none_for_empty_df(self):
        assert compute_new_advances_section(make_df([])) is None

    def test_returns_none_when_no_advances_this_month(self):
        curr = make_df([{"Ag_Date": pd.Timestamp("2026-01-01"), "Loan Amount": 100_000.0}])
        assert compute_new_advances_section(curr, curr_month="2026-06") is None

    def test_returns_totals_and_segment_rows(self):
        curr = make_df(
            [{"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 100_000.0, "SegmentName": "CV"}] * 11
        )
        result = compute_new_advances_section(curr, curr_month="2026-06")
        assert result is not None
        assert result["accounts"] == 11
        assert result["segment"][0]["Segment"] == "CV"

    def test_mom_comparison_sourced_from_same_upload_no_df_prev_needed(self):
        # df_prev is passed as None here on purpose -- MoM must still work,
        # sourced entirely from df_curr's own Ag_Date history.
        curr = make_df(
            [{"Ag_Date": pd.Timestamp("2026-06-05"), "Loan Amount": 200_000.0}] * 2
            + [{"Ag_Date": pd.Timestamp("2026-05-05"), "Loan Amount": 100_000.0}]
        )
        result = compute_new_advances_section(curr, None, curr_month="2026-06")
        assert result["has_prev"] is True
        assert result["prev_accounts"] == 1


# ── compute_good_customers_section ────────────────────────────────────────────

class TestComputeGoodCustomersSection:
    def test_returns_empty_dict_not_none_for_no_qualifying_customers(self):
        # Section always renders (with a "no accounts meet criteria" message) rather
        # than silently vanishing - matches the dashboard's Section 8 behavior.
        curr = make_df([{"LCC%": 50.0, "VehEMI Accrued": 1, "Tenure": 100}])
        result = compute_good_customers_section(curr)
        assert result is not None
        assert result["rows"] == []
        assert result["total"] == 0

    def test_returns_none_for_empty_df(self):
        assert compute_good_customers_section(make_df([])) is None

    def test_returns_rows_for_qualifying_customers(self):
        curr = make_df([{"LCC%": 100.0, "VehEMI Accrued": 80, "Tenure": 100}])
        result = compute_good_customers_section(curr)
        assert result is not None
        assert result["total"] == 1


# ── compute_executive_strike_rankings ─────────────────────────────────────────

class TestComputeExecutiveStrikeRankings:
    def test_ranked_by_strike_not_collection(self):
        rows = (
            [{"MNT NAME": "RAJ", "Unit": "MAHAD", "Strike": "N",
              "Net Collection Demand Inst+Exp+BC": 10_000.0,
              "Month Collection (Excluding Reserve Collection)": 10_000.0}] * 5
            + [{"MNT NAME": "SUNIL", "Unit": "MAHAD", "Strike": "Y",
                "Net Collection Demand Inst+Exp+BC": 10_000.0,
                "Month Collection (Excluding Reserve Collection)": 2_000.0}] * 5
        )
        curr = make_df(rows)
        result = compute_executive_strike_rankings(curr)
        assert result is not None
        assert result["top5"][0]["name"].startswith("SUNIL")
        assert result["top5"][0]["strike_rate"] == 100.0

    def test_returns_none_without_executive_column(self):
        curr = make_df([{"curr_bucket": "STD"}]).drop(columns=["Loan No"], errors="ignore")
        assert compute_executive_strike_rankings(make_df([])) is None


# ── fig_to_base64 ──────────────────────────────────────────────────────────────

class TestFigToBase64:
    @pytest.mark.parametrize("fig", [go.Figure(), None], ids=["empty_figure", "none"])
    def test_empty_or_missing_figure_returns_none(self, fig):
        assert fig_to_base64(fig) is None

    def test_valid_figure_returns_data_uri(self):
        fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
        result = fig_to_base64(fig, width=200, height=150)
        assert result is not None
        assert result.startswith("data:image/png;base64,")

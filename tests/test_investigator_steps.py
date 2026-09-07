import pandas as pd

from helpers import make_df
from investigator.steps import (
    bottom_n_per_group,
    concept_filter,
    concept_summary,
    customer_loan_book,
    concept_breakdown,
    customer_loan_summary,
    demand_summary,
    dimension_breakdown,
    distinct_executive_matches,
    entity_summary,
    executive_roster,
    fleet_defaulters,
    high_arrears_at_risk,
    fleet_exposure,
    good_customers,
    intersect_entities,
    new_advances_by_dimension,
    new_advances_summary,
    new_advances_trend,
    non_paying_customers,
    product_analysis,
    repossession_list,
    top_accounts,
    priority_accounts,
    priority_by_executive,
    priority_menu,
    resolve_executive_identity,
    top_closing_arrears,
    roll_rate_by_dimension,
    roll_rate_summary,
    summarize_worst_loans,
    underperformer_quartile,
    vintage_summary,
    worst_loans_by_metric,
)


def _branch_df(branch, region, n=3, arrears=0.0):
    return [
        {"RegionName": region, "Unit": branch, "MNT NAME": f"EXEC{i}",
         "Arrears / EMI": arrears, "curr_bucket": "STD" if arrears == 0 else "NPA"}
        for i in range(n)
    ]


class TestDimensionBreakdown:
    """Covers "how are my regions/branches/zones/BUs performing" -- a
    PLURAL/comparative question with no single entity named, distinct from
    entity_summary (one branch/executive) or underperformer_quartile (worst
    quartile only). A real live-Gemini check caught this gap: "how are my
    regions performing" was routed to entity_summary with no entity_value at
    all, which crashed the dispatcher -- there was no step type for
    "show me every entity at this level" until this one."""

    def _two_region_df(self, prev=False):
        rows = []
        for region, coll, npa in [("WEST", 90.0, "STD"), ("EAST", 40.0, "NPA")]:
            demand = 10_000.0
            collected = demand * (coll - 20 if prev else coll) / 100
            for _ in range(3):
                rows.append({
                    "RegionName": region, "Unit": f"{region}_BR",
                    "Net Collection Demand Inst+Exp+BC": demand,
                    "Month Collection (Excluding Reserve Collection)": collected,
                    "curr_bucket": npa,
                })
        return make_df(rows)

    def test_one_row_per_region_with_key_metrics(self):
        result = dimension_breakdown(self._two_region_df(), "region")
        assert set(result["RegionName"]) == {"WEST", "EAST"}
        assert "Collection%" in result.columns
        assert "NPA%" in result.columns

    def test_sorted_worst_collection_pct_first(self):
        result = dimension_breakdown(self._two_region_df(), "region")
        assert result.iloc[0]["RegionName"] == "EAST"

    def test_computes_pp_delta_not_relative_pct_change(self):
        curr = self._two_region_df(prev=False)
        prev = self._two_region_df(prev=True)
        result = dimension_breakdown(curr, "region", df_prev=prev)
        west_row = result[result["RegionName"] == "WEST"].iloc[0]
        # Curr Collection% = 90.0, prev Collection% = 70.0 -> raw pp delta = 20.0.
        assert west_row["Δ Collection%"] == 20.0

    def test_works_for_branch_dimension_too(self):
        result = dimension_breakdown(self._two_region_df(), "branch")
        assert set(result["Unit"]) == {"WEST_BR", "EAST_BR"}

    def test_unknown_dimension_returns_empty(self):
        result = dimension_breakdown(self._two_region_df(), "not_a_real_dimension")
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = dimension_breakdown(make_df([]), "region")
        assert result.empty

    def test_sorts_by_the_given_metric_worst_first_not_always_collection_pct(self):
        # "why is NPA more" scoped to branch-level must sort by NPA%
        # (worst/highest first), not always Collection% -- a real gap: this
        # function used to hardcode Collection% ascending regardless of
        # which metric the question was actually about.
        rows = []
        for branch, npa in [("W1", 5.0), ("W2", 40.0)]:
            for _ in range(3):
                rows.append({
                    "RegionName": "WEST", "Unit": branch,
                    "Net Collection Demand Inst+Exp+BC": 10_000.0,
                    "Month Collection (Excluding Reserve Collection)": 9_000.0,
                    "curr_bucket": "NPA" if npa >= 40 else "STD",
                })
        result = dimension_breakdown(make_df(rows), "branch", metric="NPA%", higher_is_better=False)
        assert result.iloc[0]["Unit"] == "W2"

    def test_metric_with_a_stray_space_is_normalized_not_silently_ignored(self):
        # dimension_breakdown's own columns are "Collection%"/"NPA%" (no
        # space) -- but callers (the LLM) sometimes emit the executive-grain
        # spelling ("Collection %", with a space). This must be normalized,
        # not silently fall back to the default sort as if no metric were
        # given at all (a real live-Gemini call produced exactly this).
        rows = []
        for branch, npa in [("W1", 5.0), ("W2", 40.0)]:
            for _ in range(3):
                rows.append({
                    "RegionName": "WEST", "Unit": branch,
                    "Net Collection Demand Inst+Exp+BC": 10_000.0,
                    "Month Collection (Excluding Reserve Collection)": 9_000.0,
                    "curr_bucket": "NPA" if npa >= 40 else "STD",
                })
        result = dimension_breakdown(make_df(rows), "branch", metric="NPA %", higher_is_better=False)
        assert result.iloc[0]["Unit"] == "W2"

    def test_scope_narrows_to_branches_within_one_region(self):
        # Drilling from a region entity_summary into "its branches" needs
        # dimension_breakdown scoped to that region -- same scope_col/
        # scope_value convention underperformer_quartile already uses.
        rows = []
        for region, branch, coll in [("WEST", "W1", 90.0), ("WEST", "W2", 50.0), ("EAST", "E1", 40.0)]:
            demand = 10_000.0
            for _ in range(3):
                rows.append({
                    "RegionName": region, "Unit": branch,
                    "Net Collection Demand Inst+Exp+BC": demand,
                    "Month Collection (Excluding Reserve Collection)": demand * coll / 100,
                })
        result = dimension_breakdown(make_df(rows), "branch", scope_col="region", scope_value="WEST")
        assert set(result["Unit"]) == {"W1", "W2"}


class TestRollRateSummary:
    """Root-cause MECHANISM, not just location: "why is NPA up" decomposes
    into roll-forward (existing accounts sliding into worse buckets --
    a collections-execution problem) vs roll-backward (fewer accounts
    recovering -- also execution) vs fresh NPA formation -- each implying a
    DIFFERENT fix than "which branch/executive is worst," which is all the
    org-hierarchy drills (dimension_breakdown/underperformer_quartile) can
    answer. Reuses analysis/roll_rate.py::compute_roll_rate_matrix/
    compute_roll_rate_kpis directly -- the SAME bucket-migration engine
    report_agent's bucket_migration section and the AI Query
    roll_rate_matrix VIEW already use, so this never becomes a third,
    independently-drifting implementation of "roll forward rate.\""""

    def _matched_df(self):
        # 2 loans matched Loan No across curr/prev: one rolls forward
        # (STD -> NPA), one rolls backward / cures (NPA -> SMA-2).
        prev = make_df([
            {"Loan No": "L1", "Unit": "MAHAD", "curr_bucket": "STD"},
            {"Loan No": "L2", "Unit": "MAHAD", "curr_bucket": "NPA"},
        ])
        curr = make_df([
            {"Loan No": "L1", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"Loan No": "L2", "Unit": "MAHAD", "curr_bucket": "SMA-2"},
        ])
        return curr, prev

    def test_returns_the_three_headline_migration_rates(self):
        curr, prev = self._matched_df()
        result = roll_rate_summary(curr, prev)
        metrics = dict(zip(result["Metric"], result["Value"]))
        assert metrics["Roll Forward Rate %"] == 50.0
        assert metrics["Roll Backward Rate %"] == 50.0

    def test_no_previous_period_returns_empty(self):
        curr, _ = self._matched_df()
        result = roll_rate_summary(curr, pd.DataFrame())
        assert result.empty

    def test_scope_narrows_to_one_branch(self):
        prev = make_df([
            {"Loan No": "L1", "Unit": "MAHAD", "curr_bucket": "STD"},
            {"Loan No": "L2", "Unit": "PUNE", "curr_bucket": "STD"},
        ])
        curr = make_df([
            {"Loan No": "L1", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"Loan No": "L2", "Unit": "PUNE", "curr_bucket": "STD"},
        ])
        result = roll_rate_summary(curr, prev, scope_col="branch", scope_value="MAHAD")
        metrics = dict(zip(result["Metric"], result["Value"]))
        # Scoped to MAHAD alone: 1 matched account, rolled forward -- 100%,
        # not diluted by PUNE's unrelated stable account.
        assert metrics["Roll Forward Rate %"] == 100.0
        assert metrics["Matched Accounts"] == 1

    def test_empty_curr_returns_empty(self):
        _, prev = self._matched_df()
        assert roll_rate_summary(pd.DataFrame(), prev).empty


class TestRollRateByDimension:
    """"Which branch had the most roll forward" -- a real gap
    roll_rate_summary alone can't answer, since it only computes ONE
    scope's rates at a time, never a ranked breakdown across many (a live
    user hit exactly this: asked the question, got "needs a bit more
    detail" because no step could produce it). Reuses the SAME
    BUCKET_SCORE bucket-severity definitions analysis/roll_rate.py's
    compute_roll_rate_matrix uses -- imported directly, not re-derived --
    so "what counts as forward vs backward" can never drift into a second
    definition here, even though the GROUPED aggregation shape (one row
    per branch, computed via one vectorized merge+groupby, never a Python
    loop calling compute_roll_rate_matrix once per branch) is new."""

    def _two_branch_df(self):
        prev = make_df([
            {"Loan No": "L1", "Unit": "MAHAD", "curr_bucket": "STD"},
            {"Loan No": "L2", "Unit": "MAHAD", "curr_bucket": "STD"},
            {"Loan No": "L3", "Unit": "PUNE", "curr_bucket": "STD"},
            {"Loan No": "L4", "Unit": "PUNE", "curr_bucket": "STD"},
        ])
        curr = make_df([
            {"Loan No": "L1", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"Loan No": "L2", "Unit": "MAHAD", "curr_bucket": "NPA"},
            {"Loan No": "L3", "Unit": "PUNE", "curr_bucket": "NPA"},
            {"Loan No": "L4", "Unit": "PUNE", "curr_bucket": "STD"},
        ])
        return curr, prev

    def test_one_row_per_branch_with_roll_forward_rate(self):
        curr, prev = self._two_branch_df()
        result = roll_rate_by_dimension(curr, prev, "branch")
        mahad = result[result["Unit"] == "MAHAD"].iloc[0]
        pune = result[result["Unit"] == "PUNE"].iloc[0]
        assert mahad["Roll Forward Rate %"] == 100.0
        assert pune["Roll Forward Rate %"] == 50.0

    def test_sorted_worst_roll_forward_first(self):
        curr, prev = self._two_branch_df()
        result = roll_rate_by_dimension(curr, prev, "branch")
        assert result.iloc[0]["Unit"] == "MAHAD"

    def test_executive_dimension_groups_on_name_and_branch_together(self):
        # Same collision class investigator/guardrails.py's ambiguous-name
        # check exists to prevent -- two different people sharing a first
        # name in different branches must never be merged into one row.
        prev = make_df([
            {"Loan No": "L1", "Unit": "PUNE", "MNT NAME": "RAHUL SHARMA", "curr_bucket": "STD"},
            {"Loan No": "L2", "Unit": "NAGPUR", "MNT NAME": "RAHUL VERMA", "curr_bucket": "STD"},
        ])
        curr = make_df([
            {"Loan No": "L1", "Unit": "PUNE", "MNT NAME": "RAHUL SHARMA", "curr_bucket": "NPA"},
            {"Loan No": "L2", "Unit": "NAGPUR", "MNT NAME": "RAHUL VERMA", "curr_bucket": "STD"},
        ])
        result = roll_rate_by_dimension(curr, prev, "executive")
        assert len(result) == 2
        assert set(zip(result["MNT NAME"], result["Unit"])) == {
            ("RAHUL SHARMA", "PUNE"), ("RAHUL VERMA", "NAGPUR"),
        }

    def test_no_previous_period_returns_empty(self):
        curr, _ = self._two_branch_df()
        assert roll_rate_by_dimension(curr, pd.DataFrame(), "branch").empty

    def test_empty_curr_returns_empty(self):
        _, prev = self._two_branch_df()
        assert roll_rate_by_dimension(pd.DataFrame(), prev, "branch").empty


class TestNewAdvancesSummary:
    """Business VOLUME (originations), not collection performance -- a
    different axis entirely from every other step in this module, matching
    the dashboard's own Business tab Section 1. Reuses
    analysis/portfolio_intelligence.py::compute_new_advances directly, so
    these numbers can never drift from what the Business tab itself shows."""

    def _adv_row(self, month: str, unit: str = "MAHAD", amount: float = 200_000.0):
        return {"Unit": unit, "RegionName": "WEST", "Ag_Date": pd.Timestamp(f"{month}-15"), "Loan Amount": amount}

    def test_returns_metric_current_previous_delta_shape(self):
        rows = [self._adv_row("2025-06"), self._adv_row("2025-06"), self._adv_row("2025-05")]
        result = new_advances_summary(make_df(rows), as_of="2025-06-30")
        metrics = result["Metric"].tolist()
        assert "New Advances (Accounts)" in metrics
        accounts_row = result[result["Metric"] == "New Advances (Accounts)"].iloc[0]
        assert accounts_row["Current"] == 2
        assert accounts_row["Previous"] == 1
        assert accounts_row["Delta"] == 1

    def test_no_previous_month_data_leaves_previous_and_delta_none(self):
        rows = [self._adv_row("2025-06")]
        result = new_advances_summary(make_df(rows), as_of="2025-06-30")
        accounts_row = result[result["Metric"] == "New Advances (Accounts)"].iloc[0]
        assert accounts_row["Previous"] is None
        assert accounts_row["Delta"] is None

    def test_scope_narrows_to_one_branch(self):
        rows = [self._adv_row("2025-06", unit="MAHAD"), self._adv_row("2025-06", unit="PUNE")]
        result = new_advances_summary(make_df(rows), scope_col="branch", scope_value="MAHAD", as_of="2025-06-30")
        accounts_row = result[result["Metric"] == "New Advances (Accounts)"].iloc[0]
        assert accounts_row["Current"] == 1

    def test_empty_df_returns_empty(self):
        assert new_advances_summary(pd.DataFrame()).empty

    def test_no_advances_at_all_returns_empty(self):
        rows = [self._adv_row("2020-01")]  # nowhere near as_of's month
        result = new_advances_summary(make_df(rows), as_of="2025-06-30")
        assert result.empty


class TestNewAdvancesByDimension:
    """WHICH region/branch/executive originated the most new business --
    the ranked-breakdown counterpart to new_advances_summary, same
    relationship dimension_breakdown has to entity_summary. Reuses
    analysis/portfolio_intelligence.py::compute_new_advances_by_dimension
    directly, the SAME table the Business tab's Section 3 renders."""

    def _adv_row(self, unit: str, exec_name: str, mob: str):
        return {
            "Unit": unit, "RegionName": "WEST", "MNT NAME": exec_name, "Cust Mob No": mob,
            "Ag_Date": pd.Timestamp("2025-06-15"), "Loan Amount": 200_000.0,
        }

    def test_branch_dimension_ranks_by_accounts_this_month(self):
        rows = (
            [self._adv_row("MAHAD", "RAHUL", f"90000000{i}") for i in range(3)]
            + [self._adv_row("PUNE", "AJAY", "9111111111")]
        )
        result = new_advances_by_dimension(make_df(rows), "branch", as_of="2025-06-30")
        assert result.iloc[0]["Branch"] == "MAHAD"
        assert result.iloc[0]["Accounts This Month"] == 3

    def test_executive_dimension(self):
        rows = [self._adv_row("MAHAD", "RAHUL", "9000000001")]
        result = new_advances_by_dimension(make_df(rows), "executive", as_of="2025-06-30")
        assert result.iloc[0]["Executive"] == "RAHUL"

    def test_invalid_dimension_returns_empty(self):
        rows = [self._adv_row("MAHAD", "RAHUL", "9000000001")]
        assert new_advances_by_dimension(make_df(rows), "customer", as_of="2025-06-30").empty

    def test_scope_narrows_the_input_first(self):
        rows = (
            [self._adv_row("MAHAD", "RAHUL", "9000000001")]
            + [self._adv_row("PUNE", "AJAY", "9111111111")]
        )
        result = new_advances_by_dimension(
            make_df(rows), "branch", scope_col="region", scope_value="WEST", as_of="2025-06-30",
        )
        assert set(result["Branch"]) == {"MAHAD", "PUNE"}

    def test_empty_df_returns_empty(self):
        assert new_advances_by_dimension(pd.DataFrame(), "branch").empty


class TestProductAnalysis:
    """Tier-1 gap-closing: exposes compute_product_analysis's "segment"/
    "fuel"/"source" keys, which vintage_summary already reuses (under the
    "vintage" key) but nothing in this module read before now -- zero new
    business logic, same shared function."""

    def test_segment_axis_ranks_worst_npa_first(self):
        rows = [
            {"SegmentName": "AUTO", "curr_bucket": "NPA"},
            {"SegmentName": "AUTO", "curr_bucket": "STD"},
            {"SegmentName": "CV", "curr_bucket": "STD"},
            {"SegmentName": "CV", "curr_bucket": "STD"},
        ]
        result = product_analysis(make_df(rows), "segment")
        assert result.iloc[0]["Segment"] == "AUTO"
        assert result.iloc[0]["NPA%"] == 50.0

    def test_fuel_axis(self):
        rows = [{"FUEL_TYPE": "DIESEL", "curr_bucket": "NPA"}, {"FUEL_TYPE": "DIESEL", "curr_bucket": "STD"}]
        result = product_analysis(make_df(rows), "fuel")
        assert result.iloc[0]["Fuel Type"] == "DIESEL"

    def test_source_axis_below_minimum_accounts_is_excluded(self):
        rows = [{"SRC Name": "DSA1", "curr_bucket": "NPA"}] * 3  # below MIN_ACCOUNTS_SOURCE_VINTAGE (10)
        result = product_analysis(make_df(rows), "source")
        assert result.empty

    def test_invalid_axis_returns_empty(self):
        rows = [{"SegmentName": "AUTO", "curr_bucket": "NPA"}]
        assert product_analysis(make_df(rows), "vintage").empty

    def test_scope_narrows_to_one_branch(self):
        rows = [
            {"Unit": "MAHAD", "SegmentName": "AUTO", "curr_bucket": "NPA"},
            {"Unit": "PUNE", "SegmentName": "AUTO", "curr_bucket": "STD"},
        ]
        result = product_analysis(make_df(rows), "segment", scope_col="branch", scope_value="MAHAD")
        assert result.iloc[0]["NPA%"] == 100.0

    def test_empty_df_returns_empty(self):
        assert product_analysis(pd.DataFrame(), "segment").empty


class TestTopAccounts:
    """Largest single exposures among DELINQUENT accounts by SOH -- distinct
    from top_closing_arrears (any status) and high_arrears_at_risk (a
    ratio). Reuses analysis/portfolio_intelligence.py::compute_top_accounts
    directly."""

    def test_ranks_delinquent_accounts_by_soh_descending(self):
        rows = [
            {"curr_bucket": "NPA", "SOH": 500_000.0},
            {"curr_bucket": "SMA-2", "SOH": 900_000.0},
            {"curr_bucket": "STD", "SOH": 2_000_000.0},  # healthy -- excluded regardless of SOH
        ]
        result = top_accounts(make_df(rows), n=5)
        assert len(result) == 2
        assert result.iloc[0]["SOH"] == 900_000.0

    def test_n_caps_the_result(self):
        rows = [{"curr_bucket": "NPA", "SOH": float(i)} for i in range(10)]
        result = top_accounts(make_df(rows), n=3)
        assert len(result) == 3

    def test_scope_narrows_to_one_branch(self):
        rows = [
            {"Unit": "MAHAD", "curr_bucket": "NPA", "SOH": 500_000.0},
            {"Unit": "PUNE", "curr_bucket": "NPA", "SOH": 900_000.0},
        ]
        result = top_accounts(make_df(rows), scope_col="branch", scope_value="MAHAD")
        assert len(result) == 1
        assert result.iloc[0]["SOH"] == 500_000.0

    def test_no_delinquent_accounts_returns_empty(self):
        rows = [{"curr_bucket": "STD", "SOH": 500_000.0}]
        assert top_accounts(make_df(rows)).empty

    def test_empty_df_returns_empty(self):
        assert top_accounts(pd.DataFrame()).empty


class TestNewAdvancesTrend:
    """"Show me the last 6 months of new business" -- a DIFFERENT axis from
    every collection-metric step (which only ever compares current-vs-
    previous UPLOADED file); this is derived entirely from ONE upload's own
    Ag_Date history, so it's not subject to the app's 2-uploaded-file
    ceiling. Reuses compute_new_advances_trend + roll_new_advances_trend
    directly -- the SAME functions the Business tab's own Section 2 uses."""

    def _adv_row(self, month: str, unit: str = "MAHAD"):
        return {"Unit": unit, "Ag_Date": pd.Timestamp(f"{month}-15"), "Loan Amount": 200_000.0}

    def test_one_row_per_month_with_accounts_and_funded(self):
        rows = [self._adv_row("2025-04"), self._adv_row("2025-04"), self._adv_row("2025-05")]
        result = new_advances_trend(make_df(rows), as_of="2025-06-30")
        apr = result[result["Month"] == "2025-04"].iloc[0]
        assert apr["Accounts"] == 2

    def test_months_window_limits_history(self):
        rows = [self._adv_row("2024-01"), self._adv_row("2025-06")]
        result = new_advances_trend(make_df(rows), months=1, as_of="2025-06-30")
        assert "2024-01" not in result["Month"].tolist()

    def test_granularity_rolls_up_the_period(self):
        rows = [self._adv_row("2025-04"), self._adv_row("2025-05"), self._adv_row("2025-06")]
        result = new_advances_trend(make_df(rows), granularity="Quarterly", as_of="2025-06-30")
        assert len(result) == 1
        assert result.iloc[0]["Accounts"] == 3

    def test_scope_narrows_to_one_branch(self):
        rows = [self._adv_row("2025-06", unit="MAHAD"), self._adv_row("2025-06", unit="PUNE")]
        result = new_advances_trend(make_df(rows), scope_col="branch", scope_value="MAHAD", as_of="2025-06-30")
        assert result.iloc[0]["Accounts"] == 1

    def test_empty_df_returns_empty(self):
        assert new_advances_trend(pd.DataFrame()).empty


class TestFleetExposure:
    """"Which customers hold multiple loans" (relationship/cross-sell) --
    distinct from fleet_defaulters (loan-level, delinquency-filtered ONLY).
    Reuses analysis/portfolio_intelligence.py::compute_fleet_exposure's own
    "top_df" aggregate directly -- ALL fleet operators regardless of
    delinquency status, ranked by Total SOH."""

    def _loan(self, mob: str, unit: str = "MAHAD", soh: float = 100_000.0, bucket: str = "STD"):
        return {"Cust Mob No": mob, "Unit": unit, "RegionName": "WEST", "SOH": soh, "curr_bucket": bucket}

    def test_customer_with_3plus_loans_is_a_fleet_operator(self):
        rows = [self._loan("9000000001") for _ in range(3)] + [self._loan("9111111111") for _ in range(2)]
        result = fleet_exposure(make_df(rows))
        assert len(result) == 1
        assert result.iloc[0]["Mobile"] == "9000000001"
        assert result.iloc[0]["Loans"] == 3

    def test_includes_a_healthy_non_delinquent_fleet_operator(self):
        # The exact distinction from fleet_defaulters: an all-STD fleet
        # customer must still show up here (no delinquency filter).
        rows = [self._loan("9000000001", bucket="STD") for _ in range(3)]
        result = fleet_exposure(make_df(rows))
        assert len(result) == 1
        assert result.iloc[0]["NPA Loans"] == 0

    def test_blank_mobile_loans_excluded_not_a_phantom_operator(self):
        rows = [self._loan("") for _ in range(5)]
        result = fleet_exposure(make_df(rows))
        assert result.empty

    def test_scope_narrows_to_one_branch(self):
        rows = (
            [self._loan("9000000001", unit="MAHAD") for _ in range(3)]
            + [self._loan("9111111111", unit="PUNE") for _ in range(3)]
        )
        result = fleet_exposure(make_df(rows), scope_col="branch", scope_value="MAHAD")
        assert len(result) == 1
        assert result.iloc[0]["Mobile"] == "9000000001"

    def test_empty_df_returns_empty(self):
        assert fleet_exposure(pd.DataFrame()).empty


class TestRepossessionList:
    """Accounts eligible for repossession -- SMA-2/NPA AND still within the
    collateral-value window. Reuses
    analysis/portfolio_intelligence.py::compute_repossession_list directly."""

    def _loan(self, bucket: str, months_old: int, soh: float = 100_000.0, unit: str = "MAHAD"):
        return {
            "curr_bucket": bucket, "Unit": unit, "RegionName": "WEST",
            "Ag_Date": pd.Timestamp("2025-06-30") - pd.DateOffset(months=months_old),
            "SOH": soh,
        }

    def test_only_sma2_and_npa_within_window_are_eligible(self):
        rows = [
            self._loan("NPA", months_old=5, soh=200_000.0),
            self._loan("STD", months_old=5, soh=999_999.0),      # not delinquent enough
            self._loan("NPA", months_old=30, soh=999_999.0),     # too old, collateral window passed
        ]
        result = repossession_list(make_df(rows), as_of="2025-06-30")
        assert len(result) == 1
        assert result.iloc[0]["SOH"] == 200_000.0

    def test_sorted_worst_first_by_soh(self):
        rows = [self._loan("NPA", months_old=5, soh=100_000.0), self._loan("SMA-2", months_old=5, soh=500_000.0)]
        result = repossession_list(make_df(rows), as_of="2025-06-30")
        assert result.iloc[0]["SOH"] == 500_000.0

    def test_scope_narrows_to_one_branch(self):
        rows = [
            self._loan("NPA", months_old=5, soh=100_000.0, unit="MAHAD"),
            self._loan("NPA", months_old=5, soh=500_000.0, unit="PUNE"),
        ]
        result = repossession_list(make_df(rows), scope_col="branch", scope_value="MAHAD", as_of="2025-06-30")
        assert len(result) == 1
        assert result.iloc[0]["SOH"] == 100_000.0

    def test_no_eligible_accounts_returns_empty(self):
        rows = [self._loan("STD", months_old=1)]
        assert repossession_list(make_df(rows), as_of="2025-06-30").empty

    def test_empty_df_returns_empty(self):
        assert repossession_list(pd.DataFrame()).empty


class TestGoodCustomers:
    """Loyal / high-quality customers eligible for refinance -- a business-
    development question, distinct from every other step in this module.
    Reuses analysis/portfolio_intelligence.py::compute_good_customers
    directly (tenure completed >= 70%, LCC% >= 100%, sorted lowest SOH
    first)."""

    def _loan(self, tenure=100.0, emi_accrued=80.0, lcc=100.0, soh=100_000.0, unit="MAHAD"):
        return {
            "Unit": unit, "RegionName": "WEST", "Tenure": tenure, "VehEMI Accrued": emi_accrued,
            "LCC%": lcc, "SOH": soh,
        }

    def test_meets_both_tenure_and_lcc_criteria(self):
        rows = [self._loan(tenure=100.0, emi_accrued=80.0, lcc=100.0)]  # 80% tenure, LCC 100%
        result = good_customers(make_df(rows))
        assert len(result) == 1
        assert result.iloc[0]["Tenure Completed %"] == 80.0

    def test_below_tenure_threshold_is_excluded(self):
        rows = [self._loan(tenure=100.0, emi_accrued=50.0, lcc=100.0)]  # 50% tenure, below 70%
        assert good_customers(make_df(rows)).empty

    def test_below_lcc_threshold_is_excluded(self):
        rows = [self._loan(tenure=100.0, emi_accrued=80.0, lcc=99.0)]  # LCC below 100%
        assert good_customers(make_df(rows)).empty

    def test_sorted_by_soh_ascending(self):
        rows = [
            self._loan(tenure=100.0, emi_accrued=80.0, lcc=100.0, soh=500_000.0),
            self._loan(tenure=100.0, emi_accrued=80.0, lcc=100.0, soh=100_000.0),
        ]
        result = good_customers(make_df(rows))
        assert result.iloc[0]["SOH"] == 100_000.0

    def test_scope_narrows_to_one_branch(self):
        rows = [
            self._loan(tenure=100.0, emi_accrued=80.0, lcc=100.0, unit="MAHAD"),
            self._loan(tenure=100.0, emi_accrued=80.0, lcc=100.0, unit="PUNE"),
        ]
        result = good_customers(make_df(rows), scope_col="branch", scope_value="MAHAD")
        assert len(result) == 1

    def test_empty_df_returns_empty(self):
        assert good_customers(pd.DataFrame()).empty


class TestVintageSummary:
    """Root-cause MECHANISM's other half: is delinquency concentrated in
    RECENT originations (an underwriting/sourcing-quality signal) or spread
    across the seasoned book (an execution signal)? Neither
    dimension_breakdown (org grain) nor roll_rate_summary (this-month
    migration only) can answer that -- this groups by DISBURSEMENT cohort
    instead. Reuses analysis/portfolio_intelligence.py::compute_product_analysis's
    own "vintage" sub-table directly -- the SAME cohort computation the
    dashboard's Vintage chart and report_agent's vintage_analysis section
    already use, so a cohort's NPA%/SMA-2% can never drift into a second,
    independently-computed definition here."""

    def _cohort_df(self, month: str, n: int, npa_n: int, unit: str = "MAHAD"):
        rows = []
        for i in range(n):
            rows.append({
                "Unit": unit, "RegionName": "WEST",
                "Ag_Date": pd.Timestamp(f"{month}-15"),
                "curr_bucket": "NPA" if i < npa_n else "STD",
                "Loan Amount": 2_00_000.0,
            })
        return rows

    def test_one_row_per_disbursement_cohort_with_npa_pct(self):
        rows = self._cohort_df("2025-01", 12, npa_n=6) + self._cohort_df("2025-02", 12, npa_n=0)
        result = vintage_summary(make_df(rows), as_of="2025-06-01")
        jan = result[result["Disbursement Month"] == "2025-01"].iloc[0]
        assert jan["NPA%"] == 50.0

    def test_cohort_below_minimum_accounts_is_excluded(self):
        rows = self._cohort_df("2025-01", 3, npa_n=1)  # below MIN_ACCOUNTS_SOURCE_VINTAGE (10)
        result = vintage_summary(make_df(rows), as_of="2025-06-01")
        assert result.empty

    def test_scope_narrows_to_one_branch(self):
        rows = (
            self._cohort_df("2025-01", 12, npa_n=6, unit="MAHAD")
            + self._cohort_df("2025-01", 12, npa_n=0, unit="PUNE")
        )
        result = vintage_summary(make_df(rows), scope_col="branch", scope_value="MAHAD", as_of="2025-06-01")
        jan = result[result["Disbursement Month"] == "2025-01"].iloc[0]
        # Scoped to MAHAD alone: 6/12 = 50%, not diluted by PUNE's clean cohort.
        assert jan["NPA%"] == 50.0
        assert jan["Accounts"] == 12

    def test_empty_df_returns_empty(self):
        assert vintage_summary(pd.DataFrame()).empty


class TestDemandSummary:
    """Root-cause MECHANISM's third lens: is poor Overall Collection% driven
    by failing to recover OLD arrears (Overdue Collection% -- a legacy/
    recovery problem) or by failing to collect THIS MONTH's own fresh EMI
    demand (Month Demand Collection% -- an ongoing execution problem)? These
    need different fixes, same reason roll_rate_summary/vintage_summary
    exist. Reuses analysis/portfolio_intelligence.py::compute_overdue_demand_scorecard
    directly -- the SAME waterfall computation the dashboard's Overdue-vs-
    Month-Demand tables and report_agent's overdue_demand section already
    use, so these percentages can never drift into a second, independently
    -computed definition here."""

    def _row(self, name, unit, overdue, overdue_collected, demand, demand_collected):
        return {
            "RegionName": "WEST", "Unit": unit, "MNT NAME": name,
            "Overdue": overdue, "OverdueCollected": overdue_collected,
            "MonthDemandExclPC": demand, "DemandCollected": demand_collected,
        }

    def _df(self):
        rows = [self._row("A", "MAHAD", 10_000.0, 2_000.0, 10_000.0, 9_000.0)] * 12
        rows += [self._row("B", "PUNE", 10_000.0, 9_000.0, 10_000.0, 2_000.0)] * 12
        return make_df(rows)

    def test_branch_dimension_shows_overdue_vs_demand_collection_pct(self):
        result = demand_summary(self._df(), dimension="branch")
        mahad = result[result["Branch"] == "MAHAD"].iloc[0]
        # MAHAD: overdue barely collected (20%), month demand collected well (90%)
        # -- a legacy-arrears problem, not a this-month execution problem.
        assert mahad["Overdue Collection %"] == 20.0
        assert mahad["Month Demand Collection %"] == 90.0

    def test_the_opposite_pattern_is_distinguishable(self):
        result = demand_summary(self._df(), dimension="branch")
        pune = result[result["Branch"] == "PUNE"].iloc[0]
        # PUNE: old arrears mostly cleared (90%), but THIS month's demand is
        # barely being collected (20%) -- an ongoing execution problem.
        assert pune["Overdue Collection %"] == 90.0
        assert pune["Month Demand Collection %"] == 20.0

    def test_region_dimension_returns_one_row_per_region(self):
        result = demand_summary(self._df(), dimension="region")
        assert set(result["Region"]) == {"WEST"}

    def test_scope_narrows_to_one_branch_before_computing_executive_rows(self):
        result = demand_summary(self._df(), dimension="executive", scope_col="branch", scope_value="MAHAD")
        assert set(result["Executive"]) == {"A"}

    def test_unknown_dimension_returns_empty(self):
        result = demand_summary(self._df(), dimension="not_a_real_dimension")
        assert result.empty

    def test_empty_df_returns_empty(self):
        assert demand_summary(pd.DataFrame(), dimension="branch").empty


class TestEntitySummaryBranch:
    def test_returns_one_row_per_metric_with_current_only_when_no_prev(self):
        rows = _branch_df("MAHAD", "WEST", n=3)
        df_curr = make_df(rows)
        result = entity_summary("branch", "MAHAD", df_curr)
        assert set(result["Metric"]) >= {"Collection%", "NPA%", "Hard Bucket%"}
        assert (result["Previous"].isna() | result["Previous"].isnull()).all() or result["Previous"].isnull().all()
        assert result["Delta"].isnull().all()

    def test_computes_pp_delta_not_relative_pct_change_for_percent_metrics(self):
        # Curr: 3 STD accounts (NPA% = 0). Prev: 3 NPA accounts (NPA% = 100).
        df_curr = make_df(_branch_df("MAHAD", "WEST", n=3, arrears=0.0))
        df_prev = make_df(_branch_df("MAHAD", "WEST", n=3, arrears=7.0))
        result = entity_summary("branch", "MAHAD", df_curr, df_prev)
        npa_row = result[result["Metric"] == "NPA%"].iloc[0]
        assert npa_row["Current"] == 0.0
        assert npa_row["Previous"] == 100.0
        # Raw pp difference (0 - 100 = -100), never a relative "% of %" figure.
        assert npa_row["Delta"] == -100.0

    def test_unknown_branch_returns_all_none_current(self):
        df_curr = make_df(_branch_df("MAHAD", "WEST", n=3))
        result = entity_summary("branch", "NO_SUCH_BRANCH", df_curr)
        assert result["Current"].isnull().all()


class TestEntitySummaryExecutive:
    def test_filters_to_the_single_executive(self):
        rows = (
            [{"MNT NAME": "RAJ", "Unit": "MAHAD", "Strike": "Y"}] * 5
            + [{"MNT NAME": "SUNIL", "Unit": "MAHAD", "Strike": "N"}] * 5
        )
        df_curr = make_df(rows)
        result = entity_summary("executive", ("RAJ", "MAHAD"), df_curr)
        coll_row = result[result["Metric"] == "Collection %"].iloc[0]
        assert coll_row["Current"] is not None

    def _pankaj_df(self):
        rows = (
            [{"MNT NAME": "PANKAJ DANDGE", "Unit": "AMRVA", "Strike": "N"}] * 20
            + [{"MNT NAME": "VIPUL RAJESH BHANGA", "Unit": "BHSWL", "Strike": "Y"}] * 20
        )
        return make_df(rows)

    def test_bare_name_string_matches_by_contains_not_a_crash(self):
        # A live Gemini call gave a name-only reference (no unit) for a
        # follow-up like "why does pankaj have so low strike rate" after an
        # executive table was already shown -- this used to crash with
        # "too many values to unpack" because entity_summary blindly did
        # name, unit = entity_value, assuming exactly a 2-element list.
        result = entity_summary("executive", "Pankaj", self._pankaj_df())
        coll_row = result[result["Metric"] == "Strike Rate %"].iloc[0]
        assert coll_row["Current"] == 0.0

    def test_one_element_list_does_not_crash(self):
        result = entity_summary("executive", ["Pankaj"], self._pankaj_df())
        assert result["Current"].notna().any()

    def test_three_element_list_does_not_crash_extra_ignored(self):
        # A live Gemini call produced exactly this shape (3 elements instead
        # of 2) for the same reason -- the fix must not depend on the LLM
        # always emitting precisely 2 elements.
        result = entity_summary("executive", ["Pankaj", "Dandge", "AMRVA"], self._pankaj_df())
        assert result["Current"].notna().any()

    def test_unit_mismatch_falls_back_to_name_only_match(self):
        # unit guessed wrong (or genuinely unknown, e.g. None) must not
        # prevent finding the executive by name.
        result = entity_summary("executive", ["Pankaj", "WRONG_UNIT"], self._pankaj_df())
        assert result["Current"].notna().any()


class TestDistinctExecutiveMatches:
    """The real bug this exists to prevent: _executive_row's name-only
    (contains) match, when it resolves to more than one DISTINCT person, used
    to silently pick compute_executive_scorecard's iloc[0] (best Collection%
    first, per that function's own sort) with no signal at all -- a
    struggling "Rahul" could be silently swapped for a well-performing
    "Rahul" in a different branch. guardrails.py uses this helper's own
    match-count BEFORE entity_summary ever runs, to ask for clarification
    instead."""

    def _two_rahuls_df(self):
        rows = (
            [{"MNT NAME": "RAHUL SHARMA", "Unit": "PUNE", "Strike": "Y"}] * 5
            + [{"MNT NAME": "RAHUL VERMA", "Unit": "NAGPUR", "Strike": "N"}] * 5
        )
        return make_df(rows)

    def test_name_matching_two_distinct_people_returns_both(self):
        matches = distinct_executive_matches(self._two_rahuls_df(), "Rahul")
        assert set(matches) == {("RAHUL SHARMA", "PUNE"), ("RAHUL VERMA", "NAGPUR")}

    def test_name_matching_one_person_returns_a_single_match(self):
        matches = distinct_executive_matches(self._two_rahuls_df(), "Sharma")
        assert matches == [("RAHUL SHARMA", "PUNE")]

    def test_unit_that_actually_matches_disambiguates_to_one(self):
        matches = distinct_executive_matches(self._two_rahuls_df(), "Rahul", unit="PUNE")
        assert matches == [("RAHUL SHARMA", "PUNE")]

    def test_no_match_returns_empty(self):
        assert distinct_executive_matches(self._two_rahuls_df(), "Nobody") == []

    def test_empty_df_returns_empty(self):
        assert distinct_executive_matches(pd.DataFrame(), "Rahul") == []


class TestResolveExecutiveIdentity:
    """The real bug this exists to fix: entity_summary resolves a typed
    name ("Taushif Khan") to the real data's canonical MNT NAME ("TAUSIF
    KHAN NAZIR KH") internally via _executive_row's contains-match, but
    used to throw that resolution away -- only the Metric/Value table
    reached the caller, never the canonical identity. Two confirmed
    consequences from a live session: (1) a follow-up asking for "the
    customers behind" that SAME already-focused executive silently
    returned EMPTY (non_paying_customers's source table lookup requires an
    exact MNT NAME/Unit match, and the raw typed name never exactly
    matches the real data), and (2) a differently-phrased follow-up
    instead pulled a stale, unrelated executive-group table from memory
    and returned the WRONG executives' customers entirely. Surfacing the
    canonical (MNT NAME, Unit) pair lets the UI layer store and reuse the
    REAL identity everywhere downstream, closing both failure modes at
    the root rather than patching either symptom individually."""

    def _df(self):
        rows = (
            [{"MNT NAME": "TAUSIF KHAN NAZIR KH", "Unit": "BHSWL", "Strike": "Y"}] * 10
            + [{"MNT NAME": "SAURABH GURUDAS PA", "Unit": "BHSWL", "Strike": "Y"}] * 10
        )
        return make_df(rows)

    def test_resolves_a_partial_typed_name_to_the_canonical_full_name(self):
        result = resolve_executive_identity(self._df(), "Tausif Khan")
        assert result == ("TAUSIF KHAN NAZIR KH", "BHSWL")

    def test_no_match_returns_none(self):
        assert resolve_executive_identity(self._df(), "Nobody") is None

    def test_ambiguous_match_returns_none_rather_than_guessing(self):
        # Guardrails is expected to have already asked for clarification
        # before this is ever called on an ambiguous name -- this must
        # never silently pick one, matching that same safety contract.
        rows = (
            [{"MNT NAME": "RAHUL SHARMA", "Unit": "PUNE"}] * 10
            + [{"MNT NAME": "RAHUL VERMA", "Unit": "NAGPUR"}] * 10
        )
        assert resolve_executive_identity(make_df(rows), "Rahul") is None

    def test_empty_df_returns_none(self):
        assert resolve_executive_identity(pd.DataFrame(), "Taushif Khan") is None


class TestEntitySummaryRegionZoneBU:
    """The exact regression a live user hit: "why is CS Nagar not performing
    good" after a region-level dimension_breakdown -- CS Nagar is a
    RegionName, not a Unit (branch), and entity_summary used to only
    support entity_type "branch"/"executive". Silently filtering
    Unit=="CS Nagar" found zero rows and returned an all-None table with no
    error at all -- the worst kind of silent failure."""

    def _hierarchy_df(self):
        rows = []
        for region, zone, bu, coll in [("CS NAGAR", "NORTH", "CV", 60.0), ("NAGPUR", "NORTH", "CV", 90.0)]:
            demand = 10_000.0
            collected = demand * coll / 100
            for _ in range(3):
                rows.append({
                    "RegionName": region, "Zone": zone, "BU": bu, "Unit": f"{region}_BR",
                    "Net Collection Demand Inst+Exp+BC": demand,
                    "Month Collection (Excluding Reserve Collection)": collected,
                })
        return make_df(rows)

    def test_region_entity_type_returns_real_values_not_none(self):
        result = entity_summary("region", "CS Nagar", self._hierarchy_df())
        coll_row = result[result["Metric"] == "Collection%"].iloc[0]
        assert coll_row["Current"] == 60.0

    def test_zone_entity_type_returns_real_values(self):
        result = entity_summary("zone", "NORTH", self._hierarchy_df())
        coll_row = result[result["Metric"] == "Collection%"].iloc[0]
        assert coll_row["Current"] is not None

    def test_bu_entity_type_returns_real_values(self):
        result = entity_summary("bu", "CV", self._hierarchy_df())
        coll_row = result[result["Metric"] == "Collection%"].iloc[0]
        assert coll_row["Current"] is not None

    def test_unknown_region_returns_all_none_not_a_crash(self):
        result = entity_summary("region", "NO_SUCH_REGION", self._hierarchy_df())
        assert result["Current"].isnull().all()


class TestUnderperformerQuartile:
    def _scorecard_df(self):
        # 4 executives, spread across Collection% and NPA% so quartile cuts
        # are unambiguous either direction.
        rows = []
        for name, coll, npa in [
            ("A", 100.0, "STD"), ("B", 80.0, "STD"), ("C", 60.0, "NPA"), ("D", 20.0, "NPA"),
        ]:
            demand = 10_000.0
            collected = demand * coll / 100
            rows += [{
                "MNT NAME": name, "Unit": "MAHAD",
                "Net Collection Demand Inst+Exp+BC": demand,
                "Month Collection (Excluding Reserve Collection)": collected,
                "curr_bucket": npa,
            }] * 5
        return make_df(rows)

    def test_higher_is_better_metric_flags_the_lowest_values(self):
        result = underperformer_quartile(self._scorecard_df(), "Collection %")
        assert set(result["MNT NAME"]) == {"D"}

    def test_lower_is_better_metric_flags_the_highest_values(self):
        result = underperformer_quartile(self._scorecard_df(), "NPA %")
        # C and D both sit at 100% NPA -- both in the "worst" (top) tier.
        assert "D" in set(result["MNT NAME"])

    def test_unknown_metric_direction_requires_explicit_flag(self):
        import pytest
        with pytest.raises(ValueError):
            underperformer_quartile(self._scorecard_df(), "Some Unregistered Metric")

    def test_a_wrong_llm_supplied_override_is_ignored_for_a_known_metric(self):
        # The exact live-Gemini bug this fixes: the model gave
        # higher_is_better=False for Collection % (semantically wrong --
        # Collection% is unambiguously higher-is-better, that's the whole
        # point of registering it in METRIC_DIRECTION). Trusting a
        # caller-supplied override that CONTRADICTS a registered, known
        # metric's direction defeats the registry's entire purpose as a
        # single source of truth immune to LLM error -- confirmed live,
        # this silently sorted best-performer-first for a question asking
        # for the WORST performer.
        result = underperformer_quartile(self._wide_spread_df(), "Collection %", higher_is_better=False)
        assert result.iloc[0]["MNT NAME"] == "H"  # still the worst (lowest) performer

    def _wide_spread_df(self):
        # 8 executives so the bottom quartile flags several rows -- enough
        # to meaningfully check ROW ORDER within the flagged set, not just
        # membership.
        rows = []
        for name, coll in [
            ("A", 100.0), ("B", 95.0), ("C", 90.0), ("D", 85.0),
            ("E", 40.0), ("F", 30.0), ("G", 20.0), ("H", 5.0),
        ]:
            demand = 10_000.0
            rows += [{
                "MNT NAME": name, "Unit": "MAHAD",
                "Net Collection Demand Inst+Exp+BC": demand,
                "Month Collection (Excluding Reserve Collection)": demand * coll / 100,
            }] * 5
        return make_df(rows)

    def test_higher_is_better_metric_sorts_worst_first(self):
        # A user asking "why is collection% so less" is focused on badness --
        # the worst performer must be the FIRST row, not buried at the
        # bottom of whatever order rank_by_metric happened to leave it in
        # (rank_by_metric always sorts descending regardless of which
        # direction is "bad" for a given metric).
        result = underperformer_quartile(self._wide_spread_df(), "Collection %")
        assert result.iloc[0]["MNT NAME"] == "H"
        assert list(result["Collection %"]) == sorted(result["Collection %"])

    def test_lower_is_better_metric_sorts_worst_first(self):
        # Fractional per-executive NPA% (not just 0/100) via a mix of
        # NPA/STD rows per executive, so ordering within the flagged set is
        # actually meaningful to check.
        rows = []
        for name, npa_of_5 in [("A", 0), ("B", 1), ("C", 2), ("D", 4), ("E", 5)]:
            for i in range(5):
                rows.append({
                    "MNT NAME": name, "Unit": "MAHAD",
                    "Net Collection Demand Inst+Exp+BC": 10_000.0,
                    "Month Collection (Excluding Reserve Collection)": 8_000.0,
                    "curr_bucket": "NPA" if i < npa_of_5 else "STD",
                })
        result = underperformer_quartile(make_df(rows), "NPA %")
        assert len(result) >= 2
        assert result.iloc[0]["NPA %"] == max(result["NPA %"])
        assert list(result["NPA %"]) == sorted(result["NPA %"], reverse=True)

    def _scope_df(self):
        # Same 4-executive spread as _scorecard_df, but split across two
        # zones (WEST/EAST) so scope_col="zone" can be exercised -- a BU or
        # Zone head must be able to scope this the same way a region_filter
        # already could, not just at the region/branch level.
        rows = []
        for name, coll, zone in [
            ("A", 100.0, "WEST"), ("B", 80.0, "WEST"), ("C", 60.0, "EAST"), ("D", 20.0, "EAST"),
        ]:
            demand = 10_000.0
            collected = demand * coll / 100
            rows += [{
                "MNT NAME": name, "Unit": "MAHAD", "Zone": zone,
                "Net Collection Demand Inst+Exp+BC": demand,
                "Month Collection (Excluding Reserve Collection)": collected,
            }] * 5
        return make_df(rows)

    def test_scope_col_narrows_to_one_zone(self):
        result = underperformer_quartile(self._scope_df(), "Collection %", scope_col="zone", scope_value="EAST")
        assert set(result["MNT NAME"]) <= {"C", "D"}
        assert "D" in set(result["MNT NAME"])

    def test_scope_col_still_supports_region_by_name(self):
        rows = [{"MNT NAME": n, "Unit": "MAHAD", "RegionName": "WEST" if n in ("A", "B") else "EAST",
                  "Net Collection Demand Inst+Exp+BC": 10_000.0,
                  "Month Collection (Excluding Reserve Collection)": 10_000.0 * c / 100}
                for n, c in [("A", 100.0), ("B", 80.0), ("C", 60.0), ("D", 20.0)] for _ in range(5)]
        result = underperformer_quartile(make_df(rows), "Collection %", scope_col="region", scope_value="EAST")
        assert set(result["MNT NAME"]) <= {"C", "D"}


class TestBottomNPerGroup:
    def _two_branch_df(self):
        rows = []
        for branch, execs in [
            ("MAHAD", [("A", 100.0), ("B", 80.0), ("C", 20.0)]),
            ("PUNE",  [("D", 90.0), ("E", 10.0)]),
        ]:
            for name, coll in execs:
                demand = 10_000.0
                collected = demand * coll / 100
                rows += [{
                    "MNT NAME": name, "Unit": branch,
                    "Net Collection Demand Inst+Exp+BC": demand,
                    "Month Collection (Excluding Reserve Collection)": collected,
                }] * 5
        return make_df(rows)

    def test_bottom_n_per_branch_by_collection_pct(self):
        result = bottom_n_per_group(self._two_branch_df(), "Collection %", n=1, group_col="branch")
        mahad_pick = result[result["Unit"] == "MAHAD"]["MNT NAME"].tolist()
        pune_pick  = result[result["Unit"] == "PUNE"]["MNT NAME"].tolist()
        assert mahad_pick == ["C"]
        assert pune_pick == ["E"]

    def test_group_smaller_than_n_returns_all_its_rows(self):
        result = bottom_n_per_group(self._two_branch_df(), "Collection %", n=5, group_col="branch")
        assert len(result[result["Unit"] == "PUNE"]) == 2

    def test_tie_at_cutoff_is_broken_deterministically_by_accounts(self):
        rows = []
        for name, coll, n_accounts in [("A", 50.0, 3), ("B", 50.0, 7), ("C", 90.0, 5)]:
            demand = 10_000.0
            collected = demand * coll / 100
            rows += [{
                "MNT NAME": name, "Unit": "MAHAD",
                "Net Collection Demand Inst+Exp+BC": demand,
                "Month Collection (Excluding Reserve Collection)": collected,
            }] * n_accounts
        result = bottom_n_per_group(make_df(rows), "Collection %", n=1, group_col="branch")
        # A and B tie at 50% (worst) -- B has more accounts, wins the tiebreak.
        assert result.iloc[0]["MNT NAME"] == "B"


class TestExecutiveRoster:
    """Covers a plain "list/show me all executives of X branch" question --
    distinct from underperformer_quartile (only the worst quartile survives)
    and bottom_n_per_group (caps each group at N). A real live-Gemini check
    caught this gap: that exact question was misrouted to entity_summary
    (which only returns one all-metrics ROW for the branch itself, not a
    per-executive roster), since entity_summary was the only step that
    matched "one named entity, no metric, no quartile/N language" -- there
    was no step for "just list everyone at this grain" until this one."""

    def _two_branch_df(self):
        rows = []
        for branch, execs in [("MAHAD", ["A", "B", "C"]), ("PUNE", ["D", "E"])]:
            for name in execs:
                rows += [{
                    "MNT NAME": name, "Unit": branch,
                    "Net Collection Demand Inst+Exp+BC": 10_000.0,
                    "Month Collection (Excluding Reserve Collection)": 8_000.0,
                }] * 5
        return make_df(rows)

    def test_returns_every_executive_in_the_scoped_branch_not_just_worst(self):
        result = executive_roster(self._two_branch_df(), scope_col="branch", scope_value="MAHAD")
        assert set(result["MNT NAME"]) == {"A", "B", "C"}

    def test_unscoped_returns_every_executive_portfolio_wide(self):
        result = executive_roster(self._two_branch_df())
        assert set(result["MNT NAME"]) == {"A", "B", "C", "D", "E"}

    def test_empty_df_returns_empty(self):
        assert executive_roster(pd.DataFrame()).empty

    def _ranked_df(self):
        rows = []
        for name, coll in [("A", 95.0), ("B", 60.0), ("C", 80.0), ("D", 40.0)]:
            rows += [{
                "MNT NAME": name, "Unit": "MAHAD",
                "Net Collection Demand Inst+Exp+BC": 10_000.0,
                "Month Collection (Excluding Reserve Collection)": 10_000.0 * coll / 100,
            }] * 5
        return make_df(rows)

    def test_top_n_by_metric_returns_best_performers_first(self):
        # "top 2 executives by collection%" -- best (highest Collection%)
        # first, matching colloquial "top N" meaning, distinct from
        # underperformer_quartile/bottom_n_per_group which are always
        # worst-first.
        result = executive_roster(self._ranked_df(), n=2, metric="Collection %")
        assert result["MNT NAME"].tolist() == ["A", "C"]

    def test_bottom_n_by_metric_returns_worst_performers_first(self):
        result = executive_roster(self._ranked_df(), n=2, metric="Collection %", worst_first=True)
        assert result["MNT NAME"].tolist() == ["D", "B"]

    def test_top_n_respects_metric_direction_for_inverse_metrics(self):
        # NPA% is lower-is-better -- "top 2 by NPA%" (best first) means the
        # LOWEST NPA% first, not the highest raw number.
        rows = []
        for name, npa in [("A", 2.0), ("B", 15.0), ("C", 8.0)]:
            rows += [{
                "MNT NAME": name, "Unit": "MAHAD",
                "curr_bucket": "NPA" if npa > 0 else "STD",
            }] * int(npa) + [{
                "MNT NAME": name, "Unit": "MAHAD", "curr_bucket": "STD",
            }] * (100 - int(npa))
        result = executive_roster(make_df(rows), n=2, metric="NPA %")
        assert result["MNT NAME"].tolist() == ["A", "C"]

    def test_n_without_metric_returns_full_unranked_roster(self):
        result = executive_roster(self._ranked_df(), n=2)
        assert set(result["MNT NAME"]) == {"A", "B", "C", "D"}


class TestIntersectEntities:
    def _table(self, names):
        return make_df([{"MNT NAME": n, "Unit": "MAHAD"} for n in names])

    def test_only_entities_in_every_table_are_kept_by_default(self):
        t1 = self._table(["A", "B", "C"])
        t2 = self._table(["B", "C", "D"])
        t3 = self._table(["B", "C", "E"])
        result = intersect_entities([t1, t2, t3])
        assert set(result["MNT NAME"]) == {"B", "C"}

    def test_no_common_entities_returns_empty(self):
        t1 = self._table(["A"])
        t2 = self._table(["B"])
        result = intersect_entities([t1, t2])
        assert result.empty

    def test_min_appearances_relaxes_the_requirement(self):
        t1 = self._table(["A", "B"])
        t2 = self._table(["B", "C"])
        t3 = self._table(["C", "D"])
        result = intersect_entities([t1, t2, t3], min_appearances=2)
        assert set(result["MNT NAME"]) == {"B", "C"}


class TestNonPayingCustomers:
    def test_filters_to_hard_bucket_customers_of_the_given_executives(self):
        rows = [
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "Cust Name": "PAYING CUST",
             "Cust Mob No": "1111111111", "Arrears / EMI": 0.0},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "Cust Name": "DEFAULTER",
             "Cust Mob No": "2222222222", "Arrears / EMI": 7.0},
            {"MNT NAME": "OTHER", "Unit": "MAHAD", "Cust Name": "NOT IN SCOPE",
             "Cust Mob No": "3333333333", "Arrears / EMI": 8.0},
        ]
        entities = make_df([{"MNT NAME": "RAJ", "Unit": "MAHAD"}])
        result = non_paying_customers(make_df(rows), entities)
        assert list(result["Cust Name"]) == ["DEFAULTER"]
        assert "Last Receipt Amount" in result.columns or "Ag_Date" in result.columns

    def test_empty_entities_returns_empty(self):
        rows = [{"MNT NAME": "RAJ", "Unit": "MAHAD", "Arrears / EMI": 7.0}]
        result = non_paying_customers(make_df(rows), pd.DataFrame())
        assert result.empty

    def test_all_columns_returns_every_source_column_not_the_curated_subset(self):
        # The exact gap a live user hit: "can you give me all columns for
        # these customers" / "show loan number too" both silently re-ran
        # the SAME curated column subset with no way to ask for more --
        # the LLM understood the intent fine, there was just no parameter
        # to act on it.
        rows = [{
            "MNT NAME": "RAJ", "Unit": "MAHAD", "Cust Mob No": "2222222222",
            "Arrears / EMI": 7.0, "SegmentName": "TWO WHEELER", "FUEL_TYPE": "PETROL",
        }]
        entities = make_df([{"MNT NAME": "RAJ", "Unit": "MAHAD"}])
        curated = non_paying_customers(make_df(rows), entities)
        full = non_paying_customers(make_df(rows), entities, all_columns=True)
        assert "SegmentName" not in curated.columns
        assert "FUEL_TYPE" not in curated.columns
        assert "SegmentName" in full.columns
        assert "FUEL_TYPE" in full.columns


class TestCustomerLoanBook:
    def test_returns_every_loan_for_the_customer(self):
        rows = [
            {"Cust Mob No": "9999999999", "curr_bucket": "NPA"},
            {"Cust Mob No": "9999999999", "curr_bucket": "STD"},
            {"Cust Mob No": "1111111111", "curr_bucket": "STD"},
        ]
        result = customer_loan_book(make_df(rows), "9999999999")
        assert len(result) == 2

    def test_blank_mobile_number_never_matches(self):
        rows = [{"Cust Mob No": "", "curr_bucket": "NPA"}] * 3
        result = customer_loan_book(make_df(rows), "")
        assert result.empty

    def test_all_columns_returns_every_source_column(self):
        rows = [{"Cust Mob No": "9999999999", "curr_bucket": "NPA", "SegmentName": "CAR"}]
        curated = customer_loan_book(make_df(rows), "9999999999")
        full = customer_loan_book(make_df(rows), "9999999999", all_columns=True)
        assert "SegmentName" not in curated.columns
        assert "SegmentName" in full.columns


class TestCustomerLoanSummary:
    def test_summarizes_delinquency_and_exposure(self):
        rows = [
            {"Cust Mob No": "9999999999", "curr_bucket": "NPA", "SOH": 50_000.0},
            {"Cust Mob No": "9999999999", "curr_bucket": "STD", "SOH": 30_000.0},
        ]
        summary = customer_loan_summary(make_df(rows), "9999999999")
        assert summary["loan_count"] == 2
        assert summary["delinquent_count"] == 1
        assert summary["total_soh"] == 80_000.0

    def test_no_loans_returns_zeroed_summary(self):
        summary = customer_loan_summary(make_df([]), "0000000000")
        assert summary == {"loan_count": 0, "delinquent_count": 0, "total_soh": 0.0}


class TestConceptFilter:
    """The real gap a live user question exposed: "are there any non
    starters in my region" -- non_starter is already a registered CONCEPT
    in registry/ontology.py (reused by graph.py's compiler AND
    smart_alerts.py), but the Investigator's step vocabulary had no way to
    filter by a concept at all. Reuses compiler.core._expand_filters +
    agents.data_executor._build_mask -- the SAME expansion/application the
    main AI Query pipeline already uses, so a concept's definition can never
    drift between the two pipelines."""

    def _df(self):
        rows = [
            {"RegionName": "WEST", "Unit": "MAHAD", "Non Starter": "Y", "VehEMI Accrued": 1, "Loan No": "L1"},
            {"RegionName": "WEST", "Unit": "MAHAD", "Non Starter": "N", "VehEMI Accrued": 1, "Loan No": "L2"},
            {"RegionName": "EAST", "Unit": "NAGPUR", "Non Starter": "Y", "VehEMI Accrued": 1, "Loan No": "L3"},
        ]
        return make_df(rows)

    def test_filters_to_matching_rows_only(self):
        result = concept_filter(self._df(), "non_starter")
        assert set(result["Loan No"]) == {"L1", "L3"}

    def test_spelled_out_yes_also_matches_not_just_the_y_abbreviation(self):
        # A real, confirmed bug (not "genuinely zero non-starters"): some
        # monthly LCC extracts spell this flag out as "Yes" instead of "Y"
        # (utils.is_yes's own docstring documents this exact variance, and
        # registry/ontology.py's strike_pct METRIC was already fixed for the
        # identical variance on the Strike column) -- but the non_starter
        # CONCEPT was still an exact "==" Y match, silently returning zero
        # rows on any file using the spelled-out form. A live user hit
        # exactly this: "are there any non starters" -> "No rows matched,"
        # indistinguishable from a genuinely clean file.
        rows = [
            {"RegionName": "WEST", "Unit": "MAHAD", "Non Starter": "Yes", "VehEMI Accrued": 1, "Loan No": "L1"},
            {"RegionName": "WEST", "Unit": "MAHAD", "Non Starter": "No", "VehEMI Accrued": 1, "Loan No": "L2"},
        ]
        result = concept_filter(make_df(rows), "non_starter")
        assert set(result["Loan No"]) == {"L1"}

    def test_vehicle_emi_accrued_must_be_exactly_one(self):
        # A customer flagged Non Starter but already 2+ EMIs into the loan
        # is a different (and arguably mis-flagged/stale) case, not a
        # genuine "never paid the 1st EMI" -- narrow to the loan's very
        # first EMI cycle specifically.
        rows = [
            {"RegionName": "WEST", "Unit": "MAHAD", "Non Starter": "Y", "VehEMI Accrued": 1, "Loan No": "L1"},
            {"RegionName": "WEST", "Unit": "MAHAD", "Non Starter": "Y", "VehEMI Accrued": 2, "Loan No": "L2"},
        ]
        result = concept_filter(make_df(rows), "non_starter")
        assert set(result["Loan No"]) == {"L1"}

    def test_scoped_to_one_region(self):
        result = concept_filter(self._df(), "non_starter", scope_col="region", scope_value="WEST")
        assert list(result["Loan No"]) == ["L1"]

    def test_unknown_concept_raises_not_silently_returns_everything(self):
        import pytest
        with pytest.raises(ValueError):
            concept_filter(self._df(), "not_a_real_concept")

    def test_empty_df_returns_empty(self):
        assert concept_filter(make_df([]), "non_starter").empty

    def test_missing_required_column_raises_not_silently_over_matches(self):
        # Real bug: agents/data_executor.py::_apply_condition SKIPS (returns
        # unfiltered) a condition whose column isn't in df.columns, rather
        # than erroring. For a multi-condition concept like
        # "colending_at_risk" ({"CoLending_Loans"=="Y"} AND
        # {"Arrears / EMI">0}), a file missing CoLending_Loans (a REQUIRED_COLS
        # column, not CRITICAL_COLS -- utils.py documents this exact column as
        # an example of one that can be legitimately absent) would silently
        # drop that condition, leaving every delinquent loan in the WHOLE
        # portfolio flagged as "co-lending at risk" -- a false-positive
        # answer, not an empty one, with nothing distinguishing it from a
        # real result. Must raise instead, before ever building the mask.
        import pytest
        df = self._df().drop(columns=["Non Starter"])
        with pytest.raises(ValueError, match="Non Starter"):
            concept_filter(df, "non_starter")


class TestConceptSummary:
    def test_summarizes_count_and_exposure_not_raw_rows(self):
        rows = [
            {"Non Starter": "Y", "VehEMI Accrued": 1, "SOH": 50_000.0, "Loan No": "L1"},
            {"Non Starter": "N", "VehEMI Accrued": 1, "SOH": 30_000.0, "Loan No": "L2"},
            {"Non Starter": "Y", "VehEMI Accrued": 1, "SOH": 20_000.0, "Loan No": "L3"},
        ]
        summary = concept_summary(make_df(rows), "non_starter")
        assert summary["count"] == 2
        assert summary["total_soh"] == 70_000.0

    def test_no_matches_returns_zeroed_summary(self):
        rows = [{"Non Starter": "N", "VehEMI Accrued": 1, "SOH": 10.0}]
        summary = concept_summary(make_df(rows), "non_starter")
        assert summary == {"count": 0, "total_soh": 0.0}


class TestConceptBreakdown:
    """concept_filter has no drill-down at all today -- "are there any non
    starters in my region" is a dead end. This is the drill: which
    branch/executive has the most rows in an ALREADY-CONCEPT-FILTERED
    table, distinct from dimension_breakdown (which recomputes each
    entity's OVERALL Collection%/NPA%/etc from the FULL portfolio -- a
    different question from "how many of THIS filtered subset's rows
    belong to it"). Groups by EVERY column resolve_dimension returns for a
    dimension, not just the first -- "executive" resolves to
    ["MNT NAME", "Unit"] together, so two different people sharing a first
    name in different branches are never merged into one row (the same
    collision class investigator/guardrails.py's ambiguous-name check
    exists to prevent, avoided here by construction since this groups real
    rows directly rather than fuzzy-matching a typed name)."""

    def _filtered_df(self):
        rows = [
            {"Unit": "MAHAD", "MNT NAME": "RAJ", "SOH": 1_20_000.0},
            {"Unit": "MAHAD", "MNT NAME": "RAJ", "SOH": 60_000.0},
            {"Unit": "PUNE", "MNT NAME": "SUNIL", "SOH": 2_00_000.0},
        ]
        return make_df(rows)

    def test_breaks_down_by_branch(self):
        result = concept_breakdown(self._filtered_df(), "branch")
        mahad = result[result["Unit"] == "MAHAD"].iloc[0]
        assert mahad["Accounts"] == 2
        assert mahad["SOH (Cr)"] == round(1_80_000.0 / 1_00_00_000, 2)

    def test_sorted_worst_offender_first_by_count(self):
        result = concept_breakdown(self._filtered_df(), "branch")
        assert result.iloc[0]["Unit"] == "MAHAD"

    def test_breaks_down_by_executive_groups_on_name_and_branch_together(self):
        # Two different people can share a first name across branches --
        # grouping must key on (MNT NAME, Unit) TOGETHER, never name alone.
        rows = [
            {"Unit": "PUNE", "MNT NAME": "RAHUL SHARMA", "SOH": 10_000.0},
            {"Unit": "NAGPUR", "MNT NAME": "RAHUL VERMA", "SOH": 20_000.0},
        ]
        result = concept_breakdown(make_df(rows), "executive")
        assert len(result) == 2
        assert set(zip(result["MNT NAME"], result["Unit"])) == {
            ("RAHUL SHARMA", "PUNE"), ("RAHUL VERMA", "NAGPUR"),
        }

    def test_empty_df_returns_empty(self):
        assert concept_breakdown(pd.DataFrame(), "branch").empty


class TestHighArrearsAtRisk:
    """A named, actively-tracked leader priority (confirmed against a real
    operational email: "High Arrears: Loan at Risk" / "TOP LOANS WITH HIGH
    CLOSING ARREARS") that had no registered CONCEPT at all -- the ratio
    (Inst+Exp+BC arrears vs a % of original loan amount) can't be expressed
    as a simple {column, op, value} CONCEPTS condition (it needs a derived
    sum-of-3-columns compared to a ratio of a 4th, and no such derived
    column exists on the raw upload), so this reuses
    smart_alerts.py::alert_high_arrears_ratio directly -- the SAME
    computation the dashboard's alert card and report_agent's risk_flags
    section already use -- as an Investigator step instead of forcing it
    into the registry's declarative schema."""

    def _row(self, unit, inst, exp, bc, loan_amt):
        return {
            "Unit": unit, "RegionName": "WEST",
            "ARREARS AGAINST INST": inst, "ARREARS AGAINST EXP": exp,
            "ARREARS AGAINST BC": bc, "Loan Amount": loan_amt,
        }

    def test_flags_accounts_over_the_ratio_threshold(self):
        rows = [
            self._row("MAHAD", 60_000.0, 0.0, 0.0, 100_000.0),  # 60% -- over 50% threshold
            self._row("MAHAD", 10_000.0, 0.0, 0.0, 100_000.0),  # 10% -- under threshold
        ]
        result = high_arrears_at_risk(make_df(rows))
        assert len(result) == 1
        assert result.iloc[0]["Unit"] == "MAHAD"

    def test_scope_narrows_to_one_branch(self):
        rows = [
            self._row("MAHAD", 60_000.0, 0.0, 0.0, 100_000.0),
            self._row("PUNE", 60_000.0, 0.0, 0.0, 100_000.0),
        ]
        result = high_arrears_at_risk(make_df(rows), scope_col="branch", scope_value="MAHAD")
        assert set(result["Unit"]) == {"MAHAD"}

    def test_no_accounts_over_threshold_returns_empty(self):
        rows = [self._row("MAHAD", 5_000.0, 0.0, 0.0, 100_000.0)]
        assert high_arrears_at_risk(make_df(rows)).empty

    def test_empty_df_returns_empty(self):
        assert high_arrears_at_risk(pd.DataFrame()).empty


class TestFleetDefaulters:
    """"Fleet owners must be in our priority as they must pay their EMI" --
    fleet identification (>= FLEET_MIN_LOANS distinct loans per customer)
    already exists (compute_fleet_exposure), but nothing combines it with a
    CURRENT delinquency filter to answer "which fleet owners are
    defaulting" at the loan level. Reuses FLEET_MIN_LOANS (config.py, the
    SAME threshold compute_fleet_exposure uses) and the SAME blank-mobile
    exclusion that function's own docstring documents as a real, confirmed
    production bug otherwise (a missing Cust Mob No collapsing many
    unrelated customers into one phantom "fleet operator")."""

    def _df(self):
        rows = []
        # RAJ: 3 loans (fleet), one delinquent -- should appear.
        for i in range(3):
            rows.append({
                "Cust Mob No": "9999999999", "Loan No": f"RAJ_{i}", "Unit": "MAHAD",
                "Arrears / EMI": 2.0 if i == 0 else 0.0, "SOH": 100_000.0,
            })
        # SUNIL: only 2 loans -- not a fleet operator, must be excluded
        # even though delinquent.
        for i in range(2):
            rows.append({
                "Cust Mob No": "8888888888", "Loan No": f"SUNIL_{i}", "Unit": "MAHAD",
                "Arrears / EMI": 5.0, "SOH": 50_000.0,
            })
        return make_df(rows)

    def test_returns_only_delinquent_loans_of_fleet_operators(self):
        result = fleet_defaulters(self._df())
        assert set(result["Loan No"]) == {"RAJ_0"}

    def test_non_fleet_customer_excluded_even_if_delinquent(self):
        result = fleet_defaulters(self._df())
        assert "SUNIL_0" not in set(result["Loan No"])

    def test_blank_mobile_number_excluded_not_treated_as_one_fleet(self):
        rows = [
            {"Cust Mob No": "", "Loan No": f"L{i}", "Unit": "MAHAD", "Arrears / EMI": 1.0, "SOH": 10_000.0}
            for i in range(5)
        ]
        assert fleet_defaulters(make_df(rows)).empty

    def test_scope_narrows_to_one_branch(self):
        rows = self._df().to_dict("records")
        for r in rows[:3]:
            r["Unit"] = "PUNE"
        result = fleet_defaulters(make_df(rows), scope_col="branch", scope_value="PUNE")
        assert set(result["Loan No"]) == {"RAJ_0"}

    def test_empty_df_returns_empty(self):
        assert fleet_defaulters(pd.DataFrame()).empty


class TestTopClosingArrears:
    """Matches the real operational priority email's own "TOP N LOANS WITH
    HIGH CLOSING ARREARS" section exactly -- a straight top-N by raw
    Closing Arrears value, distinct from high_arrears_at_risk (a RATIO:
    Inst+Exp+BC arrears exceeding a % of loan amount, "potential
    write-off," not "top N by absolute rupee value")."""

    def _df(self):
        rows = [
            {"Loan No": "L1", "Unit": "MAHAD", "Closing Arrears": 50_000.0},
            {"Loan No": "L2", "Unit": "MAHAD", "Closing Arrears": 90_000.0},
            {"Loan No": "L3", "Unit": "MAHAD", "Closing Arrears": 0.0},
            {"Loan No": "L4", "Unit": "PUNE", "Closing Arrears": 200_000.0},
        ]
        return make_df(rows)

    def test_sorted_worst_first_excludes_zero_arrears(self):
        result = top_closing_arrears(self._df(), top_n=None)
        assert list(result["Loan No"]) == ["L4", "L2", "L1"]

    def test_top_n_caps_the_result(self):
        result = top_closing_arrears(self._df(), top_n=2)
        assert list(result["Loan No"]) == ["L4", "L2"]

    def test_scope_narrows_to_one_branch(self):
        result = top_closing_arrears(self._df(), top_n=None, scope_col="branch", scope_value="MAHAD")
        assert list(result["Loan No"]) == ["L2", "L1"]

    def test_empty_df_returns_empty(self):
        assert top_closing_arrears(pd.DataFrame()).empty


class TestPriorityMenu:
    """"Ask me which category to focus on first, don't just dump every
    priority loan at once" -- matches the user's own described workflow and
    their real operational priority email, which shows a count per category
    (Non Starter, NPA, High Closing Arrears, Fleet Owners, Co-lending,
    Insurance-Only, Easy Settlement, Recent Advances, No Collection 3M)
    rather than one combined dump. Each category's count is computed
    INDEPENDENTLY (a loan CAN appear in more than one category's count) --
    deliberately NOT deduplicated to one tier per loan the way
    priority_accounts/execute_priority_mode's "highest tier only" rule
    works, because the real email this mirrors doesn't dedupe across
    sections either (each of its tables is its own independent filter)."""

    def _df(self):
        rows = [
            {"Loan No": "L1", "Unit": "MAHAD", "Non Starter": "Y", "VehEMI Accrued": 1,
             "curr_bucket": "STD", "Closing Arrears": 0.0, "Arrears / EMI": 0.0},
            {"Loan No": "L2", "Unit": "MAHAD", "Non Starter": "N", "VehEMI Accrued": 1,
             "curr_bucket": "NPA", "Closing Arrears": 90_000.0, "Arrears / EMI": 8.0},
            {"Loan No": "L3", "Unit": "MAHAD", "Non Starter": "N", "VehEMI Accrued": 1,
             "curr_bucket": "STD", "Closing Arrears": 0.0, "Arrears / EMI": 0.0},
        ]
        return make_df(rows)

    def test_returns_one_row_per_nonzero_category_with_a_count(self):
        result = priority_menu(self._df())
        categories = dict(zip(result["Category"], result["Count"]))
        assert categories.get("Non Starter") == 1
        assert categories.get("NPA Accounts") == 1
        assert categories.get("High Closing Arrears") == 1

    def test_zero_count_categories_are_excluded(self):
        result = priority_menu(self._df())
        assert "Co-lending at Risk" not in set(result["Category"])

    def test_shown_is_capped_at_the_categorys_default_but_never_exceeds_count(self):
        result = priority_menu(self._df())
        row = result[result["Category"] == "Non Starter"].iloc[0]
        assert row["Shown"] == 1  # only 1 actual match, even though the default cap is higher

    def test_scope_narrows_to_one_branch(self):
        rows = self._df().to_dict("records")
        rows.append({"Loan No": "L4", "Unit": "PUNE", "Non Starter": "Y", "VehEMI Accrued": 1,
                      "curr_bucket": "STD", "Closing Arrears": 0.0, "Arrears / EMI": 0.0})
        result = priority_menu(make_df(rows), scope_col="branch", scope_value="MAHAD")
        row = result[result["Category"] == "Non Starter"].iloc[0]
        assert row["Count"] == 1  # L4 (PUNE) excluded

    def test_empty_df_returns_empty(self):
        assert priority_menu(pd.DataFrame()).empty


class TestWorstLoansByMetric:
    """The real gap in the region->branch->executive->loan drill chain:
    the LAST hop (executive -> loan) used to always apply the SAME generic
    "not paying" threshold (non_paying_customers), regardless of WHICH
    metric was actually in focus -- "why does Rahul have so much NPA%" and
    "why is Rahul's strike rate low" drilled into the identical list. This
    maps the metric already in focus to its REGISTERED CONCEPT (reusing
    concept_filter's own compiler.core._expand_filters/
    agents.data_executor._build_mask machinery directly -- never a second,
    independently-drifting filter definition), sorted by severity
    (Arrears / EMI descending) so the worst offenders surface first."""

    def _df(self):
        rows = [
            {"MNT NAME": "RAHUL", "Unit": "PUNE", "Cust Name": "Cust NPA High", "curr_bucket": "NPA", "Arrears / EMI": 10.0, "Strike": "N"},
            {"MNT NAME": "RAHUL", "Unit": "PUNE", "Cust Name": "Cust NPA Low", "curr_bucket": "NPA", "Arrears / EMI": 4.0, "Strike": "N"},
            {"MNT NAME": "RAHUL", "Unit": "PUNE", "Cust Name": "Cust Clean", "curr_bucket": "STD", "Arrears / EMI": 0.0, "Strike": "Y"},
            {"MNT NAME": "SUNIL", "Unit": "NAGPUR", "Cust Name": "Cust Other Branch NPA", "curr_bucket": "NPA", "Arrears / EMI": 20.0, "Strike": "N"},
        ]
        return make_df(rows)

    def test_npa_metric_returns_npa_loans_sorted_worst_first(self):
        result = worst_loans_by_metric(self._df(), "NPA %", scope_col="executive", scope_value=("RAHUL", "PUNE"))
        assert list(result["Cust Name"]) == ["Cust NPA High", "Cust NPA Low"]

    def test_scoped_to_executive_excludes_other_executives_even_with_same_metric(self):
        result = worst_loans_by_metric(self._df(), "NPA %", scope_col="executive", scope_value=("RAHUL", "PUNE"))
        assert "Cust Other Branch NPA" not in set(result["Cust Name"])

    def test_hard_bucket_metric_uses_the_hard_bucket_concept(self):
        rows = [
            {"MNT NAME": "RAHUL", "Unit": "PUNE", "Cust Name": "High Arrears", "Arrears / EMI": 8.0},
            {"MNT NAME": "RAHUL", "Unit": "PUNE", "Cust Name": "Low Arrears", "Arrears / EMI": 1.0},
        ]
        result = worst_loans_by_metric(make_df(rows), "Hard Bucket %")
        assert list(result["Cust Name"]) == ["High Arrears"]

    def test_strike_rate_metric_returns_non_struck_loans(self):
        rows = [
            {"MNT NAME": "RAHUL", "Unit": "PUNE", "Cust Name": "Not Struck", "Strike": "N", "SOH": 100.0},
            {"MNT NAME": "RAHUL", "Unit": "PUNE", "Cust Name": "Struck", "Strike": "Y", "SOH": 50.0},
        ]
        result = worst_loans_by_metric(make_df(rows), "Strike Rate %")
        assert list(result["Cust Name"]) == ["Not Struck"]

    def test_unmapped_metric_returns_empty(self):
        result = worst_loans_by_metric(self._df(), "SOH (Cr)")
        assert result.empty

    def test_no_space_metric_spelling_also_works(self):
        result = worst_loans_by_metric(self._df(), "NPA%", scope_col="executive", scope_value=("RAHUL", "PUNE"))
        assert len(result) == 2

    def test_scope_by_branch_still_works_with_a_single_column_dimension(self):
        result = worst_loans_by_metric(self._df(), "NPA %", scope_col="branch", scope_value="NAGPUR")
        assert list(result["Cust Name"]) == ["Cust Other Branch NPA"]

    def test_empty_df_returns_empty(self):
        assert worst_loans_by_metric(pd.DataFrame(), "NPA %").empty

    def test_all_columns_returns_every_raw_column_not_the_curated_subset(self):
        # Same lever non_paying_customers/customer_loan_book/priority_accounts
        # already have -- a raw upload can carry 85+ columns; the curated
        # _CUSTOMER_DISPLAY_COLS subset (~13) is the sensible default, not
        # the only option.
        result = worst_loans_by_metric(
            self._df(), "NPA %", scope_col="executive", scope_value=("RAHUL", "PUNE"), all_columns=True,
        )
        assert "curr_bucket" in result.columns
        assert "Strike" in result.columns
        assert list(result["Cust Name"]) == ["Cust NPA High", "Cust NPA Low"]


class TestSummarizeWorstLoans:
    """The deterministic (never LLM-improvised) top-offender summary --
    same reasoning as investigator/suggestions.py's mechanism suggestions:
    a factual claim about which specific rows are worst must never have
    LLM variance in it."""

    def _df(self):
        return make_df([
            {"Cust Name": "Cust A", "Arrears / EMI": 10.0},
            {"Cust Name": "Cust B", "Arrears / EMI": 7.0},
            {"Cust Name": "Cust C", "Arrears / EMI": 5.0},
            {"Cust Name": "Cust D", "Arrears / EMI": 1.0},
        ])

    def test_names_top_3_offenders_by_arrears_emi_descending(self):
        summary = summarize_worst_loans(self._df())
        assert "Cust A" in summary
        assert "Cust B" in summary
        assert "Cust C" in summary
        assert "Cust D" not in summary

    def test_independent_of_the_caller_s_own_sort_order(self):
        shuffled = self._df().sample(frac=1, random_state=1).reset_index(drop=True)
        summary = summarize_worst_loans(shuffled)
        assert summary.index("Cust A") < summary.index("Cust B") < summary.index("Cust C")

    def test_empty_df_returns_empty_string(self):
        assert summarize_worst_loans(pd.DataFrame()) == ""

    def test_missing_required_columns_returns_empty_string(self):
        assert summarize_worst_loans(make_df([{"Cust Name": "A"}])[["Cust Name"]]) == ""


class TestPriorityAccounts:
    """The 7-tier priority framework ("what should my team work on today"),
    reusing agents/data_executor.py::execute_priority_mode directly -- the
    SAME framework the main AI Query pipeline's priority-mode routing
    already uses, never a reimplementation. This was wrongly assumed to
    have no drill-down value; checking the actual output (loan-level, with
    Cust Mob No/MNT NAME/Unit on every row) showed otherwise, so this
    reuses customer_loan_book's existing drill unchanged."""

    def _df(self):
        rows = [
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "RegionName": "WEST",
             "Cust Mob No": "1111111111", "curr_bucket": "NPA"},
            {"MNT NAME": "RAJ", "Unit": "MAHAD", "RegionName": "WEST",
             "Cust Mob No": "2222222222", "curr_bucket": "STD"},
            {"MNT NAME": "SUNIL", "Unit": "PUNE", "RegionName": "WEST",
             "Cust Mob No": "3333333333", "curr_bucket": "NPA"},
        ]
        return make_df(rows)

    def test_returns_loan_level_rows_with_cust_mob_no(self):
        result = priority_accounts(self._df())
        assert "Cust Mob No" in result.columns
        assert set(result["Cust Mob No"]) == {"1111111111", "3333333333"}

    def test_scoped_to_one_branch(self):
        result = priority_accounts(self._df(), scope_col="branch", scope_value="MAHAD")
        assert set(result["Cust Mob No"]) == {"1111111111"}

    def test_empty_df_returns_empty(self):
        assert priority_accounts(make_df([])).empty

    def test_all_columns_returns_every_raw_column_not_the_curated_subset(self):
        # execute_priority_mode's own display_cols curates down to ~13
        # columns -- a real user asked for the raw file's full 85+ columns
        # on THIS exact step and had no way to get them (non_paying_customers/
        # customer_loan_book already had this lever; priority_accounts did
        # not). Must still carry the computed Priority tag, just alongside
        # every other original column instead of a curated handful.
        result = priority_accounts(self._df(), all_columns=True)
        assert "Priority" in result.columns
        assert "RegionName" in result.columns
        assert set(result["Cust Mob No"]) == {"1111111111", "3333333333"}
        # More columns than the curated display set (Priority, Loan No,
        # Cust Name, Cust Mob No, RegionName, Unit, MNT NAME, Ag_Date, ...).
        assert len(result.columns) > 10

    def test_all_columns_true_but_no_priority_matches_returns_empty(self):
        rows = [{"MNT NAME": "CLEAN", "Unit": "MAHAD", "curr_bucket": "STD"}]
        assert priority_accounts(make_df(rows), all_columns=True).empty


class TestPriorityByExecutive:
    def test_ranks_executives_by_priority_case_count_most_first(self):
        rows = []
        for name, unit, n_npa, n_std in [("RAJ", "MAHAD", 1, 4), ("SUNIL", "PUNE", 3, 2)]:
            for _ in range(n_npa):
                rows.append({"MNT NAME": name, "Unit": unit, "curr_bucket": "NPA"})
            for _ in range(n_std):
                rows.append({"MNT NAME": name, "Unit": unit, "curr_bucket": "STD"})
        df_curr = make_df(rows)
        flagged = priority_accounts(df_curr)
        result = priority_by_executive(df_curr, flagged)
        assert result.iloc[0]["MNT NAME"] == "SUNIL"  # 3 priority cases, the most
        assert result.iloc[0]["Accounts"] == 3

    def test_empty_priority_df_returns_empty(self):
        assert priority_by_executive(make_df([{"Loan No": "L1"}]), make_df([])).empty

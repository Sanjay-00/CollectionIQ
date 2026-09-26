import io

import pandas as pd

from analysis.root_cause import (
    clean_contaminated_flags,
    compute_recent_advances_bucket_summary,
    compute_recent_advances_region_status,
    compute_recent_advances_status_by_grain,
    load_daily_missed_feed,
)


def _master():
    rows = [
        ("L1", "PUNE", "BR1", "Amit Rao", "2025-12-01"),
        ("L2", "PUNE", "BR1", "Amit Rao", "2026-01-01"),
        ("L3", "PUNE", "BR2", "Sunil Jha", "2026-02-01"),
        ("L4", "CS NAGAR", "BR3", "Amit Rao", "2025-11-15"),   # same name, DIFFERENT branch
        ("L5", "CS NAGAR", "BR3", "Amit Rao", "2025-11-20"),
        ("OLD", "PUNE", "BR1", "Amit Rao", "2025-06-01"),
    ]
    return pd.DataFrame(rows, columns=["Loan No", "RegionName", "Unit", "MNT NAME", "Ag_Date"])


def _feed():
    return pd.DataFrame({
        "LOAN NO": ["L1", "L2", "L4", "L5"],
        "REGIONNAME": ["PUNE", "PUNE", "CHHATRAPATI SAMBHAJI NAGAR", "CHHATRAPATI SAMBHAJI NAGAR"],
        "UNIT": ["BR1", "BR1", "BR3", "BR3"],
        "RE NAME": ["Amit Rao ", "Amit Rao ", "Amit Rao ", "Amit Rao "],
        "ARREARS / EMI": [0.5, 2.0, 3.0, 2.99],
    })


class TestRegionStatus:
    def test_counts_and_percentages(self):
        t = compute_recent_advances_region_status(_master(), _feed()).set_index("Region")
        assert t.loc["PUNE", "Total Nov'25 Onward Cases"] == 3        # OLD loan excluded from cohort
        assert t.loc["PUNE", "Delinquent Cases"] == 2
        assert t.loc["PUNE", "Delinquent %"] == 66.67
        assert t.loc["PUNE", "PNPA (SMA-2) Cases"] == 1                # exactly 2.0 is PNPA
        assert t.loc["PUNE", "NPA Cases"] == 0

    def test_pnpa_npa_boundaries(self):
        t = compute_recent_advances_region_status(_master(), _feed()).set_index("Region")
        assert t.loc["CS NAGAR", "NPA Cases"] == 1                     # exactly 3.0 is NPA
        assert t.loc["CS NAGAR", "PNPA (SMA-2) Cases"] == 1            # 2.99 is PNPA

    def test_feed_region_alias_maps_to_master_name(self):
        t = compute_recent_advances_region_status(_master(), _feed())
        assert "CHHATRAPATI SAMBHAJI NAGAR" not in t["Region"].tolist()
        assert t.set_index("Region").loc["CS NAGAR", "Delinquent Cases"] == 2

    def test_grand_total_row(self):
        t = compute_recent_advances_region_status(_master(), _feed()).set_index("Region")
        g = t.loc["Grand Total"]
        assert g["Total Nov'25 Onward Cases"] == 5
        assert g["Delinquent Cases"] == 4
        assert g["Delinquent %"] == 80.0

    def test_region_missing_from_master_has_blank_total_and_pct(self):
        feed = pd.concat([_feed(), pd.DataFrame({"LOAN NO": ["X1"], "REGIONNAME": ["THANE"], "ARREARS / EMI": [1.0]})])
        t = compute_recent_advances_region_status(_master(), feed).set_index("Region")
        assert t.loc["THANE", "Delinquent Cases"] == 1
        assert pd.isna(t.loc["THANE", "Total Nov'25 Onward Cases"])
        assert pd.isna(t.loc["THANE", "Delinquent %"])

    def test_duplicate_feed_loans_counted_once(self):
        feed = pd.concat([_feed(), _feed()])
        t = compute_recent_advances_region_status(_master(), feed).set_index("Region")
        assert t.loc["PUNE", "Delinquent Cases"] == 2

    def test_feed_missing_required_column_returns_empty(self):
        assert compute_recent_advances_region_status(_master(), _feed().drop(columns=["ARREARS / EMI"])).empty


class TestStatusByGrain:
    def test_branch_grain_counts(self):
        t = compute_recent_advances_status_by_grain(_master(), _feed(), grain="branch").set_index("Branch")
        assert t.loc["BR1", "Total Nov'25 Onward Cases"] == 2   # OLD excluded
        assert t.loc["BR1", "Delinquent Cases"] == 2
        assert t.loc["BR3", "NPA Cases"] == 1
        assert t.loc["BR3", "PNPA (SMA-2) Cases"] == 1

    def test_executive_grain_uses_composite_key_not_name_alone(self):
        t = compute_recent_advances_status_by_grain(_master(), _feed(), grain="executive")
        rows = t[t["Executive"] == "AMIT RAO"]
        # Same name, two different branches -> two separate rows, not merged into one.
        assert len(rows) == 2
        by_branch = rows.set_index("Branch")
        assert by_branch.loc["BR1", "Total Nov'25 Onward Cases"] == 2
        assert by_branch.loc["BR1", "Delinquent Cases"] == 2
        assert by_branch.loc["BR3", "Total Nov'25 Onward Cases"] == 2
        assert by_branch.loc["BR3", "NPA Cases"] == 1

    def test_executive_feed_name_whitespace_still_matches_master(self):
        # _feed()'s "RE NAME" has trailing whitespace ("Amit Rao "); master's
        # "MNT NAME" does not -- canonicalization (strip+upper) must still match,
        # not create a separate unmatched "AMIT RAO " row alongside "AMIT RAO".
        t = compute_recent_advances_status_by_grain(_master(), _feed(), grain="executive")
        assert (t["Executive"] == "AMIT RAO").sum() == 2  # BR1 row + BR3 row, no stray whitespace variant
        assert t[t["Executive"] == "Grand Total"]["Delinquent Cases"].iloc[0] == 4

    def test_sorted_worst_first(self):
        t = compute_recent_advances_status_by_grain(_master(), _feed(), grain="branch")
        data_rows = t[t["Branch"] != "Grand Total"]
        pct = data_rows["Delinquent %"].tolist()
        assert pct == sorted(pct, reverse=True)

    def test_unknown_grain_raises(self):
        import pytest
        with pytest.raises(KeyError):
            compute_recent_advances_status_by_grain(_master(), _feed(), grain="planet")


class TestLoadDailyFeed:
    def _xlsx(self, sheets: dict) -> io.BytesIO:
        buf = io.BytesIO()
        buf.name = "feed.xlsx"
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name, index=False)
        buf.seek(0)
        return buf

    def test_skips_leading_pivot_sheet(self):
        pivot = pd.DataFrame({"Row Labels": ["PUNE"], "Count of LOAN NO": [2]})
        f = self._xlsx({"Sheet2": pivot, "Sheet1": _feed()})
        df, err = load_daily_missed_feed(f)
        assert err is None and len(df) == 4 and "LOAN NO" in df.columns

    def test_no_loan_column_returns_error(self):
        df, err = load_daily_missed_feed(self._xlsx({"Sheet1": pd.DataFrame({"a": [1]})}))
        assert df is None and "Loan No" in err


class TestRecentBucketsAndCleaning:
    def test_bucket_summary_uses_arrears_emi_thresholds(self):
        df = pd.DataFrame({
            "Loan No": list("abcdef"), "Ag_Date": ["2026-01-01"] * 6,
            "Arrears / EMI": [0, 0.5, 1.0, 2.0, 3.0, None], "SOH": [1e7] * 6,
        })
        b = compute_recent_advances_bucket_summary(df).set_index("Bucket")
        assert b["Count"].to_dict() == {"STD": 1, "0-1": 1, "1-2": 1, "2-3": 1, "NPA": 1}

    def test_contaminated_flag_values_become_null_and_are_counted(self):
        df = pd.DataFrame({"Non Starter": ["Yes", "No", "NEMCV", None], "CoLending_Loans": ["Y", "ALIVE", "N", "N"]})
        cleaned, contamination = clean_contaminated_flags(df)
        assert contamination == {"Non Starter": 1, "CoLending_Loans": 1}
        assert pd.isna(cleaned.loc[2, "Non Starter"]) and cleaned.loc[0, "Non Starter"] == "Yes"

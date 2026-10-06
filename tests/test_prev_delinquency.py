"""Last month's Delinquency % beside this month's, per region / branch / executive."""
import pandas as pd

from utils import attach_prev_delinquency, compute_delinquency_pct


def _book(rows):
    # rows: (loan, region, unit, exec, arrears/emi)
    return pd.DataFrame(rows, columns=["Loan No", "RegionName", "Unit", "MNT NAME", "Arrears / EMI"])


PREV = _book([
    ("L1", "PUNE", "KARAD", "RAHUL PATIL", 1.0),
    ("L2", "PUNE", "KARAD", "RAHUL PATIL", 0.0),
    ("L3", "PUNE", "MAN", "RAHUL PATIL", 0.0),        # same name, other branch
    ("L4", "THANE ", "VASAI", "ANIL SHINDE", 2.0),
])


def test_matches_the_single_source_rule_and_computes_the_change():
    table = pd.DataFrame({"Region": ["PUNE", "Thane"], "Delinquency%": [50.0, 0.0]})
    out = attach_prev_delinquency(table, PREV, {"Region": "RegionName"})
    assert list(out.columns) == ["Region", "Delinquency%", "Prev Delinquency%", "Δ Delinquency%"]
    assert out["Prev Delinquency%"].iloc[0] == compute_delinquency_pct(PREV[PREV["RegionName"] == "PUNE"])
    assert out["Prev Delinquency%"].iloc[1] == 100.0          # "THANE " and "Thane" are one region
    assert out["Δ Delinquency%"].tolist() == [round(50 - 100 / 3, 2), -100.0]


def test_no_previous_file_adds_no_columns():
    table = pd.DataFrame({"Region": ["PUNE"], "Delinquency%": [50.0]})
    assert list(attach_prev_delinquency(table, PREV.iloc[0:0], {"Region": "RegionName"}).columns) == list(table.columns)
    assert list(attach_prev_delinquency(table, None, {"Region": "RegionName"}).columns) == list(table.columns)


def test_executive_is_matched_within_branch_and_survives_name_truncation():
    table = pd.DataFrame({"MNT NAME": ["RAHUL PAT", "RAHUL PATIL", "NEW JOINER"],
                          "Unit": ["KARAD", "MAN", "KARAD"], "Delinquency %": [0.0, 0.0, 10.0]})
    out = attach_prev_delinquency(table, PREV, {"MNT NAME": "MNT NAME", "Unit": "Unit"},
                                  pct_col="Delinquency %", fuzzy="MNT NAME")
    prev = out["Prev Delinquency %"].tolist()
    assert prev[0] == 50.0 and prev[1] == 0.0                 # KARAD book vs MAN book, not merged
    assert pd.isna(prev[2]) and pd.isna(out["Δ Delinquency %"].iloc[2])   # no history: blank, not 0


def _month(arrears):
    n = len(arrears)
    return pd.DataFrame({
        "Loan No": [f"L{i}" for i in range(n)], "RegionName": ["PUNE"] * n, "Unit": ["KARAD"] * n,
        "MNT NAME": ["RAHUL PATIL"] * n, "Arrears / EMI": arrears,
        "curr_bucket": ["STD" if a <= 0 else "1-30 DPD" for a in arrears],
    })


def test_every_region_branch_executive_table_carries_it():
    from analysis.executive_scorecard import compute_executive_scorecard
    from analysis.portfolio_intelligence import (
        compute_branch_quadrant, compute_npa_sma2_comparison, compute_region_scorecard,
    )
    curr, prev = _month([0, 0, 1, 1] * 5), _month([0, 0, 0, 1] * 5)   # 50% now, 25% last month
    assert compute_region_scorecard(curr, prev)["Δ Delinquency%"].iloc[0] == 25.0
    assert compute_branch_quadrant(curr, prev)[0]["Prev Delinquency%"].iloc[0] == 25.0
    assert compute_executive_scorecard(curr, min_accounts=1, df_prev=prev)["Prev Delinquency %"].iloc[0] == 25.0
    for table in compute_npa_sma2_comparison(curr, prev).values():
        assert table[["Delinquent", "Delinquency%", "Prev Delinquency%", "Δ Delinquency%"]].iloc[0].tolist() == [10, 50.0, 25.0, 25.0]


def test_root_cause_region_diagnosis_reuses_the_scorecard_figures():
    from analysis.portfolio_intelligence import compute_region_scorecard
    from analysis.root_cause import compute_region_why_table
    curr, prev = _month([0, 0, 1, 1] * 5), _month([0, 0, 0, 1] * 5)
    for df in (curr, prev):
        df["Cust Mob No"] = [f"9000000{i:03d}" for i in range(len(df))]
    why = compute_region_why_table(curr, compute_region_scorecard(curr, prev))
    assert why[["Delinquency%", "Prev Delinquency%", "Δ Delinquency%"]].iloc[0].tolist() == [50.0, 25.0, 25.0]
    no_prev = compute_region_why_table(curr, compute_region_scorecard(curr, prev.iloc[0:0]))
    assert "Prev Delinquency%" not in no_prev.columns


def test_unit_table_shows_change_and_counts_on_screen():
    import ui.tabs.portfolio_intelligence as pi
    from analysis.summary import unit_table
    table = unit_table(_month([0, 0, 1, 1] * 5), _month([0, 0, 0, 1] * 5), "Region")
    html = pi.units_table_html(table, "Region")
    assert "Change vs last month" in html and "▲ 25.00 pts" in html     # got worse: red up arrow
    assert "50.0% <span" in html and "(10)</span>" in html               # the % with its count in one cell


def test_root_cause_branch_tables_get_delinquency_and_last_month():
    from analysis.root_cause import add_unit_delinquency, compute_chronic_shock_split, compute_insurance_split
    curr, prev = _month([0, 0, 1, 1] * 5), _month([0, 0, 0, 1] * 5)
    curr = curr.assign(**{"ARREARS AGAINST INST": curr["Arrears / EMI"], "ARREARS AGAINST EXP": 0,
                          "No Coll 3 Months and >6 EMI": "N"})
    for split in (compute_insurance_split(curr), compute_chronic_shock_split(curr)):
        assert not split.empty
        out = add_unit_delinquency(split, curr, prev)
        at = list(out.columns).index("Delinquent Accounts")
        assert list(out.columns)[at - 1:at + 4] == ["Accounts", "Delinquent Accounts", "Delinquency%",
                                                    "Prev Delinquency%", "Δ Delinquency%"]
        assert out["Accounts"].iloc[0] == 20                          # the base behind the %
        assert out[["Delinquency%", "Prev Delinquency%", "Δ Delinquency%"]].iloc[0].tolist() == [50.0, 25.0, 25.0]
        assert "Prev Delinquency%" not in add_unit_delinquency(split, curr, None).columns


def test_total_row_delinquency_is_all_delinquent_over_all_accounts():
    from ui.components import append_total_row
    from ui.tabs.root_cause import _DELQ_RATIO
    t = pd.DataFrame({"Unit": ["A", "B"], "Accounts": [10, 490], "Delinquent Accounts": [8, 290],
                      "Delinquency%": [80.0, 59.18]})
    total = append_total_row(t, ratio_cols=_DELQ_RATIO).iloc[-1]
    assert total["Delinquency%"] == round(298 / 500 * 100, 2)          # 59.6, not the 69.6 average


def test_unit_tables_read_accounts_delinquent_pct_prev_then_buckets():
    from analysis.executive_scorecard import compute_executive_scorecard
    from analysis.portfolio_intelligence import compute_branch_quadrant, compute_region_scorecard
    curr, prev = _month([0, 0, 1, 1] * 5), _month([0, 0, 0, 1] * 5)
    lead = lambda cols, first: [c for c in cols if c in first]
    reg = list(compute_region_scorecard(curr, prev).columns)
    assert reg[:9] == ["Region", "Accounts", "Delinquent", "Delinquency%", "Prev Delinquency%",
                       "Δ Delinquency%", "SMA-2", "SMA-2%", "Δ SMA-2%"]
    br = list(compute_branch_quadrant(curr, prev)[0].columns)
    assert br[3:8] == ["Accounts", "Delinquent", "Delinquency%", "Prev Delinquency%", "Δ Delinquency%"]
    ex = list(compute_executive_scorecard(curr, min_accounts=1, df_prev=prev).columns)
    assert ex[3:8] == ["Accounts", "Delinquent", "Delinquency %", "Prev Delinquency %", "Δ Delinquency %"]
    assert ex.index("SMA-2") < ex.index("NPA") < ex.index("Collection %")

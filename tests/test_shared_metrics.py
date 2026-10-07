"""One definition per metric (utils.loan_flags / unit_metrics), so every
table agrees, and every threshold comes from config.py."""
import pandas as pd
import pytest

import config
from helpers import make_df
from utils import insurance_only_mask, unit_metrics

T = config.INSURANCE_EXP_ARREARS_MIN


def _book():
    """Two regions, three branches, four executives (two named RAHUL)."""
    rows = []
    spec = [  # region, branch, exec, n loans, how many NPA, how many SMA-2, how many 1-30 DPD
        ("WEST", "MAHAD", "RAHUL", 10, 2, 1, 2),
        ("WEST", "PEN", "RAHUL", 10, 0, 2, 1),
        ("EAST", "KARAD", "ANIL", 20, 5, 1, 4),
        ("EAST", "KARAD", "SUNIL", 8, 1, 0, 0),
    ]
    i = 0
    for region, unit, exe, n, npa, sma2, dpd in spec:
        for k in range(n):
            bucket, ae = ("NPA", 4.0) if k < npa else ("SMA-2", 2.5) if k < npa + sma2 else \
                ("1-30 DPD", 0.5) if k < npa + sma2 + dpd else ("STD", 0.0)
            rows.append({"Loan No": f"L{i}", "RegionName": region, "Unit": unit, "MNT NAME": exe,
                         "curr_bucket": bucket, "Arrears / EMI": ae, "SOH": 100_000.0 * (1 + k % 3)})
            i += 1
    return make_df(rows)


def test_every_region_branch_executive_table_uses_the_same_numbers():
    from analysis.executive_scorecard import compute_executive_scorecard
    from analysis.portfolio_intelligence import (
        compute_branch_quadrant, compute_npa_sma2_comparison, compute_region_scorecard,
    )
    df = _book()
    ref = unit_metrics(df, ["RegionName"]).set_index("RegionName")
    reg = compute_region_scorecard(df, df.iloc[0:0]).set_index("Region")
    for col in ("Accounts", "Delinquent", "Delinquency%", "NPA", "NPA%", "NPA% (SOH)", "SMA-2%"):
        assert reg[col].to_dict() == ref[col].to_dict(), col

    branch_ref = unit_metrics(df, ["Unit"]).set_index("Unit")
    quad = compute_branch_quadrant(df)[0].set_index("Branch")
    cmp = compute_npa_sma2_comparison(df, df.iloc[0:0])
    cmp_branch = cmp["branch"].set_index("Unit")
    for b in branch_ref.index:
        assert quad.at[b, "NPA%"] == branch_ref.at[b, "NPA%"]
        assert cmp_branch.at[b, "Delinquency%"] == quad.at[b, "Delinquency%"] == branch_ref.at[b, "Delinquency%"]

    sc = compute_executive_scorecard(df).set_index("Executive (Branch)")
    ce = cmp["executive"].set_index("MNT NAME")
    for name in sc.index:                                   # same people, same counts
        assert sc.at[name, "NPA"] == ce.at[name, "NPA (Curr)"]
        assert sc.at[name, "Delinquent"] == ce.at[name, "Delinquent"]


def test_same_name_in_two_branches_is_two_executives_everywhere(monkeypatch):
    import analysis.portfolio_intelligence as pi
    from analysis.executive_scorecard import compute_executive_scorecard
    monkeypatch.setattr(pi, "MIN_ACCOUNTS_OVERDUE_DEMAND_EXECUTIVE", 1)   # its own stricter floor
    from analysis.portfolio_intelligence import compute_npa_sma2_comparison, compute_overdue_demand_scorecard
    from analysis.new_business import compute_new_advances_by_dimension
    df = _book().assign(Ag_Date=pd.Timestamp("2026-08-05"))
    assert set(compute_executive_scorecard(df)["Executive (Branch)"]) >= {"RAHUL (MAHAD)", "RAHUL (PEN)"}
    assert set(compute_npa_sma2_comparison(df, df.iloc[0:0])["executive"]["MNT NAME"]) >= {"RAHUL (MAHAD)", "RAHUL (PEN)"}
    od = compute_overdue_demand_scorecard(df)["executive"]
    na = compute_new_advances_by_dimension(df, as_of="2026-08-01")["executive"]
    for t in (od, na):
        rahul = t[t["Executive"] == "RAHUL"]
        assert sorted(rahul["Branch"]) == ["MAHAD", "PEN"]    # not merged into one row


def test_every_executive_table_uses_one_minimum():
    from analysis.executive_scorecard import compute_executive_scorecard
    from analysis.portfolio_intelligence import compute_npa_sma2_comparison
    assert config.SCORECARD_MIN_ACCOUNTS == config.MIN_ACCOUNTS_EXECUTIVE
    df = _book()
    small = df[df["MNT NAME"] == "SUNIL"].head(config.MIN_ACCOUNTS_EXECUTIVE - 1)
    df = pd.concat([df[df["MNT NAME"] != "SUNIL"], small])
    assert "SUNIL (KARAD)" not in set(compute_executive_scorecard(df)["Executive (Branch)"])
    assert "SUNIL (KARAD)" not in set(compute_npa_sma2_comparison(df, df.iloc[0:0])["executive"]["MNT NAME"])


def test_npa_pct_by_soh_is_money_not_count():
    df = make_df([
        {"Loan No": "A", "curr_bucket": "NPA", "Arrears / EMI": 4.0, "SOH": 900_000.0},
        {"Loan No": "B", "curr_bucket": "STD", "SOH": 100_000.0},
    ])
    m = unit_metrics(df, []).iloc[0]
    assert m["NPA%"] == 50.0 and m["NPA% (SOH)"] == 90.0
    assert m["NPA SOH (Cr)"] == 0.09                     # the money behind the %, in crore


def test_percentages_round_ties_the_same_way_everywhere():
    # 478 / 8000 = 5.975 exactly: Python's round() (used by _safe_pct) and
    # pandas' .round() disagree on this tie; every table must agree.
    from utils import _safe_pct
    df = make_df([{"Loan No": f"L{i}", "curr_bucket": "SMA-2" if i < 478 else "STD"} for i in range(8000)])
    assert unit_metrics(df, []).iloc[0]["SMA-2%"] == _safe_pct(478, 8000)


class TestInsuranceOnly:
    def _df(self):
        return make_df([
            {"Loan No": "A", "Arrears / EMI": 0.4, "ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": T + 1},
            {"Loan No": "B", "Arrears / EMI": 0.4, "ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": T},      # not over
            {"Loan No": "C", "Arrears / EMI": 0.4, "ARREARS AGAINST INST": 900.0, "ARREARS AGAINST EXP": T + 1},
            {"Loan No": "D", "Arrears / EMI": 0.4, "ARREARS AGAINST INST": 900.0, "ARREARS AGAINST EXP": 0.0},
            {"Loan No": "E", "Arrears / EMI": 0.0, "ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": T + 1},  # not behind
        ])

    def test_threshold_comes_from_config_and_is_3000(self):
        assert T == 3_000
        assert insurance_only_mask(self._df()).tolist() == [True, False, False, False, False]

    def test_card_alert_and_root_cause_count_the_same_loans(self):
        from analysis.portfolio_intelligence import compute_pulse_kpis
        from analysis.root_cause import compute_insurance_split
        from smart_alerts import alert_insurance_delinquency
        df = self._df()
        card = next(k for k in compute_pulse_kpis(df, make_df([])) if k["label"] == "Insurance Debit Cases")
        split = compute_insurance_split(df)
        assert card["value"] == "1" and alert_insurance_delinquency(df)["count"] == 1
        assert int(split["Insurance-Only"].sum()) == 1

    def test_split_groups_add_up_to_every_delinquent_loan(self):
        from analysis.root_cause import compute_insurance_split
        row = compute_insurance_split(self._df()).iloc[0]
        parts = row["Insurance-Only"] + row["Installment-Only"] + row["Both"] + row["Other"]
        assert parts == row["Delinquent Accounts"] == 4


def test_repossession_list_skips_seized_and_sold():
    from analysis.exposure import compute_repossession_list
    assert "S&S" in config.REPOSSESSION_EXCLUDE_STATUSES
    df = make_df([
        {"Loan No": "RUN1", "curr_bucket": "NPA", "Loan Status": "RUN", "Ag_Date": pd.Timestamp("2026-01-01")},
        {"Loan No": "SS1", "curr_bucket": "NPA", "Loan Status": "S&S", "Ag_Date": pd.Timestamp("2026-01-01")},
        {"Loan No": "STD1", "curr_bucket": "STD", "Loan Status": "RUN", "Ag_Date": pd.Timestamp("2026-01-01")},
    ])
    assert compute_repossession_list(df, as_of="2026-08-01")["Loan No"].tolist() == ["RUN1"]


class TestConcernScore:
    def test_uses_the_not_paying_rate_not_the_count(self):
        from analysis.portfolio_intelligence import _concern_score
        # Same rate (10%), very different counts: the component must rank them equal.
        df = pd.DataFrame({"NPA%": [5.0, 5.0], "Hard Bucket%": [1.0, 1.0], "Roll Fwd%": [2.0, 2.0],
                           "Not Paying 3M+": [50, 1], "Not Paying 3M+%": [10.0, 10.0]})
        s = _concern_score(df)
        assert s.iloc[0] == s.iloc[1]
        assert "Not Paying 3M+%" in config.CONCERN_SCORE_WEIGHTS

    def test_missing_roll_rate_is_left_out_not_scored_best(self):
        from analysis.portfolio_intelligence import _concern_score
        df = pd.DataFrame({"NPA%": [9.0, 1.0], "Hard Bucket%": [9.0, 1.0],
                           "Roll Fwd%": [float("nan"), 5.0], "Not Paying 3M+%": [9.0, 1.0]})
        s = _concern_score(df)
        assert s.iloc[0] == 100               # worst on every part it has data for
        assert s.iloc[0] > s.iloc[1]


def test_no_jargon_on_screen():
    # "Chronic"/"shock" were replaced by plain descriptions of what the data shows.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    files = [root / "app.py", *sorted((root / "ui").rglob("*.py"))]
    import re
    hits = [f"{f.name}:{i}" for f in files
            for i, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines(), 1)
            if not line.strip().startswith("#")
            for text in re.findall(r'"[^"]*"' + "|'[^']*'", line)          # string literals only
            if re.search(r"(chronic|shock)", text, re.IGNORECASE)]
    assert hits == []


def test_tables_survive_an_empty_view():
    # Sidebar filters can leave no loans; every table must render empty, not crash.
    import analysis.portfolio_intelligence as pi
    from analysis.executive_scorecard import compute_executive_scorecard
    empty = _book().iloc[0:0]
    assert unit_metrics(empty, ["RegionName"]).empty
    from analysis.exposure import compute_concentration_treemap
    compute_concentration_treemap(empty)
    pi.compute_branch_quadrant(empty)
    pi.compute_npa_sma2_comparison(empty, empty)
    assert compute_executive_scorecard(empty).empty

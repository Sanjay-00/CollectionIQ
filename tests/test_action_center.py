"""Action overview (analysis/action_center.py): what changed, what to fix
first, where, and who to call."""
import pandas as pd
import pytest

import analysis.action_center as ac
import config


def _book(n_std=40, n_new=10, region_new="NASHIK", branch_new="MALEGAON"):
    """A portfolio where `n_new` of the loans that were STD last month are
    behind now, all in one branch, plus a few loans in every other bucket."""
    rows, i = [], 0

    def add(region, branch, exec_, prev, now, soh=100_000.0, **kw):
        nonlocal i
        ae = {"STD": 0.0, "1-30 DPD": 0.5, "SMA-1": 1.5, "SMA-2": 2.5, "NPA": 4.0}[now]
        rows.append({"Loan No": f"L{i}", "RegionName": region, "Unit": branch, "MNT NAME": exec_,
                     "prev_bucket": prev, "curr_bucket": now, "prev_SOH": soh, "SOH": soh, "Arrears / EMI": ae,
                     "Net Collection Demand Inst+Exp+BC": 10_000.0,
                     "Month Collection (Excluding Reserve Collection)": 10_000.0, **kw})
        i += 1

    for k in range(n_std):
        add("PUNE", "BARAMATI", "ANIL", "STD", "STD")
    for k in range(n_new):
        add(region_new, branch_new, "RAHUL", "STD", "1-30 DPD")
    for k in range(30):                                # the hotspot branch's healthy loans
        add(region_new, branch_new, "RAHUL", "STD", "STD")
    for prev, now in (("1-30 DPD", "SMA-1"), ("SMA-1", "SMA-2"), ("SMA-2", "NPA"), ("SMA-2", "SMA-2"),
                      ("1-30 DPD", "STD"), ("NPA", "NPA")):
        add("PUNE", "BARAMATI", "ANIL", prev, now)
    add("PUNE", "BARAMATI", "ANIL", "STD", "1-30 DPD", **{"ARREARS AGAINST INST": 0.0, "ARREARS AGAINST EXP": 5_000.0})
    return pd.DataFrame(rows)


def _prev(df):
    return df.assign(curr_bucket=df["prev_bucket"], **{"Arrears / EMI": 0.0})


def test_new_defaulters_headline_says_how_many_and_where():
    df = _book()
    heads = ac.headlines(df, _prev(df))
    new = next(h for h in heads if "new defaulters" in h["title"])
    assert new["tone"] == "bad" and new["title"].startswith("11 new defaulters")
    assert "NASHIK" in new["detail"]                          # names keep their case
    assert "worst branch MALEGAON" in new["detail"]


def test_good_news_and_quick_wins_always_get_a_slot(monkeypatch):
    monkeypatch.setattr(ac, "ACTION_HEADLINES_MAX", 3)
    heads = ac.headlines(_book(), _prev(_book()))
    tones = [h["tone"] for h in heads]
    assert tones.count("good") == 1 and tones.count("info") == 1 and len(heads) == 3


def test_pulse_compares_with_last_month():
    df = _book()
    pulse = {x["label"]: x for x in ac.pulse(df, _prev(df))}
    assert pulse["Delinquency"]["change"] > 0 and pulse["Delinquency"]["worse"] is True
    assert ac.pulse(df, df.iloc[0:0])[0]["change"] is None


def test_outlook_is_todays_loans_times_this_months_rates():
    df = _book()
    rates = ac.transition_rates(df)
    out = ac.next_month_outlook(df)
    std_now = (df["curr_bucket"] == "STD").sum()
    assert out["new_defaulters"]["n"] == round(std_now * rates["STD"]["worse"])
    assert out["new_defaulters"]["low"] <= out["new_defaulters"]["n"] <= out["new_defaulters"]["high"]
    assert ac.next_month_outlook(df.drop(columns=["prev_bucket"])) is None


def test_focus_first_order_and_if_ignored():
    focus = ac.focus_first(_book())
    names = [f["name"] for f in focus]
    assert names[:2] == ["Quick wins: insurance charge only", "New defaulters"]
    new = focus[1]
    assert new["loans"] == 11 and new["risk"]["n"] >= 0


def test_without_last_month_focus_uses_todays_position_only():
    focus = ac.focus_first(_book().drop(columns=["prev_bucket"]))
    assert "New defaulters" not in [f["name"] for f in focus]
    assert all(f["risk"] is None for f in focus)


def test_call_lists_show_the_columns_that_prove_each_group():
    df = _book()
    lists = {t["name"]: t["loans"] for t in ac.call_lists(df)}
    quick = lists["Quick wins: insurance charge only"]
    for col in ("ARREARS AGAINST INST", "ARREARS AGAINST EXP", "Arrears / EMI", "SOH", "Also In"):
        assert col in quick.columns, col
    assert (quick["ARREARS AGAINST INST"] <= 0).all() and (quick["ARREARS AGAINST EXP"] > ac.INSURANCE_EXP_ARREARS_MIN).all()
    assert quick["Also In"].iloc[0] == "New defaulters"            # the quick win is also a new defaulter
    new = lists["New defaulters"]
    assert {"Last Month Bucket", "Bucket Now"} <= set(new.columns)
    assert new["SOH"].is_monotonic_decreasing


def test_hotspot_branch_is_flagged_with_reasons_in_words():
    df = _book()
    att = ac.attention(df, _prev(df), "branch")
    top = att["worst"][0]
    assert top["name"] == "MALEGAON" and top["region"] == "NASHIK"
    assert any("STD → Behind" in r and "x the average" in r for r in top["reasons"])


def test_small_units_are_not_judged(monkeypatch):
    monkeypatch.setitem(ac.ACTION_MIN_ACCOUNTS, "branch", 10_000)
    df = _book()
    assert ac.attention(df, _prev(df), "branch")["worst"] == []


def test_thresholds_live_in_config():
    for name in ("ACTION_HEADLINES_MAX", "ACTION_TOP_N", "ACTION_MIN_ACCOUNTS", "ACTION_HOTSPOT_MULTIPLE",
                 "ACTION_MIN_ROLLED", "ACTION_RISING_PP", "ACTION_COLLECTION_GAP_PP", "ACTION_RANGE_Z"):
        assert hasattr(config, name)


def test_build_runs_on_an_empty_view():
    empty = _book().iloc[0:0]
    r = ac.build(empty, empty)
    assert r["headlines"] == [] and r["calls"] == [] and r["outlook"] is None

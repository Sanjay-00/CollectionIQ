"""metric_change: the 30+ rate both months and how its change splits (spec 7.2)."""
import pandas as pd
import pytest

from investigation.evidence import EvidenceStore, ToolError
from investigation.panel import make_pair
from investigation.tools import metric_change


def _month(buckets, prefix="L", unit="B1"):
    return pd.DataFrame({"Loan No": [f"{prefix}{i}" for i in range(len(buckets))], "curr_bucket": buckets,
                         "Loan Status": "RUN", "Unit": unit})


def _pair(prev, curr):
    return make_pair(curr, prev, "2026-09", "2026-08")


def test_rate_change_and_its_two_parts():
    prev = _month(["SMA-1"] * 2 + ["STD"] * 8 + ["NA"])                # 2 of 10 eligible, 1 no EMI yet
    curr = _month(["NPA"] * 5 + ["STD"] * 15, prefix="C")             # 5 of 20
    e = metric_change(EvidenceStore(), _pair(prev, curr))
    s = e.summary
    assert e.tool == "metric_change" and e.checks_passed
    assert s["prev"]["rate"] == pytest.approx(20.0) and s["curr"]["rate"] == pytest.approx(25.0)
    assert s["change_pp"] == pytest.approx(5.0) and s["change_rel_pct"] == pytest.approx(25.0)
    assert s["numerator_effect_pp"] == pytest.approx(15.0) and s["base_effect_pp"] == pytest.approx(-10.0)
    assert s["prev"]["rate_all_running"] == pytest.approx(200 / 11)
    assert (s["prev"]["running"], s["prev"]["eligible"], s["prev"]["numerator"]) == (11, 10, 2)
    assert s["label"].startswith("30+") and s["version"] == 1


def test_no_emi_loans_are_outside_the_base():
    prev = _month(["SMA-1"] * 2 + ["STD"] * 8)
    curr = _month(["SMA-1"] * 2 + ["STD"] * 8)
    with_new = pd.concat([curr, _month(["NA"] * 100, prefix="N")], ignore_index=True)
    a = metric_change(EvidenceStore(), _pair(prev, curr)).summary
    b = metric_change(EvidenceStore(), _pair(prev, with_new)).summary
    assert a["curr"]["rate"] == b["curr"]["rate"] == pytest.approx(20.0)
    assert b["curr"]["rate_all_running"] == pytest.approx(200 / 110)
    assert b["all_running_change_pp"] < 0 and b["change_pp"] == pytest.approx(0.0)


def test_scope_new_this_month_is_refused():
    prev = _month(["STD"] * 5)
    curr = pd.concat([_month(["STD"] * 5), _month(["NPA"] * 3, prefix="H", unit="HINGL")], ignore_index=True)
    with pytest.raises(ToolError, match="Aug 2026"):
        metric_change(EvidenceStore(), _pair(prev, curr), scope={"branch": "HINGL"})


def test_flat_previous_rate_has_no_relative_change():
    s = metric_change(EvidenceStore(), _pair(_month(["STD"] * 4), _month(["NPA", "STD", "STD", "STD"]))).summary
    assert s["change_rel_pct"] is None and s["change_pp"] == pytest.approx(25.0)


def test_a_change_inside_its_noise_band_is_not_material():
    # Spec 9.3: material = above INVESTIGATION_NO_CHANGE_PP AND outside the
    # total's own paired noise band. A 150-loan branch wobbles by points.
    from investigation.synthetic import make_book
    prev, curr = make_book(6000, seed=63, churn=0.1)             # TINYBR moves -3.2 points on churn alone
    pair = _pair(prev, curr)
    tiny = metric_change(EvidenceStore(), pair, scope={"branch": "TINYBR"}).summary
    assert abs(tiny["change_pp"]) >= 0.2 and tiny["noise_pp"] > abs(tiny["change_pp"]) and tiny["material"] is False
    whole = metric_change(EvidenceStore(), _pair(*make_book(30000, seed=62, churn=0.06, rate_up=("TRACTOR", 6)))).summary
    assert whole["material"] is True and whole["noise_pp"] < abs(whole["change_pp"])


def test_scope_narrows_both_months():
    prev = pd.concat([_month(["NPA"] * 2, unit="A"), _month(["STD"] * 2, prefix="X", unit="B")], ignore_index=True)
    s = metric_change(EvidenceStore(), _pair(prev, prev), scope={"branch": "A"}).summary
    assert s["prev"]["eligible"] == 2 and s["prev"]["rate"] == pytest.approx(100.0)

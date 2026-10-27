"""scan_dimensions: every dimension's breakdown, ranked by where the change sits (spec 7.5)."""
from investigation.evidence import EvidenceStore
from investigation.panel import make_pair
from investigation.tools import scan_dimensions
from tests.investigation.books import make_book


def _scan(prev, curr):
    store = EvidenceStore()
    return store, scan_dimensions(store, make_pair(curr, prev, "2026-09", "2026-08"))


def test_planted_product_mix_ranks_product_first():
    store, e = _scan(*make_book(6000, seed=1, product_share=3.0))
    first = e.summary["ranked"][0]
    assert e.summary["material"] is True and e.tool == "scan_dimensions"
    assert first["dimension"] == "product" and first["top"][0]["segment"] == "TRACTOR"
    assert first["coverage_pct"] >= 60 and first["mix_pp"] > first["rate_pp"]
    # A coarse dimension (few big segments) must not outrank the cause just by size.
    assert first["concentration_pct"] > 2 * e.summary["ranked"][1]["concentration_pct"]
    assert e.checks_passed


def test_planted_branch_shift_ranks_branch_first():
    _, e = _scan(*make_book(6000, seed=1, shift={"BR1": 120}))
    assert e.summary["ranked"][0]["dimension"] == "branch"
    assert e.summary["ranked"][0]["top"][0]["segment"] == "BR1"


def test_unavailable_dimensions_are_listed_not_dropped():
    prev, curr = make_book(3000, seed=2, product_share=2.0)
    _, e = _scan(prev.drop(columns="FUEL_TYPE"), curr.drop(columns="FUEL_TYPE"))
    assert {"dimension": "fuel", "reason": "no FUEL_TYPE column in last month's file"} in e.summary["unavailable"]
    assert "fuel" not in [r["dimension"] for r in e.summary["ranked"]]


def test_flat_total_ranks_by_largest_segment_move():
    prev, curr = make_book(6000, seed=3, shift={"BR1": 40, "BR2": -40})
    _, e = _scan(prev, curr)
    assert e.summary["material"] is False and abs(e.summary["total_change_pp"]) < 1e-9
    assert all(r["coverage_pct"] is None for r in e.summary["ranked"])
    assert e.summary["ranked"][0]["dimension"] == "branch"


def test_planted_cause_outranks_many_noisy_branches():
    # Review finding: with ordinary month-to-month churn, 146 branches each
    # wobbling a little summed to more "concentration" than the planted cause.
    _, e = _scan(*make_book(40000, seed=31, n_branches=146, churn=0.06, rate_up=("TRACTOR", 2.5)))
    assert e.summary["material"] is True
    assert e.summary["ranked"][0]["dimension"] == "product"
    branch = next(r for r in e.summary["ranked"] if r["dimension"] == "branch")
    assert branch["concentration_pct"] < 0.5 * e.summary["ranked"][0]["concentration_pct"]


def test_relabelled_dimension_is_set_aside_not_ranked():
    _, e = _scan(*make_book(6000, seed=22, product_share=3.0, relabel=0.3))
    ranked = [r["dimension"] for r in e.summary["ranked"]]
    reason = next(u["reason"] for u in e.summary["unavailable"] if u["dimension"] == "channel")
    assert "channel" not in ranked and ranked[0] == "product"
    # 30% of this month's loans, picked at random: about 30% of the loans in both months.
    pct = float(reason.split("for ")[1].split("%")[0])
    assert reason.startswith("labels changed for") and 25 < pct < 35


def test_each_breakdown_is_citable():
    store, e = _scan(*make_book(3000, seed=4, product_share=2.0))
    for r in e.summary["ranked"]:
        assert store.get(r["evidence_id"]).tool == "mix_rate"
        assert store.get(r["evidence_id"]).summary["dimension"] == r["dimension"]


def test_drill_only_dimensions_are_not_scanned():
    _, e = _scan(*make_book(3000, seed=5, product_share=2.0))
    scanned = {r["dimension"] for r in e.summary["ranked"]} | {u["dimension"] for u in e.summary["unavailable"]}
    assert "executive" not in scanned and "dealer" not in scanned

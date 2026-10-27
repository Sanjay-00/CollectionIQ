"""The scorer: one check per thing an investigation must get right."""
import pandas as pd

from investigation.evaluation import Driver, Expected, InvestigationResult, score
from investigation.evidence import Check, EvidenceStore


def _breakdown(store, dimension, segments, checks=None):
    return store.add("mix_rate", {"dimension": dimension}, {"dimension": dimension},
                     {"segments": pd.DataFrame({"segment": segments})}, checks)


def _store():
    store = EvidenceStore()
    _breakdown(store, "product", ["TRACTOR", "CAR"])                               # E1
    _breakdown(store, "product", ["TRACTOR"], [Check("components_add_up", False)])  # E2: failed its checks
    _breakdown(store, "executive", ["EXEC7 (BR3)"])                                 # E3
    _breakdown(store, "branch", ["BR1", "BR3"])                                     # E4
    store.add("metric_change", {}, {})                                              # E5: not a breakdown
    return store


EXPECTED = Expected(status="concluded", drivers=[("product", "TRACTOR")], forbidden=[("branch", "BR1")],
                    mechanism="rate", deep=[("executive", "EXEC7 (BR3)")], flags=["labels_changed"],
                    untestable=["fuel"], excluded_units=["HINGL"])


def _perfect(**changes):
    r = InvestigationResult(status="concluded", drivers=[Driver("product", "TRACTOR", "E1")], mechanism="rate",
                            deep=[Driver("executive", "EXEC7 (BR3)", "E3")], flags=["labels_changed", "units_new"],
                            untestable=["fuel"], excluded_units=["HINGL"])
    for k, v in changes.items():
        setattr(r, k, v)
    return r


def test_a_perfect_result_passes_every_check():
    s = score("c1", EXPECTED, _perfect(), _store())
    assert s.passed and all(s.checks.values()) and s.case_id == "c1"


def test_each_wrong_field_fails_exactly_its_own_check():
    wrong = {
        "status": {"status": "no_material_change"},
        "drivers": {"drivers": [Driver("product", "CAR", "E1")]},
        "forbidden": {"drivers": [Driver("product", "TRACTOR", "E1"), Driver("branch", "BR1", "E4")]},
        "mechanism": {"mechanism": "mix"},
        "deep": {"deep": []},
        "flags": {"flags": []},
        "untestable": {"untestable": []},
        "excluded_units": {"excluded_units": []},
    }
    for check, change in wrong.items():
        s = score("c", EXPECTED, _perfect(**change), _store())
        assert not s.passed and [k for k, ok in s.checks.items() if not ok] == [check], check


def test_citing_a_result_that_failed_its_checks_fails():
    s = score("c", EXPECTED, _perfect(drivers=[Driver("product", "TRACTOR", "E2")]), _store())
    assert s.checks["evidence_valid"] is False
    s = score("c", EXPECTED, _perfect(drivers=[Driver("product", "TRACTOR", "E99")]), _store())
    assert s.checks["evidence_valid"] is False


def test_evidence_must_be_the_breakdown_that_shows_the_driver():
    # Review: any existing id used to pass, so a guess could cite E1.
    for wrong_id in ("E4", "E5"):          # a branch breakdown; a metric change
        s = score("c", EXPECTED, _perfect(drivers=[Driver("product", "TRACTOR", wrong_id)]), _store())
        assert s.checks["evidence_valid"] is False, wrong_id


def test_a_no_story_case_fails_an_invented_driver():
    exp = Expected(status="no_material_change", no_drivers=True)
    assert score("c", exp, _perfect(status="no_material_change", drivers=[], deep=[]), _store()).passed
    assert not score("c", exp, _perfect(status="no_material_change"), _store()).checks["no_drivers"]


def test_an_expected_driver_found_one_level_deeper_counts():
    # Review: "region R1, then branch BR3, then EXEC7" is a right answer to case 13.
    exp = Expected(status="concluded", drivers=[("branch", "BR3")])
    r = _perfect(drivers=[Driver("product", "TRACTOR", "E1")], deep=[Driver("branch", "BR3", "E4")])
    assert score("c", exp, r, _store()).checks["drivers"]


def test_a_case_may_accept_more_than_one_status():
    exp = Expected(status=("concluded", "inconclusive"), mechanism="flow")
    assert score("c", exp, _perfect(status="inconclusive", mechanism="flow"), _store()).checks["status"]
    assert not score("c", exp, _perfect(status="no_material_change", mechanism="flow"), _store()).checks["status"]


def test_labels_compare_without_case():
    s = score("c", EXPECTED, _perfect(drivers=[Driver("Product", "tractor", "E1")]), _store())
    assert s.checks["drivers"] and s.checks["evidence_valid"]


def test_any_driver_of_needs_one_of_them():
    exp = Expected(status="concluded", any_driver_of=[("vintage", "2026-Q1"), ("product", "TRACTOR")])
    assert score("c", exp, _perfect(), _store()).checks["any_driver_of"]
    assert not score("c", exp, _perfect(drivers=[Driver("branch", "BR3", "E4")]), _store()).checks["any_driver_of"]


def test_unexpected_fields_are_not_judged():
    s = score("c", Expected(status="no_material_change"), _perfect(status="no_material_change"), _store())
    assert s.passed and set(s.checks) == {"status", "forbidden", "evidence_valid"}

"""The 30+ metric (spec section 5) and the evidence store every tool writes to."""
import pandas as pd
import pytest

from investigation.evidence import Check, EvidenceStore, ToolError
from investigation.metrics import get_metric

BUCKETS = ["STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA", "NA"]


def test_thirty_plus_counts_one_or_more_emis_overdue():
    m = get_metric("thirty_plus_rate")
    df = pd.DataFrame({"curr_bucket": BUCKETS, "Loan Status": ["RUN"] * 6})
    assert int(m.eligible(df).sum()) == 5            # NA (no EMI due yet) is outside the base
    assert int(m.numerator(df).sum()) == 3           # SMA-1, SMA-2, NPA
    assert int(m.running(df).sum()) == 6


def test_non_running_loans_are_not_eligible():
    m = get_metric("thirty_plus_rate")
    df = pd.DataFrame({"curr_bucket": ["NPA", "NPA"], "Loan Status": ["RUN", "CLOSED"]})
    assert list(m.running(df)) == [True, False]
    assert list(m.eligible(df)) == [True, False]
    assert list(m.numerator(df)) == [True, False]


def test_file_without_loan_status_counts_every_row_as_running():
    m = get_metric("thirty_plus_rate")
    assert bool(m.running(pd.DataFrame({"curr_bucket": ["STD"]})).all())


def test_unknown_metric_is_refused():
    with pytest.raises(ToolError, match="thirty_plus_rate"):
        get_metric("npa_rate")


def test_store_numbers_results_in_order():
    store = EvidenceStore()
    e1 = store.add("metric_change", {"metric_id": "thirty_plus_rate"}, {"change_pp": 0.78})
    e2 = store.add("flow_bridge", {}, {})
    assert (e1.result_id, e2.result_id) == ("E1", "E2")
    assert store.get("E2").tool == "flow_bridge"
    assert [e.result_id for e in store.all()] == ["E1", "E2"]
    with pytest.raises(ToolError):
        store.get("E9")


def test_one_failed_check_marks_the_result():
    store = EvidenceStore()
    e = store.add("mix_rate", {}, {}, checks=[Check("a", True), Check("b", False, "x")])
    assert e.checks_passed is False
    assert store.add("x", {}, {}).checks_passed is True

"""flow_bridge: last month's 30+ loans to this month's, every path counted (spec 7.3)."""
import numpy as np
import pandas as pd

from investigation.evidence import EvidenceStore
from investigation.metrics import ELIGIBLE_BUCKETS
from investigation.panel import make_pair
from investigation.tools import flow_bridge, metric_change

# loan: (bucket last month, bucket this month); None = not in that month's file
PATHS = {
    "A": ("STD", "SMA-1"), "B": ("1-30 DPD", "NPA"), "C": ("NA", "SMA-2"), "D": (None, "SMA-1"),
    "E": ("SMA-1", "STD"), "F": ("SMA-2", "1-30 DPD"), "G": ("NPA", "NA"), "H": ("NPA", None),
    "I": ("SMA-1", "SMA-2"),
}


def _book(paths, owners=None):
    owners = owners or {}

    def month(i):
        rows = [{"Loan No": k, "curr_bucket": v[i], "Loan Status": "RUN", "Unit": "B1",
                 "MNT NAME": owners.get(k, ("RAJ", "RAJ"))[i]} for k, v in paths.items() if v[i] is not None]
        return pd.DataFrame(rows)
    return make_pair(month(1), month(0), "2026-09", "2026-08")


def test_bridge_lines_on_a_hand_built_book():
    e = flow_bridge(EvidenceStore(), _book(PATHS))
    lines = e.summary["lines"]
    assert list(lines) == ["opening", "in_from_std", "in_from_1_30", "in_from_no_emi", "new_to_book",
                           "cured_to_std", "cured_to_1_30", "became_no_emi", "left_book", "closing"]
    assert lines["opening"] == 5 and lines["closing"] == 5                 # I stays 30+: in neither in nor out
    assert all(lines[k] == 1 for k in lines if k not in ("opening", "closing"))
    assert e.summary["left_book_last_bucket"] == {"NPA": 1}
    assert e.summary["base_lines"] == {"opening": 7, "new_to_book": 1, "in_from_no_emi": 1,
                                       "became_no_emi": 1, "left_book": 1, "closing": 7}
    assert e.tool == "flow_bridge" and e.checks_passed


def test_closing_equals_metric_change_numerator():
    rng = np.random.default_rng(7)
    buckets = list(ELIGIBLE_BUCKETS) + ["NA"]
    paths = {}
    for i in range(2000):
        prev = rng.choice(buckets + [None], p=[.5, .1, .1, .05, .1, .05, .1])
        curr = rng.choice(buckets + [None], p=[.5, .1, .1, .05, .1, .05, .1])
        if prev is not None or curr is not None:
            paths[f"L{i}"] = (prev, curr)
    pair = _book(paths)
    store = EvidenceStore()
    bridge, change = flow_bridge(store, pair), metric_change(store, pair)
    lines, base = bridge.summary["lines"], bridge.summary["base_lines"]
    assert (lines["opening"], lines["closing"]) == (change.summary["prev"]["numerator"], change.summary["curr"]["numerator"])
    assert (base["opening"], base["closing"]) == (change.summary["prev"]["eligible"], change.summary["curr"]["eligible"])
    assert bridge.checks_passed


def test_by_a_missing_column_is_refused_plainly():
    import pytest
    from investigation.evidence import ToolError
    with pytest.raises(ToolError, match="no FUEL_TYPE column"):
        flow_bridge(EvidenceStore(), _book(PATHS), by="fuel")


def test_by_executive_uses_todays_owner():
    paths = {"X": ("STD", "SMA-1"), "Y": ("NPA", None), "Z": ("SMA-1", "SMA-1")}
    owners = {"X": ("RAJ", "AMIT"), "Y": ("RAJ", None), "Z": ("RAJ", "RAJ")}
    e = flow_bridge(EvidenceStore(), _book(paths, owners), by="executive")
    by = e.tables["by"].set_index("segment")
    assert by.loc["AMIT (B1)", "in_from_std"] == 1 and by.loc["AMIT (B1)", "closing"] == 1
    assert by.loc["Left the running book", "left_book"] == 1
    assert by.loc["RAJ (B1)", "opening"] == 1 and by.loc["RAJ (B1)", "closing"] == 1
    assert e.checks_passed and any(c.name == "each_segment_closes" and c.passed for c in e.checks)

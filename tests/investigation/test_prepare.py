"""prepare: compare like for like when a branch is in one file only (user decision 2026-10-24)."""
import pytest

from investigation.evidence import EvidenceStore, ToolError
from investigation.panel import make_pair
from investigation.prepare import prepare
from investigation.tools import metric_change
from tests.investigation.books import make_book


def _prep(prev, curr, scope=None):
    store = EvidenceStore()
    return store, prepare(store, make_pair(curr, prev, "2026-09", "2026-08"), scope=scope)


def test_branch_missing_from_last_months_file_is_left_out_of_both_months():
    # The real Sept 2026 fault: HINGL in this month's file only, its loans agreed long ago.
    store, p = _prep(*make_book(6000, seed=40, add_branch=("HINGL", 0.03, "2024-05-01")))
    assert p.status == "ok" and p.adjustment is not None
    assert "HINGL" not in set(p.pair.curr["Unit"]) and "HINGL" not in set(p.pair.prev["Unit"])
    s = p.adjustment.summary
    assert [e["unit"] for e in s["excluded"]] == ["HINGL"] and s["excluded"][0]["loans_curr"] == 180
    assert s["full_change_pp"] != s["like_for_like_change_pp"]
    assert s["like_for_like_change_pp"] == pytest.approx(metric_change(store, p.pair).summary["change_pp"])
    assert p.health.summary["status"] != "halt"          # the re-check on the like-for-like pair


def test_branch_missing_this_month_is_left_out_too():
    _, p = _prep(*make_book(6000, seed=41, drop_branch="BR3"))
    assert p.status == "ok" and [e["unit"] for e in p.adjustment.summary["excluded"]] == ["BR3"]
    assert "BR3" not in set(p.pair.prev["Unit"])


def test_other_halts_still_halt():
    _, p = _prep(*make_book(3000, seed=42, grow=0.2))
    assert p.status == "halted_data" and p.adjustment is None


def test_clean_book_needs_no_adjustment():
    _, p = _prep(*make_book(3000, seed=43))
    assert p.status == "ok" and p.adjustment is None


def test_a_truncated_extract_is_not_made_like_for_like():
    # Review: six of ten branches missing this month was "fixed" by leaving
    # them out and investigating a third of the book as if all were well.
    prev, curr = make_book(20000, seed=45)
    curr = curr[~curr["Unit"].isin([f"BR{i}" for i in range(6)])]
    _, p = _prep(prev, curr)
    assert p.status == "halted_data" and p.adjustment is None


def test_a_renamed_branch_is_a_move_not_a_gap():
    # Review: TINYBR's loans now sit under BR0. Nothing is missing, so nothing
    # is left out (it used to record "TINYBR left out, 0 / 0 loans").
    prev, curr = make_book(6000, seed=46)
    curr.loc[curr["Unit"] == "TINYBR", "Unit"] = "BR0"
    store, p = _prep(prev, curr)
    f = next(f for f in p.files.summary["findings"] if f["check"] == "units_missing")
    assert f["status"] == "info" and "TINYBR" in f["detail"] and "BR0" in f["detail"] and f["units"] == []
    assert p.adjustment is None and p.status == "ok"


def test_like_for_like_keeps_the_whole_file_check():
    # Review: the coverage flags live in the whole-file check; the result must carry it.
    _, p = _prep(*make_book(6000, seed=40, add_branch=("HINGL", 0.03, "2024-05-01")))
    assert p.files.tool == "data_health" and p.files is not p.health
    assert any(f["check"] == "units_new" and f["status"] == "halt" for f in p.files.summary["findings"])


def test_a_scope_that_is_the_left_out_branch_is_refused():
    prev, curr = make_book(6000, seed=44, add_branch=("HINGL", 0.03, "2024-05-01"))
    with pytest.raises(ToolError, match="HINGL"):
        _prep(prev, curr, scope={"branch": "HINGL"})

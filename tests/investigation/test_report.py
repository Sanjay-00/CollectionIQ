"""The plain report built from a result and its evidence (spec 11)."""
import re

import pytest

from investigation.baseline import run_baseline
from investigation.eval_cases import CASES, CURR_MONTH, PREV_MONTH
from investigation.report import build_report

CAUSAL = re.compile(r"\b(caused by|due to|driven by|because of|led to|resulted in)\b", re.I)


def _run(case_id):
    prev, curr = next(c for c in CASES if c.id == case_id).build()
    result, store = run_baseline(curr, prev, CURR_MONTH, PREV_MONTH)
    return result, store, build_report(result, store)


@pytest.fixture(scope="module")
def adjusted():
    return _run(17)          # branch missing from last month's file + TRACTOR rate rise


def test_headline_numbers_are_the_evidence(adjusted):
    result, store, rep = adjusted
    change = next(e for e in store.all() if e.tool == "metric_change").summary
    assert f"{change['prev']['rate']:.2f}%" in rep["headline"] and f"{change['curr']['rate']:.2f}%" in rep["headline"]
    assert f"{change['change_pp']:+.2f}" in rep["headline"] and "TRACTOR" in rep["headline"]


def test_like_for_like_is_said_before_any_finding(adjusted):
    _, _, rep = adjusted
    md = rep["markdown"]
    assert "HINGL" in rep["adjustment"] and md.index("HINGL") < md.index("Where the change sits")


def test_no_causal_wording_and_no_dashes(adjusted):
    for case_id in (1, 2, 8, 14):
        md = _run(case_id)[2]["markdown"]
        assert not CAUSAL.search(md) and "–" not in md and "—" not in md, case_id
    assert not CAUSAL.search(adjusted[2]["markdown"])


def test_where_section_gives_each_drivers_share_of_the_change(adjusted):
    result, _, rep = adjusted
    assert "TRACTOR" in rep["where"] and "% of the change" in rep["where"]


def test_a_halted_investigation_reports_the_data_problem_only():
    _, _, rep = _run(18)
    assert rep["headline"].startswith("Stopped") and "Where the change sits" not in rep["markdown"]


def test_no_material_change_says_so_without_a_story():
    _, _, rep = _run(6)
    assert "within the normal" in rep["headline"] and "Where the change sits" not in rep["markdown"]


def test_next_steps_point_to_action_lists(adjusted):
    assert "Action Lists" in adjusted[2]["next_steps"]

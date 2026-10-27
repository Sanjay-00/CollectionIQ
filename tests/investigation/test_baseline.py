"""The fixed-script baseline, scored on every evaluation case (spec 8, 12).
Cases the baseline is not expected to pass are xfail(strict) with the reason:
they are what the Phase 4 agent must win."""
import pytest

from investigation.baseline import run_baseline
from investigation.eval_cases import CASES, CURR_MONTH, PREV_MONTH
from investigation.evaluation import score

BASELINE_MISSES: dict[int, str] = {
    11: "two causes: the fixed script names the top dimension's driver (branch BR2) only; the second cause "
        "(product TRACTOR) sits in a dimension below the concentration line once BR2 absorbs part of the change",
}


def _params():
    for c in CASES:
        marks = [pytest.mark.xfail(strict=True, reason=BASELINE_MISSES[c.id])] if c.id in BASELINE_MISSES else []
        yield pytest.param(c, id=f"case{c.id}", marks=marks)


@pytest.mark.parametrize("case", list(_params()))
def test_baseline_on_case(case):
    prev, curr = case.build()
    result, store = run_baseline(curr, prev, CURR_MONTH, PREV_MONTH)
    s = score(str(case.id), case.expected, result, store)
    assert s.passed, {k: v for k, v in s.checks.items() if not v} | {"drivers": [(d.dimension, d.segment) for d in result.drivers],
                                                                    "status": result.status}
    assert result.model_calls == 0 and result.tool_calls > 0

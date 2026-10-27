"""Every evaluation case is fair: its planted cause is visible to the tools."""
import pytest

from investigation.eval_cases import CASES, HOLDOUT_IDS, recoverable


def test_case_ids_are_unique_and_holdouts_fixed():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)) == 19
    assert {c.id for c in CASES if c.expected.status == "halted_data"} == {18, 19}   # review: a never-halting answer must fail
    assert HOLDOUT_IDS == {4, 9, 12, 15, 16}


def test_a_wrong_expectation_is_caught():
    # CAR is notable in case 2 (its share fell) but pulls against the rise: not a driver.
    import dataclasses
    from investigation.evaluation import Expected
    wrong = dataclasses.replace(CASES[1], expected=Expected("concluded", drivers=[("product", "CAR")]))
    assert recoverable(wrong)["drivers"] is False


@pytest.mark.parametrize("case_id, decoy", [
    (3, ("fleet", "1")), (12, ("make", "ASHOK")), (11, ("product", "CAR")), (7, ("product", "CAR")),
    (12, ("make", "RAREMAKE")),       # real but minor: under INVESTIGATION_DRIVER_MIN_SHARE_PCT of the change
])
def test_a_big_segment_that_only_rose_with_everything_is_not_a_driver(case_id, decoy):
    # Review: when the total rises every large segment rises beyond noise; a
    # driver must rise beyond its fair share of the change.
    import dataclasses
    from investigation.evaluation import Expected
    case = next(c for c in CASES if c.id == case_id)
    assert recoverable(dataclasses.replace(case, expected=Expected("concluded", drivers=[decoy])))["drivers"] is False


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"case{c.id}")
def test_case_is_recoverable(case):
    checks = recoverable(case)
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}

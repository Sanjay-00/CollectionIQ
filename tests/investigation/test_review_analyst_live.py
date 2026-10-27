"""Live evaluation of the analyst summary against real Gemini (opt-in:
pytest -m llm_eval). Acceptance (spec section 8): every answer passes the
verifier (after at most one retry), and the recommended region is the same in
at least 9 of 10 runs. The lens for a region on several lists can differ
(sample data: NASHIK shortfall 7, getting worse 3, both defensible); the
region is what the leader acts on."""
import os
from collections import Counter

import pytest
from dotenv import load_dotenv

from investigation.review import runner as R
from investigation.review.analyst import write_analysis
from investigation.review.synthetic import REVIEW_CURR_MONTH as CURR, REVIEW_PREV_MONTH as PREV, make_review_book
from utils import load_and_validate

pytestmark = pytest.mark.llm_eval
RUNS = 10


def _synthetic():
    prev, curr = make_review_book(shortfall={"WEST": 0.85}, risk={"NORTH": 1.6}, worsen={"EAST": 4.0})
    return R.run_review(curr, prev, CURR, PREV)


def _sample():
    with open("sample_data/Current_Month_Demo.xlsx", "rb") as fc, open("sample_data/Previous_Month_Demo.xlsx", "rb") as fp:
        curr, prev = load_and_validate(fc)[0], load_and_validate(fp)[0]
    return R.run_review(curr, prev, "2026-08", "2026-07")


@pytest.fixture(scope="module", params=["synthetic", "sample"])
def runs(request):
    load_dotenv(".env")
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("no GOOGLE_API_KEY")
    brief = {"synthetic": _synthetic, "sample": _sample}[request.param]()
    return [write_analysis(brief) for _ in range(RUNS)]


def test_every_answer_passes_the_checks(runs):
    assert [a.status for a in runs] == ["ok"] * RUNS, [a.problems for a in runs if a.status != "ok"]


def test_the_recommended_region_is_stable(runs):
    regions = Counter((a.recommended_option or "none").rsplit(":", 1)[-1] for a in runs)
    assert regions.most_common(1)[0][1] >= 9, regions

"""The analyst summary (spec 7.2): verifier, retry and fallback, with scripted
model answers (no AI call). The live Gemini eval is test_review_analyst_live.py."""
import json
import re

import pytest

from investigation.review import runner as R
from investigation.review.analyst import analyst_facts, build_prompt, verify, write_analysis
from investigation.review.synthetic import REVIEW_CURR_MONTH as CURR, REVIEW_PREV_MONTH as PREV, make_review_book

PLANTED = dict(shortfall={"WEST": 0.85}, risk={"NORTH": 1.6}, worsen={"EAST": 4.0})


@pytest.fixture(scope="module")
def brief():
    prev, curr = make_review_book(**PLANTED)
    return R.run_review(curr, prev, CURR, PREV)


def _good(brief, **override) -> str:
    """A valid answer: quotes the WEST shortfall exactly as the brief does."""
    west = next(f for f in brief.findings if f.region == "WEST")
    amount = re.search(r"₹[\d.]+[LC]r?", west.text).group(0)
    answer = {
        "summary": f"WEST collected {amount} less than a like-for-like book, and EAST is getting worse.",
        "recommended_option": f"investigate:{west.candidate_id}",
        "why": f"WEST comes first: {amount} of cash this month can be acted on now.",
        "tradeoff": "NORTH's excess 30+ exposure is a slower book-quality fix.",
        "watch": "Check next month whether EAST keeps rising.",
    }
    return json.dumps({**answer, **override}, ensure_ascii=False)


class Scripted:
    """A fake model: returns the given answers in turn and keeps the prompts."""
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TestVerifier:
    def test_a_valid_answer_is_accepted_first_time(self, brief):
        model = Scripted(_good(brief))
        a = write_analysis(brief, model)
        assert a.status == "ok" and a.attempts == 1 and a.recommended_option.startswith("investigate:")
        assert "WEST" in a.why

    def test_an_invented_number_is_retried_with_the_reason(self, brief):
        model = Scripted(_good(brief, why="WEST comes first: about ₹99.99L is at stake."), _good(brief))
        a = write_analysis(brief, model)
        assert a.status == "ok" and a.attempts == 2
        assert "numbers not written in FACTS: 99.99" in model.prompts[1]

    def test_causal_wording_twice_keeps_the_plain_brief(self, brief):
        bad = _good(brief, summary="WEST is low because collections slowed.")
        a = write_analysis(brief, Scripted(bad, bad))
        assert a.status == "fallback" and a.attempts == 2
        assert any("causal wording: because" in p for p in a.problems)

    def test_a_share_claim_the_facts_do_not_state_is_refused(self, brief):
        # Live run: "NORTH ... concentrated in 2023 agreements", when its two
        # largest parts were about a fifth of the total. The facts say "the
        # largest parts are"; a word claiming a share is a number in disguise.
        facts = analyst_facts(brief)
        for word in ("concentrated in", "mostly", "primarily", "the majority of", "most of"):
            answer = json.loads(_good(brief, tradeoff=f"NORTH's excess sits {word} older agreements."))
            assert any("a share the facts don't state" in p for p in verify(answer, facts)), word

    def test_a_share_word_before_in_becomes_the_largest_parts(self, brief):
        # Live run 2: Flash-Lite kept "primarily in STD and 1-30 DPD" through a
        # retry (true there, 90%; "concentrated in 2023 agreements" was a
        # fifth). Rewritten to what the facts say, so neither overclaims.
        a = write_analysis(brief, Scripted(_good(brief, summary="WEST's shortfall sits primarily in STD and 1-30 DPD.",
                                                 tradeoff="NORTH's excess is concentrated in 2023 agreements.")))
        assert a.status == "ok" and a.attempts == 1
        assert a.summary == "WEST's shortfall has its largest parts in STD and 1-30 DPD."
        assert a.tradeoff == "NORTH's excess has its largest parts in 2023 agreements."
        b = write_analysis(brief, Scripted(_good(brief, summary="WEST's ₹30.07L shortfall, mostly in STD, stands out.")))
        assert "shortfall, with the largest parts in STD" in b.summary

    def test_an_option_that_does_not_exist_is_refused(self, brief):
        facts = analyst_facts(brief)
        problems = verify(json.loads(_good(brief, recommended_option="investigate:excess_30plus:MARS")), facts)
        assert any("is not one of" in p for p in problems)

    def test_the_recommendation_must_name_its_region(self, brief):
        facts = analyst_facts(brief)
        answer = json.loads(_good(brief, why="This comes first.", summary="One region stands out."))
        assert any("doesn't name its region (WEST)" in p for p in verify(answer, facts))

    def test_no_recommendation_is_allowed_when_it_is_a_real_choice(self, brief):
        a = write_analysis(brief, Scripted(_good(brief, recommended_option=None,
                                                 why="WEST and NORTH are different kinds of problem.")))
        assert a.status == "ok" and a.recommended_option is None

    def test_not_json_is_retried(self, brief):
        model = Scripted("Sure! Here is my analysis...", _good(brief))
        assert write_analysis(brief, model).status == "ok"
        assert "not a JSON object" in model.prompts[1]

    def test_long_dashes_are_replaced_not_failed(self, brief):
        a = write_analysis(brief, Scripted(_good(brief, watch="Check EAST next month — is it still rising?")))
        assert a.status == "ok" and "—" not in a.watch and " - " in a.watch


class TestWhenNotToCall:
    def test_service_error_keeps_the_plain_brief(self, brief):
        a = write_analysis(brief, Scripted(TimeoutError("slow")))
        assert a.status == "fallback" and "didn't answer" in a.problems[0]

    def test_nothing_to_recommend_means_no_call(self):
        prev, curr = make_review_book(seed=3)
        quiet = R.run_review(curr, prev, CURR, PREV)
        model = Scripted()
        assert write_analysis(quiet, model).status == "skipped" and not model.prompts

    def test_no_api_key_means_no_call(self, brief, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert write_analysis(brief).status == "skipped"


class TestFactsAndPrompt:
    def test_facts_are_the_briefs_own_sentences_and_options(self, brief):
        facts = analyst_facts(brief)
        assert [f["text"] for f in facts["findings"]] == [f.text for f in brief.findings]
        assert {o["id"] for o in facts["options"]} == {o["id"] for o in brief.options if o["id"].startswith("investigate:")}

    def test_no_customer_field_reaches_the_prompt(self):
        prev, curr = make_review_book(**PLANTED)
        curr = curr.assign(**{"Cust Name": "SECRET PERSON", "Cust Mob No": "9999912345"})
        prompt = build_prompt(analyst_facts(R.run_review(curr, prev, CURR, PREV)))
        assert "SECRET PERSON" not in prompt and "9999912345" not in prompt

    def test_the_prompt_states_the_rules(self, brief):
        prompt = build_prompt(analyst_facts(brief))
        for rule in ("Quote every number exactly", "Describe, never explain causes", "Name only regions"):
            assert rule in prompt


class TestRunner:
    def test_the_background_run_adds_the_checked_summary(self, brief):
        prev, curr = make_review_book(**PLANTED)
        steps = []
        out = R.run_review(curr, prev, CURR, PREV, progress=steps.append, analyst=lambda b: write_analysis(b, Scripted(_good(b))))
        assert out.analysis.status == "ok"
        assert steps[-2:] == ["Wrote the brief", "Wrote the analyst summary (AI, checked against the figures)"]

    def test_code_only_by_default(self):
        prev, curr = make_review_book(**PLANTED)
        assert R.run_review(curr, prev, CURR, PREV).analysis is None

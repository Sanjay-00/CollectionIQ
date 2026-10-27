"""
The analyst summary over a finished review (spec 7.2): one Gemini call reads
the brief's facts (region and branch names, figures as the brief already
states them, never a loan row), then writes a short summary, recommends one
of the brief's own "Look deeper" options or explains why two findings are
different kinds of problem, and says what to watch.

The model never computes: a verifier checks every answer (shape, a real
option, every number quoted from the facts, descriptive wording only). One
retry gets the list of failures; a second failure keeps the plain template
brief and says so. Code computes, the model explains, code checks.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from agents.insight_generator import _unverified, _values
from config import PLANNER_TEMPERATURE
from gemini_client import JSONReplyError, generate_json_text, parse_json_object
from investigation.review.brief import LIST_NAMES, Brief
from investigation.review.outcomes import IMPROVED, NO_CLEAR_CHANGE, WORSE as OUTCOME_WORSE

logger = logging.getLogger(__name__)

PROMPT_VERSION = 3   # 2: no share claims (live run 1); 3: last month's outcomes
_KEYS = ("summary", "recommended_option", "why", "tradeoff", "watch")
_CAUSAL = re.compile(r"\b(caus\w*|due to|because|driven by|drives|leads? to|led to|results? in|resulting in|"
                     r"thanks to|owing to|as a result)\b", re.I)
# Words that claim a share of a total ("concentrated in 2023 agreements" for
# a fifth of it, live run 1): a number in disguise the number check can't see.
_SHARE = re.compile(r"\b(concentrat\w*|mostly|mainly|primarily|largely|predominantly|majority|most of|bulk of|"
                    r"almost all|nearly all)\b", re.I)
_MAX_CHARS = {"summary": 600, "why": 450, "tradeoff": 350, "watch": 300}


@dataclass
class Analysis:
    status: str                       # ok | fallback | skipped
    summary: str = ""
    recommended_option: str | None = None
    why: str = ""
    tradeoff: str = ""
    watch: str = ""
    attempts: int = 0
    problems: list = field(default_factory=list)   # the last attempt's failed checks (fallback)
    seconds: float = 0.0
    prompt_version: int = PROMPT_VERSION
    outcomes_seen: str = ""           # outcomes_signature of the last-month outcomes it was written with


def outcomes_signature(outcomes: list | None) -> str:
    """Which last-month outcomes an analysis was written with, to tell when a
    restored decision log has changed them."""
    return "|".join(sorted(f"{o.list_name}:{o.region}:{o.status}" for o in outcomes or []))


def analyst_facts(brief: Brief, outcomes: list | None = None) -> dict:
    """Everything the model may use: the brief's own sentences (their numbers
    are the only numbers it may quote), the options it may recommend, and
    how the regions chosen last month moved (outcomes.track_outcomes)."""
    by_candidate = {o["id"].split(":", 1)[1]: o["id"] for o in brief.options if o["id"].startswith("investigate:")}
    facts = {
        "headline": brief.headline,
        "findings": [{"list": LIST_NAMES[f.list], "region": f.region, "text": f.text,
                      "option_id": by_candidate.get(f.candidate_id)} for f in brief.findings],
        "on_several_lists": brief.overlaps,
        "data_notes": brief.data_notes,
        "options": [o for o in brief.options if o["id"].startswith("investigate:")],
    }
    if outcomes:
        facts["last_month"] = [{"region": o.region, "list": LIST_NAMES[o.list_name], "status": o.status,
                                "handled": o.handled, "text": o.text} for o in outcomes]
    return facts


_LAST_MONTH_RULE = """
LAST MONTH: FACTS.last_month lists the regions the head chose to look into last month and how each moved
since (status: improved, worse, no clear change, not comparable). Say how they moved, in a sentence that
contains the words "last month" and uses exactly the status given. A region chosen last month that is still
on this month's lists with no clear change or worse deserves attention; one that improved can be named as
such. These compare before and after: never say the choice or any action caused the change.
"""


def build_prompt(facts: dict, problems: list[str] | None = None) -> str:
    retry = ""
    if problems:
        retry = ("\nYOUR PREVIOUS ANSWER FAILED THESE CHECKS. Fix every one:\n- " + "\n- ".join(problems) + "\n")
    last_month = _LAST_MONTH_RULE if facts.get("last_month") else ""
    return f"""You are a senior collections analyst at a vehicle-finance lender, briefing the regional business head
on this month's portfolio review. The review was computed in code. You only explain it and recommend where
to look first.

FACTS (JSON, the only source you may use):
{json.dumps(facts, ensure_ascii=False, indent=1)}

How to weigh the lists:
- collection shortfall: cash not collected this month, can be acted on now
- excess 30+ exposure: book quality, larger amounts, a slower fix
- 30+ getting worse: direction of travel; a region getting worse and already on other lists is urgent
- a region on several lists usually comes first
{last_month}
Return ONLY a JSON object with exactly these keys:
{{"summary": "2-3 sentences on what stands out and how the findings fit together",
 "recommended_option": "one option_id from FACTS.options, or null if FACTS.options is empty or two findings are different kinds of problem the head should choose between",
 "why": "1-2 sentences: why this first, compared with the next most important finding",
 "tradeoff": "1 sentence on what the main alternative would address instead, or an empty string",
 "watch": "1 sentence on what to check next month"}}

Rules:
- Quote every number exactly as it is written in FACTS, with the same digits and units. Never calculate,
  round, add, convert or invent a number.
- Describe, never explain causes. Do not use: because, due to, caused, driven by, leads to, results in.
  Use phrases like "sits in", "the largest parts are", "is on two lists".
- Never claim a share of a total the facts don't state (mostly, primarily, concentrated, the majority).
  Quote the largest parts instead.
- Name only regions and branches that appear in FACTS.
- Plain business English, short sentences, no markdown, a single hyphen only (no long dashes).
{retry}"""


_SHARE_WORDS = r"(?:primarily|mostly|mainly|largely|predominantly|concentrated)"
_SHARE_WITH_VERB = re.compile(rf"\b(?:sits|sit|is|are|lies|lie)\s+{_SHARE_WORDS}\s+in\b", re.I)
_SHARE_BARE = re.compile(rf"\b{_SHARE_WORDS}\s+in\b", re.I)


def _clean(text) -> str:
    """Long dashes to a hyphen, and a share word before "in" to what the facts
    say ("the largest parts"): live run 2, Flash-Lite kept "primarily in"
    through a retry, true there (90%) but a fifth in run 1. Rewritten, it is
    accurate either way; any other share claim still fails verify."""
    text = re.sub(r"\s*[—–]\s*", " - ", str(text or "")).strip()
    text = _SHARE_WITH_VERB.sub("has its largest parts in", text)
    return _SHARE_BARE.sub("with the largest parts in", text)


def verify(answer: dict, facts: dict) -> list[str]:
    """Every check an answer must pass; empty means accepted."""
    problems = []
    missing = [k for k in _KEYS if k not in answer]
    if missing:
        return [f"missing keys: {', '.join(missing)}"]
    options = {o["id"] for o in facts["options"]}
    rec = answer["recommended_option"]
    if rec is not None and rec not in options:
        problems.append(f"recommended_option {rec!r} is not one of {sorted(options) or 'the options (there are none)'}")
    if not answer["summary"]:
        problems.append("summary is empty")
    for key, limit in _MAX_CHARS.items():
        if len(answer[key]) > limit:
            problems.append(f"{key} is longer than {limit} characters")
    text = " ".join(answer[k] for k in ("summary", "why", "tradeoff", "watch"))
    allowed = _values(json.dumps(facts, ensure_ascii=False))
    wrong = sorted({v for v in _unverified(text, allowed)})
    if wrong:
        problems.append("numbers not written in FACTS: " + ", ".join(f"{v:g}" for v in wrong))
    causal = sorted({m.group(0).lower() for m in _CAUSAL.finditer(text)})
    if causal:
        problems.append("causal wording: " + ", ".join(causal))
    shares = sorted({m.group(0).lower() for m in _SHARE.finditer(text)})
    if shares:
        problems.append("claims a share the facts don't state (" + ", ".join(shares)
                        + "): say \"the largest parts are\" and quote them")
    if rec is not None:
        region = next((f["region"] for f in facts["findings"] if f["option_id"] == rec), None)
        if region and region.upper() not in (answer["why"] + " " + answer["summary"]).upper():
            problems.append(f"the recommendation doesn't name its region ({region})")
    problems += _last_month_problems(text, facts.get("last_month") or [])
    problems += _list_count_problems(text, facts)
    return problems


_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}
_ON_LISTS = re.compile(r"\bon (?:only |just )?(one|two|three|1|2|3) lists?\b", re.I)


def _list_count_problems(text: str, facts: dict) -> list[str]:
    """"<region> is on two lists" must match how many lists the region is on
    (live: "NORTH ... is on two lists" for a region on one). Each claim is
    read as being about the last region named before it in the sentence."""
    counts: dict[str, int] = {}
    for f in facts["findings"]:
        counts[f["region"].upper()] = counts.get(f["region"].upper(), 0) + 1
    problems = []
    for sentence in _SENTENCES.split(text):
        upper = sentence.upper()
        for m in _ON_LISTS.finditer(sentence):
            named = [(upper.rfind(r, 0, m.start()), r) for r in counts if upper.rfind(r, 0, m.start()) >= 0]
            if not named:
                continue
            region = max(named)[1]
            said, actual = _COUNT_WORDS[m.group(1).lower()], counts[region]
            if said != actual:
                problems.append(f"says {region} is on {said} list(s); it is on {actual} list"
                                f"{'s' if actual != 1 else ''}")
    return problems


# What a sentence about last month says happened, as an outcome status.
_OUTCOME_CLAIMS = (
    (IMPROVED, re.compile(r"\b(improv\w*|got better|recovered)\b", re.I)),
    (OUTCOME_WORSE, re.compile(r"\b(got worse|worsened|deteriorat\w*)\b", re.I)),
    (NO_CLEAR_CHANGE, re.compile(r"\b(no clear change|unchanged|no change|flat)\b", re.I)),
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _last_month_problems(text: str, last_month: list) -> list[str]:
    """A sentence about last month must give each named region the status its
    outcome has. Only sentences saying "last month" count: "EAST is on the
    30+ getting worse list" is this month's list, not an outcome."""
    problems = []
    for sentence in _SENTENCES.split(text):
        if "last month" not in sentence.lower():
            continue
        claimed = {status for status, rx in _OUTCOME_CLAIMS if rx.search(sentence)}
        for o in last_month:
            if claimed and o["region"].upper() in sentence.upper() and o["status"] not in claimed:
                problems.append(f"says {o['region']} {' / '.join(sorted(claimed))} since last month; its outcome is "
                                f"{o['status']}")
    return problems


def _parse(raw: str) -> dict | None:
    try:
        data = parse_json_object(raw)
    except JSONReplyError:
        return None
    out = {k: _clean(data.get(k)) for k in ("summary", "why", "tradeoff", "watch")}
    rec = data.get("recommended_option")
    out["recommended_option"] = None if rec in (None, "", "null") else str(rec).strip()
    return out


def gemini_generate(prompt: str) -> str:
    """One Gemini call, JSON out (the app's model, timeout and retry policy)."""
    return generate_json_text(prompt, temperature=PLANNER_TEMPERATURE)


def write_analysis(brief: Brief, generate: Callable[[str], str] | None = None,
                   outcomes: list | None = None) -> Analysis:
    """The checked analyst summary for a finished brief. generate(prompt) ->
    raw text is the model call (Gemini by default); skipped without a key.
    outcomes: how last month's chosen regions moved (outcomes.track_outcomes)."""
    started, seen = time.time(), outcomes_signature(outcomes)
    if brief.status != "ok" or not brief.findings:
        return Analysis("skipped")
    if generate is None:
        if not os.environ.get("GOOGLE_API_KEY"):
            return Analysis("skipped", problems=["no Google API key"])
        generate = gemini_generate
    facts, problems = analyst_facts(brief, outcomes), None
    for attempt in (1, 2):
        try:
            raw = generate(build_prompt(facts, problems))
        except Exception as e:                       # the plain brief still stands
            logger.warning("Analyst summary call failed: %s", e)
            return Analysis("fallback", attempts=attempt, problems=[f"the AI service didn't answer ({type(e).__name__})"],
                            seconds=time.time() - started, outcomes_seen=seen)
        answer = _parse(raw)
        problems = ["the answer was not a JSON object with the keys asked for"] if answer is None else verify(answer, facts)
        if not problems:
            return Analysis("ok", answer["summary"], answer["recommended_option"], answer["why"], answer["tradeoff"],
                            answer["watch"], attempt, [], time.time() - started, outcomes_seen=seen)
    return Analysis("fallback", attempts=2, problems=problems, seconds=time.time() - started, outcomes_seen=seen)

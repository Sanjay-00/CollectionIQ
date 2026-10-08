"""
Optional AI wording for the report's summary (off by default).

The AI never adds facts: it only rewrites the rules-based summary sentences
into a short paragraph, and every number in its answer must already appear in
those sentences -- otherwise the answer is discarded and the report keeps the
rules-based summary. That way the AI can improve the reading, never the
figures.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

PROMPT = """You write the opening paragraph of a monthly collections report for regional managers of an Indian NBFC.
Rewrite the facts below as ONE plain paragraph of 4 to 6 short sentences: what got worse, where, what improved,
and what to focus on first. Rules:
- Use ONLY the facts and numbers given. Do not add, round, recompute or infer any number.
- Keep names of regions, branches and executives exactly as written.
- No headings, no bullet points, no markdown, no em dashes.

Facts:
{facts}"""

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    """Numbers as written, commas removed ("1,015" -> "1015"; "11.9" stays)."""
    return {n.replace(",", "") for n in _NUM.findall(text)}


def facts_from(model: dict) -> str:
    """The summary sentences the AI may rewrite: the "At a glance" story (or,
    in a report without one, the first bullets block)."""
    for b in model["blocks"]:
        if b["type"] == "glance":
            lines = [b["verdict"]["text"], *b["good"], *b["bad"],
                     *(f"Focus: {f['title']}: {f['detail']}" for f in b["focus"])]
            return "\n".join(f"- {x}" for x in lines)
    for b in model["blocks"]:
        if b["type"] == "bullets":
            return "\n".join(f"- {it['text']}" + (f": {it['detail']}" if it.get("detail") else "") for it in b["items"])
    return ""


def numbers_check(answer: str, facts: str) -> list[str]:
    """Numbers in the answer that the facts don't contain (empty = safe)."""
    allowed = _numbers(facts)
    return sorted(n for n in _numbers(answer) if n not in allowed)


def polish(model: dict, call=None) -> tuple[str | None, str]:
    """(paragraph or None, status message). `call(prompt) -> str` can be
    passed in (tests); by default Gemini is used when GOOGLE_API_KEY is set."""
    facts = facts_from(model)
    if not facts:
        return None, "No summary to rewrite."
    if call is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return None, "AI summary skipped: no GOOGLE_API_KEY set."
        from config import GEMINI_MODEL
        from gemini_client import call_gemini_with_retry, make_client
        client = make_client(api_key)
        call = lambda prompt: call_gemini_with_retry(client, GEMINI_MODEL, prompt, {}).text  # noqa: E731
    try:
        answer = (call(PROMPT.format(facts=facts)) or "").strip()
    except Exception as e:
        logger.warning("AI summary failed: %s", e)
        return None, "AI summary skipped: the AI service did not respond."
    if not answer:
        return None, "AI summary skipped: empty answer."
    answer = answer.replace("—", ",").replace("–", ",")
    bad = numbers_check(answer, facts)
    if bad:
        return None, f"AI summary discarded: it used numbers not in the data ({', '.join(bad[:5])})."
    return answer, "AI summary added (every number checked against the data)."

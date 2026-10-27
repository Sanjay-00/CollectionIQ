"""Shared Gemini client construction and retry policy.

Every Gemini call site (Logical Planner, Insight Generator, Investigator,
Risk Narrator) used its own copy of a retry loop that retried EVERY
exception -- including a bad API key or a malformed request, which can never
succeed -- and created clients with no timeout, so a hung connection blocked
a query indefinitely. One implementation here fixes both everywhere.
"""
import json
import logging
import os
import re
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeout
import threading

from config import GEMINI_HEDGE_AFTER_S, GEMINI_MODEL, GEMINI_TIMEOUT_MS

logger = logging.getLogger(__name__)

# 4xx codes that CAN succeed on retry: request timeout, rate limit.
_RETRYABLE_CLIENT_CODES = {408, 429}


def add_token_usage(response) -> None:
    """Attach Gemini token counts to the active LangSmith run, if tracing."""
    try:
        from langsmith.run_helpers import get_current_run_tree
        rt = get_current_run_tree()
        if rt is None:
            return
        um = getattr(response, "usage_metadata", None)
        if um:
            rt.add_metadata({
                "input_tokens":  int(getattr(um, "prompt_token_count",     0) or 0),
                "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                "total_tokens":  int(getattr(um, "total_token_count",      0) or 0),
            })
    except Exception:
        pass


def make_client(api_key: str):
    """A Gemini client with this app's standard per-request timeout."""
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )


def is_retryable(exc: Exception) -> bool:
    """False only for errors a retry can never fix -- a 4xx client error
    (bad key, bad request, permission) other than timeout/rate-limit.
    Server errors (5xx), network failures, and anything unrecognized are
    treated as transient."""
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) in _RETRYABLE_CLIENT_CODES
    return True


def call_gemini_with_retry(client, model: str, contents: str, config: dict, max_retries: int = 2,
                           stop: threading.Event | None = None):
    """generate_content with exponential backoff on transient failures only.
    stop: set once the reply isn't wanted any more (a hedged call's other
    request answered), so a failure here isn't retried."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if attempt == max_retries or not is_retryable(e) or (stop is not None and stop.is_set()):
                raise
            logger.warning("Gemini call failed (attempt %d/%d), retrying: %s", attempt + 1, max_retries + 1, e)
            time.sleep(2 ** attempt)


# ── Hedged calls ─────────────────────────────────────────────────────────────
# Code review item 4 (2026-10-27): the same planner question with an identical
# reply took 1.9s to 7.5s, so a slow call is random server delay, which a
# second identical request usually beats. Only for calls the user waits on
# with nothing on screen; the loser can't be cancelled (a blocking HTTP call)
# and just finishes in the background, its reply discarded.

_hedge_lock = threading.Lock()
_hedge_pool: ThreadPoolExecutor | None = None
_hedge_counts = {"calls": 0, "hedged": 0, "backup_won": 0}


def _pool() -> ThreadPoolExecutor:
    global _hedge_pool
    with _hedge_lock:
        if _hedge_pool is None:
            _hedge_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="gemini-hedge")
        return _hedge_pool


def _count(name: str) -> None:
    with _hedge_lock:
        _hedge_counts[name] += 1


def hedge_stats() -> dict:
    """calls, hedged (a backup was sent), backup_won (the backup answered first)."""
    with _hedge_lock:
        return dict(_hedge_counts)


def call_gemini_hedged(client, model: str, contents: str, config: dict, hedge_after_s: float | None = None):
    """call_gemini_with_retry, plus one identical backup request if no reply
    by hedge_after_s (default GEMINI_HEDGE_AFTER_S; 0 = off). Returns the
    first reply; an error only when both failed (the first one's error). A
    quick failure (e.g. a bad key) is raised at once, with no backup."""
    hedge_after_s = GEMINI_HEDGE_AFTER_S if hedge_after_s is None else hedge_after_s
    if hedge_after_s <= 0:
        return call_gemini_with_retry(client, model, contents, config)
    _count("calls")
    answered = threading.Event()
    first = _pool().submit(call_gemini_with_retry, client, model, contents, config, 2, answered)
    try:
        return first.result(timeout=hedge_after_s)
    except FuturesTimeout:
        pass
    _count("hedged")
    logger.info("Gemini call slower than %.1fs, sending a backup request", hedge_after_s)
    # The backup doesn't retry: the first request is still running its own retries.
    backup = _pool().submit(call_gemini_with_retry, client, model, contents, config, 0)
    pending = {first, backup}
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            if f.exception() is None:
                answered.set()      # seen live: a hung first request timed out, then retried for nobody
                if f is backup:
                    _count("backup_won")
                return f.result()
    raise first.exception()


# ── Structured (JSON) replies ────────────────────────────────────────────────
# Every call whose reply is parsed as data (the AI Query planner, the
# Investigator router, the review analyst) asks for JSON mode and parses here.
# Before (code review, 2026-10-27) only the analyst used JSON mode; the other
# two asked for free text and each stripped markdown fences with its own regex.

class JSONReplyError(ValueError):
    """The model's reply wasn't the one JSON object asked for."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def json_config(temperature: float, system_instruction: str | None = None, response_schema=None) -> dict:
    """Gemini config for a reply that must be one JSON object: JSON mode
    returns JSON text, never prose or a markdown fence."""
    config = {"temperature": temperature, "response_mime_type": "application/json"}
    if system_instruction:
        config["system_instruction"] = system_instruction
    if response_schema is not None:
        config["response_schema"] = response_schema
    return config


def parse_json_object(text) -> dict:
    """The one JSON object in a model reply. A stray markdown fence is still
    removed (another provider, or a model ignoring JSON mode). Raises
    JSONReplyError for anything else, so a caller never acts on prose."""
    raw = _FENCE.sub("", (text or "").strip())
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise JSONReplyError(f"not JSON: {raw[:80]!r}") from e
    if not isinstance(data, dict):
        raise JSONReplyError(f"expected a JSON object, got {type(data).__name__}")
    return data


def generate_json_text(contents: str, *, temperature: float, system_instruction: str | None = None,
                       response_schema=None, model: str | None = None) -> str:
    """One JSON-mode call with this app's client, timeout and retry policy;
    the raw reply text (parse it with parse_json_object)."""
    client = make_client(os.environ["GOOGLE_API_KEY"])
    response = call_gemini_with_retry(client, model or GEMINI_MODEL, contents,
                                      json_config(temperature, system_instruction, response_schema))
    add_token_usage(response)
    return response.text or ""

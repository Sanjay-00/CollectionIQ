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

from config import GEMINI_MODEL, GEMINI_TIMEOUT_MS

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


def call_gemini_with_retry(client, model: str, contents: str, config: dict, max_retries: int = 2):
    """generate_content with exponential backoff on transient failures only."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if attempt == max_retries or not is_retryable(e):
                raise
            logger.warning("Gemini call failed (attempt %d/%d), retrying: %s", attempt + 1, max_retries + 1, e)
            time.sleep(2 ** attempt)


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

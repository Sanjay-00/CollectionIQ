"""Shared Gemini client construction and retry policy.

Every Gemini call site (Logical Planner, Insight Generator, Investigator,
Risk Narrator) used its own copy of a retry loop that retried EVERY
exception -- including a bad API key or a malformed request, which can never
succeed -- and created clients with no timeout, so a hung connection blocked
a query indefinitely. One implementation here fixes both everywhere.
"""
import logging
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from config import GEMINI_TIMEOUT_MS

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

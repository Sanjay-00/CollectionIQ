"""gemini_client.py: shared retry policy and client timeout for every Gemini
call site. Before this, each call site retried EVERY error (a bad API key
burned ~3s of backoff before failing) and created clients with no timeout."""
import pytest
from google.genai import errors as genai_errors

import gemini_client
from config import GEMINI_TIMEOUT_MS


def _client_error(code: int):
    return genai_errors.ClientError(code, {"error": {"code": code, "message": "x", "status": "X"}})


def _server_error(code: int):
    return genai_errors.ServerError(code, {"error": {"code": code, "message": "x", "status": "X"}})


class TestIsRetryable:
    @pytest.mark.parametrize("code", [400, 401, 403, 404])
    def test_permanent_client_errors_are_not_retried(self, code):
        assert gemini_client.is_retryable(_client_error(code)) is False

    @pytest.mark.parametrize("code", [408, 429])
    def test_timeout_and_rate_limit_are_retried(self, code):
        assert gemini_client.is_retryable(_client_error(code)) is True

    def test_server_errors_are_retried(self):
        assert gemini_client.is_retryable(_server_error(503)) is True

    def test_unrecognized_errors_are_treated_as_transient(self):
        assert gemini_client.is_retryable(ConnectionError("reset")) is True


class _Client:
    def __init__(self, errors_then_ok):
        self.attempts = 0
        outer = self

        class models:
            @staticmethod
            def generate_content(model, contents, config):
                outer.attempts += 1
                if errors_then_ok:
                    raise errors_then_ok.pop(0)
                return "ok"

        self.models = models


class TestCallGeminiWithRetry:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(gemini_client.time, "sleep", lambda *_: None)

    def test_permanent_error_fails_fast_without_retrying(self):
        client = _Client([_client_error(400)])
        with pytest.raises(genai_errors.ClientError):
            gemini_client.call_gemini_with_retry(client, "m", "p", {})
        assert client.attempts == 1

    def test_transient_error_is_retried_then_succeeds(self):
        client = _Client([_server_error(503)])
        assert gemini_client.call_gemini_with_retry(client, "m", "p", {}) == "ok"
        assert client.attempts == 2

    def test_rate_limit_exhausts_retries_then_raises(self):
        client = _Client([_client_error(429)] * 3)
        with pytest.raises(genai_errors.ClientError):
            gemini_client.call_gemini_with_retry(client, "m", "p", {}, max_retries=2)
        assert client.attempts == 3


def test_make_client_sets_the_shared_timeout(monkeypatch):
    seen = {}
    monkeypatch.setattr(gemini_client.genai, "Client", lambda **kw: seen.update(kw) or object())
    gemini_client.make_client("key")
    assert seen["api_key"] == "key"
    assert seen["http_options"].timeout == GEMINI_TIMEOUT_MS

"""Tests for the Ollama executor, with the HTTP call mocked.

None of these tests need a running Ollama instance.
"""

from __future__ import annotations

import pytest
import requests

import vibeforge.router.executor as executor_module
from vibeforge.router.executor import OllamaExecutor
from vibeforge.types import ExecutionResult


class FakeResponse:
    """A tiny stand-in for ``requests.Response``."""

    def __init__(
        self, status_code: int = 200, json_data: object | None = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.ok = status_code < 400

    def json(self) -> object:
        if isinstance(self._json, ValueError):
            raise self._json
        return self._json


def make_executor() -> OllamaExecutor:
    """Executor bound to a throwaway URL; the HTTP call is always mocked."""
    return OllamaExecutor(base_url="http://ollama.test:11434")


def test_successful_generation_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(json_data={"response": "hello back", "eval_count": 7})

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("test-model:latest", "say hi")

    assert isinstance(result, ExecutionResult)
    assert result.ok
    assert result.output == "hello back"
    assert result.eval_count == 7
    assert result.error is None
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    assert result.tokens_per_sec is not None and result.tokens_per_sec > 0
    assert captured["url"] == "http://ollama.test:11434/api/generate"
    assert captured["json"] == {"model": "test-model:latest", "prompt": "say hi", "stream": False}
    assert captured["timeout"] == 120.0


def test_options_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        captured["json"] = json
        return FakeResponse(json_data={"response": "ok"})

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    make_executor().execute("m:latest", "p", options={"temperature": 0.0, "num_predict": 16})
    assert captured["json"]["options"] == {"temperature": 0.0, "num_predict": 16}


def test_missing_eval_count_yields_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        return FakeResponse(json_data={"response": "ok"})

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("m:latest", "p")
    assert result.eval_count is None
    assert result.tokens_per_sec is None


def test_connection_error_becomes_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("m:latest", "p")

    assert not result.ok
    assert result.output is None
    assert "connection refused" in (result.error or "")


def test_timeout_becomes_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        raise requests.Timeout("slow")

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("m:latest", "p")
    assert "timed out" in (result.error or "")


def test_model_not_pulled_is_an_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        return FakeResponse(
            status_code=404,
            json_data={"error": "model 'nope:latest' not found, try pulling it first"},
        )

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("nope:latest", "p")

    assert not result.ok
    assert "model 'nope:latest' not found" in (result.error or "")


def test_non_json_success_response_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        return FakeResponse(status_code=200, json_data=ValueError("not json"), text="<html>")

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("m:latest", "p")
    assert not result.ok
    assert "non-JSON" in (result.error or "")


def test_generic_request_error_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        raise requests.RequestException("boom")

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("m:latest", "p")
    assert not result.ok
    assert "boom" in (result.error or "")


def test_latency_is_measured_even_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        return FakeResponse(status_code=500, json_data={"error": "server blew up"})

    monkeypatch.setattr(executor_module.requests, "post", fake_post)

    result = make_executor().execute("m:latest", "p")
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    assert "server blew up" in (result.error or "")


def test_base_url_trailing_slash_is_normalized() -> None:
    executor = OllamaExecutor(base_url="http://localhost:11434/")
    assert executor.base_url == "http://localhost:11434"

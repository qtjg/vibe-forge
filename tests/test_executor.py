"""Tests for the Ollama executor, with the official client mocked.

None of these tests need a running Ollama instance. They assert the same
result contract the executor had when it used raw HTTP directly.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from httpx import ConnectError, TimeoutException
from ollama import ResponseError

import vibeforge.router.executor as executor_module
from vibeforge.router.executor import OllamaExecutor
from vibeforge.types import ExecutionResult


class FakeClient:
    """Stand-in for ``ollama.Client``; each test programs ``generate``."""

    def __init__(
        self,
        host: str | None = None,
        timeout: float | None = None,
        script: list | None = None,
        _crank: Callable[[], object] | None = None,
    ) -> None:
        self.host = host
        self.timeout = timeout
        self._script = script or []
        self._crank = _crank
        self.closed = False
        self.last_kwargs: dict = {}

    def generate(self, **kwargs: object) -> object:
        self.last_kwargs = kwargs
        if self._crank is not None:
            entry = self._crank()
        else:
            entry = self._script[-1]
        if isinstance(entry, tuple):
            exc_type, message, status_code = (list(entry) + [None])[:3]
            if status_code is None:
                raise exc_type(message)
            raise exc_type(message, status_code)
        return entry

    def close(self) -> None:
        self.closed = True


def install_client(
    monkeypatch: pytest.MonkeyPatch,
    script: list,
) -> dict[str, FakeClient]:
    """Patch ``ollama.Client`` so every ``execute`` attempt gets a fresh fake.

    Script entries are either response objects (SimpleNamespace with
    ``response``/``eval_count``) or ``(exception_type, message,
    [status_code])`` tuples. The script advances once per *attempt* across
    retries, so a one-shot failure followed by a success is retried exactly
    once. Returns a mutable holder for the first created client.
    """
    crank = {"calls": 0}
    holder: dict[str, FakeClient] = {}

    def advance() -> object:
        index = min(crank["calls"], len(script) - 1)
        crank["calls"] += 1
        return script[index]

    def factory(host: str | None = None, timeout: float | None = None) -> FakeClient:
        client = FakeClient(
            host=host,
            timeout=timeout,
            script=script,
            _crank=advance,
        )
        holder.setdefault("first", client)
        holder.setdefault("all", []).append(client)
        return client

    monkeypatch.setattr(executor_module.ollama, "Client", factory)
    return holder


def ok_response(text: str = "hello back", eval_count: int = 7) -> SimpleNamespace:
    """A successful ``generate`` payload."""
    return SimpleNamespace(response=text, eval_count=eval_count)


def test_successful_generation_is_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install_client(monkeypatch, [ok_response()])

    result = make_executor().execute("test-model:latest", "say hi")

    assert isinstance(result, ExecutionResult)
    assert result.ok
    assert result.output == "hello back"
    assert result.eval_count == 7
    assert result.error is None
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    assert result.tokens_per_sec is not None and result.tokens_per_sec > 0
    assert client["first"].host == "http://ollama.test:11434"
    assert client["first"].timeout == 120.0
    assert client["first"].last_kwargs == {
        "model": "test-model:latest",
        "prompt": "say hi",
        "stream": False,
        "options": {},
    }
    assert client["first"].closed


def make_executor() -> OllamaExecutor:
    """Executor bound to a throwaway URL; the client is always mocked."""
    return OllamaExecutor(base_url="http://ollama.test:11434")


def test_options_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install_client(monkeypatch, [ok_response()])

    make_executor().execute("m:latest", "p", options={"temperature": 0.0, "num_predict": 16})
    assert client["first"].last_kwargs["options"] == {"temperature": 0.0, "num_predict": 16}


def test_missing_eval_count_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, [SimpleNamespace(response="ok", eval_count=None)])

    result = make_executor().execute("m:latest", "p")
    assert result.eval_count is None
    assert result.tokens_per_sec is None


def test_connection_error_becomes_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)
    install_client(monkeypatch, [(ConnectError, "connection refused")])

    result = make_executor().execute("m:latest", "p")

    assert not result.ok
    assert result.output is None
    assert "connection refused" in (result.error or "")
    assert result.error_kind == "connection"


def test_timeout_becomes_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)
    install_client(monkeypatch, [(TimeoutException, "slow")])

    result = make_executor().execute("m:latest", "p")
    assert "timed out" in (result.error or "")
    assert result.error_kind == "timeout"


def test_model_not_pulled_is_an_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(
        monkeypatch,
        [(ResponseError, '{"error": "model \'nope:latest\' not found"}')],
    )

    result = make_executor().execute("nope:latest", "p")

    assert not result.ok
    assert "model 'nope:latest' not found" in (result.error or "")
    assert result.error_kind == "http"


def test_non_json_success_response_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, [(ValueError, "not json")])

    result = make_executor().execute("m:latest", "p")
    assert not result.ok
    assert "non-JSON" in (result.error or "")
    assert result.error_kind == "json"


def test_generic_request_error_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from httpx import TransportError

    install_client(monkeypatch, [(TransportError, "boom")])

    result = make_executor().execute("m:latest", "p")
    assert not result.ok
    assert "boom" in (result.error or "")


def test_latency_is_measured_even_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, [(ResponseError, "server blew up")])

    result = make_executor().execute("m:latest", "p")
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    assert "server blew up" in (result.error or "")


def test_base_url_trailing_slash_is_normalized() -> None:
    executor = OllamaExecutor(base_url="http://localhost:11434/")
    assert executor.base_url == "http://localhost:11434"


def test_transient_connection_error_is_retried_and_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(
        monkeypatch,
        [(ConnectError, "first attempt refused"), ok_response("recovered")],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(executor_module.time, "sleep", sleeps.append)

    result = make_executor().execute("m:latest", "p")

    assert result.ok
    assert result.output == "recovered"
    assert result.retries_attempted == 0  # success path reports no retries
    assert sleeps == [0.5]  # first backoff


def test_timeout_is_retried_up_to_max_then_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, [(TimeoutException, "slow")])
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)

    result = make_executor().execute("m:latest", "p")

    assert not result.ok
    assert result.error_kind == "timeout"
    assert result.retries_attempted == 2
    assert "timed out" in (result.error or "")
    assert "after 2 retries" in (result.error or "")


def test_retryable_http_status_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(
        monkeypatch,
        [(ResponseError, "warming up", 503), ok_response("ok now")],
    )
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)

    result = make_executor().execute("m:latest", "p")

    assert result.ok
    assert result.output == "ok now"


def test_http_429_is_retried_then_fails_with_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, [(ResponseError, "too many", 429)])
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)

    result = make_executor().execute("m:latest", "p")

    assert result.error_kind == "http"
    assert result.retries_attempted == 2


def test_http_404_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, [(ResponseError, "model not found", 404)])

    result = make_executor().execute("nope:latest", "p")

    assert result.error_kind == "http"
    assert result.retries_attempted == 0


def test_backoff_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, [(TimeoutException, "slow")])
    sleeps: list[float] = []
    monkeypatch.setattr(executor_module.time, "sleep", sleeps.append)

    # max_retries=4, cap=2.0: backoffs 0.5, 1.0, 2.0, 2.0
    executor = OllamaExecutor(base_url="http://ollama.test:11434", max_retries=4, backoff_cap=2.0)
    executor.execute("m:latest", "p")

    assert sleeps == [0.5, 1.0, 2.0, 2.0]


def test_per_call_timeout_overrides_instance_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)
    clients = []

    def factory(host: str | None = None, timeout: float | None = None) -> FakeClient:
        client = FakeClient(host=host, timeout=timeout, script=[(TimeoutException, "slow")])
        clients.append(client)
        return client

    monkeypatch.setattr(executor_module.ollama, "Client", factory)
    make_executor().execute("m:latest", "p", timeout=3.0)
    assert clients[0].timeout == 3.0


def test_no_retries_configured_gives_up_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, [(ConnectError, "nope")])

    executor = OllamaExecutor(base_url="http://ollama.test:11434", max_retries=0)
    result = executor.execute("m:latest", "p")

    assert result.retries_attempted == 0
    assert result.error_kind == "connection"
    assert "after" not in (result.error or "")


def test_success_reports_error_kind_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, [ok_response()])

    result = make_executor().execute("m:latest", "p")
    assert result.error_kind is None
    assert result.retries_attempted == 0
    assert result.as_dict()["error_kind"] is None
    assert result.as_dict()["retries_attempted"] == 0

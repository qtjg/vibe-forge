"""Tests for the Ollama availability probe (client mocked, no Ollama needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vibeforge.router.ollama as ollama_module
from vibeforge.router.ollama import OllamaStatus, probe_ollama


class FakeClient:
    """Stand-in for ``ollama.Client`` with just enough surface."""

    class Behavior(Exception):
        pass

    def __init__(self, host: str | None = None, timeout: float | None = None) -> None:
        self.host = host
        self.timeout = timeout
        self.closed = False

    def list(self) -> object:
        raise self.Behavior("not configured")

    def close(self) -> None:
        self.closed = True


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    behavior: object | tuple[type[Exception], str],
) -> dict[str, FakeClient]:
    """Patch ``ollama.Client`` with a fake whose ``list()`` raises ``behavior``
    or returns it. Returns a mutable holder for the created instance."""

    def _list(behavior: object) -> object:
        if isinstance(behavior, tuple):
            exc, message = behavior
            raise exc(message)
        return behavior

    holder: dict[str, FakeClient] = {}

    def factory(host: str | None = None, timeout: float | None = None) -> FakeClient:
        fake = FakeClient(host=host, timeout=timeout)
        fake.list = lambda: _list(behavior)
        holder["instance"] = fake
        return fake

    monkeypatch.setattr(ollama_module.ollama, "Client", factory)
    return holder


def two_models() -> object:
    """A ``list()`` payload with two pulled models."""
    return SimpleNamespace(
        models=[
            SimpleNamespace(model="qwen2.5:0.5b"),
            SimpleNamespace(model="llama3.1:latest"),
        ]
    )


def test_probe_reports_pulled_tags_and_connection_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch, two_models())

    status = probe_ollama(base_url="http://ollama.test:11434")

    assert status.reachable
    assert status.available
    assert status.pulled_tags == frozenset({"qwen2.5:0.5b", "llama3.1:latest"})
    assert status.error is None
    assert fake["instance"].host == "http://ollama.test:11434"
    assert fake["instance"].timeout == 3.0
    assert fake["instance"].closed


def test_probe_with_no_models_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, SimpleNamespace(models=[]))

    status = probe_ollama()
    assert status.reachable
    assert not status.available


def test_probe_connection_error_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from httpx import ConnectError

    install_fake(monkeypatch, (ConnectError, "refused"))

    status = probe_ollama()
    assert not status.reachable
    assert not status.available
    assert "cannot reach Ollama" in (status.error or "")


def test_probe_sdk_connection_error_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real ollama SDK raises the builtin ConnectionError with an
    # install hint; the probe must translate it, not let it escape.
    install_fake(monkeypatch, (ConnectionError, "Failed to connect to Ollama."))

    status = probe_ollama()
    assert not status.reachable
    assert "cannot reach Ollama" in (status.error or "")


def test_probe_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from httpx import TimeoutException

    install_fake(monkeypatch, (TimeoutException, "slow"))

    status = probe_ollama()
    assert not status.reachable
    assert "timed out" in (status.error or "")


def test_probe_http_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from ollama import ResponseError

    install_fake(monkeypatch, (ResponseError, '{"error": "boom"}'))

    status = probe_ollama()
    assert not status.reachable
    assert "HTTP" in (status.error or "")


def test_probe_non_json_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, (ValueError, "nope"))

    status = probe_ollama()
    assert not status.reachable
    assert "non-JSON" in (status.error or "")


def test_status_serializes_to_json_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, two_models())

    payload = probe_ollama().as_dict()
    for key in ("reachable", "available", "pulled_tags", "error", "checked_at"):
        assert key in payload
    assert payload["pulled_tags"] == ["llama3.1:latest", "qwen2.5:0.5b"]


def test_ollama_status_is_a_dataclass() -> None:
    status = OllamaStatus(reachable=True)
    assert not status.available
    assert status.error is None

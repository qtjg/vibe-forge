"""Tests for the Ollama availability probe (HTTP mocked, no Ollama needed)."""

from __future__ import annotations

import requests

import vibeforge.router.ollama as ollama_module
from vibeforge.router.ollama import OllamaStatus, probe_ollama


class FakeResponse:
    """Stand-in for ``requests.Response`` with just enough surface."""

    def __init__(self, status_code: int = 200, json_data: object | None = None) -> None:
        self.status_code = status_code
        self._json = json_data
        self.ok = status_code < 400

    def json(self) -> object:
        if isinstance(self._json, ValueError):
            raise self._json
        return self._json


def test_probe_reports_pulled_tags_and_version(monkeypatch) -> None:
    captured: dict = {}

    def fake_get(url: str, timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse(
            json_data={
                "version": "0.5.0",
                "models": [
                    {"name": "qwen2.5:0.5b"},
                    {"name": "llama3.1:latest"},
                ],
            }
        )

    monkeypatch.setattr(ollama_module.requests, "get", fake_get)

    status = probe_ollama(base_url="http://ollama.test:11434")

    assert status.reachable
    assert status.available
    assert status.pulled_tags == frozenset({"qwen2.5:0.5b", "llama3.1:latest"})
    assert status.server_version == "0.5.0"
    assert status.error is None
    assert captured["url"] == "http://ollama.test:11434/api/tags"
    assert captured["timeout"] == 3.0


def test_probe_with_no_models_is_unavailable(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        return FakeResponse(json_data={"models": []})

    monkeypatch.setattr(ollama_module.requests, "get", fake_get)

    status = probe_ollama()
    assert status.reachable
    assert not status.available


def test_probe_connection_error_is_reported_not_raised(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ollama_module.requests, "get", fake_get)

    status = probe_ollama()
    assert not status.reachable
    assert not status.available
    assert "cannot reach Ollama" in (status.error or "")


def test_probe_timeout_is_reported(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        raise requests.Timeout("slow")

    monkeypatch.setattr(ollama_module.requests, "get", fake_get)

    status = probe_ollama()
    assert not status.reachable
    assert "timed out" in (status.error or "")


def test_probe_non_200_is_an_error(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        return FakeResponse(status_code=500, json_data={})

    monkeypatch.setattr(ollama_module.requests, "get", fake_get)

    status = probe_ollama()
    assert not status.reachable
    assert "HTTP 500" in (status.error or "")


def test_probe_non_json_is_an_error(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        return FakeResponse(status_code=200, json_data=ValueError("nope"))

    monkeypatch.setattr(ollama_module.requests, "get", fake_get)

    status = probe_ollama()
    assert not status.reachable
    assert "non-JSON" in (status.error or "")


def test_status_serializes_to_json_shape(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        return FakeResponse(json_data={"models": [{"name": "a:latest"}]})

    monkeypatch.setattr(ollama_module.requests, "get", fake_get)

    payload = probe_ollama().as_dict()
    for key in ("reachable", "available", "pulled_tags", "server_version", "error",
                "checked_at"):
        assert key in payload
    assert payload["pulled_tags"] == ["a:latest"]


def test_ollama_status_is_a_dataclass() -> None:
    status = OllamaStatus(reachable=True)
    assert not status.available
    assert status.error is None
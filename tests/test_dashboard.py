"""Tests for the dashboard API with FastAPI's TestClient (no server needed)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from vibeforge.dashboard.app import create_app


def make_decision(
    model: str = "tiny-fast",
    task_type: str = "autocomplete",
    complexity: str = "trivial",
    latency_ms: float | None = None,
) -> dict[str, object]:
    """A valid decision payload as produced by RoutingDecision.as_dict()."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "task_type": task_type,
        "prompt": f"do something {model}",
        "file_path": None,
        "score": 0,
        "complexity": complexity,
        "reason": "baseline autocomplete=0 => score 0/3 (trivial)",
        "model": model,
        "ollama_tag": "tag:latest",
        "latency_ms": latency_ms,
        "eval_count": 42,
        "execution_error": None,
    }


@pytest.fixture
def client() -> TestClient:
    """Test client against a fresh dashboard app."""
    return TestClient(create_app())


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "vibe-forge" in response.text


def test_decisions_start_empty(client: TestClient) -> None:
    response = client.get("/api/decisions")
    assert response.status_code == 200
    assert response.json() == {"decisions": []}


def test_posted_decision_appears_in_decisions(client: TestClient) -> None:
    payload = make_decision()
    assert client.post("/api/decisions", json=payload).status_code == 200

    body = client.get("/api/decisions").json()
    assert len(body["decisions"]) == 1
    assert body["decisions"][0]["model"] == "tiny-fast"
    assert body["decisions"][0]["complexity"] == "trivial"


def test_decisions_newest_first(client: TestClient) -> None:
    for model in ("tiny-fast", "balanced", "heavy"):
        client.post("/api/decisions", json=make_decision(model=model))

    body = client.get("/api/decisions").json()
    assert [d["model"] for d in body["decisions"]] == ["heavy", "balanced", "tiny-fast"]


def test_decisions_limit_is_applied(client: TestClient) -> None:
    for i in range(10):
        client.post("/api/decisions", json=make_decision(model=f"m{i}"))

    body = client.get("/api/decisions?limit=3").json()
    assert len(body["decisions"]) == 3


def test_rejects_invalid_decision(client: TestClient) -> None:
    response = client.post("/api/decisions", json={"model": "only"})
    assert response.status_code == 400
    assert "task_type" in response.json()["detail"]

    assert client.post("/api/decisions", json=[1, 2, 3]).status_code == 400


def test_stats_aggregate_usage_and_latency(client: TestClient) -> None:
    client.post("/api/decisions", json=make_decision(model="tiny-fast", latency_ms=100.0))
    client.post("/api/decisions", json=make_decision(model="tiny-fast", latency_ms=300.0))
    client.post("/api/decisions", json=make_decision(model="balanced", latency_ms=None))

    stats = client.get("/api/stats").json()["models"]
    by_name = {m["model"]: m for m in stats}

    assert by_name["tiny-fast"]["decisions"] == 2
    assert by_name["tiny-fast"]["avg_latency_ms"] == 200.0
    assert by_name["balanced"]["decisions"] == 1
    assert by_name["balanced"]["avg_latency_ms"] is None

    total_decisions = sum(m["decisions"] for m in stats)
    assert total_decisions == 3


def test_stats_count_execution_errors(client: TestClient) -> None:
    good = make_decision()
    bad = make_decision(model="heavy")
    bad["execution_error"] = "model not found"
    client.post("/api/decisions", json=good)
    client.post("/api/decisions", json=bad)

    stats = client.get("/api/stats").json()["models"]
    by_name = {m["model"]: m for m in stats}
    assert by_name["heavy"]["errors"] == 1
    assert by_name["tiny-fast"]["errors"] == 0


def test_app_can_be_seeded_with_history() -> None:
    seeded = [make_decision(model="balanced")]
    client = TestClient(create_app(history=seeded))
    body = client.get("/api/decisions").json()
    assert len(body["decisions"]) == 1
    assert body["decisions"][0]["model"] == "balanced"

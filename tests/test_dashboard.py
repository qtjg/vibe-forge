"""Tests for the dashboard API with FastAPI's TestClient (no server needed)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vibeforge.dashboard.app import create_app
from vibeforge.router.registry import ModelRegistry
from vibeforge.types import Complexity, ExecutionResult, ModelTier


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


def test_app_persists_across_restarts(
    tmp_path: Path,
) -> None:
    """Kill simulation: app #1 writes, app #2 (fresh process) reads it back."""
    db_path = tmp_path / "dashboard.db"

    with TestClient(create_app(db_path=db_path)) as first:
        assert (
            first.post("/api/decisions", json=make_decision(model="tiny-fast")).status_code == 200
        )

    with TestClient(create_app(db_path=db_path)) as second:
        body = second.get("/api/decisions").json()
        assert len(body["decisions"]) == 1
        assert body["decisions"][0]["model"] == "tiny-fast"


def test_in_memory_app_has_no_db_file() -> None:
    app = create_app()
    assert app.state.db_path is None
    assert len(app.state.store) == 0


class FixedExecutor:
    """An executor-shaped fake returning a canned ExecutionResult."""

    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.calls: list[tuple[str, str]] = []

    def execute(self, model_tag: str, prompt: str, options: dict | None = None) -> ExecutionResult:
        self.calls.append((model_tag, prompt))
        return self._result


def fixed_registry() -> ModelRegistry:
    """A registry with one tier per complexity band, config-free."""
    return ModelRegistry(
        [
            ModelTier(
                name="balanced",
                ollama_tag="l:latest",
                complexity_ceiling=Complexity.HIGH,
                approx_ram_gb=8.0,
            ),
        ]
    )


def test_route_scores_and_stores_decision() -> None:
    client = TestClient(create_app(registry_factory=fixed_registry))

    response = client.post(
        "/api/route", json={"prompt": "explain this regex", "task_type": "explain"}
    )

    assert response.status_code == 200
    decision = response.json()
    assert decision["prompt"] == "explain this regex"
    assert decision["task_type"] == "explain"
    assert decision["model"] == "balanced"
    assert decision["complexity"] in {t.value for t in Complexity}

    stored = client.get("/api/decisions").json()["decisions"]
    assert len(stored) == 1
    assert stored[0]["prompt"] == "explain this regex"


def test_route_rejects_bad_bodies(client: TestClient) -> None:
    assert client.post("/api/route", json={}).status_code == 400
    assert client.post("/api/route", json={"prompt": ""}).status_code == 400
    assert (
        client.post("/api/route", json={"prompt": "hi", "task_type": "nonsense"}).status_code == 400
    )
    assert client.post("/api/route", json=[1, 2]).status_code == 400
    assert (
        client.post(
            "/api/route", content=b"not json", headers={"content-type": "application/json"}
        ).status_code
        == 400
    )


def test_route_reports_routing_failures() -> None:
    def boom() -> ModelRegistry:
        raise RuntimeError("registry exploded")

    client = TestClient(create_app(registry_factory=boom))
    response = client.post("/api/route", json={"prompt": "hi"})
    assert response.status_code == 500
    assert "routing failed" in response.json()["detail"]


def test_execute_returns_output_and_latency() -> None:
    result = ExecutionResult(
        model="l:latest", prompt="hi", latency_ms=12.5, eval_count=7, output="hello"
    )
    client = TestClient(create_app(executor_factory=lambda: FixedExecutor(result)))

    response = client.post("/api/execute", json={"prompt": "hi", "model_tag": "l:latest"})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "output": "hello",
        "error": None,
        "error_kind": None,
        "latency_ms": 12.5,
        "eval_count": 7,
    }


def test_execute_reports_failure_without_http_error() -> None:
    result = ExecutionResult(
        model="l:latest",
        prompt="hi",
        latency_ms=3.0,
        error="cannot reach Ollama",
        error_kind="connection",
    )
    client = TestClient(create_app(executor_factory=lambda: FixedExecutor(result)))

    response = client.post("/api/execute", json={"prompt": "hi", "model_tag": "l:latest"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error_kind"] == "connection"


def test_execute_rejects_invalid_bodies(client: TestClient) -> None:
    assert client.post("/api/execute", json={}).status_code == 400
    assert client.post("/api/execute", json={"prompt": "hi"}).status_code == 400
    assert client.post("/api/execute", json={"prompt": "hi", "model_tag": ""}).status_code == 400
    assert (
        client.post(
            "/api/execute", json={"prompt": "hi", "model_tag": "l", "options": [1]}
        ).status_code
        == 400
    )

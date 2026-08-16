"""Tests for concurrent multi-model comparison (no Ollama, fake executors)."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from vibeforge.compare_models import ModelRun, run_models_concurrently
from vibeforge.types import ExecutionResult

MODELS = [
    ("tiny-fast", "qwen2.5:0.5b"),
    ("balanced", "llama3.1:latest"),
    ("heavy", "qwen2.5-coder:14b"),
]


def ok_result(tag: str, latency: float = 10.0, output: str = "hi") -> ExecutionResult:
    """A canned success result."""
    return ExecutionResult(model=tag, prompt="p", latency_ms=latency, eval_count=5, output=output)


def make_factory() -> Callable[[], object]:
    """Factory producing a fresh executor per call."""

    def factory() -> object:
        class E:
            def execute(
                self, model_tag: str, prompt: str, options: dict | None = None
            ) -> ExecutionResult:
                time.sleep(0.15)
                return ok_result(model_tag)

        return E()

    return factory


def test_runs_every_model_exactly_once() -> None:
    executed: list[str] = []

    def factory() -> object:
        class E:
            def execute(
                self, model_tag: str, prompt: str, options: dict | None = None
            ) -> ExecutionResult:
                executed.append(model_tag)
                return ok_result(model_tag)

        return E()

    runs = run_models_concurrently(MODELS, "prompt", factory)

    assert [run.name for run in runs] == ["tiny-fast", "balanced", "heavy"]
    assert sorted(executed) == ["llama3.1:latest", "qwen2.5-coder:14b", "qwen2.5:0.5b"]


def test_models_run_in_parallel() -> None:
    started = time.perf_counter()
    runs = run_models_concurrently(MODELS, "p", make_factory())
    elapsed = time.perf_counter() - started

    # Three 150ms sleeps in parallel finish well under the 450ms serial bound.
    assert elapsed < 0.35
    assert len(runs) == 3


def test_one_failure_does_not_abort_siblings() -> None:
    def factory() -> object:
        class E:
            def execute(
                self, model_tag: str, prompt: str, options: dict | None = None
            ) -> ExecutionResult:
                if model_tag == "llama3.1:latest":
                    return ExecutionResult(
                        model=model_tag,
                        prompt="p",
                        error="model not found",
                        error_kind="http",
                        status_code=404,
                    )
                return ok_result(model_tag)

        return E()

    runs = run_models_concurrently(MODELS, "p", factory)

    assert runs[0].ok
    assert not runs[1].ok
    assert "not found" in (runs[1].result.error or "")
    assert runs[1].result.status_code == 404
    assert runs[2].ok


def test_options_are_forwarded() -> None:
    seen: list[dict | None] = []

    def factory() -> object:
        class E:
            def execute(
                self, model_tag: str, prompt: str, options: dict | None = None
            ) -> ExecutionResult:
                seen.append(options)
                return ok_result(model_tag)

        return E()

    run_models_concurrently(MODELS, "p", factory, options={"num_predict": 128})

    assert seen == [{"num_predict": 128}] * 3


def test_duplicate_tags_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate model tag"):
        run_models_concurrently([("a", "x:latest"), ("b", "x:latest")], "p", make_factory())


def test_model_run_shape() -> None:
    run = ModelRun(
        name="balanced", ollama_tag="llama3.1:latest", result=ok_result("llama3.1:latest")
    )

    assert run.ok
    assert run.latency_ms == 10.0


def test_empty_models_list_returns_empty() -> None:
    assert run_models_concurrently([], "p", make_factory()) == ()

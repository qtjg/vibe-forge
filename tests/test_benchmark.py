"""Tests for the benchmark runner, using a fake executor (no Ollama)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vibeforge.benchmark.runner import BenchmarkRunner
from vibeforge.benchmark.tasks import all_tasks, tasks_for
from vibeforge.router.registry import ModelRegistry
from vibeforge.types import ExecutionResult, TaskType

REGISTRY_YAML = """\
models:
  - name: tiny
    ollama_tag: tiny:latest
    complexity_ceiling: trivial
    approx_ram_gb: 0.6
  - name: big
    ollama_tag: big:latest
    complexity_ceiling: high
    approx_ram_gb: 9
"""


class FakeExecutor:
    """Executor that returns a canned result, optionally failing some models."""

    def __init__(self, fail_tag: str | None = None) -> None:
        self._fail_tag = fail_tag
        self.calls: list[tuple[str, str]] = []

    def execute(
        self, model_tag: str, prompt: str, options: dict[str, object] | None = None
    ) -> ExecutionResult:
        self.calls.append((model_tag, prompt))
        if model_tag == self._fail_tag:
            return ExecutionResult(model=model_tag, prompt=prompt, error="model not found")
        return ExecutionResult(
            model=model_tag,
            prompt=prompt,
            latency_ms=123.4,
            eval_count=42,
            output="hello world",
        )


@pytest.fixture
def registry() -> ModelRegistry:
    """Two-tier registry used across benchmark tests."""
    return ModelRegistry.from_yaml(REGISTRY_YAML)


def test_suite_covers_every_task_type() -> None:
    for task_type in TaskType:
        assert len(tasks_for(task_type)) >= 5, f"not enough tasks for {task_type}"


def test_suite_has_expected_size() -> None:
    total = len(all_tasks())
    assert 30 <= total <= 50
    assert len({task.id for task in all_tasks()}) == total  # ids are unique


def test_run_generates_row_per_model_per_task(registry: ModelRegistry) -> None:
    runner = BenchmarkRunner(registry=registry, executor=FakeExecutor(), tasks=all_tasks())
    rows = runner.run(silent=True)
    assert len(rows) == 2 * len(all_tasks())
    assert all(row.latency_ms == 123.4 for row in rows)
    assert all(row.error is None for row in rows)
    assert rows[0].model_name == "tiny"


def test_run_records_failures_without_aborting(registry: ModelRegistry) -> None:
    runner = BenchmarkRunner(
        registry=registry, executor=FakeExecutor(fail_tag="big:latest"), tasks=all_tasks()
    )
    rows = runner.run(silent=True)
    errors = [row for row in rows if row.error]
    assert len(errors) == len(all_tasks())
    assert all("model not found" in row.error for row in errors)


def test_csv_write_has_header_and_all_rows(registry: ModelRegistry, tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        registry=registry,
        executor=FakeExecutor(),
        tasks=tasks_for(TaskType.DEBUG),
    )
    rows = runner.run(silent=True)
    out = runner.write_csv(rows, tmp_path / "results.csv")

    with open(out, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        content = list(reader)

    assert reader.fieldnames == [
        "model_name",
        "model_tag",
        "task_id",
        "task_type",
        "latency_ms",
        "eval_count",
        "tokens_per_sec",
        "output_chars",
        "error",
    ]
    assert len(content) == len(rows)
    assert content[0]["model_name"] == "tiny"
    assert float(content[0]["tokens_per_sec"]) == pytest.approx(42 / 0.1234)


def test_summary_table_lists_models(registry: ModelRegistry, capsys: pytest.CaptureFixture) -> None:
    runner = BenchmarkRunner(
        registry=registry,
        executor=FakeExecutor(fail_tag="big:latest"),
        tasks=tasks_for(TaskType.DEBUG),
    )
    runner.run(silent=True)
    table = runner.summarize(runner.run(silent=True))
    assert "tiny" in table
    assert "big" in table
    assert "errors" in table


def test_executor_never_receives_benchmark_metadata(
    registry: ModelRegistry,
) -> None:
    executor = FakeExecutor()
    runner = BenchmarkRunner(
        registry=registry, executor=executor, tasks=tasks_for(TaskType.EXPLAIN)
    )
    rows = runner.run(silent=True)
    assert len(executor.calls) == len(rows)
    for model_tag, prompt in executor.calls:
        assert model_tag in ("tiny:latest", "big:latest")
        assert prompt.strip()  # only the raw prompt is sent

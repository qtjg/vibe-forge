"""Tests for the committed evaluation set and its loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeforge.eval.dataset import DEFAULT_EVAL_SET, EvalSetError, EvalTask, load_eval_set
from vibeforge.types import COMPLEXITY_ORDER, Complexity, TaskType

GOOD_META = """\
evaluator: tester
version: 9
tasks:
  - id: t1
    task_type: debug
    prompt: "fix the deadlock"
    ground_truth: high
    rationale: concurrency
  - id: t2
    task_type: explain
    prompt: "what is a decorator"
    ground_truth: low
    rationale: small concept
  - id: t3
    task_type: generate
    prompt: "sum a list"
    ground_truth: trivial
    rationale: one line
  - id: t4
    task_type: refactor
    prompt: "split the god function"
    ground_truth: medium
    rationale: decomposition
"""


def write_set(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_bundled_eval_set_is_committed_and_complete() -> None:
    assert DEFAULT_EVAL_SET.is_file()
    tasks = load_eval_set()

    assert len(tasks) >= 40
    assert len(tasks) == len({task.id for task in tasks})
    by_tier = {tier: 0 for tier in COMPLEXITY_ORDER}
    for task in tasks:
        by_tier[task.ground_truth] += 1
    for tier, count in by_tier.items():
        assert count >= 10, f"tier {tier} has only {count} tasks"
    assert all(task.rationale for task in tasks)


def test_loads_valid_set(tmp_path: Path) -> None:
    tasks = load_eval_set(write_set(tmp_path, GOOD_META))

    assert len(tasks) == 4
    first = tasks[0]
    assert first.id == "t1"
    assert first.task_type == "debug"
    assert first.ground_truth == Complexity.HIGH
    assert isinstance(first, EvalTask)


def test_as_routing_task_preserves_prompt_and_type() -> None:
    task = EvalTask(
        id="x",
        task_type="debug",
        prompt="fix the deadlock",
        ground_truth=Complexity.HIGH,
        rationale="concurrency",
    )

    routing = task.as_routing_task()

    assert routing.prompt == task.prompt
    assert routing.type == TaskType.DEBUG


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="cannot read"):
        load_eval_set(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="not valid YAML"):
        load_eval_set(write_set(tmp_path, "tasks: [unclosed"))


def test_missing_metadata_raises(tmp_path: Path) -> None:
    body = "tasks:\n  - id: t\n    prompt: x\n    task_type: debug\n    ground_truth: high\n"
    with pytest.raises(EvalSetError, match="metadata"):
        load_eval_set(write_set(tmp_path, body))


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    body = GOOD_META.replace("t2", "t1")
    with pytest.raises(EvalSetError, match="duplicate"):
        load_eval_set(write_set(tmp_path, body))


def test_invalid_ground_truth_raises(tmp_path: Path) -> None:
    body = GOOD_META.replace("ground_truth: high", "ground_truth: extreme")
    with pytest.raises(EvalSetError, match="invalid 'ground_truth'"):
        load_eval_set(write_set(tmp_path, body))


def test_uncovered_tier_raises(tmp_path: Path) -> None:
    body = GOOD_META.replace("ground_truth: high", "ground_truth: medium")
    with pytest.raises(EvalSetError, match="cover every tier"):
        load_eval_set(write_set(tmp_path, body))


def test_empty_tasks_list_raises(tmp_path: Path) -> None:
    body = "evaluator: tester\nversion: 1\ntasks: []\n"
    with pytest.raises(EvalSetError, match="non-empty"):
        load_eval_set(write_set(tmp_path, body))

"""Tests for the eval runner (fake scorers, no Ollama)."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from vibeforge.eval.dataset import EvalTask
from vibeforge.eval.runner import CSV_COLUMNS, Evaluator, ScorerReport
from vibeforge.types import Complexity

T = Complexity.TRIVIAL
L = Complexity.LOW
M = Complexity.MEDIUM
H = Complexity.HIGH


def make_tasks() -> list[EvalTask]:
    return [
        EvalTask(id="a", task_type="debug", prompt="p1", ground_truth=H, rationale="r"),
        EvalTask(id="b", task_type="explain", prompt="p2", ground_truth=T, rationale="r"),
        EvalTask(id="c", task_type="generate", prompt="p3", ground_truth=L, rationale="r"),
    ]


class PerfectScorer:
    """Predicts the ground truth tier by reading it off the task id."""

    def score(self, task: object) -> tuple[Complexity, str]:
        by_id = {"p1": H, "p2": T, "p3": L}
        return by_id[task.prompt], "matched"

    def confidence(self, task: object) -> float:
        return 0.9


class FallingBackScorer:
    """Always "falls back" (reason contains the marker) and predicts low."""

    def score(self, task: object) -> tuple[Complexity, str]:
        return L, "embedding unavailable; fell back to heuristic"

    def confidence(self, task: object) -> float | None:
        return None


class SlowScorer:
    """Predicts medium with a measurable delay."""

    def score(self, task: object) -> tuple[Complexity, str]:
        time.sleep(0.001)
        return M, "slow but honest"


def test_report_covers_every_task_with_metrics() -> None:
    tasks = make_tasks()
    evaluator = Evaluator({"perfect": PerfectScorer()})

    reports = evaluator.run(tasks, silent=True)

    report = reports["perfect"]
    assert isinstance(report, ScorerReport)
    assert len(report.rows) == 3
    assert report.metrics.total == 3
    assert report.metrics.correct == 3
    assert report.metrics.accuracy == 1.0
    assert report.fallback_count == 0
    supported = [m for m in report.metrics.per_tier if m.is_supported]
    assert len(supported) == 3  # the set covers T, L, H
    assert all(m.f1 == 1.0 for m in supported)


def test_fallback_detection_counts_matching_tasks() -> None:
    tasks = make_tasks()
    evaluator = Evaluator({"flaky": FallingBackScorer()})

    reports = evaluator.run(tasks, silent=True)

    report = reports["flaky"]
    assert report.fallback_count == 3
    assert all(row.fell_back for row in report.rows)


def test_latency_is_measured_per_scorer() -> None:
    tasks = make_tasks()
    evaluator = Evaluator({"slow": SlowScorer(), "perfect": PerfectScorer()})

    reports = evaluator.run(tasks, silent=True)

    assert reports["slow"].mean_latency_ms is not None
    assert reports["slow"].mean_latency_ms >= 0.9  # 1ms sleep per task avg
    assert reports["slow"].total_latency_ms >= 3.0
    assert reports["perfect"].mean_latency_ms is not None
    assert reports["perfect"].mean_latency_ms < reports["slow"].mean_latency_ms


def test_write_csv_shape(tmp_path: Path) -> None:
    tasks = make_tasks()
    evaluator = Evaluator({"perfect": PerfectScorer(), "flaky": FallingBackScorer()})
    reports = evaluator.run(tasks, silent=True)

    path = evaluator.write_csv(reports, tmp_path / "sub" / "eval.csv")

    assert path.is_file()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6  # 2 scorers x 3 tasks
    assert rows[0]["scorer"] == "perfect"
    assert rows[0]["task_id"] == "a"
    assert rows[0]["ground_truth"] == "high"
    assert rows[0]["predicted"] == "high"
    assert rows[0]["correct"] == "True"
    assert rows[0]["fell_back_to_heuristic"] == "False"
    assert float(rows[0]["latency_ms"]) >= 0.0
    assert rows[3]["scorer"] == "flaky"
    assert rows[3]["fell_back_to_heuristic"] == "True"
    assert rows[3]["predicted"] == "low"
    assert rows[3]["correct"] == "False"


def test_csv_headers_match_contract() -> None:
    assert CSV_COLUMNS == (
        "scorer",
        "task_id",
        "task_type",
        "ground_truth",
        "predicted",
        "correct",
        "fell_back_to_heuristic",
        "latency_ms",
    )


def test_multiple_scorers_report_independently() -> None:
    tasks = make_tasks()
    evaluator = Evaluator({"first": PerfectScorer(), "second": FallingBackScorer()})

    reports = evaluator.run(tasks, silent=True)

    assert list(reports) == ["first", "second"]
    assert reports["first"].metrics.correct == 3
    # 'flaky' predicts low everywhere; task c happens to BE low.
    assert reports["second"].metrics.correct == 1
    assert reports["second"].fallback_count == 3


def test_empty_scorers_raise() -> None:
    with pytest.raises(ValueError):
        Evaluator({})


def test_summarize_reports_the_facts() -> None:
    tasks = make_tasks()
    evaluator = Evaluator({"perfect": PerfectScorer(), "flaky": FallingBackScorer()})
    reports = evaluator.run(tasks, silent=True)

    summary = evaluator.summarize(reports)

    assert "scorer: perfect" in summary
    assert "accuracy:   3/3 (100.0%)" in summary
    assert "macro-F1:" in summary
    assert "confusion matrix" in summary
    assert "scorer: flaky" in summary
    assert "fallbacks:  3 task(s)" in summary
    assert "NOT produced by this scorer" in summary
    assert "latency:" in summary

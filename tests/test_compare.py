"""Tests for the scorer comparison runner (fake scorers, no Ollama)."""

from __future__ import annotations

import csv
from pathlib import Path

from vibeforge.benchmark.compare import COLUMNS, ScorerComparison
from vibeforge.benchmark.tasks import all_tasks
from vibeforge.types import Complexity, Task


class FixedScorer:
    """A scorer with a canned tier per task id."""

    def __init__(self, by_id: dict[str, Complexity]) -> None:
        self._by_id = by_id

    def score(self, task: Task) -> tuple[Complexity, str]:
        return self._by_id.get(task.id, Complexity.LOW), "canned"


def test_rows_cover_every_task_with_stable_columns(tmp_path: Path) -> None:
    tasks = all_tasks()
    heuristic = FixedScorer({task.id: Complexity.MEDIUM for task in tasks})
    embedding = FixedScorer({task.id: Complexity.MEDIUM for task in tasks})
    comparison = ScorerComparison(heuristic=heuristic, embedding=embedding)

    rows = comparison.run(tasks, silent=True)
    path = comparison.write_csv(rows, tmp_path / "compare.csv")

    assert len(rows) == len(tasks)
    assert all(row["agree"] for row in rows)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COLUMNS)
        assert sum(1 for _ in reader) == len(tasks)


def test_disagreements_are_recorded(tmp_path: Path) -> None:
    tasks = all_tasks()
    heuristic = FixedScorer({task.id: Complexity.HIGH for task in tasks})
    embedding = FixedScorer({task.id: Complexity.TRIVIAL for task in tasks})
    comparison = ScorerComparison(heuristic=heuristic, embedding=embedding)

    rows = comparison.run(tasks, silent=True)

    assert all(not row["agree"] for row in rows)
    assert all(row["heuristic_tier"] == "high" for row in rows)
    assert all(row["embedding_tier"] == "trivial" for row in rows)


def test_latency_is_positive_and_measured(tmp_path: Path) -> None:
    tasks = all_tasks()
    comparison = ScorerComparison(heuristic=FixedScorer({}), embedding=FixedScorer({}))

    rows = comparison.run(tasks[:5], silent=True)

    for row in rows:
        assert row["heuristic_latency_ms"] >= 0.0
        assert row["embedding_latency_ms"] >= 0.0


def test_summarize_reports_agreement_rate() -> None:
    tasks = all_tasks()
    comparison = ScorerComparison(heuristic=FixedScorer({}), embedding=FixedScorer({}))

    rows = comparison.run(tasks[:10], silent=True)
    summary = comparison.summarize(rows)

    assert "agreement" in summary
    assert "10/10" in summary
    assert "tasks compared:  10" in summary


def test_summarize_lists_mismatch_pairs() -> None:
    tasks = all_tasks()
    heuristic = FixedScorer({task.id: Complexity.HIGH for task in tasks})
    embedding = FixedScorer({task.id: Complexity.TRIVIAL for task in tasks})
    comparison = ScorerComparison(heuristic=heuristic, embedding=embedding)

    rows = comparison.run(tasks[:5], silent=True)
    summary = comparison.summarize(rows)

    assert "mismatches by tier pair" in summary
    assert "high -> trivial" in summary


def test_empty_run_summarizes_safely() -> None:
    comparison = ScorerComparison(heuristic=FixedScorer({}), embedding=FixedScorer({}))
    assert "no rows" in comparison.summarize([])

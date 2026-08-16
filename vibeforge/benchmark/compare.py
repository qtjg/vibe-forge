"""Side-by-side comparison of scoring strategies on the benchmark suite.

:class:`ScorerComparison` runs the heuristic and the embedding scorer over
every benchmark task and records each one's tier choice and latency. The
output CSV is the committed research artifact for the v0.2.0 release notes:
agreement between the strategies is reported honestly as *agreement*, never
as accuracy against an external ground truth.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from vibeforge.router.complexity import HeuristicScorer, Scorer
from vibeforge.router.embedding import EmbeddingScorer
from vibeforge.types import Task

__all__ = ["ScorerComparison"]

#: CSV columns; stable, never rename once committed.
COLUMNS: tuple[str, ...] = (
    "task_id",
    "task_type",
    "heuristic_tier",
    "embedding_tier",
    "agree",
    "heuristic_latency_ms",
    "embedding_latency_ms",
)


class ScorerComparison:
    """Runs both scorers over a task set and writes a comparison CSV."""

    def __init__(
        self,
        heuristic: Scorer | None = None,
        embedding: Scorer | None = None,
    ) -> None:
        """Build a comparison with default scorer instances."""
        self._heuristic = heuristic if heuristic is not None else HeuristicScorer()
        self._embedding = embedding if embedding is not None else EmbeddingScorer()

    def run(self, tasks: list[Task], silent: bool = False) -> list[dict[str, Any]]:
        """Score every task with both strategies, timing each call.

        Args:
            tasks: The benchmark task set.
            silent: Suppress per-task progress output.

        Returns:
            One row dict per task (see :data:`COLUMNS`).
        """
        rows: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            h_tier, h_ms = self._time_it(self._heuristic, task)
            e_tier, e_ms = self._time_it(self._embedding, task)
            rows.append(
                {
                    "task_id": task.id,
                    "task_type": task.type.value,
                    "heuristic_tier": h_tier.value,
                    "embedding_tier": e_tier.value,
                    "agree": h_tier is e_tier,
                    "heuristic_latency_ms": round(h_ms, 3),
                    "embedding_latency_ms": round(e_ms, 3),
                }
            )
            if not silent:
                mark = "agree" if h_tier is e_tier else "DIFF"
                print(
                    f"[{index}/{len(tasks)}] {task.id}: {h_tier.value} vs {e_tier.value} ({mark})"
                )
        return rows

    def write_csv(self, rows: list[dict[str, Any]], path: Path) -> Path:
        """Write rows to ``path`` in the stable column order."""
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def summarize(self, rows: list[dict[str, Any]]) -> str:
        """Human-readable summary of agreement and per-strategy latency."""
        total = len(rows)
        if total == 0:
            return "no rows to summarize"
        agreements = sum(1 for row in rows if row["agree"])
        h_lat = sum(float(row["heuristic_latency_ms"]) for row in rows)
        e_lat = sum(float(row["embedding_latency_ms"]) for row in rows)
        lines = [
            f"tasks compared:  {total}",
            f"tier agreement:  {agreements}/{total} ({agreements / total:.1%})",
            f"heuristic total: {h_lat:.0f} ms",
            f"embedding total: {e_lat:.0f} ms (incl. model warm-up)",
        ]
        diffs: dict[tuple[str, str], int] = {}
        for row in rows:
            if not row["agree"]:
                key = (row["heuristic_tier"], row["embedding_tier"])
                diffs[key] = diffs.get(key, 0) + 1
        if diffs:
            lines.append("mismatches by tier pair:")
            for (h, e), count in sorted(diffs.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {h} -> {e}: {count}")
        return "\n".join(lines)

    def _time_it(self, scorer: Scorer, task: Task) -> tuple[Any, float]:
        """Score ``task`` with ``scorer`` and return (tier, latency_ms)."""
        started = time.perf_counter()
        tier, _ = scorer.score(task)
        return tier, (time.perf_counter() - started) * 1000.0

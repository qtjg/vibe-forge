"""Run scorers against the labeled evaluation set and collect evidence.

The runner is the piece that turns "heuristic vs embedding, trust me"
into a reproducible result: it scores every task with every given
scorer, times each call, detects silent fallbacks (an embedding scorer
that fell back to the heuristic must be reported, not hidden), and
writes a CSV any researcher can diff against.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vibeforge.eval.dataset import EvalTask
from vibeforge.eval.metrics import EvaluationMetrics, evaluate
from vibeforge.router.complexity import Scorer
from vibeforge.types import Complexity

__all__ = ["EvalRow", "ScorerReport", "Evaluator"]

#: CSV columns (stable; changes are breaking for downstream analysis).
CSV_COLUMNS = (
    "scorer",
    "task_id",
    "task_type",
    "ground_truth",
    "predicted",
    "correct",
    "fell_back_to_heuristic",
    "latency_ms",
)


@dataclass(frozen=True)
class EvalRow:
    """One (scorer, task) scoring result."""

    scorer: str
    task: EvalTask
    predicted: Complexity
    latency_ms: float
    fell_back: bool


@dataclass(frozen=True)
class ScorerReport:
    """The evidence for one scorer: predictions, metrics, and timing."""

    name: str
    rows: tuple[EvalRow, ...]
    metrics: EvaluationMetrics
    fallback_count: int
    mean_latency_ms: float | None
    median_latency_ms: float | None
    total_latency_ms: float

    @property
    def latency_ms_per_task(self) -> tuple[float, ...]:
        """Per-task latencies, including any warm-up cost."""
        return tuple(row.latency_ms for row in self.rows)


class Evaluator:
    """Score a labeled set with several scorers and collect evidence.

    Args:
        scorers: Ordered mapping of scorer name -> scorer instance.
            Order is preserved in reports and CSVs.
        fallback_marker: Substring of a scorer's reason string that
            means "this prediction was produced by falling back to the
            heuristic" (the EmbeddingScorer's documented marker).
    """

    def __init__(
        self,
        scorers: Mapping[str, Scorer],
        fallback_marker: str = "fell back",
    ) -> None:
        """Build an evaluator for the given scorers."""
        if not scorers:
            raise ValueError("at least one scorer is required")
        self._scorers: dict[str, Scorer] = dict(scorers)
        self._fallback_marker = fallback_marker

    def run(self, tasks: Sequence[EvalTask], silent: bool = False) -> dict[str, ScorerReport]:
        """Score every task with every scorer.

        Args:
            tasks: The labeled tasks to evaluate.
            silent: When True, suppress the progress line.

        Returns:
            Mapping of scorer name -> :class:`ScorerReport`, in the
            order the scorers were given.
        """
        reports: dict[str, ScorerReport] = {}
        for name, scorer in self._scorers.items():
            truth_list: list[Complexity] = []
            predicted_list: list[Complexity] = []
            rows: list[EvalRow] = []
            fallbacks = 0
            if not silent:
                print(f"evaluating scorer {name!r}: {len(tasks)} tasks")

            for task in tasks:
                started = time.perf_counter()
                tier, reason = scorer.score(task.as_routing_task())
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                truth_list.append(task.ground_truth)
                predicted_list.append(tier)
                fell_back = self._fallback_marker in reason
                if fell_back:
                    fallbacks += 1
                rows.append(
                    EvalRow(
                        scorer=name,
                        task=task,
                        predicted=tier,
                        latency_ms=elapsed_ms,
                        fell_back=fell_back,
                    )
                )

            metrics = evaluate(tuple(truth_list), tuple(predicted_list), name=name)
            latencies = [row.latency_ms for row in rows]
            reports[name] = ScorerReport(
                name=name,
                rows=tuple(rows),
                metrics=metrics,
                fallback_count=fallbacks,
                mean_latency_ms=_mean(latencies),
                median_latency_ms=_median(latencies),
                total_latency_ms=sum(latencies),
            )
        return reports

    @staticmethod
    def write_csv(reports: Mapping[str, ScorerReport], path: str | Path) -> Path:
        """Write one row per (scorer, task) to ``path``.

        Returns:
            The resolved output path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for report in reports.values():
                for row in report.rows:
                    writer.writerow(
                        {
                            "scorer": row.scorer,
                            "task_id": row.task.id,
                            "task_type": row.task.task_type,
                            "ground_truth": row.task.ground_truth.value,
                            "predicted": row.predicted.value,
                            "correct": (
                                "True" if row.predicted is row.task.ground_truth else "False"
                            ),
                            "fell_back_to_heuristic": "True" if row.fell_back else "False",
                            "latency_ms": f"{row.latency_ms:.3f}",
                        }
                    )
        return output

    def summarize(self, reports: Mapping[str, ScorerReport]) -> str:
        """Render a compact, honest summary of all reports."""
        lines: list[str] = []
        for report in reports.values():
            m = report.metrics
            lines.append(f"\nscorer: {report.name}")
            lines.append("-" * (len(report.name) + 8))
            accuracy = f"{m.accuracy:.1%}" if m.accuracy is not None else "--"
            macro = f"{m.macro_f1:.3f}" if m.macro_f1 is not None else "--"
            lines.append(f"accuracy:   {m.correct}/{m.total} ({accuracy})")
            lines.append(f"macro-F1:   {macro}")
            if report.fallback_count:
                lines.append(
                    f"fallbacks:  {report.fallback_count} task(s) {self._fallback_marker!r} "
                    f"(results for those tasks are NOT produced by this scorer)"
                )
            lines.append("per-tier precision/recall/F1:")
            lines.append("  tier        precision  recall  f1       support")
            for tier_metrics in m.per_tier:
                lines.append(
                    f"  {tier_metrics.tier.value:<12}"
                    f"{_fmt(tier_metrics.precision):>11}"
                    f"{_fmt(tier_metrics.recall):>9}"
                    f"{_fmt(tier_metrics.f1):>8}"
                    f"{tier_metrics.support:>9}"
                )
            lines.append("confusion matrix (rows=true, cols=predicted):")
            lines.append(str(m.confusion))
            latency = "n/a"
            if report.mean_latency_ms is not None:
                latency = (
                    f"mean {report.mean_latency_ms:.1f} ms, "
                    f"median {report.median_latency_ms:.1f} ms"
                )
            lines.append(
                f"latency:    {latency}, total {report.total_latency_ms:.1f} ms (incl. warm-up)"
            )
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    """Render a metric as 3 decimals, or a dash placeholder when undefined."""
    return f"{value:.3f}" if value is not None else "   --  "


def _mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None for empty input."""
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    """Median, or None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0

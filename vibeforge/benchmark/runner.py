"""Benchmark runner: every configured model against the fixed task set.

Results are written to CSV in a flat, pandas-friendly shape -- one row per
(model, task) run -- and printed as a per-model summary table.

The CSV is flushed incrementally (once per model) so a Ctrl+C mid-run still
leaves the runs completed so far on disk for the research paper's raw data.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from vibeforge.benchmark.tasks import BenchmarkTask
from vibeforge.router.registry import ModelRegistry
from vibeforge.types import ExecutionResult, ModelTier

__all__ = ["BenchmarkRow", "BenchmarkRunner", "BenchmarkInterrupted"]

#: CSV columns; keep this stable across releases so results stay comparable.
CSV_COLUMNS: tuple[str, ...] = (
    "model_name",
    "model_tag",
    "task_id",
    "task_type",
    "latency_ms",
    "eval_count",
    "tokens_per_sec",
    "output_chars",
    "error",
)


class RunnerExecutor(Protocol):
    """Anything that can execute a prompt against an Ollama tag."""

    def execute(
        self, model_tag: str, prompt: str, options: dict[str, object] | None = None
    ) -> ExecutionResult: ...


class BenchmarkInterrupted(Exception):
    """Raised when a benchmark run is stopped mid-way (Ctrl+C).

    The runs completed up to that point are already flushed to the CSV.
    """

    def __init__(self, completed: int, total: int, csv_path: Path) -> None:
        """Record how far the run got before being interrupted."""
        super().__init__(f"benchmark interrupted after {completed}/{total} runs")
        self.completed = completed
        self.total = total
        self.csv_path = csv_path


class BenchmarkRow:
    """One (model, task) run, ready for CSV or pandas."""

    __slots__ = (
        "model_name",
        "model_tag",
        "task_id",
        "task_type",
        "latency_ms",
        "eval_count",
        "tokens_per_sec",
        "output_chars",
        "error",
    )

    def __init__(self, model: ModelTier, task: BenchmarkTask, result: ExecutionResult) -> None:
        """Build a row from the model, task, and execution result."""
        self.model_name = model.name
        self.model_tag = model.ollama_tag
        self.task_id = task.id
        self.task_type = task.type.value
        self.latency_ms = result.latency_ms
        self.eval_count = result.eval_count
        self.tokens_per_sec = result.tokens_per_sec
        self.output_chars = len(result.output) if result.output else 0
        self.error = result.error

    def as_dict(self) -> dict[str, object]:
        """Serialize the row to a plain dict aligned with :data:`CSV_COLUMNS`."""
        return {column: getattr(self, column) for column in CSV_COLUMNS}


class BenchmarkRunner:
    """Runs the benchmark task set against every registered model.

    Examples:
        >>> runner = BenchmarkRunner(
        ...     registry=ModelRegistry.load_default(),
        ...     executor=OllamaExecutor(),
        ...     tasks=all_tasks(),
        ... )
        >>> rows = runner.run(output_path="benchmark_results.csv")
    """

    def __init__(
        self,
        registry: ModelRegistry,
        executor: RunnerExecutor,
        tasks: Sequence[BenchmarkTask],
    ) -> None:
        """Bind a registry, an executor, and the task set to run.

        Args:
            registry: Models to benchmark, in cheapest-first order.
            executor: How to run prompts (real Ollama or a fake in tests).
            tasks: The benchmark task set to run.
        """
        self._registry = registry
        self._executor = executor
        self._tasks = tuple(tasks)

    def run(
        self,
        output_path: str | Path | None = None,
        silent: bool = False,
    ) -> list[BenchmarkRow]:
        """Run every task against every model, in a model-major loop.

        Args:
            output_path: When given, the rows completed so far are flushed to
                this CSV after every model, and on interruption.
            silent: When true, don't print per-task progress lines.

        Returns:
            One :class:`BenchmarkRow` per (model, task) pair, including rows
            where the run failed (``error`` set).

        Raises:
            BenchmarkInterrupted: On Ctrl+C; completed rows are already
                flushed to ``output_path`` when it was provided.
        """
        rows: list[BenchmarkRow] = []
        total = len(self._registry.models) * len(self._tasks)
        start = time.perf_counter()
        try:
            for model in self._registry.models:
                for task in self._tasks:
                    if not silent:
                        print(f"[{model.name}] {task.id} ... ", end="", flush=True)
                    result = self._executor.execute(
                        model.ollama_tag,
                        task.prompt,
                        options={"temperature": 0.0, "num_predict": 2048},
                    )
                    rows.append(BenchmarkRow(model=model, task=task, result=result))
                    if not silent:
                        print("ok" if result.ok else "ERROR", flush=True)
                if output_path is not None:
                    self.write_csv(rows, output_path)
            if not silent:
                elapsed = time.perf_counter() - start
                print(f"\ndone in {elapsed:.1f}s across {len(self._registry.models)} model(s)")
            return rows
        except KeyboardInterrupt as exc:
            path = Path(output_path) if output_path is not None else Path()
            if rows and output_path is not None:
                self.write_csv(rows, path)
            raise BenchmarkInterrupted(completed=len(rows), total=total, csv_path=path) from exc

    def write_csv(self, rows: Sequence[BenchmarkRow], path: str | Path) -> Path:
        """Write the benchmark rows to a pandas-friendly CSV.

        Args:
            rows: Rows produced by :meth:`run`.
            path: Output file (e.g. ``benchmark_results.csv``).

        Returns:
            The path the CSV was written to.
        """
        output = Path(path)
        with open(output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.as_dict())
        return output

    def summarize(self, rows: Sequence[BenchmarkRow]) -> str:
        """Render a compact per-model summary table (simple, parseable output)."""
        by_model: dict[str, dict[str, float | int]] = {}
        for row in rows:
            stats = by_model.setdefault(
                row.model_name,
                {"runs": 0, "errors": 0, "latency_sum": 0.0, "tokens": 0},
            )
            stats["runs"] += 1
            if row.error:
                stats["errors"] += 1
            if row.latency_ms is not None:
                stats["latency_sum"] += row.latency_ms
            if row.eval_count is not None:
                stats["tokens"] += row.eval_count

        header = (
            f"{'model':<14} {'runs':>5} {'errors':>7} {'avg ms':>10} {'tok/s':>10} {'tokens':>8}"
        )
        lines = [header, "-" * len(header)]
        for name, stats in by_model.items():
            ok_runs = stats["runs"] - stats["errors"]
            avg_ms = stats["latency_sum"] / ok_runs if ok_runs and stats["latency_sum"] else 0.0
            secs = avg_ms / 1000.0
            tok_ps = stats["tokens"] / secs if secs else 0.0
            lines.append(
                f"{name:<14} {stats['runs']:>5} {stats['errors']:>7} "
                f"{avg_ms:>10.1f} {tok_ps:>10.1f} {stats['tokens']:>8}"
            )
        return "\n".join(lines)

"""Dataset construction and model artifact training for the ML scorer.

The pipeline has two halves, each independently runnable:

1. :func:`build_dataset` -- pull labeled rows from the benchmark CSV
   (one row per unique task of the fixed 36-task suite, labeled by
   :func:`vibeforge.router.ml.features.heuristic_tier`) and from the
   dashboard's SQLite history (via :class:`vibeforge.dashboard.store.HistoryDB`,
   the *same* access path the dashboard uses), weakly validating history
   rows: any decision that errored or took unreasonably long for its tier
   is dropped and counted.

2. :func:`train_and_save` -- fit a classifier on the feature vectors and
   persist two artifacts to disk: ``dataset.csv`` (the labeled rows) and
   ``scorer.joblib`` (the fitted estimator). Nothing here touches the
   router: the model becomes routable only in the next phase (3.2).

The model fit is the only sklearn-touching step and it is injected as a
factory so tests stub it; the default factory imports scikit-learn lazily,
so the base install (no ``[ml]`` extra) never loads it.
"""

from __future__ import annotations

import csv
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from vibeforge.dashboard.store import HistoryDB
from vibeforge.router.ml.features import feature_vector, heuristic_tier
from vibeforge.types import Complexity, Task

__all__ = [
    "DatasetRow",
    "TrainingDataset",
    "build_dataset",
    "train_and_save",
    "default_model_factory",
    "TIER_LATENCY_CAP_MS",
]

#: Weakly-validated history rows must not have errors, and must be quicker
#: than this many ms for their tier. Generous, but a 60 s "trivial" call is
#: almost certainly a stalled model, not a real routing signal.
TIER_LATENCY_CAP_MS: dict[str, float] = {
    Complexity.TRIVIAL: 30_000,
    Complexity.LOW: 60_000,
    Complexity.MEDIUM: 180_000,
    Complexity.HIGH: 600_000,
}


@dataclass(frozen=True)
class DatasetRow:
    """One labeled row feeding the classifier.

    Attributes:
        task_id: Stable id; ``benchmark-<id>`` for fixed-suite rows and
            ``history-<n>`` for SQLite rows.
        source: ``"benchmark"`` or ``"history"``.
        task_type: Routing task type of the prompt.
        features: ``[baseline_rank, word_count, keyword_hits]``.
        ground_truth: Complexity tier label.
    """

    task_id: str
    source: str
    task_type: str
    features: list[float]
    ground_truth: Complexity


@dataclass
class TrainingDataset:
    """The assembled labeled dataset plus provenance counts.

    Attributes:
        rows: One entry per kept row, in build order (benchmark first,
            then history).
        benchmark_rows: Number of unique fixed-suite tasks included.
        history_rows: Number of kept (weakly validated) history rows.
        history_rejected: List of reasons from dropped history rows.
    """

    rows: list[DatasetRow] = field(default_factory=list)
    benchmark_rows: int = 0
    history_rows: int = 0
    history_rejected: list[str] = field(default_factory=list)

    @property
    def X(self) -> list[list[float]]:
        """Feature matrix, one row per dataset row."""
        return [row.features for row in self.rows]

    @property
    def y(self) -> list[int]:
        """Target ranks (0..3), one per dataset row."""
        return [row.ground_truth.rank for row in self.rows]

    def __len__(self) -> int:
        """Number of labeled rows."""
        return len(self.rows)


def _iter_benchmark_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_dataset(
    benchmark_csv: Path,
    history_db: Path | None = None,
    benchmark_reader: Callable[[Path], list[dict[str, str]]] | None = None,
    history_factory: Callable[[Path], HistoryDB] | None = None,
    log: Callable[[str], None] = print,
) -> TrainingDataset:
    """Build the labeled dataset from the benchmark CSV and optional history.

    Args:
        benchmark_csv: Path to ``benchmark_results.csv`` (columns
            ``model_name, model_tag, task_id, task_type, ...``).
        history_db: SQLite database path, or ``None`` to skip history
            (the command must still work off the benchmark CSV alone).
        benchmark_reader: Injected CSV reader (default: stdlib
            ``csv.DictReader``).
        history_factory: Injected store opener (default: ``HistoryDB``).
        log: Line sink for progress/rejection notes.

    Returns:
        The assembled dataset. History may contribute zero rows when the
        database has no decision history yet; that is not an error.
    """
    from vibeforge.benchmark.tasks import TASKS

    dataset = TrainingDataset()

    rows = (
        benchmark_reader(benchmark_csv) if benchmark_reader else _iter_benchmark_rows(benchmark_csv)
    )
    tasks_by_id = {task.id: task for task in TASKS}
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        benchmark_task = tasks_by_id.get(task_id)
        if benchmark_task is None:
            continue
        task = Task(
            type=benchmark_task.type, prompt=benchmark_task.prompt, context=benchmark_task.context
        )
        dataset.rows.append(
            DatasetRow(
                task_id=f"benchmark-{task_id}",
                source="benchmark",
                task_type=task.type,
                features=feature_vector(task),
                ground_truth=heuristic_tier(task),
            )
        )
    dataset.benchmark_rows = len(dataset.rows)

    if history_db is not None:
        history = history_factory(history_db) if history_factory else HistoryDB(history_db)
        try:
            decisions = history.recent(10_000)
        finally:
            history.close()
        for index, decision in enumerate(decisions):
            reason = _validate_history(decision)
            if reason is not None:
                dataset.history_rejected.append(reason)
                continue
            task_type = str(decision["task_type"])
            task = Task(type=task_type, prompt=str(decision["prompt"]))
            dataset.rows.append(
                DatasetRow(
                    task_id=f"history-{index}",
                    source="history",
                    task_type=task_type,
                    features=feature_vector(task),
                    ground_truth=Complexity(decision["complexity"]),
                )
            )
        dataset.history_rows = len(dataset.rows) - dataset.benchmark_rows
        log(
            f"history: {len(decisions)} decisions found, "
            f"{dataset.history_rows} kept, {len(dataset.history_rejected)} rejected"
        )

    return dataset


def _validate_history(decision: dict) -> str | None:
    """Return a rejection reason for a history row, or ``None`` to keep it.

    Weak validation: the decision must have the routing fields, must not
    have errored, and must be within the per-tier latency cap.
    """
    if "prompt" not in decision or not str(decision.get("prompt", "")).strip():
        return "missing prompt"
    if "task_type" not in decision or not str(decision.get("task_type", "")).strip():
        return "missing task_type"
    try:
        tier = Complexity(str(decision["complexity"]))
    except (KeyError, ValueError):
        return "invalid complexity"
    if decision.get("execution_error"):
        return "execution error"
    latency = decision.get("latency_ms")
    if latency is not None and float(latency) > TIER_LATENCY_CAP_MS[tier]:
        return f"latency {latency}ms over {TIER_LATENCY_CAP_MS[tier]}ms cap for {tier.value}"
    return None


def default_model_factory() -> Callable[[], object]:
    """Build the classifier factory used when none is injected.

    Imports scikit-learn lazily: the base install never sees it. A small
    random forest is a deliberate choice: it trains in well under a second
    on the ~40-1000 rows this pipeline produces, and predicts in the
    microsecond range -- comfortably below the ~100 ms embedding scorer.
    """
    from sklearn.ensemble import RandomForestClassifier

    return lambda: RandomForestClassifier(n_estimators=200, random_state=0)


def train_and_save(
    dataset: TrainingDataset,
    output_dir: Path,
    model_factory: Callable[[], object] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    """Fit a classifier and write ``dataset.csv`` + ``scorer.joblib``.

    Args:
        dataset: The assembled labeled dataset.
        output_dir: Directory to write artifacts into (created if missing).
        model_factory: Callable returning a fresh fit-able estimator; the
            default is :func:`default_model_factory` (lazy sklearn).
        log: Line sink for progress notes.

    Returns:
        ``(dataset_csv, model_path)``.

    Raises:
        ImportError: When no model factory is given and scikit-learn is not
            installed (callers turn this into an install hint).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "source",
                "task_type",
                "baseline_rank",
                "word_count",
                "keyword_hits",
                "ground_truth",
            ]
        )
        for row in dataset.rows:
            writer.writerow(
                [row.task_id, row.source, row.task_type, *row.features, row.ground_truth.value]
            )

    factory = model_factory if model_factory is not None else default_model_factory()
    model = factory()
    model.fit(dataset.X, dataset.y)

    model_path = output_dir / "scorer.joblib"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    rows_note = (
        f"trained on {len(dataset)} rows: {dataset.benchmark_rows} benchmark, "
        f"{dataset.history_rows} history"
    )
    log(rows_note)
    log(f"artifacts: {csv_path} and {model_path}")
    return csv_path, model_path

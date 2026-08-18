"""Tests for the Phase 3.1 ML scorer training data pipeline.

Verified here: dataset construction from a fixture benchmark CSV and a
fixture SQLite decision store, weak validation of history rows, graceful
behavior with empty history (the command must work off the benchmark CSV
alone), and artifact writing. The model fit is stubbed -- CI verifies
pipeline wiring, not training honesty, so no sklearn dependency is
required to run this module.
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import vibeforge.cli.main as cli_main
from vibeforge.dashboard.store import HistoryDB
from vibeforge.router.ml.features import feature_vector, heuristic_tier
from vibeforge.router.ml.pipeline import build_dataset, train_and_save

runner = CliRunner()

CSV_HEADER = [
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


def write_csv(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    return path


def bench_row(*fields: str) -> list[str]:
    return list(fields)


@pytest.fixture
def bench_csv(tmp_path: Path) -> Path:
    r = bench_row
    rows = [
        r(
            "tiny-fast",
            "qwen2.5:0.5b",
            "autocomplete-01",
            "autocomplete",
            "42",
            "5",
            "119",
            "12",
            "",
        ),
        r(
            "tiny-fast",
            "qwen2.5:0.5b",
            "autocomplete-01",
            "autocomplete",
            "41",
            "5",
            "121",
            "12",
            "",
        ),
        r("heavy", "qwen2.5-coder:14b", "debug-04", "debug", "120000", "300", "2", "800", ""),
        r(
            "balanced",
            "llama3.1:latest",
            "debug-05",
            "debug",
            "90000",
            "150",
            "1",
            "400",
            "model timed out",
        ),
        r("tiny-fast", "qwen2.5:0.5b", "unknown-task-99", "debug", "42", "5", "119", "12", ""),
    ]
    return write_csv(tmp_path / "bench.csv", rows)


def add_decision(db: HistoryDB, **overrides: object) -> None:
    base: dict[str, object] = {
        "task_type": "debug",
        "prompt": "why does this crash with a race condition?",
        "complexity": "high",
        "latency_ms": 500,
        "execution_error": "",
    }
    base.update(**overrides)
    db.add(base)


@pytest.fixture
def history_db(tmp_path: Path) -> Path:
    path = tmp_path / "history.db"
    db = HistoryDB(path)
    add_decision(db)
    add_decision(db, task_type="explain", prompt="explain the GIL", complexity="low", latency_ms=50)
    add_decision(db, execution_error="connection refused")
    add_decision(db, latency_ms=60_000, complexity="trivial")  # over trivial cap
    add_decision(db, prompt="   ")
    add_decision(db, complexity="not-a-tier")
    db.close()
    return path


def test_benchmark_csv_rows_become_labeled_feature_rows(
    bench_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = build_dataset(benchmark_csv=bench_csv)

    assert dataset.benchmark_rows == 3  # autocomplete-01 only once (deduped)
    assert dataset.history_rows == 0
    ids = [row.task_id for row in dataset.rows]
    assert ids == ["benchmark-autocomplete-01", "benchmark-debug-04", "benchmark-debug-05"]

    first = dataset.rows[0]
    assert first.source == "benchmark"
    assert first.features[0] == 0.0  # autocomplete baseline
    assert first.features[1] == 13.0  # prompt word count
    assert first.features[2] == 0.0  # no keywords in autocomplete-01
    assert first.ground_truth == heuristic_tier(_task(first.task_type))
    assert len(dataset.X) == 3
    assert dataset.y == [row.ground_truth.rank for row in dataset.rows]


def test_history_rows_weakly_validated_and_counted(bench_csv: Path, history_db: Path) -> None:
    dataset = build_dataset(benchmark_csv=bench_csv, history_db=history_db, log=lambda _: None)

    assert dataset.history_rows == 2
    assert len(dataset.history_rejected) == 4
    reasons = " ".join(dataset.history_rejected)
    assert "execution error" in reasons
    assert "over" in reasons  # latency cap
    assert "missing prompt" in reasons
    assert "invalid complexity" in reasons
    history = [row for row in dataset.rows if row.source == "history"]
    assert {row.ground_truth.value for row in history} == {"high", "low"}


def test_empty_history_still_builds_from_benchmark_alone(
    bench_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.db"
    HistoryDB(path).close()
    dataset = build_dataset(benchmark_csv=bench_csv, history_db=path)

    assert dataset.benchmark_rows == 3
    assert dataset.history_rows == 0
    assert len(dataset.history_rejected) == 0
    out = capsys.readouterr().out
    assert "0 decisions found" in out


class StubModel:
    """Dumb estimator: records the fit, returns itself. No sklearn."""

    def fit(self, X: list[list[float]], y: list[int]) -> StubModel:
        self.X = X
        self.y = y
        return self


def test_train_and_save_writes_both_artifacts_with_injected_factory(
    bench_csv: Path, tmp_path: Path
) -> None:
    dataset = build_dataset(benchmark_csv=bench_csv)
    output = tmp_path / "artifacts"
    csv_path, model_path = train_and_save(dataset, output, model_factory=StubModel)

    assert csv_path.exists()
    assert model_path.exists()
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["task_id"] == "benchmark-autocomplete-01"
    assert rows[0]["ground_truth"] == "trivial"
    assert rows[0]["baseline_rank"] == "0.0"

    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    assert isinstance(model, StubModel)
    assert len(model.X) == 3


def test_train_and_save_import_error_surfaces_cleanly_without_sklearn(
    bench_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    dataset = build_dataset(benchmark_csv=bench_csv)
    with pytest.raises(ImportError):
        train_and_save(dataset, tmp_path / "out")


def test_cli_command_builds_dataset_and_writes_artifacts(
    bench_csv: Path, history_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibeforge.router.ml.pipeline.default_model_factory", lambda: StubModel)
    output = tmp_path / "out"
    result = runner.invoke(
        cli_main.app,
        [
            "train-scorer",
            "--input",
            str(bench_csv),
            "--history",
            str(history_db),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "2 kept, 4 rejected" in result.output
    assert "not routed until Phase 3.2" in result.output
    assert (output / "dataset.csv").is_file()
    assert (output / "scorer.joblib").is_file()


def test_cli_command_is_fine_without_history_store(
    bench_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibeforge.router.ml.pipeline.default_model_factory", lambda: StubModel)
    missing = tmp_path / "no-history.db"
    result = runner.invoke(
        cli_main.app,
        [
            "train-scorer",
            "--input",
            str(bench_csv),
            "--history",
            str(missing),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "no store at" in result.output
    assert (tmp_path / "out" / "dataset.csv").is_file()


def test_cli_command_reports_missing_benchmark_csv(tmp_path: Path) -> None:
    result = runner.invoke(
        cli_main.app,
        ["train-scorer", "--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 1
    assert "benchmark CSV not found" in result.output


def test_feature_vector_matches_field_order_and_values() -> None:
    from vibeforge.types import Task

    task = Task(type="debug", prompt="fix this race condition in the distributed cache")
    vec = feature_vector(task)
    assert len(vec) == 3
    assert vec[0] == 2.0  # debug baseline
    assert vec[1] == 8.0
    assert vec[2] >= 1.0  # 'race' is a high-signal keyword


def _task(task_type: str) -> object:
    from vibeforge.types import Task

    return Task(type=task_type, prompt="test prompt", context="extra context")

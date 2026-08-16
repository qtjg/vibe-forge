"""CLI-level tests for `vibeforge eval` (offline and deterministic).

The heuristic runs are pure rule logic; the embedding runs are forced
into their fallback path by stubbing the Ollama client, so these tests
never touch the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import vibeforge.cli.main as cli_main
import vibeforge.router.embedding as embedding_module

runner = CliRunner()


def test_eval_heuristic_is_offline_and_reports_metrics(tmp_path: Path) -> None:
    output = tmp_path / "eval.csv"

    result = runner.invoke(cli_main.app, ["eval", "--scorer", "heuristic", "--output", str(output)])

    assert result.exit_code == 0
    assert "eval set:  48 labeled tasks" in result.stdout
    assert "scorer: heuristic" in result.stdout
    assert "accuracy:" in result.stdout
    assert "confusion matrix" in result.stdout
    assert "latency:" in result.stdout
    assert output.is_file()


def test_eval_embedding_falls_back_offline_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_connection(*args: object, **kwargs: object) -> object:
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(embedding_module.ollama, "Client", lambda **_: raise_connection())
    output = tmp_path / "eval.csv"

    result = runner.invoke(cli_main.app, ["eval", "--scorer", "embedding", "--output", str(output)])

    assert result.exit_code == 0
    assert "scorer: embedding" in result.stdout
    assert "fallbacks:" in result.stdout
    assert "NOT produced by this scorer" in result.stdout
    assert output.is_file()


def test_eval_rejects_unknown_scorer(tmp_path: Path) -> None:
    result = runner.invoke(cli_main.app, ["eval", "--scorer", "bogus"])

    assert result.exit_code == 1
    assert "unknown scorer" in result.stderr


def test_eval_rejects_empty_scorer_list() -> None:
    result = runner.invoke(cli_main.app, ["eval", "--scorer", ",,,"])

    assert result.exit_code == 1
    assert "at least one scorer" in result.stderr

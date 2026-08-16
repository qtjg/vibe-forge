"""CLI-level tests: config errors and failure modes must be readable.

These run the real ``vibeforge`` typer app via ``CliRunner`` (no server,
no Ollama). The intent: a stranger hitting a broken config or a dead
Ollama sees a one-line message with a hint, never a traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import vibeforge.cli.main as cli_main
from vibeforge.types import ExecutionResult

runner = CliRunner()

MODELS_OK = """\
models:
  - name: tiny-fast
    ollama_tag: qwen2.5:0.5b
    complexity_ceiling: trivial
    approx_ram_gb: 0.6
"""


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point VIBEFORGE_MODELS at a temp config; return its path."""
    path = tmp_path / "models.yaml"
    monkeypatch.setenv("VIBEFORGE_MODELS", str(path))
    return path


def test_route_ok_with_valid_config(isolated: Path) -> None:
    isolated.write_text(MODELS_OK)
    result = runner.invoke(cli_main.app, ["route", "fix the race condition", "--type", "debug"])

    assert result.exit_code == 0
    assert "chosen model:  tiny-fast" in result.stdout
    assert "config:" in result.stdout
    assert "Traceback" not in result.stderr


def test_route_reports_malformed_yaml_cleanly(isolated: Path) -> None:
    isolated.write_text("models: [unclosed")

    result = runner.invoke(cli_main.app, ["route", "hello", "--type", "debug"])

    assert result.exit_code == 1
    assert "not valid YAML" in result.stderr
    assert "Traceback" not in result.stderr


def test_route_reports_invalid_complexity_tier_cleanly(isolated: Path) -> None:
    isolated.write_text(
        "models:\n"
        "  - name: bad\n"
        "    ollama_tag: bad:latest\n"
        "    complexity_ceiling: extreme\n"
        "    approx_ram_gb: 1\n"
    )

    result = runner.invoke(cli_main.app, ["route", "hello", "--type", "debug"])

    assert result.exit_code == 1
    assert "models[0].complexity_ceiling" in result.stderr
    assert "trivial" in result.stderr
    assert "Traceback" not in result.stderr


def test_route_reports_missing_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "no-such.yaml"
    monkeypatch.setenv("VIBEFORGE_MODELS", str(missing))

    result = runner.invoke(cli_main.app, ["route", "hello", "--type", "debug"])

    assert result.exit_code == 1
    assert "missing file" in result.stderr


def test_route_reports_unknown_scorer(isolated: Path) -> None:
    isolated.write_text(MODELS_OK)

    result = runner.invoke(cli_main.app, ["route", "hello", "--scorer", "bogus"])

    assert result.exit_code == 1
    assert "unknown scorer" in result.stderr


def _install_fake_executor(monkeypatch: pytest.MonkeyPatch, result: ExecutionResult) -> None:
    """Replace the CLI's executor factory with one returning ``result``."""

    def fake_factory(*args: object, **kwargs: object) -> object:
        class FakeExecutor:
            def execute(
                self, model_tag: str, prompt: str, options: dict | None = None
            ) -> ExecutionResult:
                return result

        return FakeExecutor()

    monkeypatch.setattr(cli_main, "OllamaExecutor", fake_factory)


def test_route_execute_hints_model_not_pulled(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated.write_text(MODELS_OK)
    _install_fake_executor(
        monkeypatch,
        ExecutionResult(
            model="nope:latest",
            prompt="hello",
            error="Ollama error (HTTP 404): model 'nope:latest' not found",
            error_kind="http",
            status_code=404,
        ),
    )

    result = runner.invoke(cli_main.app, ["route", "hello", "--execute"])

    assert result.exit_code == 0
    assert "error: Ollama error (HTTP 404)" in result.stderr
    assert "not pulled" in result.stderr
    assert "ollama pull nope:latest" in result.stderr


def test_route_execute_hints_when_ollama_down(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated.write_text(MODELS_OK)
    _install_fake_executor(
        monkeypatch,
        ExecutionResult(
            model="qwen2.5:0.5b",
            prompt="hello",
            error="cannot reach Ollama at http://localhost:11434",
            error_kind="connection",
        ),
    )

    result = runner.invoke(cli_main.app, ["route", "hello", "--execute"])

    assert result.exit_code == 0
    assert "ollama serve" in result.stderr


def test_route_execute_prints_output_and_latency(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated.write_text(MODELS_OK)
    _install_fake_executor(
        monkeypatch,
        ExecutionResult(
            model="qwen2.5:0.5b",
            prompt="hello",
            latency_ms=123.4,
            eval_count=42,
            output="hello back",
        ),
    )

    result = runner.invoke(cli_main.app, ["route", "hello", "--execute"])

    assert result.exit_code == 0
    assert "execution:     123ms, 42 tokens" in result.stdout
    assert "hello back" in result.stdout


def test_bench_rejects_unknown_task_type(isolated: Path) -> None:
    isolated.write_text(MODELS_OK)

    result = runner.invoke(cli_main.app, ["bench", "--task-type", "nonsense"])

    assert result.exit_code != 0
    assert "nonsense" in result.output

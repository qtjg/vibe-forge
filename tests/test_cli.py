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

    assert result.exit_code == 1
    assert "unknown task type 'nonsense'" in result.stderr


def _install_fake_executor_factory(
    monkeypatch: pytest.MonkeyPatch, results_by_tag: dict[str, ExecutionResult]
) -> None:
    """Replace OllamaExecutor with one returning per-tag canned results."""
    import vibeforge.cli.main as _cli

    class FakeExecutor:
        def __init__(self, base_url: str = "") -> None:
            pass

        def execute(
            self, model_tag: str, prompt: str, options: dict | None = None
        ) -> ExecutionResult:
            return results_by_tag.get(
                model_tag,
                ExecutionResult(model=model_tag, prompt=prompt, output="generic"),
            )

    monkeypatch.setattr(_cli, "OllamaExecutor", FakeExecutor)


COMPARE_MODELS = """\
models:
  - name: tiny-fast
    ollama_tag: qwen2.5:0.5b
    complexity_ceiling: trivial
    approx_ram_gb: 0.6
  - name: balanced
    ollama_tag: llama3.1:latest
    complexity_ceiling: medium
    approx_ram_gb: 4.9
  - name: heavy
    ollama_tag: qwen2.5-coder:14b
    complexity_ceiling: high
    approx_ram_gb: 9.0
"""


def test_route_compare_runs_all_models_side_by_side(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated.write_text(COMPARE_MODELS)
    _install_fake_executor_factory(
        monkeypatch,
        {
            "qwen2.5:0.5b": ExecutionResult(
                model="qwen2.5:0.5b",
                prompt="p",
                latency_ms=100.0,
                eval_count=8,
                output="tiny says hi",
            ),
            "llama3.1:latest": ExecutionResult(
                model="llama3.1:latest",
                prompt="p",
                latency_ms=400.0,
                eval_count=51,
                output="balanced says hi",
            ),
            "qwen2.5-coder:14b": ExecutionResult(
                model="qwen2.5-coder:14b",
                prompt="p",
                latency_ms=1200.0,
                eval_count=200,
                output="heavy says hi",
            ),
        },
    )

    result = runner.invoke(
        cli_main.app,
        [
            "route",
            "explain this pattern",
            "--type",
            "explain",
            "--compare",
            "tiny-fast,balanced,heavy",
        ],
    )

    assert result.exit_code == 0
    assert "comparing 3 models concurrently" in result.stdout
    assert "tiny-fast (qwen2.5:0.5b):" in result.stdout
    assert "balanced (llama3.1:latest):" in result.stdout
    assert "heavy (qwen2.5-coder:14b):" in result.stdout
    assert "tiny says hi" in result.stdout
    assert "heavy says hi" in result.stdout
    assert "summary (concurrent run" in result.stdout
    assert "latency" in result.stdout


def test_route_compare_reports_model_failure_without_aborting(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated.write_text(COMPARE_MODELS)
    _install_fake_executor_factory(
        monkeypatch,
        {
            "qwen2.5:0.5b": ExecutionResult(
                model="qwen2.5:0.5b", prompt="p", latency_ms=50.0, eval_count=3, output="ok"
            ),
            "llama3.1:latest": ExecutionResult(
                model="llama3.1:latest",
                prompt="p",
                error="model not found",
                error_kind="http",
                status_code=404,
            ),
            "qwen2.5-coder:14b": ExecutionResult(
                model="qwen2.5-coder:14b",
                prompt="p",
                latency_ms=80.0,
                eval_count=7,
                output="ok too",
            ),
        },
    )

    result = runner.invoke(
        cli_main.app,
        ["route", "explain this", "--type", "explain", "--compare", "tiny-fast,balanced,heavy"],
    )

    assert result.exit_code == 0
    assert "FAILED: model not found" in result.stdout
    assert "ok too" in result.stdout


def test_route_compare_rejects_unknown_tier(
    isolated: Path,
) -> None:
    isolated.write_text(COMPARE_MODELS)

    result = runner.invoke(
        cli_main.app, ["route", "explain this", "--type", "explain", "--compare", "tiny-fast,nope"]
    )

    assert result.exit_code == 1
    assert "unknown tier name(s) nope" in result.stderr
    assert "balanced" in result.stderr


def test_route_compare_conflicts_with_execute(isolated: Path) -> None:
    isolated.write_text(COMPARE_MODELS)

    result = runner.invoke(
        cli_main.app,
        ["route", "explain this", "--type", "explain", "--compare", "tiny-fast", "--execute"],
    )

    assert result.exit_code == 1
    assert "drop --execute" in result.stderr


CUSTOM_TYPES_MODELS = """\
models:
  - name: tiny-fast
    ollama_tag: qwen2.5:0.5b
    complexity_ceiling: low
    approx_ram_gb: 0.6
  - name: heavy
    ollama_tag: qwen2.5-coder:14b
    complexity_ceiling: high
    approx_ram_gb: 9.0
custom_task_types:
  - name: translate
    baseline_rank: 1
    description: "Translate code comments between languages."
"""


def test_route_accepts_a_custom_task_type_from_config(isolated: Path) -> None:
    isolated.write_text(CUSTOM_TYPES_MODELS)

    result = runner.invoke(
        cli_main.app, ["route", "turn these comments into French", "--type", "translate"]
    )

    assert result.exit_code == 0
    assert "translate" in result.stdout
    assert "baseline translate=1" in result.stdout


def test_route_rejects_unknown_task_type(isolated: Path) -> None:
    isolated.write_text(MODELS_OK)

    result = runner.invoke(cli_main.app, ["route", "hi", "--type", "nonsense"])

    assert result.exit_code == 1
    assert "unknown task type 'nonsense'" in result.stderr
    assert "autocomplete" in result.stderr


def test_route_unknown_type_error_lists_custom_types(isolated: Path) -> None:
    isolated.write_text(CUSTOM_TYPES_MODELS)

    result = runner.invoke(cli_main.app, ["route", "hi", "--type", "nonsense"])

    assert result.exit_code == 1
    assert "translate" in result.stderr

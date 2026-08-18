"""Tests for ``vibeforge doctor`` (read-only health checks)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vibeforge.doctor import ERROR, OK, WARN, Doctor, Finding
from vibeforge.router.registry import ModelRegistry

MODELS = """\
models:
  - name: tiny-fast
    ollama_tag: qwen2.5:0.5b
    complexity_ceiling: high
    approx_ram_gb: 0.6
  - name: balanced
    ollama_tag: llama3.1:latest
    complexity_ceiling: high
    approx_ram_gb: 4.9
"""

#: Built with `complexity_ceiling: high`, so coverage always passes.
GOOD_CONFIG = MODELS.replace("complexity_ceiling: high", "complexity_ceiling: high")


def make_ollama_client(*pulled: str) -> object:
    """A stub matching the bits of the ollama SDK doctor touches."""

    def list() -> SimpleNamespace:  # noqa: A001
        return SimpleNamespace(models=[SimpleNamespace(model=tag) for tag in pulled])

    return SimpleNamespace(list=list)


def make_registry(text: str = MODELS) -> ModelRegistry:
    return ModelRegistry.from_yaml(text)


def test_healthy_install_reports_all_ok() -> None:
    registry = make_registry()
    findings = Doctor(
        registry=registry,
        ollama_client_factory=lambda: make_ollama_client("qwen2.5:0.5b", "llama3.1:latest"),
    ).run()

    assert all(f.level == OK for f in findings)
    assert any("every configured tier is pulled" in f.message for f in findings)
    assert any("all 4 complexity tiers covered" in f.message for f in findings)


def test_unreachable_ollama_is_a_hard_error_with_hint() -> None:
    def dead() -> object:
        raise ConnectionError("Failed to connect to Ollama")

    findings = Doctor(registry=make_registry(), ollama_client_factory=dead).run()

    errors = [f for f in findings if f.level == ERROR]
    assert errors
    assert any("cannot reach Ollama" in f.message for f in errors)
    assert any("ollama serve" in f.message for f in errors)


def test_missing_pulled_model_warns_with_pull_command() -> None:
    findings = Doctor(
        registry=make_registry(),
        ollama_client_factory=lambda: make_ollama_client("qwen2.5:0.5b"),
    ).run()

    warns = [f for f in findings if f.level == WARN]
    assert any(
        "balanced" in f.message and "ollama pull llama3.1:latest" in f.message for f in warns
    )


def test_no_pulled_models_at_all_is_a_hard_error() -> None:
    findings = Doctor(
        registry=make_registry(),
        ollama_client_factory=lambda: make_ollama_client(),
    ).run()

    errors = [f for f in findings if f.level == ERROR]
    assert any("no configured model is pulled" in f.message for f in errors)


def test_invalid_config_short_circuits_with_field_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_file = tmp_path / "models.yaml"
    bad_file.write_text(MODELS.replace("approx_ram_gb: 4.9", "approx_ram_gb: -1"))
    monkeypatch.setenv("VIBEFORGE_MODELS", str(bad_file))

    def never_called() -> object:
        raise AssertionError("ollama must not be probed when the config is broken")

    findings = Doctor(ollama_client_factory=never_called).run()

    errors = [f for f in findings if f.level == ERROR]
    assert any("approx_ram_gb" in f.message for f in errors)
    assert not any(f.check == "ollama" for f in findings)


def test_uncovered_tiers_warn() -> None:
    low_only = MODELS.replace("complexity_ceiling: high", "complexity_ceiling: low")
    registry = ModelRegistry.from_yaml(low_only)

    findings = Doctor(
        registry=registry,
        ollama_client_factory=lambda: make_ollama_client("qwen2.5:0.5b", "llama3.1:latest"),
    ).run()

    warns = [f for f in findings if f.level == WARN]
    assert any(f.check == "tiers" and "medium" in f.message and "high" in f.message for f in warns)


def test_finding_levels_exposed() -> None:
    assert Finding(OK, "c", "m").level == OK
    assert Finding(WARN, "c", "m").level == WARN
    assert Finding(ERROR, "c", "m").level == ERROR

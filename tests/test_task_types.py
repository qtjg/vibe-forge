"""Tests for the task type registry (built-ins + user-registered types)."""

from __future__ import annotations

import pytest

from vibeforge.router.registry import ModelRegistry
from vibeforge.router.schema import ConfigError
from vibeforge.router.task_types import (
    BUILTIN_TASK_TYPES,
    DEFAULT_BASELINE_RANK,
    TaskTypeDefinition,
    TaskTypeRegistry,
)
from vibeforge.types import TaskType

CUSTOMS = [
    TaskTypeDefinition("translate", 1, "Translate code comments between languages."),
    TaskTypeDefinition("migrate", 3, "Migrate code across frameworks."),
]

CUSTOMS_YAML = """\
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
  - name: migrate
    baseline_rank: 3
    description: "Migrate code across frameworks."
"""


def test_builtins_match_the_public_catalog() -> None:
    registry = TaskTypeRegistry.builtins()
    assert set(registry.names) == {member.value for member in TaskType}
    assert registry.names == tuple(d.name for d in BUILTIN_TASK_TYPES)


def test_builtin_baselines() -> None:
    registry = TaskTypeRegistry.builtins()
    assert registry.baseline_rank("autocomplete") == 0
    assert registry.baseline_rank("explain") == 1
    assert registry.baseline_rank("generate") == 1
    assert registry.baseline_rank("refactor") == 2
    assert registry.baseline_rank("debug") == 2
    assert registry.baseline_rank("review") == 2


def test_custom_types_merge_after_builtins() -> None:
    registry = TaskTypeRegistry.from_config(CUSTOMS)
    assert registry.names == (
        "autocomplete",
        "explain",
        "generate",
        "refactor",
        "debug",
        "review",
        "translate",
        "migrate",
    )
    assert registry.baseline_rank("translate") == 1
    assert registry.baseline_rank("migrate") == 3
    assert registry.definition("translate").description == (
        "Translate code comments between languages."
    )


def test_custom_type_shadowing_a_builtin_is_rejected() -> None:
    with pytest.raises(ConfigError, match="shadows a built-in"):
        TaskTypeRegistry.from_config([TaskTypeDefinition("review", 0)])


def test_duplicate_custom_types_are_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate task type"):
        TaskTypeRegistry.from_config(
            [TaskTypeDefinition("translate", 1), TaskTypeDefinition("translate", 2)]
        )


def test_membership_and_lookup() -> None:
    registry = TaskTypeRegistry.from_config(CUSTOMS)
    assert "translate" in registry
    assert "nonsense" not in registry
    assert registry.definition("nonsense") is None
    assert len(registry) == 8


def test_unregistered_type_gets_the_default_baseline() -> None:
    registry = TaskTypeRegistry.builtins()
    assert registry.baseline_rank("totally-unknown") == DEFAULT_BASELINE_RANK


def test_registry_from_models_config_surfaces_custom_types() -> None:
    registry = ModelRegistry.from_yaml(CUSTOMS_YAML)
    assert set(registry.task_types.names) >= {"translate", "migrate"}
    assert registry.task_types.baseline_rank("migrate") == 3


def test_registry_config_rejects_invalid_custom_baseline() -> None:
    bad = CUSTOMS_YAML.replace("baseline_rank: 3", "baseline_rank: 9")
    with pytest.raises(ConfigError, match="baseline_rank"):
        ModelRegistry.from_yaml(bad)


def test_custom_type_routes_end_to_end() -> None:
    from vibeforge.router.complexity import HeuristicScorer
    from vibeforge.router.policy import PolicyRouter
    from vibeforge.types import Complexity, Task

    registry = ModelRegistry.from_yaml(CUSTOMS_YAML)
    router = PolicyRouter(scorer=HeuristicScorer(task_types=registry.task_types), registry=registry)

    decision = router.route(Task(type="translate", prompt="short prompt"))
    assert decision.complexity is Complexity.LOW
    assert decision.model.name == "tiny-fast"
    assert "baseline translate=1" in decision.reason

    decision = router.route(Task(type="migrate", prompt="short prompt"))
    assert decision.complexity is Complexity.HIGH
    assert decision.model.name == "heavy"

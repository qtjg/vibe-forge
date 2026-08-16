"""Tests for availability-aware model selection and fallback reasons."""

from __future__ import annotations

import pytest

from vibeforge.router.complexity import HeuristicScorer
from vibeforge.router.policy import PolicyRouter
from vibeforge.router.registry import ModelRegistry
from vibeforge.types import Complexity, Task, TaskType

THREE_TIER_YAML = """\
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


@pytest.fixture
def registry() -> ModelRegistry:
    """Standard three-tier registry used across availability tests."""
    return ModelRegistry.from_yaml(THREE_TIER_YAML)


def test_everything_pulled_picks_cheapest_covering(registry: ModelRegistry) -> None:
    available = {"qwen2.5:0.5b", "llama3.1:latest", "qwen2.5-coder:14b"}

    pick = registry.pick(Complexity.TRIVIAL, available_tags=available)
    assert pick.model.name == "tiny-fast"
    assert pick.fallback_reason is None

    pick = registry.pick(Complexity.MEDIUM, available_tags=available)
    assert pick.model.name == "balanced"
    assert pick.fallback_reason is None


def test_covering_tier_missing_falls_back_to_most_capable_pulled(
    registry: ModelRegistry,
) -> None:
    available = {"qwen2.5:0.5b"}  # no covering tier pulled for LOW tasks

    pick = registry.pick(Complexity.LOW, available_tags=available)

    assert pick.model.name == "tiny-fast"
    assert "no configured model covering low is pulled" in (pick.fallback_reason or "")


def test_bigger_model_matching_availability_is_no_fallback(
    registry: ModelRegistry,
) -> None:
    """trivial task: tiny-fast absent, but balanced is pulled and covers it."""
    available = {"llama3.1:latest"}

    pick = registry.pick(Complexity.TRIVIAL, available_tags=available)

    assert pick.model.name == "balanced"
    assert pick.fallback_reason is None


def test_nothing_pulled_falls_back_to_configured(registry: ModelRegistry) -> None:
    pick = registry.pick(Complexity.HIGH, available_tags=set())

    assert pick.model.name == "heavy"
    assert "no configured models are pulled" in (pick.fallback_reason or "")


def test_without_availability_info_uses_old_behavior(registry: ModelRegistry) -> None:
    pick = registry.pick(Complexity.HIGH, available_tags=None)
    assert pick.model.name == "heavy"
    assert pick.fallback_reason is None

    pick = registry.pick(Complexity.HIGH)  # explicit None is the default
    assert pick.model.name == "heavy"


def test_pick_for_keeps_backward_compatible_signature(registry: ModelRegistry) -> None:
    assert registry.pick_for(Complexity.LOW).name == "balanced"
    assert registry.pick_for(Complexity.TRIVIAL).name == "tiny-fast"


def test_partial_availability_uses_cheapest_covering_pulled(
    registry: ModelRegistry,
) -> None:
    """MEDIUM task: balanced missing, but heavy is pulled and covers it."""
    available = {"qwen2.5:0.5b", "qwen2.5-coder:14b"}

    pick = registry.pick(Complexity.MEDIUM, available_tags=available)
    assert pick.model.name == "heavy"
    assert pick.fallback_reason is None


def test_router_records_fallback_reason_with_availability() -> None:
    router = PolicyRouter(
        scorer=HeuristicScorer(),
        registry=ModelRegistry.from_yaml(THREE_TIER_YAML),
    )
    decision = router.route(
        Task(type=TaskType.DEBUG, prompt="fix the race condition in the worker pool"),
        available_tags={"qwen2.5:0.5b", "llama3.1:latest"},
    )
    assert decision.model.name == "balanced"
    assert decision.fallback_reason is not None
    assert decision.as_dict()["fallback_reason"] == decision.fallback_reason


def test_router_picks_available_model_for_debug_task() -> None:
    router = PolicyRouter(
        scorer=HeuristicScorer(),
        registry=ModelRegistry.from_yaml(THREE_TIER_YAML),
    )
    decision = router.route(
        Task(type=TaskType.DEBUG, prompt="fix the race condition in the worker pool"),
        available_tags={"qwen2.5:0.5b", "qwen2.5-coder:14b", "llama3.1:latest"},
    )
    assert decision.model.name == "heavy"
    assert decision.fallback_reason is None


def test_router_without_availability_has_no_fallback_reason() -> None:
    router = PolicyRouter(
        scorer=HeuristicScorer(),
        registry=ModelRegistry.from_yaml(THREE_TIER_YAML),
    )
    decision = router.route(
        Task(type=TaskType.DEBUG, prompt="fix the race condition in the worker pool")
    )
    assert decision.fallback_reason is None
    assert decision.as_dict()["fallback_reason"] is None
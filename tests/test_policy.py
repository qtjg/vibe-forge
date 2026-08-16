"""Tests for the policy router."""

from __future__ import annotations

from vibeforge.router.complexity import HeuristicScorer
from vibeforge.router.policy import PolicyRouter
from vibeforge.router.registry import ModelRegistry
from vibeforge.types import Complexity, ModelTier, RoutingDecision, Task, TaskType


class FakeScorer:
    """A scorer with a fixed opinion, for isolating the router."""

    def __init__(self, complexity: Complexity, reason: str = "fake reason") -> None:
        self._complexity = complexity
        self._reason = reason

    def score(self, task: Task) -> tuple[Complexity, str]:
        return self._complexity, self._reason


class FakeRegistry:
    """A registry that always answers with one model, for isolating the router."""

    def __init__(self, model: ModelTier) -> None:
        self._model = model

    def pick_for(self, complexity: Complexity) -> ModelTier:
        return self._model


def make_fake_parts() -> tuple[FakeScorer, FakeRegistry, ModelTier]:
    """Standard fake scorer + registry + model for router tests."""
    scorer = FakeScorer(Complexity.MEDIUM)
    model = ModelTier(
        name="test-tier",
        ollama_tag="test-model:latest",
        complexity_ceiling=Complexity.HIGH,
        approx_ram_gb=4.0,
    )
    registry = FakeRegistry(model)
    return scorer, registry, model


def make_task() -> Task:
    """A plain debug task used across router tests."""
    return Task(type=TaskType.DEBUG, prompt="why does this crash?")


def test_route_returns_decision_from_scorer_and_registry() -> None:
    scorer, registry, model = make_fake_parts()
    router = PolicyRouter(scorer=scorer, registry=registry)

    decision = router.route(make_task())

    assert isinstance(decision, RoutingDecision)
    assert decision.complexity is Complexity.MEDIUM
    assert decision.score == 2
    assert decision.reason == "fake reason"
    assert decision.model is model


def test_route_records_every_decision_in_history() -> None:
    scorer, registry, _ = make_fake_parts()
    router = PolicyRouter(scorer=scorer, registry=registry)

    router.route(make_task())
    router.route(make_task())

    assert len(router.history) == 2
    assert all(isinstance(d, RoutingDecision) for d in router.history)


def test_recent_returns_newest_first() -> None:
    scorer, registry, _ = make_fake_parts()
    router = PolicyRouter(scorer=scorer, registry=registry)

    first = router.route(make_task())
    second = router.route(make_task())

    assert router.recent() == [second, first]


def test_recent_respects_limit() -> None:
    scorer, registry, _ = make_fake_parts()
    router = PolicyRouter(scorer=scorer, registry=registry)
    for _ in range(5):
        router.route(make_task())

    assert len(router.recent(limit=3)) == 3


def test_clear_history_empties_it() -> None:
    scorer, registry, _ = make_fake_parts()
    router = PolicyRouter(scorer=scorer, registry=registry)
    router.route(make_task())
    router.clear_history()
    assert router.history == ()


def test_history_can_be_seeded() -> None:
    scorer, registry, _ = make_fake_parts()
    seeded = [
        RoutingDecision(
            task=make_task(),
            score=0,
            complexity=Complexity.TRIVIAL,
            reason="seed",
            model=registry.pick_for(Complexity.TRIVIAL),
        )
    ]
    router = PolicyRouter(scorer=scorer, registry=registry, history=seeded)
    assert router.history == (seeded[0],)


def test_end_to_end_with_real_scorer_and_registry() -> None:
    router = PolicyRouter(
        scorer=HeuristicScorer(),
        registry=ModelRegistry.from_yaml("""\
            models:
              - name: tiny-fast
                ollama_tag: t:latest
                complexity_ceiling: trivial
                approx_ram_gb: 0.6
              - name: heavy
                ollama_tag: h:latest
                complexity_ceiling: high
                approx_ram_gb: 9
            """),
    )

    easy = router.route(Task(type=TaskType.AUTOCOMPLETE, prompt="add_customer(db, name)"))
    hard = router.route(
        Task(
            type=TaskType.DEBUG,
            prompt="fix the race condition in the worker pool",
            context=" ".join(["worker.process(item)" for _ in range(300)]),
        )
    )

    assert easy.model.name == "tiny-fast"
    assert easy.complexity is Complexity.TRIVIAL
    assert hard.model.name == "heavy"
    assert hard.complexity is Complexity.HIGH
    assert len(router.history) == 2


def test_end_to_end_with_bundled_default_registry() -> None:
    router = PolicyRouter(scorer=HeuristicScorer(), registry=ModelRegistry.load_default())

    decision = router.route(Task(type=TaskType.GENERATE, prompt="write a tiny helper function"))

    assert decision.model.name in ("tiny-fast", "balanced", "heavy")
    assert decision.reason
    assert decision.timestamp is not None


def test_decision_serializes_to_dashboard_shape() -> None:
    scorer, registry, _ = make_fake_parts()
    router = PolicyRouter(scorer=scorer, registry=registry)

    payload = router.route(make_task()).as_dict()

    for key in (
        "timestamp",
        "task_type",
        "prompt",
        "score",
        "complexity",
        "reason",
        "model",
        "ollama_tag",
    ):
        assert key in payload
    assert payload["complexity"] == "medium"
    assert payload["model"] == "test-tier"

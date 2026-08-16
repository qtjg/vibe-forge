"""The routing policy: score a task, pick a model, keep the history.

:class:`PolicyRouter` is the tiny orchestration layer that the CLI and the
dashboard talk to. It depends on the :class:`Scorer` protocol and the
:class:`ModelRegistry` -- swap either one without touching the other layers.
"""

from __future__ import annotations

from collections.abc import Sequence

from vibeforge.router.complexity import Scorer
from vibeforge.router.registry import ModelRegistry
from vibeforge.types import RoutingDecision, Task

__all__ = ["PolicyRouter"]


class PolicyRouter:
    """Routes :class:`Task` objects to model tiers and keeps decision history.

    Examples:
        >>> router = PolicyRouter(scorer=HeuristicScorer(), registry=ModelRegistry.load_default())
        >>> decision = router.route(Task(type=TaskType.DEBUG, prompt="fix the race"))
        >>> decision.model.name
        'heavy'
    """

    def __init__(
        self,
        scorer: Scorer,
        registry: ModelRegistry,
        history: list[RoutingDecision] | None = None,
    ) -> None:
        """Build a router from a scorer and a model registry.

        Args:
            scorer: Anything implementing the :class:`Scorer` protocol.
            registry: The model registry to pick tiers from.
            history: Optional pre-seeded decision history (e.g. dashboard
                replay); a fresh list is created when omitted.
        """
        self._scorer = scorer
        self._registry = registry
        self._history: list[RoutingDecision] = history if history is not None else []

    def route(self, task: Task) -> RoutingDecision:
        """Score ``task``, pick a model, record, and return the decision.

        Args:
            task: The coding subtask to route.

        Returns:
            A :class:`RoutingDecision` describing complexity, reason, and the
            chosen model tier.
        """
        complexity, reason = self._scorer.score(task)
        model = self._registry.pick_for(complexity)
        decision = RoutingDecision(
            task=task,
            score=complexity.rank,
            complexity=complexity,
            reason=reason,
            model=model,
        )
        self._history.append(decision)
        return decision

    def recent(self, limit: int = 50) -> list[RoutingDecision]:
        """Return the last ``limit`` decisions, newest first.

        Args:
            limit: Maximum number of decisions to return.

        Returns:
            Newest-first list of :class:`RoutingDecision` objects.
        """
        return list(reversed(self._history[-limit:]))

    def clear_history(self) -> None:
        """Drop all recorded decisions."""
        self._history.clear()

    @property
    def history(self) -> Sequence[RoutingDecision]:
        """All recorded decisions, oldest first (read-only view)."""
        return tuple(self._history)

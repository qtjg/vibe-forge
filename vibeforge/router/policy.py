"""The routing policy: score a task, pick a model, keep the history.

:class:`PolicyRouter` is the tiny orchestration layer that the CLI and the
dashboard talk to. It depends on the :class:`Scorer` protocol and the
:class:`ModelRegistry` -- swap either one without touching the other layers.

History is kept in memory (the dashboard's fast path) and, when a
:class:`~vibeforge.history.HistoryStore` is supplied, also appended to a
durable JSONL file. Persistence is best-effort: a store failure must never
fail a route.
"""

from __future__ import annotations

from collections.abc import Sequence

from vibeforge.history import HistoryStore
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
        history_store: HistoryStore | None = None,
    ) -> None:
        """Build a router from a scorer and a model registry.

        Args:
            scorer: Anything implementing the :class:`Scorer` protocol.
            registry: The model registry to pick tiers from.
            history: Optional pre-seeded decision history (e.g. dashboard
                replay); a fresh list is created when omitted.
            history_store: Optional durable JSONL store; every routed
                decision is appended to it (best-effort, failures ignored).
        """
        self._scorer = scorer
        self._registry = registry
        self._history: list[RoutingDecision] = history if history is not None else []
        self._history_store = history_store

    def route(
        self, task: Task, available_tags: set[str] | None = None
    ) -> RoutingDecision:
        """Score ``task``, pick a model, record, and return the decision.

        Args:
            task: The coding subtask to route.
            available_tags: Tags Ollama reports as pulled; when given, the
                registry avoids picking models that cannot execute and
                records a ``fallback_reason`` when it must deviate.

        Returns:
            A :class:`RoutingDecision` describing complexity, reason, and the
            chosen model tier.
        """
        complexity, reason = self._scorer.score(task)
        pick = self._registry.pick(complexity, available_tags=available_tags)
        confidence = getattr(self._scorer, "confidence", None)
        decision = RoutingDecision(
            task=task,
            score=complexity.rank,
            complexity=complexity,
            reason=reason,
            model=pick.model,
            fallback_reason=pick.fallback_reason,
            confidence=confidence(task) if callable(confidence) else None,
        )
        self._history.append(decision)
        self._persist(decision)
        return decision

    def _persist(self, decision: RoutingDecision) -> None:
        """Append the decision to the durable store, ignoring failures."""
        if self._history_store is None:
            return
        try:
            self._history_store.append(decision.as_dict())
        except OSError:
            pass

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

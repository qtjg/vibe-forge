"""Core data types shared across vibe-forge packages.

Everything in the routing pipeline flows through the dataclasses and enums
defined here: :class:`Task` flows into the router, :class:`RoutingDecision`
is its output, and :class:`ExecutionResult` is what comes back from Ollama.

Enums are :class:`str` subclasses so they serialize cleanly to JSON and YAML
without custom encoders.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime

__all__ = [
    "TaskType",
    "Complexity",
    "COMPLEXITY_ORDER",
    "Task",
    "ModelTier",
    "RoutingDecision",
    "ExecutionResult",
]


class TaskType(enum.StrEnum):
    """The kind of coding subtask being routed."""

    AUTOCOMPLETE = "autocomplete"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    GENERATE = "generate"
    DEBUG = "debug"
    REVIEW = "review"


class Complexity(enum.StrEnum):
    """Complexity tiers from cheapest to most demanding."""

    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """Position in the tier scale, 0 (trivial) to 3 (high)."""
        return Complexity._RANKS[self]

    @classmethod
    def at(cls, index: int) -> Complexity:
        """Return the tier at ``index``, clamped to the valid 0..3 range."""
        return COMPLEXITY_ORDER[max(0, min(index, len(COMPLEXITY_ORDER) - 1))]


Complexity._RANKS = {tier: i for i, tier in enumerate(Complexity)}

#: Tier scale in increasing order; used for clamping and comparisons.
COMPLEXITY_ORDER: tuple[Complexity, ...] = tuple(Complexity)


@dataclass(frozen=True)
class Task:
    """A single coding subtask to route.

    Attributes:
        type: Kind of subtask, e.g. ``debug`` or ``refactor``.
        prompt: The instruction or question text.
        context: Surrounding code/relevant file contents, if any.
        file_path: Optional path of the file the task refers to.

    """

    type: TaskType
    prompt: str
    context: str = ""
    file_path: str | None = None


@dataclass(frozen=True)
class ModelTier:
    """A configured model tier loaded from ``models.yaml``.

    Attributes:
        name: Short human-readable tier name, e.g. ``tiny-fast``.
        ollama_tag: The tag Ollama knows this model by, e.g. ``llama3.1:latest``.
        complexity_ceiling: The hardest complexity this tier should ever serve.
        approx_ram_gb: Approximate RAM usage; the router uses it as the
            proxy for "cheapest" when choosing between eligible tiers.
        notes: Free-form user notes (shown in reports, never parsed).

    """

    name: str
    ollama_tag: str
    complexity_ceiling: Complexity
    approx_ram_gb: float
    notes: str = ""


@dataclass(frozen=True)
class RoutingDecision:
    """The output of :meth:`PolicyRouter.route`.

    Attributes:
        task: The task that was routed.
        score: Effective complexity score (clamped 0..3).
        complexity: Clamped complexity tier.
        reason: Human-readable explanation of *why* this tier was chosen.
        model: The model tier selected for the task.
        timestamp: UTC wall-clock time the decision was made.

    """

    task: Task
    score: int
    complexity: Complexity
    reason: str
    model: ModelTier
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, object]:
        """Serialize to a plain JSON-ready dict (the dashboard API shape)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "task_type": self.task.type.value,
            "prompt": self.task.prompt,
            "file_path": self.task.file_path,
            "score": self.score,
            "complexity": self.complexity.value,
            "reason": self.reason,
            "model": self.model.name,
            "ollama_tag": self.model.ollama_tag,
        }


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of running a prompt against Ollama.

    Either ``error`` is set (Ollama down, model not pulled, timeout, ...) or
    ``output`` is set; the executor never raises for expected failure modes.

    Attributes:
        model: The Ollama tag that was called.
        prompt: The prompt that was sent.
        latency_ms: Wall-clock latency of the generation call, if it completed.
        eval_count: Number of tokens generated, if reported by Ollama.
        output: The generated text, if the call succeeded.
        error: Error description, if the call failed.

    """

    model: str
    prompt: str
    latency_ms: float | None = None
    eval_count: int | None = None
    output: str | None = None
    error: str | None = None

    @property
    def tokens_per_sec(self) -> float | None:
        """Generation throughput; ``None`` when there is nothing to measure."""
        if self.eval_count is None or self.latency_ms is None or self.latency_ms <= 0:
            return None
        return self.eval_count / (self.latency_ms / 1000.0)

    @property
    def ok(self) -> bool:
        """True when the call completed without errors."""
        return self.error is None and self.output is not None

    def as_dict(self) -> dict[str, object]:
        """Serialize to a plain JSON-ready dict."""
        return {
            "model": self.model,
            "prompt": self.prompt,
            "latency_ms": self.latency_ms,
            "eval_count": self.eval_count,
            "tokens_per_sec": self.tokens_per_sec,
            "output_chars": len(self.output) if self.output else 0,
            "error": self.error,
        }

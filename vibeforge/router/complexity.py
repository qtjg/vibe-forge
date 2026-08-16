"""Rule-based task complexity scoring.

The :class:`HeuristicScorer` estimates how hard a coding task is using three
cheap, explainable signals that need no ML dependencies:

1. **Task-type baseline** -- ``autocomplete`` starts cheap, ``debug`` and
   ``review`` start heavy.
2. **Length** -- a long prompt + context means more reasoning is required.
3. **High-signal keywords** -- words like *race condition* or *distributed*
   flag genuinely hard problems.

Every score returns a plain-English reason string so routing stays
explainable end to end, plus a deterministic confidence estimate based on
how much evidence the signals supplied.

All scoring knobs (baselines, keywords, length thresholds) can be overridden
per-instance, which lets projects tune routing without forking the code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from vibeforge.router.task_types import (
    DEFAULT_BASELINE_RANK,
    TaskTypeRegistry,
)
from vibeforge.types import Complexity, Task

#: Rank (0..3) used as the scoring baseline per built-in task type.
#: Cheap tasks (autocomplete) start at trivial; debug/review start heavy.
#: Keys are plain type strings -- custom types registered in models.yaml
#: are merged on top at registry build time.
BASELINE_RANKS: dict[str, int] = {
    "autocomplete": 0,  # trivial
    "explain": 1,  # low
    "generate": 1,  # low -- "say hi" is easy, length bumps it up
    "refactor": 2,  # medium
    "debug": 2,  # medium
    "review": 2,  # medium
}

#: High-signal keywords that mark out hard problems. Multi-word phrases and
#: single words are matched on word boundaries, case-insensitively.
HIGH_SIGNAL_KEYWORDS: tuple[str, ...] = (
    "concurrency",
    "concurrent",
    "race condition",
    "race",
    "memory leak",
    "leak",
    "deadlock",
    "lock contention",
    "lock",
    "thread-safe",
    "threadsafe",
    "multithreading",
    "distributed",
    "consensus",
    "async",
    "asynchronous",
    "coroutine",
    "architecture",
    "microservices",
    "security",
    "authentication",
    "encryption",
    "performance",
    "bottleneck",
    "optimization",
    "segfault",
    "undefined behavior",
    "garbage collection",
)

#: Prompt + context word counts above these thresholds add score points.
#: ``>200`` words: +1. ``>800`` words: +2.
LENGTH_BUMPS: tuple[tuple[int, int], ...] = ((200, 1), (800, 2))

#: More than this many *distinct* keyword hits adds a second bump.
KEYWORD_DOUBLE_HIT_THRESHOLD = 3

#: Confidence floor when only the baseline applies; each extra signal adds
#: toward the cap. 0..1, deterministic.
_CONFIDENCE_BASE = 0.5
_CONFIDENCE_CAP = 0.9
_CONFIDENCE_LENGTH_STEP = 0.15
_CONFIDENCE_KEYWORD_STEP = 0.15
_CONFIDENCE_KEYWORD_DOUBLE_STEP = 0.2
_CONFIDENCE_CONTEXT_STEP = 0.05

_PROTOCOL = """Choose a score() method returning (Complexity, str)."""


class Scorer:
    """Anything that turns a :class:`Task` into a complexity tier."""

    def score(self, task: Task) -> tuple[Complexity, str]:
        """Return the complexity tier and a human-readable justification."""
        raise NotImplementedError(_PROTOCOL)

    def confidence(self, task: Task) -> float | None:
        """Optional 0..1 estimate of how reliable ``score()`` is.

        ``None`` means the scorer does not produce a confidence estimate.
        """
        return None


class HeuristicScorer:
    """Rule-based scorer: baseline + length + keyword signals, clamped to 0..3.

    Deterministic and dependency-free, which keeps routing explainable,
    testable, and fast (no ML inference to score a task). Every tuning knob
    defaults to the module-level constants and can be overridden per
    instance.

    Examples:
        >>> strict = HeuristicScorer(
        ...     baseline_ranks={TaskType.REVIEW: 3},
        ...     length_bumps=((100, 1),),
        ... )
        >>> complexity, _ = strict.score(Task(TaskType.REVIEW, "short"))
        >>> complexity is Complexity.HIGH
        True
    """

    def __init__(
        self,
        baseline_ranks: Mapping[str, int] | None = None,
        length_bumps: Sequence[tuple[int, int]] | None = None,
        high_signal_keywords: Sequence[str] | None = None,
        keyword_double_hit_threshold: int = KEYWORD_DOUBLE_HIT_THRESHOLD,
        task_types: TaskTypeRegistry | None = None,
    ) -> None:
        """Configure the scoring signals.

        Args:
            baseline_ranks: Per-task-type baseline ranks; missing task types
                fall back to the registry defaults.
            length_bumps: ``(word_threshold, score_bump)`` pairs applied in
                descending threshold order (largest applicable threshold
                wins).
            high_signal_keywords: Keyword literals matched on word
                boundaries.
            keyword_double_hit_threshold: Distinct hits needed for a second
                keyword bump.
            task_types: The registered task-type catalog. When given,
                every registered type (including user-defined ones) is
                seeded at its baseline before ``baseline_ranks`` overrides
                apply; unregistered types fall back to the registry
                default baseline.
        """
        if task_types is not None:
            merged_baselines = {
                definition.name: definition.baseline_rank
                for definition in task_types.definitions
            }
        else:
            merged_baselines = dict(BASELINE_RANKS)
        if baseline_ranks:
            merged_baselines.update(baseline_ranks)
        self._baseline_ranks = merged_baselines

        self._length_bumps = tuple(length_bumps) if length_bumps else LENGTH_BUMPS
        keywords = high_signal_keywords if high_signal_keywords else HIGH_SIGNAL_KEYWORDS
        self._keywords = tuple(keywords)
        self._patterns = tuple(
            re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE) for keyword in self._keywords
        )
        self._double_hit_threshold = keyword_double_hit_threshold

    def score(self, task: Task) -> tuple[Complexity, str]:
        """Score ``task`` and return ``(complexity, reason)``.

        The reason string records each bump that contributed, e.g.::

            baseline debug=2 (medium) +1 length (412 words)
            +1 keywords (race condition) => score 3 (high)
        """
        complexity, reason, _ = self._evaluate(task)
        return complexity, reason

    def confidence(self, task: Task) -> float:
        """0..1 estimate of score confidence given the available evidence.

        A plain default task with no extra signals lands at the base; every
        additional signal class (length, keywords, context) adds toward the
        cap, and a double keyword hit signals strong evidence.
        """
        _, _, confidence = self._evaluate(task)
        return confidence

    def _evaluate(self, task: Task) -> tuple[Complexity, str, float]:
        """Internal evaluation returning (complexity, reason, confidence)."""
        baseline = self._baseline_ranks.get(task.type, DEFAULT_BASELINE_RANK)
        score = baseline
        confidence = _CONFIDENCE_BASE
        events: list[str] = []

        words = len(f"{task.prompt} {task.context}".split())
        ordered_bumps = sorted(self._length_bumps, reverse=True)
        for threshold, bump in ordered_bumps:
            if words > threshold:
                score += bump
                confidence += _CONFIDENCE_LENGTH_STEP
                events.append(f"+{bump} length ({words} words)")
                break  # the largest applicable threshold wins

        hits = [
            keyword
            for keyword, pattern in zip(self._keywords, self._patterns, strict=True)
            if pattern.search(task.prompt) or pattern.search(task.context)
        ]
        if hits:
            double = len(hits) >= self._double_hit_threshold
            bump = 2 if double else 1
            score += bump
            confidence += _CONFIDENCE_KEYWORD_DOUBLE_STEP if double else _CONFIDENCE_KEYWORD_STEP
            shown = ", ".join(hits[:4])
            events.append(f"+{bump} keywords ({shown})")

        if task.context.strip():
            confidence += _CONFIDENCE_CONTEXT_STEP

        clamped = max(0, min(score, 3))
        if clamped != score:
            events.append(f"clamped {score} -> 3")
            score = clamped
        confidence = min(confidence, _CONFIDENCE_CAP)

        complexity = Complexity.at(score)
        reason = f"baseline {task.type}={baseline}"
        if events:
            reason += " " + " ".join(events)
        reason += f" => score {score}/3 ({complexity.value})"
        return complexity, reason, round(confidence, 2)


__all__: Sequence[str] = [
    "Scorer",
    "HeuristicScorer",
    "BASELINE_RANKS",
    "HIGH_SIGNAL_KEYWORDS",
]

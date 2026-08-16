"""Rule-based task complexity scoring.

The :class:`HeuristicScorer` estimates how hard a coding task is using three
cheap, explainable signals that need no ML dependencies:

1. **Task-type baseline** -- ``autocomplete`` starts cheap, ``debug`` and
   ``review`` start heavy.
2. **Length** -- a long prompt + context means more reasoning is required.
3. **High-signal keywords** -- words like *race condition* or *distributed*
   flag genuinely hard problems.

Every score returns a plain-English reason string so routing stays
explainable end to end.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from vibeforge.types import Complexity, Task, TaskType

#: Rank (0..3) used as the scoring baseline per task type.
#: Cheap tasks (autocomplete) start at trivial; debug/review start heavy.
BASELINE_RANKS: dict[TaskType, int] = {
    TaskType.AUTOCOMPLETE: 0,  # trivial
    TaskType.EXPLAIN: 1,  # low
    TaskType.GENERATE: 1,  # low -- "say hi" is easy, length bumps it up
    TaskType.REFACTOR: 2,  # medium
    TaskType.DEBUG: 2,  # medium
    TaskType.REVIEW: 2,  # medium
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

_KEYWORD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE) for keyword in HIGH_SIGNAL_KEYWORDS
)


class Scorer(Protocol):
    """Anything that turns a :class:`Task` into a complexity tier."""

    def score(self, task: Task) -> tuple[Complexity, str]:
        """Return the complexity tier and a human-readable justification."""
        ...


class HeuristicScorer:
    """Rule-based scorer: baseline + length + keyword signals, clamped to 0..3.

    Deterministic and dependency-free, which keeps routing explainable,
    testable, and fast (no ML inference to score a task).
    """

    def score(self, task: Task) -> tuple[Complexity, str]:
        """Score ``task`` and return ``(complexity, reason)``.

        The reason string records each bump that contributed, e.g.::

            baseline debug=2 (medium) +1 length (412 words)
            +1 keywords (race condition) => score 3 (high)
        """
        baseline = BASELINE_RANKS[task.type]
        score = baseline
        events: list[str] = []

        words = len(f"{task.prompt} {task.context}".split())
        for threshold, bump in sorted(LENGTH_BUMPS, reverse=True):
            if words > threshold:
                score += bump
                events.append(f"+{bump} length ({words} words)")
                break  # the largest applicable threshold wins

        hits = [
            keyword
            for keyword, pattern in zip(
                HIGH_SIGNAL_KEYWORDS, _KEYWORD_PATTERNS, strict=True
            )
            if pattern.search(task.prompt) or pattern.search(task.context)
        ]
        if hits:
            bump = 2 if len(hits) >= KEYWORD_DOUBLE_HIT_THRESHOLD else 1
            score += bump
            shown = ", ".join(hits[:4])
            events.append(f"+{bump} keywords ({shown})")

        clamped = max(0, min(score, 3))
        if clamped != score:
            events.append(f"clamped {score} -> 3")
            score = clamped

        complexity = Complexity.at(score)
        reason = f"baseline {task.type.value}={baseline}"
        if events:
            reason += " " + " ".join(events)
        reason += f" => score {score}/3 ({complexity.value})"
        return complexity, reason


__all__: Sequence[str] = [
    "Scorer",
    "HeuristicScorer",
    "BASELINE_RANKS",
    "HIGH_SIGNAL_KEYWORDS",
]

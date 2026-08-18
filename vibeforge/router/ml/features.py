"""Deterministic feature extraction shared by the training pipeline.

The classifier is trained on the *same raw signals* the heuristic scorer
already uses -- task-type baseline rank, prompt+context word count, and the
number of distinct high-signal keyword hits -- as plain floats so the model
can learn its own thresholds instead of ours. Storing the ground-truth tier
for benchmark tasks here (``heuristic_tier``) pins down what the "task/tier
pairs" of the fixed suite mean: the deterministic tier the suite's design
intended for each task.
"""

from __future__ import annotations

import re

from vibeforge.router.complexity import BASELINE_RANKS, HIGH_SIGNAL_KEYWORDS
from vibeforge.router.task_types import DEFAULT_BASELINE_RANK
from vibeforge.types import Complexity, Task

__all__ = ["feature_vector", "heuristic_tier", "keyword_hit_count"]

_KEYWORD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in HIGH_SIGNAL_KEYWORDS
)


def baseline_rank(task_type: str) -> float:
    """Scoring baseline (0..3) for a task type; custom types default to 1."""
    return float(BASELINE_RANKS.get(task_type, DEFAULT_BASELINE_RANK))


def keyword_hit_count(task: Task) -> float:
    """Number of distinct high-signal keywords present in prompt/context."""
    hits = 0
    for pattern in _KEYWORD_PATTERNS:
        if pattern.search(task.prompt) is not None or pattern.search(task.context) is not None:
            hits += 1
    return float(hits)


def word_count(task: Task) -> float:
    """Word count of prompt + context (the heuristic's length signal)."""
    return float(len(f"{task.prompt} {task.context}".split()))


def feature_vector(task: Task) -> list[float]:
    """The labeled-row feature vector for one task.

    Order is stable: ``[baseline_rank, word_count, keyword_hit_count]``.
    Inference later feeds the identical vector; the model learns its own
    weighting, so it is not simply a re-encoding of ``HeuristicScorer``'s
    exact thresholds.
    """
    return [baseline_rank(task.type), word_count(task), keyword_hit_count(task)]


def heuristic_tier(task: Task) -> Complexity:
    """Ground-truth tier for a benchmark task (heuristic, deterministic).

    The fixed 36-task suite has no per-task label field; its rows
    translate to task/tier pairs via the tool the suite was designed to
    exercise. This stays the source of truth for benchmark CSV rows until
    3.2 introduces optional human-labeled overrides.
    """
    from vibeforge.router.complexity import HeuristicScorer

    complexity, _ = HeuristicScorer().score(task)
    return complexity

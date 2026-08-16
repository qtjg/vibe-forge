"""Classification metrics for scorer evaluation.

Pure, dependency-free math so the numbers in the paper are auditable:
given per-task true/predicted tiers, compute per-tier precision/recall/
F1, plus accuracy and macro-F1. The confusion matrix is exposed raw so
any downstream tool can render it however it wants.

Conventions:

- True positives for tier ``T`` = tasks whose true tier is ``T`` and
  were predicted ``T``.
- Precision(T) = TP / (TP + FP); when there are no positives predicted
  for ``T``, precision is None (not 0) -- saying "0 correct out of 0
  predicted" is meaningless.
- Recall(T) = TP / (TP + FN); when no task actually has tier ``T``,
  recall is None.
- Accuracy = correct / total; macro-F1 averages the F1 of tiers that
  have an F1 (a tier with no instances is skipped, not scored 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from vibeforge.types import COMPLEXITY_ORDER, Complexity

__all__ = ["ConfusionMatrix", "TierMetrics", "EvaluationMetrics", "evaluate"]


class ConfusionMatrix:
    """A sparse 4x4 confusion matrix for the four complexity tiers."""

    def __init__(self) -> None:
        """Start with an empty matrix."""
        self._counts: dict[tuple[Complexity, Complexity], int] = {}

    def add(self, truths: Complexity, predicted: Complexity) -> None:
        """Record one prediction whose ground truth is ``truths``."""
        self._counts[(truths, predicted)] = self._counts.get((truths, predicted), 0) + 1

    def count(self, truths: Complexity, predicted: Complexity) -> int:
        """Number of tasks with this (true, predicted) pair."""
        return self._counts.get((truths, predicted), 0)

    def total(self) -> int:
        """Number of recorded tasks."""
        return sum(self._counts.values())

    def raw(self) -> dict[tuple[Complexity, Complexity], int]:
        """The full sparse matrix as a dict keyed by (true, predicted)."""
        return dict(self._counts)

    def __repr__(self) -> str:
        """A readable 4x4 grid (rows are true tiers, columns predicted)."""
        labels = [tier.value for tier in COMPLEXITY_ORDER]
        width = max(len(label) for label in labels) + 1
        lines = [" " * (width + 2) + "".join(f"{label:>{width}}" for label in labels)]
        for truth in COMPLEXITY_ORDER:
            cells = "".join(
                f"{self.count(truth, predicted):>{width}}" for predicted in COMPLEXITY_ORDER
            )
            lines.append(f"{truth.value:>{width + 2}}{cells}")
        return "\n".join(lines)


@dataclass(frozen=True)
class TierMetrics:
    """Precision / recall / F1 for one complexity tier.

    ``None`` values mean the quantity is undefined for this tier (no
    true positives of that kind), and callers should render them as
    "--" rather than guess a number.
    """

    tier: Complexity
    precision: float | None
    recall: float | None
    f1: float | None
    support: int  # number of tasks whose true tier is this tier

    @property
    def is_supported(self) -> bool:
        """Whether support > 0."""
        return self.support > 0


@dataclass(frozen=True)
class EvaluationMetrics:
    """The complete evaluation of one scorer against the labeled set.

    Attributes:
        name: Scorer name as reported.
        total: Number of scored tasks.
        correct: Fully accurate predictions (predicted == ground truth).
        accuracy: correct / total (None when total == 0).
        per_tier: One :class:`TierMetrics` per tier, in tier order.
        confusion: The raw confusion matrix.
        macro_f1: Mean F1 across tiers with both precision and recall;
            None when no tier has an F1.
    """

    name: str
    total: int
    correct: int
    accuracy: float | None
    per_tier: tuple[TierMetrics, ...]
    confusion: ConfusionMatrix
    macro_f1: float | None


def evaluate(
    truths: tuple[Complexity, ...],
    predicted: tuple[Complexity, ...],
    name: str = "scorer",
) -> EvaluationMetrics:
    """Compute all metrics from parallel true/predicted tier lists.

    Args:
        truths: Ground-truth tier per task.
        predicted: Predicted tier per task (same length).
        name: Scorer label for reports.

    Raises:
        ValueError: When the lists have different lengths.
    """
    if len(truths) != len(predicted):
        raise ValueError(
            f"truths ({len(truths)}) and predicted ({len(predicted)}) must have equal length"
        )

    matrix = ConfusionMatrix()
    for truth, pred in zip(truths, predicted, strict=True):
        matrix.add(truth, pred)

    total = matrix.total()
    correct = sum(matrix.count(tier, tier) for tier in COMPLEXITY_ORDER)
    accuracy = None
    if total:
        accuracy = Fraction(correct, total)

    per_tier: list[TierMetrics] = []
    for tier in COMPLEXITY_ORDER:
        true_positives = matrix.count(tier, tier)
        pred_positives = sum(matrix.count(other, tier) for other in COMPLEXITY_ORDER)
        true_positive_possible = sum(matrix.count(tier, other) for other in COMPLEXITY_ORDER)

        precision = None
        if pred_positives:
            precision = Fraction(true_positives, pred_positives)
        recall = None
        if true_positive_possible:
            recall = Fraction(true_positives, true_positive_possible)
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = Fraction(2 * precision * recall, precision + recall)
        per_tier.append(
            TierMetrics(
                tier=tier,
                precision=float(precision) if precision is not None else None,
                recall=float(recall) if recall is not None else None,
                f1=float(f1) if f1 is not None else None,
                support=true_positive_possible,
            )
        )

    with_f1 = [m.f1 for m in per_tier if m.f1 is not None]
    macro_f1 = None
    if with_f1:
        macro_f1 = sum(with_f1) / len(with_f1)

    return EvaluationMetrics(
        name=name,
        total=total,
        correct=correct,
        accuracy=float(accuracy) if accuracy is not None else None,
        per_tier=tuple(per_tier),
        confusion=matrix,
        macro_f1=macro_f1,
    )

"""Tests for the classification metrics (hand-computed expectations)."""

from __future__ import annotations

import pytest

from vibeforge.eval.metrics import ConfusionMatrix, TierMetrics, evaluate
from vibeforge.types import Complexity

T = Complexity.TRIVIAL
L = Complexity.LOW
M = Complexity.MEDIUM
H = Complexity.HIGH


def test_perfect_scorer_scores_perfectly() -> None:
    truths = (T, L, M, H)
    metrics = evaluate(truths, truths, name="gold")

    assert metrics.name == "gold"
    assert metrics.total == 4
    assert metrics.correct == 4
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert all(tier_m.f1 == 1.0 for tier_m in metrics.per_tier)
    assert metrics.confusion.total() == 4
    assert metrics.confusion.count(T, T) == 1


def test_always_trivial_predictor_matches_hand_computed() -> None:
    truths = (T, T, L, M)
    predicted = (T, T, T, T)

    metrics = evaluate(truths, predicted, name="all-trivial")

    assert metrics.accuracy == 0.5
    assert metrics.correct == 2

    by_tier = {tier_m.tier: tier_m for tier_m in metrics.per_tier}
    trivial = by_tier[T]
    assert trivial.precision == 0.5  # 2 TP / 4 predicted-positive
    assert trivial.recall == 1.0  # 2 TP / 2 actually-trivial
    assert trivial.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)

    low = by_tier[L]
    assert low.precision is None  # nothing predicted low
    assert low.recall == 0.0  # 0 of 1 actually-low found
    assert low.f1 is None
    assert low.support == 1

    # only trivial has an F1 -> macro-F1 equals trivial's F1
    assert metrics.macro_f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_undefined_recall_when_no_support() -> None:
    metrics = evaluate((T, T), (L, L))
    by_tier = {tier_m.tier: tier_m for tier_m in metrics.per_tier}
    assert by_tier[L].recall is None  # nothing actually-low
    assert by_tier[L].f1 is None
    assert by_tier[L].precision == 0.0  # 0 of 2 predicted-low were right
    assert by_tier[T].recall == 0.0


def test_empty_input_is_safe() -> None:
    metrics = evaluate((), (), name="empty")

    assert metrics.total == 0
    assert metrics.accuracy is None
    assert metrics.macro_f1 is None
    assert all(tier_m.is_supported is False for tier_m in metrics.per_tier)


def test_unequal_lengths_raise() -> None:
    with pytest.raises(ValueError):
        evaluate((T, L), (T,))


def test_confusion_matrix_counts_are_correct() -> None:
    metrics = evaluate((T, T, L, L, M, H), (T, L, L, M, H, H))

    matrix = metrics.confusion
    assert matrix.total() == 6
    assert matrix.count(T, T) == 1
    assert matrix.count(T, L) == 1
    assert matrix.count(L, L) == 1
    assert matrix.count(L, M) == 1
    assert matrix.count(M, H) == 1
    assert matrix.count(H, H) == 1


def test_confusion_repr_is_a_4x4_grid() -> None:
    metrics = evaluate((T, L, M, H), (T, L, M, H))
    render = str(metrics.confusion)

    lines = render.splitlines()
    assert len(lines) == 5  # header + 4 rows
    assert lines[0].strip().startswith("trivial")
    assert all(line.strip() for line in lines)


def test_tier_metrics_dataclass_shape() -> None:
    metrics = evaluate((T,), (T,))
    tier_metrics = next(m for m in metrics.per_tier if m.tier is T)

    assert isinstance(tier_metrics, TierMetrics)
    assert tier_metrics.tier == T
    assert tier_metrics.precision == 1.0
    assert tier_metrics.recall == 1.0
    assert tier_metrics.f1 == 1.0
    assert tier_metrics.support == 1


def test_matrix_can_be_queried_directly() -> None:
    matrix = ConfusionMatrix()
    matrix.add(T, T)
    matrix.add(T, T)
    matrix.add(L, T)

    assert matrix.count(T, T) == 2
    assert matrix.count(T, L) == 0
    assert matrix.total() == 3
    assert matrix.raw() == {(T, T): 2, (L, T): 1}

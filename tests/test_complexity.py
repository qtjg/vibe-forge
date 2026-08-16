"""Tests for the heuristic complexity scorer."""

from __future__ import annotations

import pytest

from vibeforge.router.complexity import BASELINE_RANKS, HeuristicScorer
from vibeforge.types import Complexity, Task, TaskType


def make_task(
    task_type: TaskType = TaskType.AUTOCOMPLETE,
    prompt: str = "complete this line",
    context: str = "",
) -> Task:
    """Build a task with sensible defaults."""
    return Task(type=task_type, prompt=prompt, context=context)


@pytest.fixture
def scorer() -> HeuristicScorer:
    """A fresh scorer per test."""
    return HeuristicScorer()


def test_trivial_autocomplete_lands_on_trivial(scorer: HeuristicScorer) -> None:
    task = make_task(prompt="add_customer(db, name, email)")
    complexity, reason = scorer.score(task)
    assert complexity is Complexity.TRIVIAL
    assert "autocomplete" in reason


def test_hard_debug_task_lands_on_high(scorer: HeuristicScorer) -> None:
    context = " ".join(["worker.process(item)" for _ in range(300)])
    task = make_task(
        task_type=TaskType.DEBUG,
        prompt="Fix the race condition in the worker pool",
        context=context,
    )
    complexity, reason = scorer.score(task)
    assert complexity is Complexity.HIGH
    assert "race condition" in reason


def test_known_easy_prompt_is_not_high(scorer: HeuristicScorer) -> None:
    task = make_task(task_type=TaskType.EXPLAIN, prompt="what does this return?")
    complexity, _ = scorer.score(task)
    assert complexity in (Complexity.TRIVIAL, Complexity.LOW)
    assert complexity is not Complexity.HIGH


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        (TaskType.AUTOCOMPLETE, Complexity.TRIVIAL),
        (TaskType.EXPLAIN, Complexity.LOW),
        (TaskType.GENERATE, Complexity.LOW),
        (TaskType.REFACTOR, Complexity.MEDIUM),
        (TaskType.DEBUG, Complexity.MEDIUM),
        (TaskType.REVIEW, Complexity.MEDIUM),
    ],
)
def test_baseline_tier_per_task_type(
    scorer: HeuristicScorer, task_type: TaskType, expected: Complexity
) -> None:
    complexity, _ = scorer.score(make_task(task_type=task_type, prompt="short prompt"))
    assert complexity is expected


def test_baseline_map_covers_every_task_type() -> None:
    assert set(BASELINE_RANKS) == set(TaskType)
    assert all(0 <= rank <= 3 for rank in BASELINE_RANKS.values())


def test_length_bump_pushes_medium_task_to_high(scorer: HeuristicScorer) -> None:
    short = make_task(task_type=TaskType.REVIEW, prompt="review this loop")
    long_context = " ".join(["line_of_code();" for _ in range(500)])
    long = make_task(task_type=TaskType.REVIEW, prompt="review this loop", context=long_context)

    short_tier, _ = scorer.score(short)
    long_tier, reason = scorer.score(long)

    assert short_tier is Complexity.MEDIUM
    assert long_tier is Complexity.HIGH
    assert "length" in reason


def test_keyword_bump_triggers_only_with_high_signal_words(
    scorer: HeuristicScorer,
) -> None:
    plain = make_task(task_type=TaskType.REFACTOR, prompt="refactor the loop")
    racy = make_task(
        task_type=TaskType.REFACTOR, prompt="refactor the loop, it has a race condition"
    )

    plain_tier, _ = scorer.score(plain)
    racy_tier, reason = scorer.score(racy)

    assert plain_tier is Complexity.MEDIUM
    assert racy_tier is Complexity.HIGH
    assert "race condition" in reason


def test_autocomplete_stays_trivial_under_length_threshold(
    scorer: HeuristicScorer,
) -> None:
    context = " ".join(["x += 1" for _ in range(60)])
    complexity, _ = scorer.score(
        make_task(task_type=TaskType.AUTOCOMPLETE, prompt="finish this loop", context=context)
    )
    assert complexity is Complexity.TRIVIAL


def test_absurdly_long_prompt_clamps_to_high(scorer: HeuristicScorer) -> None:
    prompt = " ".join(["word" for _ in range(50_000)])
    complexity, reason = scorer.score(make_task(task_type=TaskType.REVIEW, prompt=prompt))
    assert complexity is Complexity.HIGH
    assert "clamped" in reason


def test_multiple_distinct_keywords_double_bump(scorer: HeuristicScorer) -> None:
    prompt = (
        "the distributed architecture has a race condition, "
        "a memory leak, and a deadlock in the async path"
    )
    complexity, reason = scorer.score(make_task(task_type=TaskType.REVIEW, prompt=prompt))
    assert complexity is Complexity.HIGH
    assert "+2" in reason


def test_reason_is_explanatory_for_simple_task(scorer: HeuristicScorer) -> None:
    _, reason = scorer.score(make_task(prompt="add_customer(db, name, email)"))
    assert "baseline autocomplete=0" in reason
    assert "=> score 0/3" in reason


def test_context_words_count_toward_length(scorer: HeuristicScorer) -> None:
    prompt = make_task(prompt="short")
    with_context = make_task(prompt="short", context=" ".join(["w" for _ in range(300)]))
    _, reason = scorer.score(prompt)
    _, context_reason = scorer.score(with_context)
    assert "length" not in reason
    assert "length" in context_reason


def test_keyword_match_is_case_insensitive(scorer: HeuristicScorer) -> None:
    task = make_task(
        task_type=TaskType.DEBUG,
        prompt="Fix the RACE CONDITION in the worker pool",
    )
    complexity, reason = scorer.score(task)
    assert complexity is Complexity.HIGH
    assert "race condition" in reason


def test_substring_noise_does_not_bump(scorer: HeuristicScorer) -> None:
    task = make_task(
        task_type=TaskType.GENERATE,
        prompt="generate a function that blocks until the lockfile is gone",
    )
    tier_a, _ = scorer.score(task)
    task_b = make_task(
        task_type=TaskType.GENERATE,
        prompt="generate a function that polls until the file is gone",
    )
    tier_b, _ = scorer.score(task_b)
    assert tier_a is tier_b


def test_baseline_override_starts_higher() -> None:
    strict = HeuristicScorer(baseline_ranks={TaskType.REVIEW: 3})
    complexity, _ = strict.score(make_task(task_type=TaskType.REVIEW, prompt="short"))
    assert complexity is Complexity.HIGH


def test_empty_baseline_override_keeps_defaults(scorer: HeuristicScorer) -> None:
    untouched = HeuristicScorer(baseline_ranks={})
    short = make_task(task_type=TaskType.EXPLAIN, prompt="short")
    assert untouched.score(short)[0] is scorer.score(short)[0]


def test_custom_length_bumps_are_tighter() -> None:
    strict = HeuristicScorer(length_bumps=((5, 1),))
    complexity, reason = strict.score(
        make_task(task_type=TaskType.EXPLAIN, prompt="a b c d e f g h")
    )
    assert complexity is Complexity.MEDIUM
    assert "length" in reason


def test_custom_keywords_are_recognized() -> None:
    custom = HeuristicScorer(high_signal_keywords=("quantum entanglement",))
    plain, _ = custom.score(make_task(prompt="store the value directly"))
    tricky = make_task(prompt="handle the quantum entanglement carefully")
    complexity, reason = custom.score(tricky)
    assert complexity.rank > plain.rank
    assert "quantum entanglement" in reason


def test_custom_keyword_double_hit_threshold() -> None:
    one_hit_keywords = HeuristicScorer(
        high_signal_keywords=("alpha", "beta"), keyword_double_hit_threshold=2
    )
    task = make_task(prompt="alpha and beta are both involved")
    _, reason = one_hit_keywords.score(task)
    assert "+2" in reason


def test_confidence_base_for_plain_task(scorer: HeuristicScorer) -> None:
    assert scorer.confidence(make_task(prompt="add_customer(db, name, email)")) == 0.5


def test_confidence_rises_with_each_signal(scorer: HeuristicScorer) -> None:
    plain = make_task(prompt="complete this line")
    assert scorer.confidence(plain) == 0.5

    racy = make_task(
        task_type=TaskType.DEBUG, prompt="race condition in the worker pool"
    )
    assert scorer.confidence(racy) == pytest.approx(0.65)

    long_and_racy = make_task(
        task_type=TaskType.DEBUG,
        prompt="race condition in the worker pool",
        context=" ".join(["w" for _ in range(300)]),
    )
    assert scorer.confidence(long_and_racy) == pytest.approx(0.85)


def test_confidence_is_capped(scorer: HeuristicScorer) -> None:
    maxed = make_task(
        task_type=TaskType.DEBUG,
        prompt=(
            "the distributed architecture has a race condition, "
            "a memory leak, a deadlock, and lock contention"
        ),
        context=" ".join(["c" for _ in range(900)]),
    )
    assert scorer.confidence(maxed) == pytest.approx(0.9)


def test_confidence_is_deterministic(scorer: HeuristicScorer) -> None:
    task = make_task(
        task_type=TaskType.DEBUG, prompt="race condition in the worker pool"
    )
    assert scorer.confidence(task) == scorer.confidence(task)


def test_token_budget_rises_with_complexity() -> None:
    assert Complexity.TRIVIAL.token_budget == 128
    assert Complexity.LOW.token_budget == 256
    assert Complexity.MEDIUM.token_budget == 1024
    assert Complexity.HIGH.token_budget == 4096

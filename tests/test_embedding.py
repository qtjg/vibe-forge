"""Tests for the embedding scorer, with embeddings mocked (no Ollama).

The fake embed maps every seed example to a distinct one-hot vector in a
16-dim space; a task vector identical to a seed therefore maps to the
intended tier with perfect determinism.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import TimeoutException
from ollama import ResponseError

import vibeforge.router.embedding as embedding_module
from vibeforge.router.complexity import HeuristicScorer
from vibeforge.router.embedding import DEFAULT_EXAMPLES_FILE, EmbeddingScorer
from vibeforge.types import Complexity, Task, TaskType

#: Space in which all fake vectors live; seeds occupy dims 0..7 in order.
SPACE = 16

SEEDS = {
    "trivial": ["complete this line of code", "write the docstring for this helper"],
    "low": ["explain what this decorator does", "add a utility to read a config file"],
    "medium": ["refactor into smaller classes", "add retries and backoff to this call"],
    "high": ["fix the race condition in the worker pool", "memory leak in the event loop"],
}


def one_hot(index: int) -> list[float]:
    """A vector with a single 1.0 at ``index % SPACE``."""
    vec = [0.0] * SPACE
    vec[index % SPACE] = 1.0
    return vec


def make_scorer(monkeypatch: pytest.MonkeyPatch) -> EmbeddingScorer:
    """Scorer with deterministic one-hot embeddings over the SEEDS set."""

    def factory(host: str | None = None, timeout: float | None = None) -> FakeOllama:
        return FakeOllama(host=host, timeout=timeout)

    monkeypatch.setattr(embedding_module.ollama, "Client", factory)
    return EmbeddingScorer(example_prompts=SEEDS)


class FakeOllama:
    """Drops-in for ``ollama.Client``: deterministic one-hot embeddings.

    Every distinct text gets a stable dim (first-seen order), so identical
    tasks and seeds share a vector while different texts differ.
    """

    _dims_by_text: dict[str, int] = {}

    def __init__(self, host: str | None = None, timeout: float | None = None) -> None:
        self.host = host
        self.timeout = timeout

    def embed(self, model: str, input: list[str], **kwargs: object) -> object:
        del model, kwargs
        vectors: list[list[float]] = []
        for text in input:
            dim = FakeOllama._dims_by_text.setdefault(text, len(FakeOllama._dims_by_text))
            vectors.append(one_hot(dim))
        return SimpleNamespace(embeddings=vectors)


def test_nearest_picks_the_matching_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = make_scorer(monkeypatch)
    task = Task(type=TaskType.DEBUG, prompt="memory leak in the event loop")

    tier, reason = scorer.score(task)

    assert tier is Complexity.HIGH
    assert "closest example: 'memory leak in the event loop'" in reason
    assert "high" in reason


def test_reason_records_tier_similarities(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = make_scorer(monkeypatch)
    _, reason = scorer.score(
        Task(type=TaskType.GENERATE, prompt="explain what this decorator does")
    )
    assert "avg per tier" in reason
    assert "low similarity" in reason


def test_confidence_is_high_for_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = make_scorer(monkeypatch)
    task = Task(type=TaskType.GENERATE, prompt="complete this line of code")
    assert scorer.confidence(task) == pytest.approx(0.9)


def test_confidence_is_lower_for_ambiguous_task(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = make_scorer(monkeypatch)
    ambiguous = Task(type=TaskType.GENERATE, prompt="improve the code quality a bit")
    assert scorer.confidence(ambiguous) < scorer.confidence(
        Task(type=TaskType.GENERATE, prompt="complete this line of code")
    )


def test_fallback_used_when_embedding_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> object:
        raise TimeoutException("slow")

    scorer = make_scorer(monkeypatch)
    monkeypatch.setattr(embedding_module.ollama, "Client", lambda **_: raise_timeout())

    tier, reason = scorer.score(Task(type=TaskType.DEBUG, prompt="fix the race"))
    assert "fell back to heuristic" in reason
    assert isinstance(tier, Complexity)


def test_fallback_used_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_response(*args: object, **kwargs: object) -> object:
        raise ResponseError("model not found", 404)

    scorer = make_scorer(monkeypatch)
    monkeypatch.setattr(embedding_module.ollama, "Client", lambda **_: raise_response())

    tier, reason = scorer.score(Task(type=TaskType.GENERATE, prompt="small task"))
    assert "embedding unavailable" in reason
    assert isinstance(tier, Complexity)


def test_fallback_confidence_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_response(*args: object, **kwargs: object) -> object:
        raise ResponseError("nope", 500)

    scorer = make_scorer(monkeypatch)
    monkeypatch.setattr(embedding_module.ollama, "Client", lambda **_: raise_response())

    task = Task(type=TaskType.DEBUG, prompt="fix the race condition in the pool")
    fallback_conf = HeuristicScorer().confidence(task)
    assert scorer.confidence(task) == fallback_conf


def test_custom_example_tiers_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unknown complexity tier"):
        EmbeddingScorer(example_prompts={"bogus": ["x"]})


def test_unknown_custom_tier_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown complexity tier"):
        EmbeddingScorer(example_prompts={"high": ["only valid one"], "nope": ["x"]})


def test_bundled_examples_file_is_complete() -> None:
    import yaml

    with DEFAULT_EXAMPLES_FILE.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert set(raw) == {"trivial", "low", "medium", "high"}
    for tier, prompts in raw.items():
        assert len(prompts) >= 3, f"{tier} needs more example prompts"
        assert all(len(prompt.split()) >= 3 for prompt in prompts)

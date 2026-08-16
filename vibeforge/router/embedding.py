"""Embedding-based complexity scoring (opt-in, experimental).

:class:`EmbeddingScorer` compares a task's prompt against a small labeled
set of example prompts (one set per complexity tier), embedding both sides
with a local Ollama embedding model and picking the nearest tier. It is the
research alternative to :class:`~vibeforge.router.complexity.HeuristicScorer`
for measuring how far a learned similarity signal gets on the benchmark
suite.

Design rules:

- **Never raises.** Embedding needs a live Ollama server with the model
  pulled; when that fails, the scorer delegates to a fallback scorer and
  says so in the reason string. Routing must not depend on the embedding
  model being present.
- **Caches seed embeddings** per model tag so only the task prompt is
  embedded per call after warm-up.
- **Explainable**: the reason records the winning tier, its average
  similarity, and the single closest example prompt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
import ollama

from vibeforge.router.complexity import HeuristicScorer, Scorer
from vibeforge.router.executor import DEFAULT_OLLAMA_URL
from vibeforge.types import Complexity, Task

__all__ = ["EmbeddingScorer", "DEFAULT_EMBEDDING_MODEL"]

#: Embedding model used unless overridden; must be pullable on this host.
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

#: Where the labeled example prompts ship inside the installed package.
DEFAULT_EXAMPLES_FILE = Path(__file__).resolve().parent.parent / "data" / "embedding-examples.yaml"

#: Per-embedding timeout; local embedding models load lazily, so be generous.
DEFAULT_EMBED_TIMEOUT = 60.0

#: Error kinds that mean "embeddings are unavailable right now".
_UNAVAILABLE_ERRORS = (
    httpx.TimeoutException,
    httpx.TransportError,
    ollama.ResponseError,
    ConnectionError,  # the ollama SDK's translation of httpx.ConnectError
    ValueError,  # non-JSON body from a flaky server
)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingScorer:
    """Nearest-neighbor complexity scorer over labeled example prompts.

    Args:
        client_host: Ollama server base URL.
        model: Embedding model tag to use.
        example_prompts: Mapping of tier name -> list of example prompts;
            defaults to the bundled :data:`DEFAULT_EXAMPLES_FILE`.
        fallback: Scorer used when embeddings are unavailable.
        timeout: Per-embed timeout in seconds.

    Examples:
        >>> scorer = EmbeddingScorer()
        >>> complexity, reason = scorer.score(
        ...     Task(TaskType.DEBUG, "fix the race condition in the worker pool")
        ... )
    """

    def __init__(
        self,
        client_host: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        example_prompts: Mapping[str, Sequence[str]] | None = None,
        fallback: Scorer | None = None,
        timeout: float = DEFAULT_EMBED_TIMEOUT,
    ) -> None:
        """Configure the embedding scorer and its fallback."""
        self._client_host = client_host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._fallback = fallback if fallback is not None else HeuristicScorer()
        if example_prompts is not None:
            self._examples = {tier: list(prompts) for tier, prompts in example_prompts.items()}
            self._validate_tiers()
        else:
            self._examples = _load_examples(DEFAULT_EXAMPLES_FILE)
        self._cache: dict[str, list[tuple[Complexity, list[float], str]]] = {}

    def _validate_tiers(self) -> None:
        """Reject custom example sets that name unknown tiers."""
        valid = {tier.value for tier in Complexity}
        unknown = set(self._examples) - valid
        if unknown:
            raise ValueError(f"unknown complexity tier in examples: {sorted(unknown)}")

    def score(self, task: Task) -> tuple[Complexity, str]:
        """Return ``(complexity, reason)`` via nearest-neighbor matching.

        When embeddings cannot be computed, the fallback scorer's answer is
        returned with the failure noted in the reason.
        """
        try:
            tier, sims, best_example = self._nearest(task)
        except _UNAVAILABLE_ERRORS as exc:
            complexity, fallback_reason = self._fallback.score(task)
            reason = (
                f"embedding unavailable ({exc.__class__.__name__}); "
                f"fell back to heuristic -> {fallback_reason}"
            )
            return complexity, reason

        per_tier = {name: round(sim, 3) for tier, sim in sims.items() if (name := tier.value)}
        reason = (
            f"embedding nearest-neighbor (model {self._model}): "
            f"{tier.value} similarity {round(sims[tier], 3)} "
            f"(avg per tier {per_tier}); closest example: {best_example!r}"
        )
        return tier, reason

    def confidence(self, task: Task) -> float:
        """Margin-based 0..1 estimate, or the fallback's when it delegates."""
        try:
            _, sims, _ = self._nearest(task)
        except _UNAVAILABLE_ERRORS:
            fallback_confidence = getattr(self._fallback, "confidence", None)
            return fallback_confidence(task) if callable(fallback_confidence) else 0.5

        ranked = sorted(sims.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
        return round(min(0.9, max(0.5, 0.5 + margin * 2.5)), 2)

    def _nearest(self, task: Task) -> tuple[Complexity, dict[Complexity, float], str]:
        """Embed the task and find the winning tier and its closest example."""
        task_text = f"{task.prompt} {task.context}".strip() or task.prompt
        client = ollama.Client(host=self._client_host, timeout=self._timeout)
        task_vec = client.embed(model=self._model, input=[task_text]).embeddings[0]

        best_example = ""
        best_sim = -math.inf
        per_tier: dict[Complexity, list[float]] = {}
        for tier, seed_vec, example in self._seed_rows():
            sim = _cosine(task_vec, seed_vec)
            per_tier.setdefault(tier, []).append(sim)
            if sim > best_sim:
                best_sim = sim
                best_example = example

        avg_sims = {tier: sum(scores) / len(scores) for tier, scores in per_tier.items()}
        winner = max(avg_sims, key=avg_sims.get)  # type: ignore[arg-type]
        return winner, avg_sims, best_example

    def _seed_rows(self) -> list[tuple[Complexity, list[float], str]]:
        """Cached (tier, embedding, example-text) rows for the seed set."""
        if self._model not in self._cache:
            rows = [
                (Complexity(tier), prompt)
                for tier, prompts in self._examples.items()
                for prompt in prompts
            ]
            texts = [prompt for _, prompt in rows]
            client = ollama.Client(host=self._client_host, timeout=self._timeout)
            vectors = client.embed(model=self._model, input=texts).embeddings
            self._cache[self._model] = [
                (tier, list(vec), prompt) for (tier, prompt), vec in zip(rows, vectors, strict=True)
            ]
        return self._cache[self._model]


def _load_examples(path: Path) -> dict[str, list[str]]:
    """Load the bundled example prompts (YAML: tier -> list of strings)."""
    import yaml

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    examples: dict[str, list[str]] = {}
    for tier, prompts in raw.items():
        if tier not in ("trivial", "low", "medium", "high"):
            raise ValueError(f"unknown complexity tier in examples file: {tier}")
        examples[tier] = [str(prompt) for prompt in prompts]
    return examples

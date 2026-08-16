"""Run one prompt against several models concurrently (comparison mode).

Used by ``vibeforge route --compare``: the demo feature that answers
"what would llama vs coder actually *say*" in parallel. This is an
execution utility -- routing itself stays single-model. Each model gets
its own executor instance (its own HTTP client), so no client is shared
across threads.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from vibeforge.types import ExecutionResult

__all__ = ["ModelRun", "run_models_concurrently", "MAX_CONCURRENT_MODELS"]

#: Cap on how many models run at once (GPU memory is rarely the limit;
#: this is about Ollama loading several models into RAM greedily).
MAX_CONCURRENT_MODELS = 8


@dataclass(frozen=True)
class ModelRun:
    """The result of running one model on the shared prompt.

    Attributes:
        name: Registry tier name (``tiny-fast``, ...).
        ollama_tag: The model tag that was actually called.
        result: The execution result (success or failure).
    """

    name: str
    ollama_tag: str
    result: ExecutionResult

    @property
    def ok(self) -> bool:
        """Whether the model responded."""
        return self.result.ok

    @property
    def latency_ms(self) -> float | None:
        """Wall-clock latency of the generation call."""
        return self.result.latency_ms


def run_models_concurrently(
    models: Sequence[tuple[str, str]],
    prompt: str,
    executor_factory: Callable[[], object],
    options: dict[str, object] | None = None,
    max_workers: int | None = None,
) -> tuple[ModelRun, ...]:
    """Execute ``prompt`` against every model in parallel.

    Args:
        models: Pairs of (tier name, ollama tag) to run.
        prompt: The shared prompt.
        executor_factory: Callable returning an object with
            ``execute(model_tag, prompt, options) -> ExecutionResult``.
            Called once per model so each thread owns its client.
        options: Generation options passed to every model.
        max_workers: Number of parallel calls; defaults to the number of
            models, capped at :data:`MAX_CONCURRENT_MODELS`.

    Returns:
        One :class:`ModelRun` per model, in input order. Failures are
        per-model (a dead model never aborts its siblings).
    """
    if not models:
        return ()

    names, tags = zip(*models, strict=True)
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            raise ValueError(f"duplicate model tag {tag!r} in comparison set")
        seen.add(tag)

    worker_count = min(len(models), max_workers or MAX_CONCURRENT_MODELS) or 1
    runs: list[ModelRun | None] = [None] * len(models)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_execute_one, executor_factory, tag, prompt, options): index
            for index, tag in enumerate(tags)
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            runs[index] = ModelRun(name=names[index], ollama_tag=tags[index], result=result)

    return tuple(run for run in runs if run is not None)


def _execute_one(
    executor_factory: Callable[[], object],
    model_tag: str,
    prompt: str,
    options: dict[str, object] | None,
) -> ExecutionResult:
    """Run one generation (thread body)."""
    started = time.perf_counter()
    executor = executor_factory()
    result = executor.execute(  # type: ignore[attr-defined]
        model_tag=model_tag, prompt=prompt, options=options
    )
    if result.latency_ms is None:
        elapsed = (time.perf_counter() - started) * 1000.0
        return ExecutionResult(
            model=result.model,
            prompt=result.prompt,
            latency_ms=elapsed,
            eval_count=result.eval_count,
            output=result.output,
            error=result.error,
            error_kind=result.error_kind,
            retries_attempted=result.retries_attempted,
            status_code=result.status_code,
        )
    return result

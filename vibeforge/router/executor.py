"""Talk to a local Ollama server and time the generation.

:class:`OllamaExecutor` is the only layer that touches the network. It talks
to Ollama through the official ``ollama`` Python client (never raw HTTP), so
the project rides the ecosystem's actual tooling instead of reimplementing
it. The executor never raises for expected failure modes (Ollama down, model
not pulled, timeout, garbage response): it catches them and returns an
:class:`ExecutionResult` with ``error`` and ``error_kind`` set, so the CLI
and benchmark can keep going.

Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried
with exponential backoff before giving up.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import ollama

from vibeforge.types import ExecutionResult

__all__ = ["OllamaExecutor"]

#: Default endpoint of a local Ollama server.
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: How long to wait for the full generation before giving up.
DEFAULT_TIMEOUT_SECONDS = 120.0

#: Number of automatic retries for transient failures (first attempt + N).
DEFAULT_MAX_RETRIES = 2

#: Backoff starts here and doubles per attempt, capped at :data:`_CAP`.
_DEFAULT_BACKOFF_BASE = 0.5

#: Upper bound for a single backoff pause, in seconds.
_DEFAULT_BACKOFF_CAP = 8.0

_RETRYABLE_HTTP_STATUS = frozenset({429}) | frozenset(range(500, 600))


class OllamaExecutor:
    """Sends single-turn generation requests to a local Ollama server.

    Examples:
        >>> executor = OllamaExecutor()
        >>> result = executor.execute("llama3.1:latest", "explain this regex")
        >>> result.ok
        True
        >>> result.latency_ms
        812.4
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        backoff_cap: float = _DEFAULT_BACKOFF_CAP,
    ) -> None:
        """Build an executor bound to one Ollama server.

        Args:
            base_url: Scheme + host + port of the Ollama server.
            timeout: Default per-request timeout in seconds; individual
                ``execute`` calls may override it.
            max_retries: How many times to retry transient failures.
            backoff_base: Initial backoff in seconds before the first retry.
            backoff_cap: Upper bound on the backoff pause, in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap

    def execute(
        self,
        model_tag: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Generate ``prompt`` with ``model_tag``, measuring latency.

        Transient failures are retried with exponential backoff up to
        ``max_retries`` times.

        Args:
            model_tag: Tag of the model to call, e.g. ``llama3.1:latest``.
            prompt: The prompt to send.
            options: Optional Ollama sampler options, e.g.
                ``{"temperature": 0.0}``. Passed through as-is.
            timeout: Per-call timeout in seconds; overrides the executor
                default when given.

        Returns:
            An :class:`ExecutionResult`; ``error`` is set when the request
            could not complete.
        """
        per_call = self._timeout if timeout is None else timeout

        for attempt in range(self._max_retries + 1):
            if attempt:
                time.sleep(self._backoff_seconds(attempt))
            started = time.perf_counter()
            client = ollama.Client(host=self._base_url, timeout=per_call)
            try:
                response = client.generate(
                    model=model_tag,
                    prompt=prompt,
                    stream=False,
                    options=options if options else {},
                )
            except httpx.TimeoutException:
                if attempt < self._max_retries:
                    continue
                return self._failure(
                    model_tag, prompt, "timeout", f"timed out after {per_call}s", attempt
                )
            except ConnectionError:
                # The ollama SDK translates httpx.ConnectError into the
                # builtin ConnectionError with an install-hint message.
                if attempt < self._max_retries:
                    continue
                detail = (
                    f"cannot reach Ollama at {self._base_url} "
                    f"(is the server running?). Failed to connect, and retries did not help"
                )
                return self._failure(model_tag, prompt, "connection", detail, attempt)
            except httpx.ConnectError as exc:
                if attempt < self._max_retries:
                    continue
                detail = (
                    f"cannot reach Ollama at {self._base_url} "
                    f"(is the server running?). {exc.__class__.__name__}: {exc}"
                )
                return self._failure(model_tag, prompt, "connection", detail, attempt)
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    continue
                return self._failure(
                    model_tag, prompt, "request", f"Ollama request failed: {exc}", attempt
                )
            except ollama.ResponseError as exc:
                if exc.status_code in _RETRYABLE_HTTP_STATUS and attempt < self._max_retries:
                    continue
                return self._failure(
                    model_tag,
                    prompt,
                    "http",
                    _describe_ollama_error(exc),
                    attempt,
                    latency_ms=_elapsed_ms(started),
                    status_code=(
                        exc.status_code if exc.status_code and exc.status_code > 0 else None
                    ),
                )
            except ValueError:
                return self._failure(
                    model_tag,
                    prompt,
                    "json",
                    "Ollama returned a non-JSON response",
                    attempt,
                    latency_ms=_elapsed_ms(started),
                )
            finally:
                client.close()
                elapsed_ms = _elapsed_ms(started)

            return ExecutionResult(
                model=model_tag,
                prompt=prompt,
                latency_ms=elapsed_ms,
                eval_count=response.eval_count,
                output=response.response,
            )

        raise AssertionError("unreachable: loop always returns")  # pragma: no cover

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff for retry ``attempt`` (1-based)."""
        return min(self._backoff_base * 2 ** (attempt - 1), self._backoff_cap)

    def _failure(
        self,
        model_tag: str,
        prompt: str,
        kind: str,
        message: str,
        retries: int,
        latency_ms: float | None = None,
        status_code: int | None = None,
    ) -> ExecutionResult:
        """Build an error result, mentioning the retry count when relevant."""
        if retries:
            label = "retry" if retries == 1 else "retries"
            message = f"{message} (after {retries} {label})"
        return ExecutionResult(
            model=model_tag,
            prompt=prompt,
            latency_ms=latency_ms,
            error=message,
            error_kind=kind,
            retries_attempted=retries,
            status_code=status_code,
        )

    @property
    def base_url(self) -> str:
        """The Ollama endpoint this executor is bound to."""
        return self._base_url


def _describe_ollama_error(exc: ollama.ResponseError) -> str:
    """Extract the readable error body from an Ollama error response."""
    try:
        body = json.loads(exc.error)
        if isinstance(body, dict) and body.get("error"):
            detail = body["error"]
        else:
            detail = exc.error
    except (json.JSONDecodeError, TypeError):
        detail = exc.error
    if exc.status_code is not None and exc.status_code > 0:
        return f"Ollama error (HTTP {exc.status_code}): {detail}"
    return f"Ollama error: {detail}"


def _elapsed_ms(started: float) -> float:
    """Wall-clock milliseconds since ``started``."""
    return (time.perf_counter() - started) * 1000.0

"""Talk to a local Ollama server and time the generation.

:class:`OllamaExecutor` is the only layer that touches the network. It never
raises for expected failure modes (Ollama down, model not pulled, timeout,
garbage response): it catches them and returns an :class:`ExecutionResult`
with ``error`` and ``error_kind`` set, so the CLI and benchmark can keep
going.

Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried
with exponential backoff before giving up.
"""

from __future__ import annotations

import time
from typing import Any

import requests

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
        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {"model": model_tag, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        per_call = self._timeout if timeout is None else timeout

        for attempt in range(self._max_retries + 1):
            if attempt:
                time.sleep(self._backoff_seconds(attempt))
            started = time.perf_counter()
            try:
                response = requests.post(url, json=payload, timeout=per_call)
            except requests.Timeout:
                if attempt < self._max_retries:
                    continue
                return self._failure(
                    model_tag, prompt, "timeout", f"timed out after {per_call}s", attempt
                )
            except requests.ConnectionError as exc:
                if attempt < self._max_retries:
                    continue
                detail = (
                    f"cannot reach Ollama at {self._base_url} "
                    f"(is the server running?). {exc.__class__.__name__}: {exc}"
                )
                return self._failure(model_tag, prompt, "connection", detail, attempt)
            except requests.RequestException as exc:
                if attempt < self._max_retries:
                    continue
                return self._failure(
                    model_tag, prompt, "request", f"Ollama request failed: {exc}", attempt
                )
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0

            if not response.ok:
                if (
                    response.status_code in _RETRYABLE_HTTP_STATUS
                    and attempt < self._max_retries
                ):
                    continue
                return self._failure(
                    model_tag,
                    prompt,
                    "http",
                    _describe_http_error(response),
                    attempt,
                    latency_ms=elapsed_ms,
                )

            try:
                body: dict[str, Any] = response.json()
            except ValueError:
                return self._failure(
                    model_tag,
                    prompt,
                    "json",
                    "Ollama returned a non-JSON response",
                    attempt,
                    latency_ms=elapsed_ms,
                )

            return ExecutionResult(
                model=model_tag,
                prompt=prompt,
                latency_ms=elapsed_ms,
                eval_count=body.get("eval_count"),
                output=body.get("response") or body.get("output"),
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
        )

    @property
    def base_url(self) -> str:
        """The Ollama endpoint this executor is bound to."""
        return self._base_url


def _describe_http_error(response: requests.Response) -> str:
    """Build a readable error message from a non-200 Ollama response."""
    try:
        body = response.json()
        if isinstance(body, dict) and "error" in body:
            return f"Ollama error (HTTP {response.status_code}): {body['error']}"
    except ValueError:
        pass
    return f"Ollama error (HTTP {response.status_code}): {response.text[:200]}"

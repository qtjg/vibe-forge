"""Talk to a local Ollama server and time the generation.

:class:`OllamaExecutor` is the only layer that touches the network. It never
raises for expected failure modes (Ollama down, model not pulled, timeout,
garbage response): it catches them and returns an :class:`ExecutionResult`
with ``error`` set, so the CLI and benchmark can keep going.
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
        self, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS
    ) -> None:
        """Build an executor bound to one Ollama server.

        Args:
            base_url: Scheme + host + port of the Ollama server.
            timeout: Per-request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def execute(
        self,
        model_tag: str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Generate ``prompt`` with ``model_tag``, measuring latency.

        Args:
            model_tag: Tag of the model to call, e.g. ``llama3.1:latest``.
            prompt: The prompt to send.
            options: Optional Ollama sampler options, e.g.
                ``{"temperature": 0.0}``. Passed through as-is.

        Returns:
            An :class:`ExecutionResult`; ``error`` is set when the request
            could not complete.
        """
        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {"model": model_tag, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options

        started = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
        except requests.Timeout:
            return ExecutionResult(
                model=model_tag,
                prompt=prompt,
                error=f"timed out after {self._timeout}s",
            )
        except requests.ConnectionError as exc:
            detail = (
                f"cannot reach Ollama at {self._base_url} "
                f"(is the server running?). {exc.__class__.__name__}: {exc}"
            )
            return ExecutionResult(model=model_tag, prompt=prompt, error=detail)
        except requests.RequestException as exc:
            return ExecutionResult(
                model=model_tag, prompt=prompt, error=f"Ollama request failed: {exc}"
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not response.ok:
            return ExecutionResult(
                model=model_tag,
                prompt=prompt,
                latency_ms=elapsed_ms,
                error=_describe_http_error(response),
            )

        try:
            body: dict[str, Any] = response.json()
        except ValueError:
            return ExecutionResult(
                model=model_tag,
                prompt=prompt,
                latency_ms=elapsed_ms,
                error="Ollama returned a non-JSON response",
            )

        return ExecutionResult(
            model=model_tag,
            prompt=prompt,
            latency_ms=elapsed_ms,
            eval_count=body.get("eval_count"),
            output=body.get("response") or body.get("output"),
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

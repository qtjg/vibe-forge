"""Read-only probing of a local Ollama server.

Separated from :class:`~vibeforge.router.executor.OllamaExecutor` because
*checking* what is available (``Client.list``) is a different concern from
*generating*: the router needs availability to pick models, while generation
happens only after a pick.

Like the executor, the probe never raises for expected failures -- it returns
an :class:`OllamaStatus` describing reachability instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import ollama

from vibeforge.router.executor import DEFAULT_OLLAMA_URL

__all__ = ["OllamaStatus", "probe_ollama"]

#: How long to wait for the availability probe.
DEFAULT_PROBE_TIMEOUT = 3.0


@dataclass(frozen=True)
class OllamaStatus:
    """The result of probing one Ollama server.

    Attributes:
        reachable: True when the server answered the probe at all.
        available: True when reachable *and* at least one model is pulled.
        pulled_tags: Set of model tags the server reports as pulled.
        error: Description of the failure when not reachable.
        checked_at: UTC timestamp of the probe.
    """

    reachable: bool
    pulled_tags: frozenset[str] = frozenset()
    error: str | None = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def available(self) -> bool:
        """True when the server is reachable and has pulled models."""
        return self.reachable and bool(self.pulled_tags)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-ready dict."""
        return {
            "reachable": self.reachable,
            "available": self.available,
            "pulled_tags": sorted(self.pulled_tags),
            "error": self.error,
            "checked_at": self.checked_at.isoformat(),
        }


def probe_ollama(
    base_url: str = DEFAULT_OLLAMA_URL, timeout: float = DEFAULT_PROBE_TIMEOUT
) -> OllamaStatus:
    """Ask the server which models are pulled.

    Args:
        base_url: Scheme + host + port of the Ollama server.
        timeout: Probe timeout in seconds.

    Returns:
        An :class:`OllamaStatus`; failures are reported, never raised.
    """
    client = ollama.Client(host=base_url.rstrip("/"), timeout=timeout)
    try:
        listing = client.list()
    except httpx.TimeoutException:
        return OllamaStatus(reachable=False, error=f"probe timed out after {timeout}s")
    except httpx.ConnectError:
        return OllamaStatus(
            reachable=False,
            error=f"cannot reach Ollama at {base_url} (is the server running?)",
        )
    except ollama.ResponseError as exc:
        return OllamaStatus(
            reachable=False,
            error=f"Ollama error (HTTP {exc.status_code}): {exc.error}",
        )
    except ValueError:
        return OllamaStatus(reachable=False, error="Ollama returned a non-JSON response")
    except httpx.TransportError as exc:
        return OllamaStatus(reachable=False, error=f"Ollama probe failed: {exc}")
    finally:
        client.close()

    tags = frozenset(str(model.model) for model in listing.models if getattr(model, "model", None))
    return OllamaStatus(reachable=True, pulled_tags=tags)

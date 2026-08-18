"""``vibeforge doctor``: read-only health checks for config + Ollama.

Doctor answers "is this install in a working state?" without changing
anything. It never mutates configs or pulls models — fixing is a separate,
explicitly confirmed step (see ROADMAP 5.3). Findings are collected
instead of printed, so the CLI and tests share one path.

Checks:

1. **Config** — resolves the models file (env var / ``./models.yaml`` /
   bundled), parses it with the same pydantic validation the router uses,
   and reports the tier + task-type inventory.
2. **Ollama reachability** — can the server be reached at all?
3. **Pulled models** — every configured tier's tag must exist in
   ``ollama list``; a missing tag is a warning with the exact pull command.
4. **Tier coverage** — every complexity tier (trivial..high) should be
   covered by at least one configured ceiling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vibeforge.router.executor import DEFAULT_OLLAMA_URL
from vibeforge.router.registry import (
    ConfigError,
    ModelRegistry,
    find_models_file,
)
from vibeforge.types import COMPLEXITY_ORDER

__all__ = ["Finding", "Doctor"]

OK = "ok"
WARN = "warn"
ERROR = "error"


@dataclass(frozen=True)
class Finding:
    """One doctor check result.

    Attributes:
        level: ``ok``, ``warn``, or ``error``.
        check: Short machine-readable check name (``config``, ``ollama``,
            ``model``, ``tiers``).
        message: Human-readable detail, with an actionable hint when the
            finding is not ``ok``.
    """

    level: str
    check: str
    message: str


#: Ollama client constructor; swapped in tests for a stub.
OllamaClientFactory = Callable[[], object]


class Doctor:
    """Runs the read-only checks and collects findings.

    Args:
        host: Base URL of the Ollama server.
        registry: Registry to check; ``None`` loads the resolved default
            (used so tests can inject one).
        ollama_client_factory: Builds the Ollama client (``None`` uses the
            official ``ollama.Client``).
    """

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_URL,
        registry: ModelRegistry | None = None,
        ollama_client_factory: OllamaClientFactory | None = None,
    ) -> None:
        """Configure the checks.

        Args:
            host: Base URL of the Ollama server.
            registry: Registry to check; ``None`` loads the resolved default
                (used so tests can inject one).
            ollama_client_factory: Builds the Ollama client (``None`` uses
                the official ``ollama.Client``).
        """
        self._host = host
        self._registry = registry
        self._client_factory = ollama_client_factory or self._default_client

    @staticmethod
    def _default_client() -> object:
        import ollama

        return ollama.Client()

    def run(self) -> tuple[Finding, ...]:
        """Run every check.

        The config check short-circuits the rest when the config is
        unusable (nothing else is meaningful then).
        """
        findings = self._check_config()
        if any(f.level == ERROR for f in findings):
            return tuple(findings)
        findings += self._check_ollama()
        findings += self._check_pulled_models()
        findings += self._check_tier_coverage()
        return tuple(findings)

    def _check_config(self) -> tuple[Finding, ...]:
        """Resolve and validate the models config, report its inventory."""
        try:
            path = find_models_file()
            registry = (
                self._registry if self._registry is not None else ModelRegistry.load_default()
            )
        except ConfigError as exc:
            return (
                Finding(ERROR, "config", str(exc)),
                Finding(
                    ERROR,
                    "config",
                    "hint: fix the file above; see the README Configuration section",
                ),
            )
        self._registry = registry
        tiers = len(registry.models)
        names = ", ".join(m.name for m in registry.models)
        types = len(registry.task_types)
        tiers_word = "tiers" if tiers != 1 else "tier"
        return (
            Finding(
                OK,
                "config",
                f"{path} ({tiers} {tiers_word}: {names}, {types} task types)",
            ),
        )

    def _check_ollama(self) -> tuple[Finding, ...]:
        """Probe the Ollama server; a dead server is a hard error."""
        try:
            self._client_factory().list()
        except ConnectionError:
            return (
                Finding(
                    ERROR,
                    "ollama",
                    f"cannot reach Ollama at {self._host}",
                ),
                Finding(
                    ERROR,
                    "ollama",
                    "hint: is the Ollama server running? Start it with `ollama serve`",
                ),
            )
        except Exception as exc:
            return (
                Finding(ERROR, "ollama", f"Ollama check failed: {exc}"),
                Finding(
                    ERROR,
                    "ollama",
                    "hint: is the Ollama server running? Start it with `ollama serve`",
                ),
            )
        return (Finding(OK, "ollama", f"reachable at {self._host}"),)

    def _check_pulled_models(self) -> tuple[Finding, ...]:
        """Compare configured tags against ``ollama list``; missing tags warn."""
        assert self._registry is not None
        try:
            response = self._client_factory().list()
            pulled = {getattr(model, "model", None) for model in response.models}
        except Exception:
            # Reachability was already reported; skip model-level checks.
            return ()
        pulled = {tag for tag in pulled if tag}

        missing = [
            (tier.name, tier.ollama_tag)
            for tier in self._registry.models
            if tier.ollama_tag not in pulled
        ]
        if not missing:
            return (Finding(OK, "model", "every configured tier is pulled"),)
        findings = [
            Finding(
                WARN,
                "model",
                f"tier {name!r} ({tag}) is not pulled — run: ollama pull {tag}",
            )
            for name, tag in missing
        ]
        if not any(tier.ollama_tag in pulled for tier in self._registry.models):
            findings.append(
                Finding(
                    ERROR, "model", "no configured model is pulled — routing would always fall back"
                )
            )
        return tuple(findings)

    def _check_tier_coverage(self) -> tuple[Finding, ...]:
        """Every complexity tier should be covered by some ceiling."""
        assert self._registry is not None
        covered = {
            tier
            for model in self._registry.models
            for tier in COMPLEXITY_ORDER
            if tier.rank <= model.complexity_ceiling.rank
        }
        uncovered = [tier for tier in COMPLEXITY_ORDER if tier not in covered]
        if not uncovered:
            return (Finding(OK, "tiers", "all 4 complexity tiers covered"),)
        names = ", ".join(t.value for t in uncovered)
        return (
            Finding(
                WARN,
                "tiers",
                f"no model ceiling covers {names} — those tasks fall back "
                f"to the most capable model",
            ),
        )

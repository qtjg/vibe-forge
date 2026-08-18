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
3. **Pulled models** — tags from ``ollama list`` are cross-referenced both
   ways: configured tiers that are not pulled warn with the exact pull
   command, and pulled models that are not configured warn with a
   suggested tier entry (guessed name/ceiling/RAM — read-only advice).
4. **Hardware fit** — detected RAM/VRAM (psutil, nvidia-smi; optional
   probes that degrade gracefully) against each tier's ``approx_ram_gb``.
5. **Tier coverage** — every complexity tier (trivial..high) should be
   covered by at least one configured ceiling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vibeforge.router.executor import DEFAULT_OLLAMA_URL
from vibeforge.router.hardware import (
    HardwareInfo,
    SubprocessRunner,
    detect,
    guess_ceiling,
    guess_ram_gb,
)
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
            ``model``, ``hardware``, ``tiers``).
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
        psutil_module: object | None = None,
        subprocess_runner: SubprocessRunner | None = None,
        hardware: HardwareInfo | None = None,
    ) -> None:
        """Configure the checks.

        Args:
            host: Base URL of the Ollama server.
            registry: Registry to check; ``None`` loads the resolved default
                (used so tests can inject one).
            ollama_client_factory: Builds the Ollama client (``None`` uses
                the official ``ollama.Client``).
            psutil_module: Injected psutil module for RAM probing
                (``None`` imports it lazily; never a hard dependency).
            subprocess_runner: Injected subprocess runner for nvidia-smi
                (see :func:`vibeforge.router.hardware.nvidia_vram_gb`).
            hardware: Precomputed probe results; when given, no probes run
                (tests inject this instead of real hardware).
        """
        self._host = host
        self._registry = registry
        self._client_factory = ollama_client_factory or self._default_client
        self._psutil_module = psutil_module
        self._subprocess_runner = subprocess_runner
        self._hardware = hardware

    @staticmethod
    def _default_client() -> object:
        import ollama

        return ollama.Client()

    def run(self) -> tuple[Finding, ...]:
        """Run every check.

        The config check short-circuits the rest when the config is
        unusable (nothing else is meaningful then).
        """
        results = self._check_config()
        if any(f.level == ERROR for f in results):
            return tuple(results)
        findings = list(results)
        findings += self._check_ollama()
        pulled = self._pull_reported()
        if pulled is not None:
            tags, sizes = pulled
            findings += self._check_pulled_models(tags)
            findings += self._check_unconfigured_models(tags, sizes)
        findings += self._check_hardware_fit()
        findings += self._check_tier_coverage()
        return tuple(findings)

    def _pull_reported(self) -> tuple[set[str], dict[str, int]] | None:
        """Tags (and byte sizes) Ollama reports; ``None`` when unreachable.

        Returns a ``(tags, tag -> size_bytes)`` pair, or ``None`` so the
        caller can skip the dependent checks when the probe failed.
        """
        try:
            response = self._client_factory().list()
        except Exception:
            return None
        sizes: dict[str, int] = {}
        tags: set[str] = set()
        for model in getattr(response, "models", ()):
            tag = getattr(model, "model", None)
            if not tag:
                continue
            tags.add(tag)
            size = getattr(model, "size", None)
            if isinstance(size, int) and size > 0:
                sizes[tag] = size
        return tags, sizes

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

    def _check_pulled_models(self, pulled: set[str]) -> tuple[Finding, ...]:
        """Warn for configured tiers whose tag is not pulled."""
        assert self._registry is not None
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

    def _check_unconfigured_models(
        self, pulled: set[str], sizes: dict[str, int]
    ) -> tuple[Finding, ...]:
        """Suggest tier entries for pulled models missing from models.yaml."""
        assert self._registry is not None
        configured = {tier.ollama_tag for tier in self._registry.models}
        extras = sorted(pulled - configured)
        if not extras:
            return ()
        findings = []
        for tag in extras:
            ram = guess_ram_gb(sizes.get(tag), tag)
            ceiling = guess_ceiling(ram)
            name = tag.replace(":", "-")
            block = (
                f"pulled model {tag!r} is not in models.yaml — suggested tier:\n"
                f"  - name: {name}\n"
                f"    ollama_tag: {tag}\n"
                f"    complexity_ceiling: {ceiling}"
            )
            if ram is not None:
                block += f"\n    approx_ram_gb: {ram}"
            findings.append(Finding(WARN, "model", block))
        return tuple(findings)

    def _check_hardware_fit(self) -> tuple[Finding, ...]:
        """Cross-reference detected RAM/VRAM against tier sizes.

        Probes degrade silently: no psutil means a warning pointing at the
        ``[hardware]`` extra, no nvidia-smi means VRAM checks are skipped.
        """
        assert self._registry is not None
        hardware = self._hardware or detect(
            psutil_module=self._psutil_module,
            subprocess_runner=self._subprocess_runner,
        )
        findings: list[Finding] = []

        if hardware.missing_psutil:
            findings.append(
                Finding(
                    WARN,
                    "hardware",
                    "RAM: psutil not installed — pip install 'vibe-forge[hardware]' "
                    "for full detection",
                )
            )
        else:
            ram = hardware.total_ram_gb or 0.0
            findings.append(Finding(OK, "hardware", f"system RAM: {hardware.total_ram_gb} GB"))
            for tier in self._registry.models:
                if tier.approx_ram_gb > ram:
                    findings.append(
                        Finding(
                            WARN,
                            "hardware",
                            f"tier {tier.name!r} needs ~{tier.approx_ram_gb} GB RAM "
                            f"but only {ram:.1f} GB detected",
                        )
                    )

        if hardware.vram_detected:
            vram = hardware.total_vram_gb or 0.0
            findings.append(Finding(OK, "hardware", f"GPU VRAM: {vram} GB (nvidia-smi)"))
            for tier in self._registry.models:
                if tier.approx_ram_gb > vram:
                    findings.append(
                        Finding(
                            WARN,
                            "hardware",
                            f"tier {tier.name!r} needs ~{tier.approx_ram_gb} GB RAM "
                            f"which exceeds {vram:.1f} GB GPU VRAM",
                        )
                    )
        elif hardware.ram_detected:
            findings.append(
                Finding(
                    OK,
                    "hardware",
                    "GPU: no nvidia-smi on PATH — VRAM checks skipped",
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

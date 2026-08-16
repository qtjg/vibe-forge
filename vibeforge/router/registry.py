"""Model tier registry: loads ``models.yaml`` and picks models for tiers.

The registry is the single source of truth for what models exist and how
capable they are. Its one job: given a complexity tier, return the *cheapest*
model that can cover it -- never route to a bigger model than necessary.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import yaml

from vibeforge.types import Complexity, ModelTier

__all__ = ["ConfigError", "ModelRegistry", "find_models_file"]

#: Environment variable that overrides the models config file location.
CONFIG_ENV_VAR = "VIBEFORGE_MODELS"

#: User-facing config searched in the working directory first.
USER_CONFIG_NAME = "models.yaml"

#: Path of the config bundled with the package (fallback when no user file).
PACKAGE_DEFAULT: Path = Path(__file__).resolve().parent.parent / "data" / "models.yaml"


class ConfigError(Exception):
    """Raised when the models config is missing, malformed, or inconsistent."""


def find_models_file() -> Path:
    """Locate the models config: env var, then ``./models.yaml``, then bundled.

    Resolution order:
    1. ``$VIBEFORGE_MODELS`` if set and the file exists.
    2. ``./models.yaml`` in the current working directory.
    3. The config bundled with the installed package.
    """
    from_env = os.environ.get(CONFIG_ENV_VAR)
    if from_env:
        env_path = Path(from_env)
        if env_path.is_file():
            return env_path
        raise ConfigError(f"VIBEFORGE_MODELS points at a missing file: {env_path}")

    user_path = Path.cwd() / USER_CONFIG_NAME
    if user_path.is_file():
        return user_path

    return PACKAGE_DEFAULT


class ModelRegistry:
    """Loads model tiers and selects the cheapest adequate one per complexity.

    Examples:
        >>> registry = ModelRegistry.load_default()
        >>> registry.pick_for(Complexity.MEDIUM)
        ModelTier(name='balanced', ...)
    """

    def __init__(self, models: Sequence[ModelTier]) -> None:
        """Construct a registry from an explicit tier list.

        Args:
            models: At least one model tier; must not be empty.

        Raises:
            ConfigError: When ``models`` is empty or has duplicate names.
        """
        if not models:
            raise ConfigError("models.yaml defines no models — add at least one tier")
        names = [model.name for model in models]
        if len(names) != len(set(names)):
            raise ConfigError(f"duplicate model tier names: {sorted(set(names))}")
        self._models: tuple[ModelTier, ...] = tuple(
            sorted(models, key=lambda m: (m.approx_ram_gb, m.name))
        )

    @classmethod
    def load_default(cls) -> ModelRegistry:
        """Load the registry from the resolved default config location."""
        return cls.from_yaml_file(find_models_file())

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ModelRegistry:
        """Load tiers from a YAML file on disk.

        Raises:
            ConfigError: When the file is unreadable, not YAML, or invalid.
        """
        try:
            with open(path, encoding="utf-8") as handle:
                return cls.from_yaml(handle)
        except OSError as exc:
            raise ConfigError(f"cannot read models config {path}: {exc}") from exc

    @classmethod
    def from_yaml(cls, stream: str | IO[str]) -> ModelRegistry:
        """Parse tiers from a YAML string or file-like object."""
        try:
            raw = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ConfigError(f"models config is not valid YAML: {exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: object) -> ModelRegistry:
        """Parse tiers from the raw structure produced by YAML loading."""
        if not isinstance(raw, dict) or "models" not in raw:
            raise ConfigError('models config must contain a top-level "models:" list')

        entries = raw["models"]
        if not isinstance(entries, list) or not entries:
            raise ConfigError('"models:" must be a non-empty list of tiers')
        if not all(isinstance(entry, dict) for entry in entries):
            raise ConfigError('"models:" entries must be mappings (name, ollama_tag, ...)')

        tiers = tuple(_parse_tier(entry, index) for index, entry in enumerate(entries))
        return cls(tiers)

    @property
    def models(self) -> tuple[ModelTier, ...]:
        """All tiers, sorted cheapest (lowest RAM) first."""
        return self._models

    def pick_for(self, complexity: Complexity) -> ModelTier:
        """Return the cheapest tier whose ceiling covers ``complexity``.

        Falls back to the most capable tier (most RAM) when no configured
        tier covers the requested complexity, e.g. a registry whose highest
        ceiling is below the requested tier.

        Args:
            complexity: The complexity tier that must be covered.

        Returns:
            The selected :class:`ModelTier`.
        """
        eligible = [m for m in self._models if m.complexity_ceiling.rank >= complexity.rank]
        if eligible:
            return eligible[0]
        return self._models[-1]


def _parse_tier(entry: dict[str, Any], index: int) -> ModelTier:
    """Validate and convert one raw YAML mapping into a :class:`ModelTier`."""
    required = ("name", "ollama_tag", "complexity_ceiling", "approx_ram_gb")
    missing = [field for field in required if field not in entry]
    if missing:
        raise ConfigError(f"tier #{index + 1} is missing required field(s): {', '.join(missing)}")

    name = str(entry["name"]).strip()
    if not name:
        raise ConfigError(f"tier #{index + 1} has an empty name")

    raw_ceiling = entry["complexity_ceiling"]
    try:
        ceiling = Complexity(raw_ceiling)
    except ValueError as exc:
        valid = ", ".join(tier.value for tier in Complexity)
        raise ConfigError(
            f"tier {name!r} has invalid complexity_ceiling {raw_ceiling!r} " f"(valid: {valid})"
        ) from exc

    try:
        ram = float(entry["approx_ram_gb"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"tier {name!r} has invalid approx_ram_gb {entry['approx_ram_gb']!r}"
        ) from exc
    if ram <= 0:
        raise ConfigError(f"tier {name!r} has approx_ram_gb {ram}, must be > 0")

    return ModelTier(
        name=name,
        ollama_tag=str(entry["ollama_tag"]),
        complexity_ceiling=ceiling,
        approx_ram_gb=ram,
        notes=str(entry.get("notes", "")),
    )

"""Model tier registry: loads ``models.yaml`` and picks models for tiers.

The registry is the single source of truth for what models exist and how
capable they are. Its one job: given a complexity tier, return the *cheapest*
model that can cover it -- never route to a bigger model than necessary.

Config parsing and validation live in :mod:`vibeforge.router.schema`
(pydantic); this module turns the validated config into the runtime
registry and keeps the *picking* logic.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import yaml

from vibeforge.router.schema import ConfigError, config_from_dict
from vibeforge.types import Complexity, ModelTier

__all__ = ["ConfigError", "ModelPick", "ModelRegistry", "find_models_file"]

#: Environment variable that overrides the models config file location.
CONFIG_ENV_VAR = "VIBEFORGE_MODELS"

#: User-facing config searched in the working directory first.
USER_CONFIG_NAME = "models.yaml"

#: Path of the config bundled with the package (fallback when no user file).
PACKAGE_DEFAULT: Path = Path(__file__).resolve().parent.parent / "data" / "models.yaml"


@dataclass(frozen=True)
class ModelPick:
    """The result of choosing a model, including why a fallback happened.

    Attributes:
        model: The chosen :class:`ModelTier`.
        fallback_reason: Why the strict cheapest-covering rule was relaxed,
            or ``None`` when the pick was the cheapest covering model.
    """

    model: ModelTier
    fallback_reason: str | None = None


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
        """Parse tiers from the raw structure produced by YAML loading.

        The structure is validated field-by-field (pydantic) and every
        problem is reported with its exact location, e.g.
        ``models[1].approx_ram_gb: Input should be greater than 0``.

        Raises:
            ConfigError: When the config is missing, malformed, or has
                invalid tiers.
        """
        config = config_from_dict(raw, source="models config")
        tiers = tuple(
            ModelTier(
                name=tier.name,
                ollama_tag=tier.ollama_tag,
                complexity_ceiling=tier.complexity_ceiling,
                approx_ram_gb=tier.approx_ram_gb,
                notes=tier.notes,
            )
            for tier in config.models
        )
        return cls(tiers)

    @property
    def models(self) -> tuple[ModelTier, ...]:
        """All tiers, sorted cheapest (lowest RAM) first."""
        return self._models

    def pick_for(self, complexity: Complexity) -> ModelTier:
        """Return the cheapest tier whose ceiling covers ``complexity``.

        Convenience wrapper around :meth:`pick` that drops the fallback
        reason; see :meth:`pick` for the full semantics.

        Args:
            complexity: The complexity tier that must be covered.

        Returns:
            The selected :class:`ModelTier`.
        """
        return self.pick(complexity).model

    def pick(
        self,
        complexity: Complexity,
        available_tags: set[str] | None = None,
    ) -> ModelPick:
        """Choose a model for ``complexity``, aware of what is actually pulled.

        Selection rules, in order:

        1. Cheapest tier whose ceiling covers ``complexity``.
        2. When ``available_tags`` is given and no covering tier is pulled,
           the most capable *pulled* tier is used as a fallback.
        3. When nothing is pulled at all, or no availability info was given,
           the most capable configured tier is the last-resort fallback.

        Every deviation from rule 1 is reported in :attr:`ModelPick.fallback_reason`.

        Args:
            complexity: The complexity tier that must be covered.
            available_tags: Set of model tags Ollama reports as pulled;
                ``None`` disables availability awareness.

        Returns:
            The selected model plus an optional fallback explanation.
        """
        eligible = [m for m in self._models if m.complexity_ceiling.rank >= complexity.rank]

        if available_tags is not None:
            pulled = {m for m in eligible if m.ollama_tag in available_tags}
            if pulled:
                return ModelPick(model=min(pulled, key=lambda m: (m.approx_ram_gb, m.name)))

            # A pulled model that can actually execute beats a configured one
            # that is guaranteed to 404.
            any_pulled = [m for m in self._models if m.ollama_tag in available_tags]
            if any_pulled:
                best = max(any_pulled, key=lambda m: m.approx_ram_gb)
                return ModelPick(
                    model=best,
                    fallback_reason=(
                        f"no configured model covering {complexity.value} is pulled; "
                        f"using most capable pulled model {best.name}"
                    ),
                )
            if eligible:
                return ModelPick(
                    model=eligible[0],
                    fallback_reason=(
                        f"no configured models are pulled; "
                        f"using configured fallback {eligible[0].name}"
                    ),
                )
            return ModelPick(
                model=self._models[-1],
                fallback_reason=(
                    f"no configured model covers {complexity.value} and none are pulled; "
                    f"using configured fallback {self._models[-1].name}"
                ),
            )

        if eligible:
            return ModelPick(model=eligible[0])
        return ModelPick(
            model=self._models[-1],
            fallback_reason=(
                f"no configured model covers {complexity.value}; "
                f"using most capable configured model {self._models[-1].name}"
            ),
        )


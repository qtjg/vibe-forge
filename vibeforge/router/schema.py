"""Pydantic schema for the ``models.yaml`` config format.

This is the contract every config must satisfy. Keeping validation here
(both for the bundled default and for user files) means one canonical set
of rules, and lets the CLI translate schema errors into actionable
messages instead of raw stack traces.

The top level of a valid config looks like::

    models:
      - name: tiny-fast
        ollama_tag: qwen2.5:0.5b
        complexity_ceiling: trivial
        approx_ram_gb: 0.6
        notes: "optional"
    custom_task_types:        # optional, experimental (v0.3)
      - name: translation
        baseline_rank: 1
        description: "Translate text between languages"

Anything else is rejected with a structured error that names the exact
field and path (``models[1].complexity_ceiling``, ...).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vibeforge.types import Complexity

__all__ = [
    "ConfigError",
    "CustomTaskTypeConfig",
    "TierConfig",
    "ConfigFile",
    "config_from_dict",
]


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or inconsistent."""


#: pydantic model settings shared by every config model.
#: ``extra="forbid"``: a typo'd key (``complexity_ceilingg``) is a hard
#: error, not a silently ignored field.
_MODEL_CONFIG = ConfigDict(extra="forbid")


class TierConfig(BaseModel):
    """One model tier entry."""

    model_config = _MODEL_CONFIG

    name: str = Field(min_length=1)
    ollama_tag: str = Field(min_length=1)
    complexity_ceiling: Complexity
    approx_ram_gb: float = Field(gt=0, allow_inf_nan=False)
    notes: str = ""


class CustomTaskTypeConfig(BaseModel):
    """A user-registered task type (optional, merged over the built-ins)."""

    model_config = _MODEL_CONFIG

    name: str = Field(min_length=1)
    baseline_rank: int = Field(ge=0, le=3)
    description: str = ""


class ConfigFile(BaseModel):
    """The full ``models.yaml`` document."""

    model_config = _MODEL_CONFIG

    models: list[TierConfig] = Field(min_length=1)
    custom_task_types: list[CustomTaskTypeConfig] = Field(default_factory=list)


def config_from_dict(raw: object, source: str = "models config") -> ConfigFile:
    """Validate the raw structure produced by YAML loading.

    Raises:
        ConfigError: With a field-level message naming the exact location
            of every problem, e.g.
            ``models config: models[1].approx_ram_gb: Input should be
            greater than 0``.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must be a mapping at the top level, got {type(raw).__name__}")
    try:
        return ConfigFile.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(_format_error(error) for error in exc.errors())
        raise ConfigError(f"{source} is invalid: {details}") from exc


def _format_error(error: dict[str, Any]) -> str:
    """Render one pydantic error with a ``models[1].field`` style path."""
    path = ""
    for part in error["loc"]:
        if isinstance(part, int):
            path += f"[{part}]"
        elif path:
            path += f".{part}"
        else:
            path = str(part)
    return f"{path or '<root>'}: {error['msg']}"

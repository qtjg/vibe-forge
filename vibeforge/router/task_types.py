"""Task type registry: the built-ins plus user-registered types.

Task types used to be a closed enum baked into the core. This module makes
them a registry so a project can add new types (e.g. ``translate`` or
``migrate``) from its own ``models.yaml`` *without touching vibeforge
code* -- the "plugin" story behind v0.3.

Every task type carries the signal the heuristic scorer needs: a
:attr:`TaskTypeDefinition.baseline_rank` (0..3, how heavy the type starts).
The built-ins are seeded from the public catalog in
:mod:`vibeforge.types`; custom definitions only merge on top, they can
never shadow a built-in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vibeforge.router.schema import ConfigError
from vibeforge.types import TaskType

__all__ = [
    "TaskTypeDefinition",
    "TaskTypeRegistry",
    "BUILTIN_TASK_TYPES",
    "DEFAULT_BASELINE_RANK",
]

#: Benchmark/fallback baseline for a type the registry does not know.
#: Defensive only: everything reachable from a validated config passes
#: through the registry.
DEFAULT_BASELINE_RANK = 1


@dataclass(frozen=True)
class TaskTypeDefinition:
    """One registered task type.

    Attributes:
        name: The type as it appears in ``--type``, models.yaml, and CSVs.
        baseline_rank: Starting complexity rank (0=trivial .. 3=high) used
            by the heuristic scorer.
        description: Free-text purpose, surfaced in docs and errors.
    """

    name: str
    baseline_rank: int
    description: str = ""


#: The six built-in task types, in display order. Derived programmatically
#: from the public catalog so the registry can never drift from
#: :class:`vibeforge.types.TaskType`.
BUILTIN_TASK_TYPES: tuple[TaskTypeDefinition, ...] = (
    TaskTypeDefinition("autocomplete", 0, "Complete a partial line or expression."),
    TaskTypeDefinition("explain", 1, "Explain code, errors, or concepts."),
    TaskTypeDefinition("generate", 1, "Write new code from a description."),
    TaskTypeDefinition("refactor", 2, "Restructure code without changing behavior."),
    TaskTypeDefinition("debug", 2, "Find and fix a bug."),
    TaskTypeDefinition("review", 2, "Review code or a design for problems."),
)

#: Description used for types that came from user config.
_CUSTOM_DESCRIPTION = "Custom task type registered by the user."


class TaskTypeRegistry:
    """Ordered catalog of every task type routing understands.

    Built-ins always come first, then custom types in config order.
    Custom names must not collide with built-ins or with each other;
    either case is a :class:`ConfigError` with the offending name.
    """

    def __init__(self, definitions: Sequence[TaskTypeDefinition]) -> None:
        """Build a registry from explicit definitions.

        Args:
            definitions: The full catalog (built-ins are *not* merged
                automatically; start from :meth:`builtins` and add customs
                via :meth:`from_config`).

        Raises:
            ConfigError: When names are empty or duplicated.
        """
        self._definitions = tuple(definitions)
        names = [d.name for d in self._definitions]
        if any(not name for name in names):
            raise ConfigError("task type names must not be empty")
        if len(names) != len(set(names)):
            raise ConfigError(f"duplicate task type names: {sorted(set(names))}")

    @classmethod
    def builtins(cls) -> TaskTypeRegistry:
        """The registry containing only the six built-in types."""
        return cls(BUILTIN_TASK_TYPES)

    @classmethod
    def from_config(
        cls, custom_types: Sequence[object] | None = None
    ) -> TaskTypeRegistry:
        """Built-ins merged with the ``custom_task_types`` config entries.

        Args:
            custom_types: Validated ``CustomTaskTypeConfig`` objects from
                the models config (``None``/empty means built-ins only).

        Raises:
            ConfigError: When a custom type shadows a built-in or another
                custom type.
        """
        custom = tuple(
            TaskTypeDefinition(
                name=entry.name,
                baseline_rank=entry.baseline_rank,
                description=entry.description or _CUSTOM_DESCRIPTION,
            )
            for entry in (custom_types or ())
        )
        builtin_names = {d.name for d in BUILTIN_TASK_TYPES}
        seen: set[str] = set()
        for entry in custom:
            if entry.name in builtin_names:
                raise ConfigError(
                    f"custom task type {entry.name!r} shadows a built-in type"
                )
            if entry.name in seen:
                raise ConfigError(
                    f"duplicate task type names: {sorted({entry.name})}"
                )
            seen.add(entry.name)
        return cls((*BUILTIN_TASK_TYPES, *custom))

    @property
    def definitions(self) -> tuple[TaskTypeDefinition, ...]:
        """Every definition, built-ins first."""
        return self._definitions

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered type name, the valid values for ``--type``."""
        return tuple(d.name for d in self._definitions)

    def definition(self, name: str) -> TaskTypeDefinition | None:
        """Return the definition for ``name``, or ``None`` if unregistered."""
        return next((d for d in self._definitions if d.name == name), None)

    def baseline_rank(self, name: str) -> int:
        """Scoring baseline for ``name``; the default when unregistered."""
        definition = self.definition(name)
        if definition is None:
            return DEFAULT_BASELINE_RANK
        return definition.baseline_rank

    def __len__(self) -> int:
        """Number of registered task types."""
        return len(self._definitions)

    def __contains__(self, name: object) -> bool:
        """True when ``name`` is a registered task type."""
        return name in self.names


def _assert_catalog_in_sync() -> None:
    """Guard against the built-in tuple drifting from the enum."""
    enum_names = {member.value for member in TaskType}
    tuple_names = {d.name for d in BUILTIN_TASK_TYPES}
    if enum_names != tuple_names:
        raise RuntimeError("BUILTIN_TASK_TYPES drifted from TaskType enum")


_assert_catalog_in_sync()

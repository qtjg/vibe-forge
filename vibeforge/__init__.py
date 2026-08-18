"""vibe-forge: local-first policy router for coding assistants.

Routes coding subtasks to the cheapest adequate local LLM served by Ollama,
so trivial tasks hit small fast models and hard tasks hit stronger ones --
fully offline.
"""

from __future__ import annotations

__version__ = "0.3.0"

from vibeforge.types import (
    Complexity,
    ExecutionResult,
    ModelTier,
    RoutingDecision,
    Task,
    TaskType,
)

__all__ = [
    "__version__",
    "TaskType",
    "Complexity",
    "Task",
    "ModelTier",
    "RoutingDecision",
    "ExecutionResult",
]

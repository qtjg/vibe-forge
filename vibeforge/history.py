"""Persistent routing history.

Decisions are appended as JSON lines to a local file so the CLI, dashboard,
and ``stats`` command all share one durable view of what was routed -- while
the in-memory history on :class:`~vibeforge.router.policy.PolicyRouter`
stays the fast path for a single process.

Storage is plain append-only JSONL: crash-safe (each line is flushed), easy
to inspect, trivial to consume with pandas. Corrupt or partial trailing
lines are skipped, never fatal.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

__all__ = ["HistoryStore", "default_history_path"]

#: Environment variable that overrides the history file location.
HISTORY_ENV_VAR = "VIBEFORGE_HISTORY"

#: Files smaller than this are read whole; larger files use a tail scan.
_TAIL_SCAN_THRESHOLD_BYTES = 256 * 1024

#: Read this many bytes per tail-scan step.
_TAIL_SCAN_CHUNK_BYTES = 16 * 1024


def default_history_path() -> Path:
    """Return the default history file location (XDG-aware, user-overridable).

    Resolution order:
    1. ``$VIBEFORGE_HISTORY`` if set.
    2. ``$XDG_DATA_HOME/vibeforge/history.jsonl``.
    3. ``~/.local/share/vibeforge/history.jsonl``.
    """
    from_env = os.environ.get(HISTORY_ENV_VAR)
    if from_env:
        return Path(from_env)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "vibeforge" / "history.jsonl"


class HistoryStore:
    """Append-only JSONL store of routing decisions.

    Examples:
        >>> store = HistoryStore(Path("/tmp/history.jsonl"))
        >>> store.append({"task_type": "debug", "model": "heavy", ...})
        >>> store.recent(limit=10)
        [...]
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Bind the store to a file.

        Args:
            path: File to read/write; defaults to
                :func:`default_history_path`.
        """
        self._path = Path(path) if path is not None else default_history_path()
        self._lock = threading.Lock()
        #: Number of unparseable lines skipped during the last read.
        self.skipped_lines: int = 0

    @property
    def path(self) -> Path:
        """The history file this store reads and writes."""
        return self._path

    def append(self, decision: Mapping[str, Any]) -> None:
        """Append one decision dict as a JSON line and flush.

        Args:
            decision: JSON-serializable mapping (see
                :meth:`RoutingDecision.as_dict`).

        Raises:
            OSError: When the file cannot be written.
        """
        line = json.dumps(decision, separators=(",", ":"))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def load(self) -> list[dict[str, Any]]:
        """Return every stored decision, oldest first.

        Unparseable lines are skipped and counted in :attr:`skipped_lines`.

        Returns:
            List of decision dicts in file order.
        """
        self.skipped_lines = 0
        records: list[dict[str, Any]] = []
        for line in self._iter_lines(from_end=False):
            record = self._parse_line(line)
            if record is not None:
                records.append(record)
        return records

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the last ``limit`` decisions, newest first.

        Args:
            limit: Maximum number of decisions to return.

        Returns:
            Newest-first list of decision dicts.
        """
        self.skipped_lines = 0
        records: list[dict[str, Any]] = []
        for line in self._iter_lines(from_end=True):
            record = self._parse_line(line)
            if record is not None:
                records.append(record)
                if len(records) >= limit:
                    break
        return records

    def count(self) -> int:
        """Number of stored decisions (unparseable lines excluded)."""
        return len(self.load())

    def clear(self) -> None:
        """Delete the history file (no-op when it does not exist)."""
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass

    def _iter_lines(self, *, from_end: bool) -> Iterator[str]:
        """Yield raw lines, newest-first when ``from_end`` is true."""
        try:
            size = self._path.stat().st_size
        except OSError:
            return

        if not from_end or size <= _TAIL_SCAN_THRESHOLD_BYTES:
            try:
                with open(self._path, encoding="utf-8") as handle:
                    lines = handle.readlines()
            except OSError:
                return
            yield from reversed(lines) if from_end else lines
            return

        # Tail scan: read the last chunks, split on newlines, walk backwards.
        with open(self._path, encoding="utf-8") as handle:
            position = size
            buffer = ""
            while position > 0:
                position = max(0, position - _TAIL_SCAN_CHUNK_BYTES)
                handle.seek(position)
                chunk = handle.read(_TAIL_SCAN_CHUNK_BYTES)
                buffer = chunk + buffer
                lines = buffer.splitlines()
                buffer = lines[0] if lines and not chunk.endswith("\n") else ""
                yield from reversed(lines[1:] if buffer else lines)
            if buffer:
                yield buffer

    def _parse_line(self, line: str) -> dict[str, Any] | None:
        """Parse one JSON line; count and ignore malformed lines."""
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            self.skipped_lines += 1
            return None
        if not isinstance(record, dict):
            self.skipped_lines += 1
            return None
        return record

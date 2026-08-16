"""SQLite-backed persistence for dashboard decisions.

:class:`HistoryDB` replaces the dashboard's in-memory list with a durable
store: every decision survives a process restart, so the dashboard no longer
loses history when it stops. The class mirrors the ``add`` / ``recent`` /
``__len__`` surface of the in-memory store so the FastAPI app treats both the
same way.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

__all__ = ["HistoryDB", "default_db_path"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""


class HistoryDB:
    """A thread-safe SQLite-backed store of decision dicts.

    The full decision dict is stored as one JSON row; ``timestamp`` is
    denormalized for cheap ordering. Rows are returned newest-first by
    insertion order (rowid), matching the in-memory store's semantics.

    Examples:
        >>> with HistoryDB(":memory:") as db:
        ...     db.add({"timestamp": "t", "model": "tiny-fast"})
        ...     db.recent(10)
        [{'timestamp': 't', 'model': 'tiny-fast'}]
    """

    def __init__(self, path: str | Path) -> None:
        """Open (and initialize) the database at ``path``.

        Args:
            path: SQLite file path; ``":memory:"`` creates a throwaway
                database.
        """
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def add(self, decision: dict[str, Any]) -> None:
        """Append one decision dict (validated by the caller)."""
        payload = json.dumps(decision, separators=(",", ":"))
        timestamp = str(decision.get("timestamp", ""))
        with self._lock:
            self._conn.execute(
                "INSERT INTO decisions (timestamp, payload) VALUES (?, ?)",
                (timestamp, payload),
            )
            self._conn.commit()

    def recent(self, limit: int) -> list[dict[str, Any]]:
        """Return the last ``limit`` decisions, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def __len__(self) -> int:
        """Number of stored decisions."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> HistoryDB:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the database when the ``with`` block exits."""
        self.close()


def default_db_path() -> Path:
    """The default database location: ``~/.vibeforge/history.db``.

    Respects ``$XDG_DATA_HOME`` when set (per XDG base-directory spec).
    """
    base = Path.home() / ".vibeforge"
    if data_home := os.getenv("XDG_DATA_HOME"):
        base = Path(data_home) / "vibeforge"
    return base / "history.db"

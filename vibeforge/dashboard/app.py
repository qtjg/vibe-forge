"""FastAPI backend for the vibe-forge live dashboard.

The dashboard owns a decision history that persists to SQLite: decisions
arrive either from the same process (e.g. tests) or -- the normal flow --
from the CLI via ``vibeforge route ... --dashboard http://localhost:8420``,
which POSTs a JSON decision to :meth:`post_decision`. A restart keeps the
history intact because it lives in ``~/.vibeforge/history.db`` by default
(override with ``vibeforge serve --db-path``).

API contract (stable; ``index.html`` polls these directly, no build step):

- ``GET /api/decisions?limit=50`` -> ``{"decisions": [...]}``, newest first.
- ``GET /api/stats`` -> ``{"models": [...]}`` per-model usage + latency.
- ``POST /api/decisions`` accepts one decision dict (see the CLI for the
  exact shape produced by ``RoutingDecision.as_dict()``) and stores it.

Add new fields to decision dicts, never rename existing ones.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vibeforge.dashboard.store import HistoryDB

#: Where the no-build static frontend lives inside the installed package.
STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Fields every stored decision must carry (the dashboard API shape).
REQUIRED_DECISION_FIELDS: tuple[str, ...] = (
    "timestamp",
    "task_type",
    "prompt",
    "score",
    "complexity",
    "reason",
    "model",
    "ollama_tag",
)


class DecisionStore:
    """A thread-safe in-memory list of decision dicts (tests, no-db runs)."""

    def __init__(self, initial: list[dict[str, Any]] | None = None) -> None:
        """Start with an optional pre-seeded list of decision dicts."""
        self._decisions: list[dict[str, Any]] = list(initial) if initial else []
        self._lock = threading.Lock()

    def add(self, decision: dict[str, Any]) -> None:
        """Append one decision dict (validated by the caller)."""
        with self._lock:
            self._decisions.append(decision)

    def recent(self, limit: int) -> list[dict[str, Any]]:
        """Return the last ``limit`` decisions, newest first."""
        with self._lock:
            return list(reversed(self._decisions[-limit:]))

    def __len__(self) -> int:
        """Number of stored decisions."""
        with self._lock:
            return len(self._decisions)


def create_app(
    history: list[dict[str, Any]] | None = None,
    db_path: str | Path | None = None,
) -> FastAPI:
    """Build the dashboard application.

    Args:
        history: Optional pre-seeded decisions (in-memory runs only).
        db_path: Optional SQLite file path; when given, decisions persist
            across restarts. Without it the dashboard stays in-memory.

    Returns:
        A configured :class:`FastAPI` application.
    """
    if db_path is not None:
        store: Any = HistoryDB(db_path)
    else:
        store = DecisionStore(initial=history)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Close the store on shutdown (context manager for FastAPI)."""
        yield
        closer = getattr(store, "close", None)
        if callable(closer):
            closer()

    app = FastAPI(
        title="vibe-forge dashboard",
        description="Live view of routing decisions for local-first LLM routing.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.db_path = str(db_path) if db_path is not None else None

    @app.get("/")
    def index() -> FileResponse:
        """Serve the single-page dashboard frontend."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/decisions")
    def list_decisions(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, list[Any]]:
        """Return the last ``limit`` routing decisions, newest first."""
        return {"decisions": store.recent(limit)}

    @app.get("/api/stats")
    def stats() -> dict[str, list[dict[str, object]]]:
        """Return per-model usage counts, average latency, and error counts."""
        return {"models": _compute_stats(store.recent(limit=10_000))}

    @app.post("/api/decisions")
    async def post_decision(request: Request) -> dict[str, str]:
        """Record a decision dict pushed by the CLI (``--dashboard`` flag)."""
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="decision must be a JSON object")
        missing = [field for field in REQUIRED_DECISION_FIELDS if field not in body]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"decision missing required field(s): {', '.join(missing)}",
            )
        store.add(body)
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _compute_stats(decisions: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Aggregate per-model usage, latency, and errors from decisions."""
    by_model: dict[str, dict[str, object]] = {}
    for decision in decisions:
        name = str(decision["model"])
        stats = by_model.setdefault(
            name,
            {
                "model": name,
                "ollama_tag": decision.get("ollama_tag", ""),
                "decisions": 0,
                "latency_sum": 0.0,
                "latency_runs": 0,
                "errors": 0,
                "last_seen": "",
            },
        )
        stats["decisions"] = int(stats["decisions"]) + 1
        if decision.get("execution_error"):
            stats["errors"] = int(stats["errors"]) + 1
        latency = decision.get("latency_ms")
        if isinstance(latency, (int, float)) and latency is not None:
            stats["latency_sum"] = float(stats["latency_sum"]) + float(latency)
            stats["latency_runs"] = int(stats["latency_runs"]) + 1
        timestamp = decision.get("timestamp")
        if timestamp and str(timestamp) > str(stats["last_seen"]):
            stats["last_seen"] = str(timestamp)

    result: list[dict[str, object]] = []
    for name in sorted(by_model):
        stats = by_model[name]
        runs = int(stats["latency_runs"])
        avg = stats["latency_sum"] / runs if runs else None
        result.append(
            {
                "model": name,
                "ollama_tag": stats["ollama_tag"],
                "decisions": stats["decisions"],
                "avg_latency_ms": round(avg, 1) if avg is not None else None,
                "errors": stats["errors"],
                "last_seen": stats["last_seen"],
            }
        )
    return result


def now_iso() -> str:
    """UTC now as an ISO string (used by tests and clients)."""
    return datetime.now(UTC).isoformat()

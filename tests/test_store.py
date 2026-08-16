"""Tests for the SQLite-backed decision store (no server needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeforge.dashboard.store import HistoryDB, default_db_path


def test_add_and_recent_roundtrip(tmp_path: Path) -> None:
    db = HistoryDB(tmp_path / "history.db")
    db.add({"timestamp": "2026-01-01T00:00:00+00:00", "model": "tiny-fast"})
    db.add({"timestamp": "2026-01-01T00:00:01+00:00", "model": "heavy"})
    db.close()

    db = HistoryDB(tmp_path / "history.db")
    try:
        rows = db.recent(10)
        assert [r["model"] for r in rows] == ["heavy", "tiny-fast"]
        assert len(db) == 2
    finally:
        db.close()


def test_recent_limit_and_newest_first(tmp_path: Path) -> None:
    db = HistoryDB(tmp_path / "history.db")
    for i in range(5):
        db.add({"timestamp": f"t{i}", "model": f"m{i}"})

    rows = db.recent(2)
    assert [r["model"] for r in rows] == ["m4", "m3"]
    db.close()


def test_restart_survival(tmp_path: Path) -> None:
    """Kill-and-restart semantics: write, close, reopen, history intact."""
    path = tmp_path / "history.db"
    db = HistoryDB(path)
    db.add({"timestamp": "t1", "model": "balanced", "prompt": "survive me"})
    db.close()

    reopened = HistoryDB(path)
    try:
        assert len(reopened) == 1
        assert reopened.recent(10)[0]["prompt"] == "survive me"
    finally:
        reopened.close()


def test_empty_db_is_empty(tmp_path: Path) -> None:
    db = HistoryDB(tmp_path / "history.db")
    try:
        assert len(db) == 0
        assert db.recent(10) == []
    finally:
        db.close()


def test_full_payload_dict_is_preserved(tmp_path: Path) -> None:
    payload = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "task_type": "debug",
        "prompt": "fix the race",
        "file_path": None,
        "score": 3,
        "complexity": "high",
        "reason": "baseline debug=2 => score 3/3 (high)",
        "confidence": 0.7,
        "token_budget": 4096,
        "fallback_reason": None,
        "model": "heavy",
        "ollama_tag": "qwen2.5-coder:14b",
    }
    db = HistoryDB(tmp_path / "history.db")
    db.add(payload)
    row = db.recent(1)[0]
    assert row == payload
    db.close()


def test_context_manager_closes(tmp_path: Path) -> None:
    with HistoryDB(tmp_path / "history.db") as db:
        db.add({"timestamp": "t", "model": "m"})
    # reopening the same file must succeed after close
    with HistoryDB(tmp_path / "history.db") as db:
        assert len(db) == 1


def test_create_then_reopen_creates_schema(tmp_path: Path) -> None:
    """Opening an existing DB file does not wipe prior data."""
    path = tmp_path / "history.db"
    db = HistoryDB(path)
    db.add({"timestamp": "t", "model": "a"})
    db.close()

    db = HistoryDB(path)  # second open must be a no-op migration
    try:
        assert len(db) == 1
    finally:
        db.close()


def test_default_db_path_is_under_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/home/example"))

    assert default_db_path() == Path("/home/example/.vibeforge/history.db")


def test_default_db_path_respects_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")

    assert default_db_path() == Path("/custom/data/vibeforge/history.db")

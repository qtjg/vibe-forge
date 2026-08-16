"""Tests for the persistent JSONL history store."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vibeforge.history import HistoryStore, default_history_path
from vibeforge.router.complexity import HeuristicScorer
from vibeforge.router.policy import PolicyRouter
from vibeforge.router.registry import ModelRegistry
from vibeforge.types import Task, TaskType


def make_decision_dict(model: str = "tiny-fast", seq: int = 0) -> dict[str, object]:
    """A decision-shaped dict, unique per call."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "task_type": "autocomplete",
        "prompt": f"prompt {seq}",
        "file_path": None,
        "score": 0,
        "complexity": "trivial",
        "reason": "baseline autocomplete=0 => score 0/3 (trivial)",
        "model": model,
        "ollama_tag": "tag:latest",
    }


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    store.append(make_decision_dict(seq=1))
    store.append(make_decision_dict(seq=2))

    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0]["prompt"] == "prompt 1"
    assert loaded[1]["prompt"] == "prompt 2"
    assert store.count() == 2


def test_recent_returns_newest_first(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    for seq in range(5):
        store.append(make_decision_dict(seq=seq))

    recent = store.recent(limit=3)
    assert [r["prompt"] for r in recent] == ["prompt 4", "prompt 3", "prompt 2"]


def test_recent_without_limit_returns_all_newest_first(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    for seq in range(3):
        store.append(make_decision_dict(seq=seq))
    assert [r["prompt"] for r in store.recent()] == ["prompt 2", "prompt 1", "prompt 0"]


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "nope.jsonl")
    assert store.load() == []
    assert store.recent() == []
    assert store.count() == 0


def test_corrupt_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps(make_decision_dict(seq=1))
        + "\n"
        + '{"truncated":\n'
        + json.dumps(make_decision_dict(seq=2))
        + "\n",
        encoding="utf-8",
    )
    store = HistoryStore(path)
    loaded = store.load()
    assert [r["prompt"] for r in loaded] == ["prompt 1", "prompt 2"]
    assert store.skipped_lines == 2
    assert store.count() == 2


def test_non_dict_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")
    assert HistoryStore(path).load() == []


def test_clear_removes_history(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    store.append(make_decision_dict())
    assert store.count() == 1
    store.clear()
    assert store.count() == 0
    store.clear()  # clearing an empty store is a no-op
    assert store.count() == 0


def test_append_creates_parent_directories(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "deep" / "nested" / "history.jsonl")
    store.append(make_decision_dict())
    assert store.path.is_file()


def test_partial_trailing_line_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(make_decision_dict(seq=1)) + "\n{partial", encoding="utf-8")
    store = HistoryStore(path)
    assert [r["prompt"] for r in store.recent()] == ["prompt 1"]
    assert store.skipped_lines == 1


def test_router_persists_decisions_to_store(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    router = PolicyRouter(
        scorer=HeuristicScorer(),
        registry=ModelRegistry.from_yaml("""\
            models:
              - name: only
                ollama_tag: only:latest
                complexity_ceiling: high
                approx_ram_gb: 1
            """),
        history_store=store,
    )
    router.route(Task(type=TaskType.DEBUG, prompt="fix the race condition"))

    stored = store.load()
    assert len(stored) == 1
    assert stored[0]["task_type"] == "debug"
    assert stored[0]["complexity"] == "high"


def test_router_survives_store_write_failure(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    router = PolicyRouter(
        scorer=HeuristicScorer(),
        registry=ModelRegistry.from_yaml("""\
            models:
              - name: only
                ollama_tag: only:latest
                complexity_ceiling: high
                approx_ram_gb: 1
            """),
        history_store=store,
    )
    # A path whose parent is a regular file makes mkdir/append fail with OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    store._path = blocker / "history.jsonl"

    decision = router.route(Task(type=TaskType.EXPLAIN, prompt="explain this"))
    assert decision.complexity.value == "low"


def test_default_path_is_env_overridable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VIBEFORGE_HISTORY", str(tmp_path / "custom.jsonl"))
    assert default_history_path() == tmp_path / "custom.jsonl"


def test_default_path_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIBEFORGE_HISTORY", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_history_path() == tmp_path / "vibeforge" / "history.jsonl"


def test_append_is_usable_across_store_instances(tmp_path: Path) -> None:
    first = HistoryStore(tmp_path / "history.jsonl")
    second = HistoryStore(tmp_path / "history.jsonl")
    first.append(make_decision_dict(seq=1))
    second.append(make_decision_dict(seq=2))
    assert len(first.load()) == 2

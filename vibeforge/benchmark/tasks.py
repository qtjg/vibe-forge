"""Fixed benchmark task set used to measure every configured model.

The set spans all six task types with a spread of difficulty per type, so the
benchmark results can be sliced by type or by complexity in the research
paper (``benchmark_results.csv`` is designed to load straight into pandas).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vibeforge.types import Task, TaskType

__all__ = ["BenchmarkTask", "TASKS", "all_tasks", "tasks_for"]


@dataclass(frozen=True)
class BenchmarkTask:
    """One benchmark prompt plus its routing metadata.

    Attributes:
        id: Stable identifier (used in the CSV ``task_id`` column).
        type: The task type; defines the scorer baseline.
        prompt: The prompt sent to the model.
        context: Optional context (file text, surrounding code).
        note: Human-readable description of what the task stresses.
    """

    id: str
    type: TaskType
    prompt: str
    context: str = ""
    note: str = ""


def _t(
    id_prefix: str, task_type: TaskType, prompt: str, context: str = "", note: str = ""
) -> BenchmarkTask:
    """Build a task with the prefixed stable id."""
    return BenchmarkTask(
        id=f"{id_prefix}", type=task_type, prompt=prompt, context=context, note=note
    )


A = TaskType.AUTOCOMPLETE
E = TaskType.EXPLAIN
R = TaskType.REFACTOR
G = TaskType.GENERATE
D = TaskType.DEBUG
V = TaskType.REVIEW

#: The full, fixed benchmark suite (stable order and ids across releases).
TASKS: tuple[BenchmarkTask, ...] = (
    # --- autocomplete -------------------------------------------------------
    _t(
        "autocomplete-01",
        A,
        "def add_customer(db, name, email):\n    # insert the customer row and return the id",
        note="trivial single-function completion",
    ),
    _t(
        "autocomplete-02",
        A,
        "def format_iso_date(dt):\n    # return the datetime as an ISO string",
        note="one-liner completion",
    ),
    _t(
        "autocomplete-03",
        A,
        "for line in open(path):\n    # strip, skip blanks, collect tokens",
        note="short loop body completion",
    ),
    _t(
        "autocomplete-04",
        A,
        "async def fetch_json(url, timeout=5):\n    # fetch and decode, catch network errors",
        note="async-aware completion (keyword bump)",
    ),
    _t(
        "autocomplete-05",
        A,
        "with ThreadPoolExecutor(max_workers=4) as pool:\n    # submit tasks and collect futures",
        note="concurrency-aware completion (keyword bump)",
    ),
    _t(
        "autocomplete-06",
        A,
        "def retry(fn, attempts=3):\n    # retry with backoff on transient failures",
        note="small general-purpose helper",
    ),
    # --- explain ------------------------------------------------------------
    _t(
        "explain-01",
        E,
        "explain what this regex matches: (?P<year>\\d{4})-(?P<month>\\d{2})",
        note="short pattern walkthrough",
    ),
    _t(
        "explain-02",
        E,
        "explain the difference between a list comprehension and a generator expression",
        note="language fundamentals",
    ),
    _t(
        "explain-03",
        E,
        "explain what this decorator does and when it would be useful:\n@retry_on_failure(times=3)",
        note="middleweight concept",
    ),
    _t(
        "explain-04",
        E,
        "explain why this code leaks memory over a long run:\n"
        "# repeatedly appends to a global list inside a background thread",
        note="harder: memory + threading concepts",
    ),
    _t(
        "explain-05",
        E,
        "explain how distributed consensus requires a quorum and how a network partition "
        "breaks availability (CAP)",
        note="advanced distributed-systems concepts",
    ),
    _t(
        "explain-06",
        E,
        "explain the GIL, its effect on concurrency, and when threads still help",
        note="concurrency deep-dive",
    ),
    # --- refactor -----------------------------------------------------------
    _t(
        "refactor-01",
        R,
        "rename this variable to something clearer:\n"
        "x = get_user()  # x is the current user object",
        note="mechanical rename",
    ),
    _t(
        "refactor-02",
        R,
        "extract the validation logic from this function into its own helper:\n"
        "def save(data):\n"
        "    if not data:\n        raise ValueError('empty')\n"
        "    if len(data) > 100:\n        raise ValueError('too long')\n"
        "    db.write(data)",
        note="small extraction",
    ),
    _t(
        "refactor-03",
        R,
        "split this 140-line function into smaller focused functions, preserving behavior",
        context="\n".join(f"    line_{i} = process_step_{i}(input_data)" for i in range(140)),
        note="long function split (length bump)",
    ),
    _t(
        "refactor-04",
        R,
        "convert this callback-based API to async/await without changing the public surface",
        context="def on_data(callback):\n    callback(load())\n",
        note="sync -> async conversion (keyword bump)",
    ),
    _t(
        "refactor-05",
        R,
        "decouple payment module from the DB layer using dependency injection",
        note="architecture-level refactor (keyword bump)",
    ),
    _t(
        "refactor-06",
        R,
        "this dedup loop is O(n^2), refactor it to be O(n) using a hash set",
        context="def dedup(items):\n    out = []\n    for a in items:\n        dup = False\n"
        "        for b in out:\n            if a == b:\n                dup = True\n"
        "        if not dup:\n            out.append(a)\n    return out\n",
        note="performance refactor (keyword bump)",
    ),
    # --- generate -----------------------------------------------------------
    _t(
        "generate-01",
        G,
        "write a hello-world python script",
        note="from-scratch trivia",
    ),
    _t(
        "generate-02",
        G,
        "write a python function that validates an email address format",
        note="small stdlib-only utility",
    ),
    _t(
        "generate-03",
        G,
        "write a small REST API with routes for creating and listing todos (in-memory store)",
        note="multi-route application",
    ),
    _t(
        "generate-04",
        G,
        "write a CLI tool that watches a directory and runs a linter on changed files",
        note="event-driven tool",
    ),
    _t(
        "generate-05",
        G,
        "write a concurrent web scraper using a thread pool that respects a rate limit",
        note="concurrency requirements (keyword bump)",
    ),
    _t(
        "generate-06",
        G,
        "design and sketch a microservice that rebalances shards across workers with "
        "distributed locking and safe failover",
        note="architecture-heavy generation (keyword bump)",
    ),
    # --- debug --------------------------------------------------------------
    _t(
        "debug-01",
        D,
        "why does this raise TypeError: unsupported operand type(s) for +:\n"
        "'int' and 'str'?\n"
        "total = price + discount  # discount is read from argv",
        note="single-cause type error",
    ),
    _t(
        "debug-02",
        D,
        "this loop misses the last element of the list, why?:\n"
        "for i in range(len(items) - 1):\n    print(items[i])",
        note="off-by-one",
    ),
    _t(
        "debug-03",
        D,
        "the process crashes after days of uptime; free memory slowly drops. Find the "
        "memory leak in the C extension keepref loops.",
        context="# pseudo-code of the C extension\n"
        "while (queue) {\n    Py_INCREF(item);\n    push(queue, item);\n}\n",
        note="memory-leak hunt (keyword bump)",
    ),
    _t(
        "debug-04",
        D,
        "two threads increment a shared counter and the final value is wrong.\n"
        "def worker():\n    for _ in range(1000):\n        counter[0] += 1",
        note="race condition (keyword bump)",
    ),
    _t(
        "debug-05",
        D,
        "the service deadlocks under load; two mutexes are acquired in different orders "
        "by different paths. Identify the cycle and fix it.",
        note="deadlock hunt (keyword bump)",
    ),
    _t(
        "debug-06",
        D,
        "a distributed job sometimes fails with stale results after leader re-election; "
        "trace the consistency issue.",
        note="distributed consistency (keyword bump)",
    ),
    # --- review -------------------------------------------------------------
    _t(
        "review-01",
        V,
        "review this small function for correctness and style:\n"
        "def avg(nums):\n    return sum(nums) / len(nums)",
        note="10-minute review",
    ),
    _t(
        "review-02",
        V,
        "review this 60-line PR; focus on error handling and edge cases",
        context="\n".join(f"    step_{i} = transform(input_data, param_{i})" for i in range(60)),
        note="mid-size review",
    ),
    _t(
        "review-03",
        V,
        "security review of this login endpoint:\n"
        "def login(request):\n    user = db.find(request.form['user'])\n    "
        "if user['password'] == request.form['pass']:\n        return session(user)\n"
        "    raise Unauthorized()",
        note="security review (keyword bump)",
    ),
    _t(
        "review-04",
        V,
        "review this singleton for thread-safety; is double-checked locking enough here?",
        context="class Config:\n    _instance = None\n    def __new__(cls):\n"
        "        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n"
        "        return cls._instance\n",
        note="thread-safety review (keyword bump)",
    ),
    _t(
        "review-05",
        V,
        "review the architecture: is batching over HTTP between these two microservices "
        "safe under partial failures, and how should retries be scoped?",
        note="architecture review (keyword bump)",
    ),
    _t(
        "review-06",
        V,
        "review this caching layer for correctness under memory pressure: eviction, "
        "expiry, and crash-consistency",
        note="memory/consistency review (keyword bump)",
    ),
)


def all_tasks() -> Sequence[BenchmarkTask]:
    """Return the full benchmark task set."""
    return TASKS


def tasks_for(task_type: TaskType) -> Sequence[BenchmarkTask]:
    """Return the benchmark tasks for a single task type."""
    return tuple(task for task in TASKS if task.type is task_type)


def as_routing_tasks(tasks: Sequence[BenchmarkTask]) -> Sequence[Task]:
    """Convert benchmark tasks into :class:`Task` objects for the router."""
    return tuple(Task(type=item.type, prompt=item.prompt, context=item.context) for item in tasks)

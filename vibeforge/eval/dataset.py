"""Dataset loading for scorer evaluation.

The evaluation set is a *held-out* collection of labeled prompts
(``vibeforge/data/eval-set.yaml``): prompts are deliberately distinct
from the benchmark suite, and ground-truth complexity is a human label,
not a heuristic output. Committing the labels to the repo is what makes
``vibeforge eval`` results reproducible by anyone who clones it.

Label format, per task:

- ``id``: unique stable id (used as the CSV row key).
- ``task_type``: the routing task type the prompt belongs to.
- ``prompt``: the prompt to score.
- ``ground_truth``: human-assigned complexity tier.
- ``rationale``: why that tier (auditability of the labels).

The whole file also carries ``evaluator`` and ``version`` so label
revisions are trackable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from vibeforge.types import Complexity, Task, TaskType

__all__ = ["EvalTask", "load_eval_set", "DEFAULT_EVAL_SET"]

#: Where the committed ground-truth set lives inside the package.
DEFAULT_EVAL_SET: Path = Path(__file__).resolve().parent.parent / "data" / "eval-set.yaml"

#: Expected document metadata. Anything else means the file was edited
#: in a way that weakens the "committed, reproducible" contract.
MANDATORY_META = ("evaluator", "version")


class EvalSetError(ValueError):
    """Raised when the evaluation set is missing or malformed."""


@dataclass(frozen=True)
class EvalTask:
    """One labeled evaluation item.

    Attributes:
        id: Stable unique id (CSV row key).
        task_type: Routing task type of the prompt.
        prompt: The prompt to score.
        ground_truth: Human-assigned complexity tier.
        rationale: Why the tier was assigned (for label auditing).
    """

    id: str
    task_type: str
    prompt: str
    ground_truth: Complexity
    rationale: str

    def as_routing_task(self) -> Task:
        """Convert to a router :class:`Task` for scoring."""
        return Task(type=TaskType(self.task_type), prompt=self.prompt)


def load_eval_set(path: str | Path | None = None) -> tuple[EvalTask, ...]:
    """Load and validate the committed evaluation set.

    Args:
        path: YAML file to load; defaults to the bundled eval set.

    Returns:
        The loaded tasks, in file order.

    Raises:
        EvalSetError: When the file is missing, malformed, or violates
            the reproducibility contract (missing metadata, duplicate or
            missing ids, unknown tiers, no tasks).
    """
    source = Path(path) if path is not None else DEFAULT_EVAL_SET
    try:
        with source.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise EvalSetError(f"cannot read eval set {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise EvalSetError(f"eval set {source} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise EvalSetError(f"eval set {source} must be a mapping at the top level")
    for key in MANDATORY_META:
        if not raw.get(key):
            raise EvalSetError(f"eval set {source} is missing required metadata '{key}'")

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise EvalSetError(f"eval set {source} 'tasks:' must be a non-empty list")

    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(tasks_raw):
        if not isinstance(entry, dict):
            raise EvalSetError(f"eval set {source}: tasks[{index}] is not a mapping")
        try:
            task_id = str(entry["id"]).strip()
        except KeyError as exc:
            raise EvalSetError(f"eval set {source}: tasks[{index}] is missing 'id'") from exc
        if not task_id:
            raise EvalSetError(f"eval set {source}: tasks[{index}] has an empty 'id'")
        if task_id in seen_ids:
            raise EvalSetError(f"eval set {source}: duplicate task id {task_id!r}")
        seen_ids.add(task_id)

        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise EvalSetError(f"eval set {source}: task {task_id!r} needs a non-empty 'prompt'")
        task_type = entry.get("task_type")
        if not isinstance(task_type, str) or not task_type.strip():
            raise EvalSetError(f"eval set {source}: task {task_id!r} needs a 'task_type'")
        try:
            tier = Complexity(entry["ground_truth"])
        except (KeyError, ValueError) as exc:
            valid = ", ".join(t.value for t in Complexity)
            raise EvalSetError(
                f"eval set {source}: task {task_id!r} has invalid 'ground_truth' (valid: {valid})"
            ) from exc
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str):
            raise EvalSetError(f"eval set {source}: task {task_id!r} 'rationale' must be a string")

        tasks.append(
            EvalTask(
                id=task_id,
                task_type=task_type,
                prompt=prompt,
                ground_truth=tier,
                rationale=rationale,
            )
        )

    tiers = {task.ground_truth for task in tasks}
    if len(tiers) != len(tuple(Complexity)):
        missing = sorted(set(t.value for t in Complexity) - {t.value for t in tiers})
        raise EvalSetError(f"eval set {source} must cover every tier; missing {missing}")
    return tuple(tasks)

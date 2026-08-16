"""Scorer evaluation harness (v0.3): rigorous, reproducible evidence.

Built for the research claim in vibe-forge: instead of "the embedding
scorer agrees with the heuristic", this package produces measurable
precision/recall/F1, a confusion matrix, and latency overhead for each
scorer against a *committed, human-labeled, held-out* test set
(``vibeforge/data/eval-set.yaml``).

Modules:

- ``dataset`` -- labeled task loading and validation.
- ``metrics`` -- classification math (precision/recall/F1, confusion).
- ``runner`` -- scores tasks with every scorer, times calls, writes CSV.
"""

from vibeforge.eval.dataset import EvalTask, load_eval_set
from vibeforge.eval.metrics import ConfusionMatrix, EvaluationMetrics, TierMetrics, evaluate
from vibeforge.eval.runner import CSV_COLUMNS, EvalRow, Evaluator, ScorerReport

__all__ = [
    "CSV_COLUMNS",
    "ConfusionMatrix",
    "EvalRow",
    "EvaluationMetrics",
    "EvalTask",
    "Evaluator",
    "ScorerReport",
    "TierMetrics",
    "evaluate",
    "load_eval_set",
]

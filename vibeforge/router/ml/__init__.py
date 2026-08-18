"""ML scorer training-data pipeline (Phase 3.1, v0.4.0 target).

This package builds a *labeled dataset* and a *trained classifier* from
benchmark CSV output and accumulated routing history -- and saves both to
disk. It deliberately does **not** wire into :class:`PolicyRouter`: the
scorer implementation that consumes these artifacts is the next phase
(3.2), and routing behavior must not change until that lands.

Dependency policy (same rule as the dashboard extra):

- Nothing in this package may import ML libraries at module load. The
  model-fit step imports scikit-learn lazily inside the function that
  needs it, so the base install (no ``[ml]`` extra) never sees sklearn;
  ``vibeforge train-scorer`` explains how to install the extra when the
  fit step is reached without it.
- All probes (CSV reading, history store, model factory) take injectable
  references so CI tests verify pipeline wiring without training anything
  real.
"""

from __future__ import annotations

from vibeforge.router.ml.features import feature_vector, heuristic_tier
from vibeforge.router.ml.pipeline import DatasetRow, TrainingDataset, build_dataset, train_and_save

__all__ = [
    "DatasetRow",
    "TrainingDataset",
    "build_dataset",
    "feature_vector",
    "heuristic_tier",
    "train_and_save",
]

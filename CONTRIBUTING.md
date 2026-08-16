# Contributing to vibe-forge

Thanks for considering a contribution. This project is small on purpose: the
routing pipeline has only four layers, and contributions should stay in that
shape — one file per concern, fully tested.

## Development setup

```bash
git clone https://github.com/qtjg/vibe-forge
cd vibe-forge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest            # run the suite (no Ollama needed — HTTP is mocked)
ruff check .      # lint
ruff format --check .   # format check
```

## Adding a new TaskType

1. Add the value to the `TaskType` enum in `vibeforge/types.py`
   (e.g. `DOCUMENT = "document"`).
2. Add a baseline rank in `BASELINE_RANKS` in
   `vibeforge/router/complexity.py` — think about whether it's cheap
   (`autocomplete`-like) or heavy (`debug`/`review`-like).
3. Add at least 5 benchmark prompts in `vibeforge/benchmark/tasks.py`
   (ids must be unique, one per difficulty level).
4. Add a `test_baseline_tier_per_task_type` parameter to
   `tests/test_complexity.py` asserting its baseline tier.

## Adding a new Scorer

Implement the `Scorer` protocol from `vibeforge/router/complexity.py`:

```python
class MyScorer:
    def score(self, task: Task) -> tuple[Complexity, str]:
        # return (complexity, human-readable reason)
        ...
```

Wire it into `PolicyRouter.__init__` **as an optional swap-in** — the default
`HeuristicScorer` path must never break:

```python
router = PolicyRouter(scorer=MyScorer(), registry=registry)
```

If your scorer makes ML or data calls, keep it lazy: the CLI should still
work offline with the default scorer.

## Adding benchmark tasks

In `vibeforge/benchmark/tasks.py`, append a `_t(...)` entry with:

- a **unique, stable** `id` (`<type>-NN`) — ids are referenced in
  `benchmark_results.csv` across releases;
- a `prompt`, optional `context`, and a `note` describing what the task
  stresses.

Keep the whole set between 30–50 tasks. `tests/test_benchmark.py` enforces
uniqueness and per-type minimums.

## Model registry rules

- The **cheapest covering model wins**. Never route to a bigger model than
  necessary.
- `models.yaml` is user-editable — never hardcode tiers in code; always read
  them through `ModelRegistry`.
- Adding a tier is just a YAML entry; keep `complexity_ceiling` honest.

## Dashboard API contract

`GET /api/decisions` and `GET /api/stats` are polled directly by the static
`index.html` with no build step. **Add new fields, don't rename existing
ones** — the page and the CLI both rely on the current shapes.

## Tests

- Every module gets a test module in `tests/`.
- `test_executor.py` mocks `requests.post` — never require a live Ollama.
- Run `pytest` locally before pushing; CI runs ruff + pytest + a package
  build check on Python 3.11–3.13.

## Commit messages

Descriptive, imperative, scope-prefixed:

```
feat: add heuristic complexity scorer
fix: clamp length bump above the largest threshold
test: assert registry fallback above max ceiling
docs: document dashboard API contract
```

## Before opening a PR

- [ ] Tests added for new/changed behavior
- [ ] `pytest` passes
- [ ] `ruff check .` clean
- [ ] `ruff format --check .` clean
- [ ] Docs updated (README/CONTRIBUTING) where behavior changed
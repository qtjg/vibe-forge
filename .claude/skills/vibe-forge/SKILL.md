---
name: vibe-forge-dev
description: Use when adding routing logic, task types, scorers, or benchmark cases to vibe-forge. Covers the router/scorer/registry contract, benchmark task format, and dashboard API shape.
---

# vibe-forge development guide

## Project shape

- Core pipeline: `Task` -> `HeuristicScorer` (complexity) -> `ModelRegistry`
  (`pick_for`, cheapest-covering-model) -> `PolicyRouter` (records decisions) ->
  optional `OllamaExecutor` (async execution).
- Types live in `vibeforge/types.py`; each router layer is one file under
  `vibeforge/router/`.
- Everything is `dataclass`/`enum` based, fully annotated, ruff+black clean
  (line length 100, google docstrings).
- Tests never need a live Ollama: `test_executor.py` mocks `requests.post`,
  others use fakes. Run `pytest`, `ruff check .`, `black --check .` before
  any commit.

## Adding a new TaskType

1. Add to `TaskType` enum in `vibeforge/types.py` (a `str` value).
2. Update `BASELINE_RANKS` in `vibeforge/router/complexity.py` if it has an
   obvious baseline (autocomplete is cheap, debug/review are heavy).
3. Add at least 5 benchmark prompts for it in `vibeforge/benchmark/tasks.py`
   (unique `<type>-NN` ids; `tests/test_benchmark.py` enforces this).
4. Add a parameter + expected tier to `test_baseline_tier_per_task_type` in
   `tests/test_complexity.py`.

## Adding a new Scorer

Implement the `Scorer` protocol (`score(task) -> tuple[Complexity, str]`) in
`vibeforge/router/complexity.py` or a new file under `vibeforge/router/`.
Wire it into `PolicyRouter.__init__` as an optional swap-in — never break
the default `HeuristicScorer` path. Keep the reason string human-readable;
the dashboard and CLI surface it verbatim.

## Model registry rules

- Cheapest-covering-model wins: `ModelRegistry.pick_for` returns the
  lowest-`approx_ram_gb` tier whose `complexity_ceiling.rank` covers the
  requested complexity, falling back to the most capable on no match.
- `models.yaml` is user-editable — don't hardcode tiers anywhere else.
- Adding a tier = one YAML entry; keep ceilings honest.

## Benchmark format

- Tasks: `BenchmarkTask(id, type, prompt, context, note)` in
  `vibeforge/benchmark/tasks.py`, 30-50 total.
- Runner writes `benchmark_results.csv` (columns: model_name, model_tag,
  task_id, task_type, latency_ms, eval_count, tokens_per_sec, output_chars,
  error) to feed the research paper — keep CSV columns stable across
  releases; failures become `error` rows, never crashes.

## Dashboard API contract

`/api/decisions` and `/api/stats` shapes must stay backward compatible — the
static `index.html` polls them directly with no build step. Add new fields,
don't rename existing ones. The CLI pushes decisions via
`POST /api/decisions` (see `vibeforge route --dashboard`).
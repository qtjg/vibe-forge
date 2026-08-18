## vibe-forge v0.1.0

Local-first policy router for coding assistants. Routes coding subtasks to
different local LLMs (via Ollama) based on task complexity — fully offline,
no cloud API calls.

### What's included
- **Router**: rule-based complexity scorer (task type + length + keyword
  signals) → cheapest-covering-model selection with graceful fallback
- **CLI**: `vibeforge route`, `vibeforge bench`, `vibeforge serve`
- **Dashboard**: FastAPI + vanilla-JS live view of routing decisions and
  per-model stats
- **Benchmark suite**: 36 tasks across 6 task types, results committed as
  `benchmark_results.csv`
- **80 tests**, ruff clean, CI on Python 3.11–3.13

### Verified against real Ollama
- Trivial task → tiny-fast tier (qwen2.5:0.5b, ~22.6 tok/s)
- Race-condition/debug task → heavy tier
- Unreachable model → clean error, no crash

### Known limitations
- Dashboard history is per-process (not persisted) — CLI pushes decisions to
  the dashboard via `--dashboard`; see ROADMAP for persistence plans
- Model tiers are heuristic, not learned — a swappable `Scorer` protocol is
  in place for a future ML-based scorer

### Links
- Benchmark data: `benchmark_results.csv` (tied to this tag)
- Architecture: `docs/architecture.md`
- Citation: `CITATION.cff`
## vibe-forge v0.3.0

From "works for me" to "defensible and shareable": every routing claim now
has a committed, reproducible result behind it, and every config failure
mode explains itself.

### What's included
- **Config validation (pydantic)** — malformed YAML, invalid tiers, a
  missing server, and an unpulled model each fail with a one-line error and
  an actionable hint, never a traceback
- **Deterministic eval harness** — 48 hand-labeled tasks (12 per tier)
  committed as `vibeforge/data/eval-set.yaml`, plus `vibeforge eval` with
  per-scorer CSVs and confusion matrices
- **`route --compare`** — run several tiers against the same prompt
  concurrently and see the outputs side by side (thread pool, per-model
  failure isolation)
- **Plugin-style task types** — `custom_task_types` in `models.yaml` adds
  new `--type` values with zero core edits
- **`vibeforge doctor`** — read-only health checks: config validity,
  Ollama reachability, pulled-model gaps (with the exact `ollama pull`
  command), and tier coverage; exit 1 on hard errors
- **PyPI packaging** — `vibe-forge` on PyPI with a lean base install and an
  optional `[dashboard]` extra; release workflow publishes from tags via
  trusted publishing

### Scorer evaluation results (committed, reproducible)
Ran `vibeforge eval` on the 48-task labeled set against the local Ollama
(`results/eval-results.csv`):

| Scorer | Accuracy | Macro-F1 | Latency (mean/median) | Fallbacks |
|---|---|---|---|---|
| heuristic | 17/48 (35.4%) | 0.366 | 0.1 ms / 0.1 ms | 0 |
| embedding | 27/48 (56.2%) | 0.565 | 228.9 ms / 94.0 ms | 0 |

The heuristic over-predicts `medium` (precision 0.250) — a known weakness
of keyword-based scoring; the embedding scorer is the better default when
Ollama is available.

### Verified against real Ollama
- All four config failure modes produce clean, hint-carrying errors
- `route --compare` ran 3 tiers concurrently in ~108 s (vs ~246 s serial)
  with a summary table
- Wheel smoke tests in CI: base install (`vibeforge --version`, offline
  route) and `[dashboard]` install (`vibeforge serve`) both pass
- 248 tests, ruff clean, CI green on Python 3.11–3.13

### Known limitations
- `vibeforge doctor` is read-only advice; auto-fix (`doctor --fix`) ships
  behind explicit confirmation in a later release
- Embedding scorer needs a live Ollama with `nomic-embed-text` pulled; it
  falls back to the heuristic silently-but-honestly (counted in eval output)
- Dashboard chain grouping (`chain_id`) and task chaining are v0.5.0 work

### Links
- Eval data: `vibeforge/data/eval-set.yaml`, results `results/eval-results.csv`
- Scorer comparison (benchmark suite): `results/scorer-comparison.csv`
- Architecture: `docs/architecture.md`
- Roadmap: `ROADMAP.md`

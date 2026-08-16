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
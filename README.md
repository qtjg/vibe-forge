# vibe-forge

Local-first policy router for coding assistants.
Routes coding subtasks to different local LLMs (via [Ollama](https://ollama.com)) based on task
complexity — so trivial tasks hit small fast models and hard tasks hit stronger ones.
**Fully offline. Zero cloud API costs. Zero code leaves your machine.**

[![CI](https://github.com/maya/vibe-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/maya/vibe-forge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Ollama required](https://img.shields.io/badge/ollama-required-orange)](https://ollama.com)

---

## Why

- **No cloud API costs** — every inference runs on your own hardware.
- **Works offline** — the router itself is pure rule-based logic; no telemetry, no phone home.
- **Hardware-aware** — pick models that fit your RAM; the router always picks the *cheapest*
  model that can do the job, so you never waste seconds on a 14B model for a one-liner.

```
             Task                 Scorer                 Registry              Executor
  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
  │ "fix this race    │   │ HeuristicScorer   │   │ ModelRegistry     │   │ OllamaExecutor    │
  │  condition"       │──▶│ baseline + length │──▶│ cheapest tier     │──▶│ POST /api/generate│──▶ llama3.1
  │ type=debug        │   │ + keyword signals │   │ covering tier     │   │ times latency     │    qwen2.5 ...
  └───────────────────┘   └───────────────────┘   └───────────────────┘   └───────────────────┘
                                explainable              user-editable
                                reason string            models.yaml
```

Each layer is a swappable unit: a new Scorer, a new model lineup, or a new executor can be
dropped in without touching the rest. Every decision carries a plain-English `reason`, so
routing is auditable end to end.

## Quickstart

```bash
# 1. Install Ollama and pull your models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b        # tiny-fast tier (trivial tasks)
ollama pull llama3.1:latest     # balanced tier (everyday tasks)
ollama pull qwen2.5-coder:14b   # heavy tier (hard tasks) — optional

# 2. Install vibe-forge
pip install -e ".[dev]"   # from the repo root

# 3. Route a task
vibeforge route "explain this regex: (?P<year>\d{4})" --type explain
vibeforge route "fix this race condition" --type debug --file mycode.py --execute
```

> `route` without `--execute` makes **no Ollama call at all** — pure local scoring, works
> with Ollama offline.

### Watch it live

```bash
vibeforge serve                   # terminal 1: dashboard on http://localhost:8420
vibeforge route "quick one-liner 3D-dash" --type autocomplete \
    --execute --dashboard http://localhost:8420    # terminal 2: appears live
```

![vibe-forge dashboard](docs/dashboard_view.svg)

## Configuration

Model tiers live in `models.yaml` (user-editable, or set `VIBEFORGE_MODELS` to point
elsewhere). The router never hardcodes tiers.

```yaml
models:
  - name: tiny-fast
    ollama_tag: qwen2.5:0.5b
    complexity_ceiling: trivial    # trivial < low < medium < high
    approx_ram_gb: 0.6             # "cheapest" = lowest RAM
    notes: "Blazing fast; ideal for autocomplete and one-liners"

  - name: balanced
    ollama_tag: llama3.1:latest
    complexity_ceiling: medium
    approx_ram_gb: 4.9
    notes: "The recommended default; handles most everyday tasks"

  - name: heavy
    ollama_tag: qwen2.5-coder:14b
    complexity_ceiling: high
    approx_ram_gb: 9.0
    notes: "Strongest tier; architecture, debugging, and reviews"
```

Tiers are assigned by `approx_ram_gb` — the smallest model whose ceiling covers the task
wins. If nothing covers the task, the most capable model is the fallback (never a crash).

## How scoring works

The `HeuristicScorer` (no ML dependencies, fully deterministic) produces a complexity in
`{trivial, low, medium, high}` from three signals:

| Signal | Rule |
|---|---|
| Task-type baseline | `autocomplete` → trivial; `explain`/`generate` → low; `refactor`/`debug`/`review` → medium |
| Prompt + context length | > 200 words: +1, > 800 words: +2 |
| High-signal keywords | `concurrency`, `race condition`, `memory leak`, `deadlock`, `architecture`, `security`, `async`, `distributed`, ... → +1 (≥ 3 hits: +2) |

The score is clamped to the 4-tier scale, and every decision records *why*:

```
baseline debug=2 (medium) +1 length (412 words) +1 keywords (race condition) => score 3 (high)
```

## Benchmarking

`vibeforge bench` runs the fixed 36-task suite against every model in your registry
and writes a pandas-friendly `benchmark_results.csv` — the data source for the
[research paper](CITATION.cff):

```bash
vibeforge bench                              # all models x all tasks
vibeforge bench --task-type debug            # scope to one task type
vibeforge bench --output results-2026.csv    # custom output
```

Each row: `model_name, model_tag, task_id, task_type, latency_ms, eval_count,
tokens_per_sec, output_chars, error`. Failures are recorded in the `error` column, never
aborting the run.

## Project layout

```
vibeforge/
├── types.py              # Task, Complexity, ModelTier, RoutingDecision, ExecutionResult
├── router/
│   ├── complexity.py     # Scorer protocol + HeuristicScorer (rule-based)
│   ├── registry.py       # ModelRegistry: models.yaml -> cheapest covering tier
│   ├── policy.py         # PolicyRouter: score -> pick -> history
│   └── executor.py       # OllamaExecutor: /api/generate, latency, graceful errors
├── benchmark/            # fixed 36-task suite + runner (CSV out)
├── dashboard/            # FastAPI app + vanilla-JS static page
└── cli/                  # vibeforge route | bench | serve
```

## Development

```bash
pip install -e ".[dev]"
pytest            # 80+ tests, no Ollama required (HTTP is mocked)
ruff check .
black --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add task types, scorers, and tasks.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

[MIT](LICENSE) © 2026 Maya. If you use the benchmark data in a paper, cite via
[CITATION.cff](CITATION.cff).
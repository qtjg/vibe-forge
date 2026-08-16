# Architecture

vibe-forge is a small, deliberately layered pipeline. Each layer is swappable
on its own, which keeps the code readable and the failure modes contained.

## Routing flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             PolicyRouter (policy.py)                        │
│                                                                              │
│  ┌──────────┐  complexity,reason  ┌──────────────┐  model   ┌─────────────┐  │
│  │  Scorer  │ ──────────────────▶ │     pick_for │ ───────▶ │  Registry   │  │
│  │ (protocol)│                    │  cheapest    │          │ (models.yaml)│ │
│  └──────────┘                    │  covering    │          └─────────────┘  │
│       ▲                          └──────────────┘                           │
│       │ scores                                                              │
│  ┌──────────┐    route(task) ──▶ RoutingDecision ──▶ history (in-memory)     │
│  │   Task   │                                          │                    │
│  └──────────┘                                          ▼                    │
│                                          dashboard /api/decisions          │
└────────────────────────────────────────────────────────────────────────────┘
                                        │ chosen model tag
                                        ▼
                              ┌─────────────────────┐
                              │   OllamaExecutor    │  POST /api/generate
                              │  (executor.py)      │  latency + eval_count
                              └─────────────────────┘  ──▶ ExecutionResult
```

## Why each layer is separate

**Scorer (complexity.py).** Purely functional: `Task -> (Complexity, reason)`.
The default `HeuristicScorer` is rule-based and dependency-free, so routing
works offline even when the model backend is down. Because it implements the
`Scorer` protocol, a future ML scorer can be swapped in without touching any
other layer — and the protocol forces it to return an explainable reason, not
just a number.

**Registry (registry.py).** The only module that knows what models exist. It
validates `models.yaml`, sorts tiers by RAM, and implements the
cheapest-covering-model rule with a most-capable fallback. Keeping this in one
place means adding a model is an edit to a YAML file, not a code change.

**PolicyRouter (policy.py).** The orchestration: score, pick, record. It owns
the in-memory decision history that the dashboard reads and the CLI pushes
from. It has no I/O by design — tests can drive it entirely with fakes.

**OllamaExecutor (executor.py).** All network I/O and timeouts live here.
Errors (server down, model missing, timeouts) are converted into
`ExecutionResult.error` rather than exceptions, so the CLI, benchmark runner,
and dashboard can all surface failures as data.

**Benchmark (benchmark/).** Turns the pipeline into a measurement instrument:
a fixed, stable task set run against every model, written to a flat CSV
(model-major, one row per run) designed to be read directly by pandas.

**Dashboard (dashboard/).** A thin FastAPI app over the decision history. The
frontend is a single static HTML file that polls two JSON endpoints — no build
step, no framework, nothing that can rot.

## Failure handling

- Ollama unreachable / model not pulled → `ExecutionResult.error`, never a
  crash; the CLI prints it, the benchmark records it in the `error` column.
- Invalid `models.yaml` → `ConfigError` with a precise message; the CLI exits
  with code 1 before any routing happens.
- No tier covers the task complexity → registry falls back to the most
  capable model, so routing always returns a decision.

## API contract

Stable, documented in `vibeforge/dashboard/app.py`:

- `GET /api/decisions?limit=N` — last N decisions, newest first
- `GET /api/stats` — per-model usage counts, average latency, error counts
- `POST /api/decisions` — pushed by `vibeforge route --dashboard URL`

The static `index.html` poll both GET endpoints directly; new fields may be
added but existing ones must not be renamed.
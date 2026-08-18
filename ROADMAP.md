# Roadmap

Tracked against the product plan. Items are handed to the coding agent as
self-contained units: implement -> test -> lint -> docs -> PR. The suite
grows monotonically and never requires a live Ollama (HTTP is mocked).

Status legend: `[x]` shipped · `[~]` in progress · `[ ]` not started

## v0.2.0 — History & Editors (shipped)

- [x] **1.1 History persistence & replay** — decisions persist to SQLite;
      the dashboard survives restarts and sessions are replayable.
- [x] 1.x Extra work pulled in: dashboard API (`/api/route`, `/api/execute`),
      VS Code extension (`vscode-extension/`), embedding scorer + honest
      comparison CSV (33.3% agreement).

## v0.3.0 — Defensible defaults (shipped, pending release)

- [x] **2.1 Deterministic eval harness** — committed labeled set
      (`vibeforge/data/eval-set.yaml`, 48 tasks, 12 per tier) + `vibeforge eval`
      with real results checked in (`results/eval-results.csv`): heuristic
      35.4% vs embedding 56.2% accuracy, macro-F1 0.366 vs 0.565.
- [x] **2.2 Config validation & doctor** — pydantic validation with
      field-level errors and CLI hints (W3-1) plus `vibeforge doctor`
      (read-only: config, Ollama reachability, pulled models, tier
      coverage, exit 1 on hard errors). Auto-fix stays out until 5.3.
- [x] Extras pulled in: `route --compare` concurrent multi-model runs,
      plugin-style custom task types (no core edits), 4 failure modes with
      actionable hints.
- [~] **5.1 PyPI packaging** — metadata/extras and the release workflow are
      in; actual publish to TestPyPI + PyPI pending (trusted publishing).
- [ ] **5.3 Follow-up** — `vibeforge init` (detect `ollama list`, scaffold
      `models.yaml`) can absorb `doctor --fix` later.

## v0.4.0 — ML scorer (not started)

- [ ] **3.1 Scorer extension point** — the `Scorer` protocol is in place;
      add an optional on-device model within it.
- [ ] **3.2 Training signal** — fine-tune on the benchmark/eval CSVs, keeping
      the rule-based scorer as fallback/ensemble.
- [ ] **3.3 Honest public numbers** — opt-in only if it beats the heuristic;
      publish the comparison the way the embedding scorer was handled.
- [ ] **5.5 Alternate backend: llama.cpp server** — executor implementing the
      same interface against llama.cpp's OpenAI-compatible endpoint,
      selected per-tier (default remains ollama).
- [ ] **5.6 Docs site** — mkdocs build: architecture, benchmark methodology
      (36 tasks, honest caveats), scorer comparison writeups.

## v0.5.0 — Task chaining (not started)

- [ ] **4.1 Chain model** — multi-step workflows (plan -> implement -> test);
      each step reuses the task-type registry (plan ~ explain/generate,
      implement ~ generate, test ~ debug/review), no parallel system.
- [ ] **4.2 Escalation** — `vibeforge/router/chain.py`: ChainRunner executes
      steps in order and escalates tiers mid-chain on signals (diff size,
      failed test step), with the reason string explaining the escalation.
      Steps still route through PolicyRouter; the runner only orchestrates.
- [ ] **4.3 CLI + dashboard** — `vibeforge chain "..." --steps plan,implement,test
      --execute`; chain runs render grouped in `/api/decisions` (chain id
      linking steps) and in the dashboard UI; `chain_id` nullable column in
      the history store; chains work dry (`--execute` omitted).

## Phase 5 — Adoption & polish (interleaved anywhere)

- [~] **5.1 PyPI** — done except the actual publish (see v0.3.0 row).
- [ ] **5.2 Docker trial** — `docker-compose.yml` bundling Ollama +
      vibe-forge + dashboard; documented `docker compose up` in the README.
- [ ] **5.3 First-run setup** — `vibeforge init` + `doctor` (see v0.3.0).
- [ ] **5.4 Shell completions** — `vibeforge --install-completion`
      (typer/click built-in mechanism).
- [ ] **5.5 llama.cpp backend** — see v0.4.0.
- [ ] **5.6 Docs site** — see v0.4.0.

## Cross-cutting

- Every phase ships `tests/test_<module>.py`; the suite grows monotonically
  (tracked in commit/PR descriptions, e.g. "229 -> 248 tests").
- CI matrix (Python 3.11-3.13) green throughout; optional extras never
  break the base install path (verified per release by wheel smoke tests:
  base install and `[dashboard]` install).
- Follow-ups discovered while building: none open besides 2.2's `doctor`.
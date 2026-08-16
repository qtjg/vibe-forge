# Roadmap

Near-term direction (roughly in order of importance). Pull requests for any
of these are very welcome.

1. **ML-based scorer** — a tiny on-device scoring model (distilled, quantized)
   fine-tuned on the benchmark CSV, replacing the hand-tuned keyword bumps,
   while keeping the explainable rule-based scorer as a fallback/ensemble.

2. **Multi-turn task chaining** — route multi-step coding workflows
   (plan -> implement -> test) as a sequence of subtasks, escalating the model
   tier only when a step's diff/stats warrant it.

3. **History persistence & replay** — write decisions to a local JSONL/SQLite
   store so the dashboard survives restarts and sessions are replayable,
   and tag benchmark results per release for the research paper.

4. **Editors & tooling integrations** — VS Code extension / LSP that sends
   autocomplete and quick-explain subtasks on demand, plus an interactive
   `vibeforge tui` for piped-in diffs.

5. **Hardware-aware tier migration** — read `nvidia-smi`/`ollama ps` to
   auto-tune `approx_ram_gb` caps so models fit a machine's VRAM without
   editing `models.yaml` by hand.

6. **Deterministic eval harness** — regenerate benchmark tables on PRs
   (CI-only, mocked executor) so routing regressions are caught before merge.

Release cadence: tag `v0.1.0` once core routing + benchmark are stable; bump
minor versions per feature, and keep `benchmark_results.csv` outputs versioned
per tag so the paper can cite exact numbers tied to a commit.
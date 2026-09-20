# Tests for vibe-forge

Run the full suite:

```bash
python3 -m pytest tests/ -v
```

Or with unittest:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## Layout

- `test_cli.py`, `test_cli_eval.py` — command-line surface
- `test_benchmark.py`, `test_compare*.py` — benchmark and model comparison
- `test_dashboard.py` — the local dashboard
- `test_embedding.py`, `test_complexity.py` — feature modules
- `test_availability.py`, `test_doctor.py` — diagnostics and health checks
- `test_hardware.py`, `test_ml_pipeline.py`, `test_ollama.py` — require
  Ollama running locally; skipped in CI

## Conventions

- Tests use `pytest` fixtures; unittest-style is also supported.
- Tests that need a running Ollama should `pytest.skip` when unavailable
  rather than fail.

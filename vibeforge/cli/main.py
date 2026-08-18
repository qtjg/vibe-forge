"""``vibeforge`` command-line interface.

Commands:
    ``route`` -- score a prompt and pick a model for it.
    ``bench`` -- run the benchmark suite against every configured model.
    ``serve`` -- start the live dashboard on ``localhost:8420``.

Every command reads model tiers from ``./models.yaml`` (or
``$VIBEFORGE_MODELS``), falling back to the copy bundled with the package.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
import typer

from vibeforge import __version__
from vibeforge.benchmark.runner import BenchmarkInterrupted, BenchmarkRunner
from vibeforge.benchmark.tasks import all_tasks, tasks_for
from vibeforge.compare_models import ModelRun
from vibeforge.router.complexity import HeuristicScorer, Scorer
from vibeforge.router.executor import DEFAULT_OLLAMA_URL, OllamaExecutor
from vibeforge.router.policy import PolicyRouter
from vibeforge.router.registry import ConfigError, ModelRegistry, find_models_file
from vibeforge.types import RoutingDecision, Task

app = typer.Typer(
    name="vibeforge",
    help="Local-first policy router for coding assistants (Ollama-powered).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is passed."""
    if value:
        typer.echo(f"vibe-forge {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Print version."
    ),
) -> None:
    """CLI entry point; options here must come before the subcommand."""


def _load_registry() -> ModelRegistry:
    """Load the model registry from the resolved config file.

    Exits with an error message when the config cannot be read, rather than
    raising a traceback into the user's face.
    """
    try:
        registry = ModelRegistry.load_default()
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return registry


@app.command()
def route(
    prompt: str = typer.Argument(..., help="The coding prompt to route."),
    task_type: str = typer.Option(
        "generate", "--type", help="Kind of subtask (default: generate)."
    ),
    file_path: Path | None = typer.Option(
        None, "--file", help="File whose contents act as task context."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the decision as JSON."),
    execute: bool = typer.Option(
        False, "--execute", help="Also run the prompt against the chosen model."
    ),
    compare: str | None = typer.Option(
        None,
        "--compare",
        help="Comma-separated tier names to run the prompt against concurrently "
        "(e.g. tiny-fast,balanced,heavy). Implies execution; not compatible with --execute.",
    ),
    dashboard: str | None = typer.Option(
        None,
        "--dashboard",
        help="Push the decision to a running dashboard, e.g. http://localhost:8420.",
    ),
    host: str = typer.Option(
        DEFAULT_OLLAMA_URL,
        "--host",
        show_default=True,
        help="Base URL of the local Ollama server (for --execute).",
    ),
    scorer: str = typer.Option(
        "heuristic",
        "--scorer",
        help="Routing scorer: 'heuristic' (default) or 'embedding' (experimental).",
    ),
    embedding_model: str = typer.Option(
        "nomic-embed-text",
        "--embedding-model",
        help="Ollama embedding model used by --scorer embedding.",
    ),
) -> None:
    """Score ``PROMPT`` and pick the cheapest model that can handle it.

    The decision itself is pure logic: no Ollama call happens unless
    ``--execute`` is passed, so routing works even when Ollama is offline.
    ``--scorer embedding`` needs a live Ollama with the embedding model
    pulled; it falls back to the heuristic scorer when embeddings are
    unavailable.
    """
    registry = _load_registry()
    if task_type not in registry.task_types:
        typer.echo(
            f"error: unknown task type '{task_type}' — "
            f"known: {', '.join(registry.task_types.names)}",
            err=True,
        )
        raise typer.Exit(code=1)

    context = ""
    if file_path is not None:
        if not file_path.is_file():
            typer.echo(f"error: file not found: {file_path}", err=True)
            raise typer.Exit(code=1)
        context = file_path.read_text(encoding="utf-8", errors="replace")

    task = Task(
        type=task_type,
        prompt=prompt,
        context=context,
        file_path=str(file_path) if file_path is not None else None,
    )
    if scorer == "embedding":
        from vibeforge.router.embedding import EmbeddingScorer

        router = PolicyRouter(
            scorer=EmbeddingScorer(client_host=host, model=embedding_model),
            registry=registry,
        )
    elif scorer == "heuristic":
        router = PolicyRouter(
            scorer=HeuristicScorer(task_types=registry.task_types), registry=registry
        )
    else:
        typer.echo(f"error: unknown scorer '{scorer}' (use 'heuristic' or 'embedding')", err=True)
        raise typer.Exit(code=1)
    decision = router.route(task)

    if compare and execute:
        typer.echo("error: --compare already runs the prompt; drop --execute", err=True)
        raise typer.Exit(code=1)

    if compare:
        _run_comparison(compare, prompt, decision, registry, host, as_json)
        return

    result = None
    if execute:
        executor = OllamaExecutor(base_url=host)
        result = executor.execute(
            decision.model.ollama_tag,
            prompt,
            options={"temperature": 0.0, "num_predict": decision.token_budget},
        )
        if result.error:
            typer.echo(f"error: {result.error}", err=True)
            if result.status_code == 404:
                typer.echo(
                    f"hint: model '{result.model}' is not pulled — run: ollama pull {result.model}",
                    err=True,
                )
            elif result.error_kind == "connection":
                typer.echo(
                    f"hint: is the Ollama server running? Start it with `ollama serve` "
                    f"and check it answers at {host}",
                    err=True,
                )

    payload = decision.as_dict()
    if result is not None:
        payload["latency_ms"] = result.latency_ms
        payload["eval_count"] = result.eval_count
        payload["execution_error"] = result.error

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"task:          {task.type}: {task.prompt}")
        typer.echo(f"complexity:    {decision.complexity.value} (score {decision.score}/3)")
        typer.echo(f"chosen model:  {decision.model.name} ({decision.model.ollama_tag})")
        typer.echo(f"reason:        {decision.reason}")
        if decision.confidence is not None:
            typer.echo(f"confidence:    {decision.confidence:.2f}")
        if result is not None:
            if result.error:
                typer.echo(f"execution:     FAILED ({result.error})")
            else:
                typer.echo(
                    f"execution:     {result.latency_ms:.0f}ms, {result.eval_count} tokens "
                    f"({result.tokens_per_sec:.1f} tok/s)"
                )
                typer.echo(result.output)
        typer.echo(f"config:        {find_models_file()}")

    if dashboard:
        url = f"{dashboard.rstrip('/')}/api/decisions"
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            typer.echo(f"pushed decision to dashboard: {dashboard}")
        except requests.RequestException as exc:
            typer.echo(f"warning: could not reach dashboard at {dashboard}: {exc}", err=True)


def _run_comparison(
    compare: str,
    prompt: str,
    decision: RoutingDecision,
    registry: ModelRegistry,
    host: str,
    as_json: bool,
) -> None:
    """Run the prompt against several tiers concurrently and render results."""
    from vibeforge.compare_models import run_models_concurrently

    names = [name.strip() for name in compare.split(",") if name.strip()]
    known = {model.name for model in registry.models}
    unknown = [name for name in names if name not in known]
    if unknown:
        typer.echo(
            f"error: unknown tier name(s) {', '.join(unknown)} (known: {', '.join(sorted(known))})",
            err=True,
        )
        raise typer.Exit(code=1)

    models = [(model.name, model.ollama_tag) for model in registry.models if model.name in names]
    options: dict[str, object] = {"temperature": 0.0, "num_predict": decision.token_budget}

    typer.echo(f"comparing {len(models)} models concurrently on: {prompt!r}")
    typer.echo("")

    def make_executor() -> object:
        return OllamaExecutor(base_url=host)

    runs = run_models_concurrently(
        models,
        prompt=prompt,
        executor_factory=make_executor,
        options=options,
    )

    if as_json:
        payload = decision.as_dict()
        payload["comparisons"] = [
            {
                "name": run.name,
                "ollama_tag": run.ollama_tag,
                "status": "ok" if run.ok else "error",
                "latency_ms": run.latency_ms,
                "eval_count": run.result.eval_count,
                "output": run.result.output,
                "error": run.result.error,
            }
            for run in runs
        ]
        typer.echo(json.dumps(payload, indent=2))
        return

    for run in runs:
        _render_model_run(run)
        typer.echo("")

    typer.echo("summary (concurrent run, lower latency is not the point -- outputs are):")
    typer.echo(f"  {'model':<16}{'tag':<28}{'latency':>10}{'tokens':>8}  status")
    for run in runs:
        latency = f"{run.latency_ms:.0f}ms" if run.latency_ms is not None else "--"
        tokens = str(run.result.eval_count) if run.result.eval_count is not None else "--"
        status = "ok" if run.ok else "FAILED"
        typer.echo(f"  {run.name:<16}{run.ollama_tag:<28}{latency:>10}{tokens:>8}  {status}")


def _render_model_run(run: ModelRun) -> None:
    """Print one model's comparison block (output or the error)."""
    typer.echo(f"{run.name} ({run.ollama_tag}):")
    if run.ok:
        latency = f"{run.latency_ms:.0f}ms" if run.latency_ms is not None else "?"
        tokens = run.result.eval_count if run.result.eval_count is not None else "?"
        rate = run.result.tokens_per_sec
        rate_text = f" ({rate:.1f} tok/s)" if rate is not None else ""
        typer.echo(f"  {latency}, {tokens} tokens{rate_text}:")
        typer.echo(f"  {run.result.output}")
    else:
        typer.echo(f"  FAILED: {run.result.error}")


@app.command("bench")
def run(
    task_type: str | None = typer.Option(
        None,
        "--task-type",
        help="Benchmark only this task type (default: all).",
    ),
    output: Path = typer.Option(
        Path("benchmark_results.csv"),
        "--output",
        help="Where to write the CSV results.",
    ),
    host: str = typer.Option(
        DEFAULT_OLLAMA_URL,
        "--host",
        show_default=True,
        help="Base URL of the local Ollama server.",
    ),
) -> None:
    """Run every registered model against the fixed benchmark task set.

    Writes one row per (model, task) run to ``benchmark_results.csv`` and
    prints a summary table. Failing runs are recorded in the CSV with an
    ``error`` column rather than aborting the benchmark.
    """
    registry = _load_registry()
    if task_type is not None and task_type not in registry.task_types:
        typer.echo(
            f"error: unknown task type '{task_type}' — "
            f"known: {', '.join(registry.task_types.names)}",
            err=True,
        )
        raise typer.Exit(code=1)
    tasks = tasks_for(task_type) if task_type is not None else all_tasks()

    typer.echo(f"config:   {find_models_file()}")
    typer.echo(f"models:   {', '.join(m.name for m in registry.models)}")
    typer.echo(f"tasks:    {len(tasks)} ({task_type if task_type else 'all types'})")
    typer.echo("")

    runner = BenchmarkRunner(registry=registry, executor=OllamaExecutor(base_url=host), tasks=tasks)
    try:
        rows = runner.run(output_path=output)
    except BenchmarkInterrupted as exc:
        typer.echo(f"\ninterrupted after {exc.completed}/{exc.total} runs", err=True)
        typer.echo(f"partial results saved to {exc.csv_path}", err=True)
        raise typer.Exit(code=130) from exc
    path = runner.write_csv(rows, output)
    typer.echo(runner.summarize(rows))
    typer.echo(f"\nresults written to {path}")


@app.command("eval")
def eval_command(
    scorer_names: str = typer.Option(
        "heuristic,embedding",
        "--scorer",
        help="Comma-separated scorers to evaluate: 'heuristic', 'embedding'.",
    ),
    output: Path = typer.Option(
        Path("results/eval-results.csv"),
        "--output",
        help="Where to write the per-task CSV.",
    ),
    host: str = typer.Option(
        DEFAULT_OLLAMA_URL,
        "--host",
        show_default=True,
        help="Base URL of the local Ollama server (for the embedding scorer).",
    ),
    embedding_model: str = typer.Option(
        "nomic-embed-text",
        "--embedding-model",
        help="Ollama embedding model used by the embedding scorer.",
    ),
) -> None:
    """Evaluate scorers against the committed human-labeled test set.

    Scores every prompt in ``vibeforge/data/eval-set.yaml`` with each
    given scorer, then reports accuracy, per-tier precision/recall/F1,
    the confusion matrix, and latency overhead. The CSV written to
    ``--output`` is the reproducible evidence: anyone who clones the
    repo can rerun this and diff the file.
    """
    from vibeforge.eval.dataset import EvalSetError, load_eval_set
    from vibeforge.eval.runner import Evaluator
    from vibeforge.router.complexity import HeuristicScorer
    from vibeforge.router.embedding import EmbeddingScorer

    names = [name.strip() for name in scorer_names.split(",") if name.strip()]
    if not names:
        typer.echo("error: --scorer needs at least one scorer name", err=True)
        raise typer.Exit(code=1)

    known = {"heuristic", "embedding"}
    unknown = [name for name in names if name not in known]
    if unknown:
        typer.echo(
            f"error: unknown scorer(s) {', '.join(unknown)} (known: {', '.join(sorted(known))})",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        tasks = load_eval_set()
    except EvalSetError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    scorers: dict[str, Scorer] = {}
    task_types = _load_registry().task_types
    if "heuristic" in names:
        scorers["heuristic"] = HeuristicScorer(task_types=task_types)
    if "embedding" in names:
        scorers["embedding"] = EmbeddingScorer(client_host=host, model=embedding_model)

    typer.echo(f"eval set:  {len(tasks)} labeled tasks")
    typer.echo(f"scorers:   {', '.join(scorers)}")
    typer.echo("")

    evaluator = Evaluator(scorers)
    reports = evaluator.run(tasks)
    path = evaluator.write_csv(reports, output)
    typer.echo(evaluator.summarize(reports))
    typer.echo(f"\nresults written to {path}")


@app.command("compare-scorers")
def compare_scorers(
    output: Path = typer.Option(
        Path("scorer-comparison.csv"),
        "--output",
        help="Where to write the comparison CSV.",
    ),
    host: str = typer.Option(
        DEFAULT_OLLAMA_URL,
        "--host",
        show_default=True,
        help="Base URL of the local Ollama server (for embeddings).",
    ),
    embedding_model: str = typer.Option(
        "nomic-embed-text",
        "--embedding-model",
        help="Ollama embedding model to compare against.",
    ),
) -> None:
    """Compare heuristic vs. embedding scoring across the benchmark suite.

    Scores every benchmark task with both strategies and writes one row per
    task to ``scorer-comparison.csv`` with tier choices and per-call
    latency. Requires a live Ollama with the embedding model pulled.
    """
    from vibeforge.benchmark.compare import ScorerComparison
    from vibeforge.router.complexity import HeuristicScorer
    from vibeforge.router.embedding import EmbeddingScorer

    tasks = all_tasks()
    typer.echo(f"tasks:    {len(tasks)}")
    typer.echo(f"embedding model: {embedding_model} (host {host})")
    typer.echo("")

    comparison = ScorerComparison(
        heuristic=HeuristicScorer(task_types=_load_registry().task_types),
        embedding=EmbeddingScorer(client_host=host, model=embedding_model),
    )
    rows = comparison.run(tasks)
    path = comparison.write_csv(rows, output)
    typer.echo(comparison.summarize(rows))
    typer.echo(f"\nresults written to {path}")


@app.command("doctor")
def doctor(
    host: str = typer.Option(
        DEFAULT_OLLAMA_URL,
        "--host",
        show_default=True,
        help="Base URL of the local Ollama server.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print findings as JSON."),
) -> None:
    """Run read-only health checks on config + Ollama.

    Checks that the models config is valid, the Ollama server is reachable,
    every configured tier is pulled, and all complexity tiers are covered.
    Nothing is modified: fixing is a separate, confirmed step. Exit code is
    1 when any check is a hard error, 0 otherwise.
    """
    from vibeforge.doctor import ERROR, Doctor

    findings = Doctor(host=host).run()
    if as_json:
        typer.echo(
            json.dumps(
                [{"level": f.level, "check": f.check, "message": f.message} for f in findings],
                indent=2,
            )
        )
    else:
        for finding in findings:
            icon = {"ok": "[ok]  ", "warn": "[warn]", "error": "[err] "}[finding.level]
            typer.echo(f"{icon} {finding.check:<7} {finding.message}")
        counts = {"ok": 0, "warn": 0, "error": 0}
        for finding in findings:
            counts[finding.level] += 1
        typer.echo(
            f"\ndoctor: {counts['ok']} ok, {counts['warn']} warning(s), {counts['error']} error(s)",
            err=True,
        )
    if any(finding.level == ERROR for finding in findings):
        raise typer.Exit(code=1)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind."),
    port: int = typer.Option(8420, "--port", help="Port to listen on."),
    db_path: Path | None = typer.Option(
        None,
        "--db-path",
        help="SQLite history file (default: ~/.vibeforge/history.db).",
    ),
) -> None:
    """Start the live dashboard at ``http://localhost:8420``.

    Decisions persist to SQLite and survive restarts. Push decisions to it
    from the CLI with ``vibeforge route ... --dashboard http://localhost:8420``.
    """
    try:
        import uvicorn

        from vibeforge.dashboard.app import create_app
        from vibeforge.dashboard.store import default_db_path
    except ImportError as exc:
        typer.echo(
            "error: the dashboard needs 'vibe-forge[dashboard]' — "
            "reinstall with: pip install 'vibe-forge[dashboard]'",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    resolved = db_path if db_path is not None else default_db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"dashboard serving at http://{host}:{port}  (Ctrl+C to stop)")
    typer.echo(f"history db:         {resolved}")
    uvicorn.run(create_app(db_path=resolved), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()

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
from vibeforge.router.complexity import HeuristicScorer
from vibeforge.router.executor import DEFAULT_OLLAMA_URL, OllamaExecutor
from vibeforge.router.policy import PolicyRouter
from vibeforge.router.registry import ConfigError, ModelRegistry, find_models_file
from vibeforge.types import Task, TaskType

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
    task_type: TaskType = typer.Option(
        TaskType.GENERATE, "--type", help="Kind of subtask (default: generate)."
    ),
    file_path: Path | None = typer.Option(
        None, "--file", help="File whose contents act as task context."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the decision as JSON."),
    execute: bool = typer.Option(
        False, "--execute", help="Also run the prompt against the chosen model."
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
        router = PolicyRouter(scorer=HeuristicScorer(), registry=registry)
    else:
        typer.echo(f"error: unknown scorer '{scorer}' (use 'heuristic' or 'embedding')", err=True)
        raise typer.Exit(code=1)
    decision = router.route(task)

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
                    f"hint: model '{result.model}' is not pulled — run: "
                    f"ollama pull {result.model}",
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
        typer.echo(f"task:          {task.type.value}: {task.prompt}")
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


@app.command("bench")
def run(
    task_type: TaskType | None = typer.Option(
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
    tasks = tasks_for(task_type) if task_type is not None else all_tasks()

    typer.echo(f"config:   {find_models_file()}")
    typer.echo(f"models:   {', '.join(m.name for m in registry.models)}")
    typer.echo(f"tasks:    {len(tasks)} ({task_type.value if task_type else 'all types'})")
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
        heuristic=HeuristicScorer(),
        embedding=EmbeddingScorer(client_host=host, model=embedding_model),
    )
    rows = comparison.run(tasks)
    path = comparison.write_csv(rows, output)
    typer.echo(comparison.summarize(rows))
    typer.echo(f"\nresults written to {path}")


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
    import uvicorn

    from vibeforge.dashboard.app import create_app
    from vibeforge.dashboard.store import default_db_path

    resolved = db_path if db_path is not None else default_db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"dashboard serving at http://{host}:{port}  (Ctrl+C to stop)")
    typer.echo(f"history db:         {resolved}")
    uvicorn.run(create_app(db_path=resolved), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()

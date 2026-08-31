from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from cc_agent.config import AgentConfig
from cc_agent.data_builder import (
    build_humaneval,
    build_mbpp,
    build_swebench_lite,
    local_tasks_to_sft,
    swebench_to_sft,
    traces_to_sft,
)
from cc_agent.graph import run_agent
from cc_agent.repo_indexer import RepoIndexer
from cc_agent.retrieval_eval import evaluate_retrieval_dataset
from cc_agent.task_runner import run_task_manifest
from cc_agent.trace_stats import compute_trace_stats

app = typer.Typer(help="Claude Code style repository agent built with LangGraph.")
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@app.command()
def index(
    repo: Path = typer.Option(..., "--repo", "-r", help="Target repository path."),
    query: str | None = typer.Option(None, "--query", "-q", help="Optional task query for hybrid retrieval."),
) -> None:
    """Preview repository context without calling an LLM."""
    snapshot = RepoIndexer(repo).snapshot(query=query)
    console.print(Panel(snapshot.tree, title="Repository Tree"))
    console.print(Panel(snapshot.symbols, title="Symbol Index"))
    console.print(Panel(snapshot.retrieval_notes, title="Retrieved Context"))
    console.print(Panel(snapshot.project_rules, title="Project Rules"))


@app.command()
def retrieve(
    repo: Path = typer.Option(..., "--repo", "-r", help="Target repository path."),
    query: str = typer.Option(..., "--query", "-q", help="Search query."),
    top_k: int = typer.Option(5, "--top-k", help="Number of retrieved code chunks."),
    path: str | None = typer.Option(None, "--path", help="Optional repository path filter."),
    language: str | None = typer.Option(None, "--language", help="Optional language filter."),
    symbol: str | None = typer.Option(None, "--symbol", help="Optional symbol-name filter."),
) -> None:
    """Run hybrid BM25 and dense retrieval over the repository index."""
    result = RepoIndexer(repo).retrieve(
        query=query,
        top_k=top_k,
        path_filter=path,
        language_filter=language,
        symbol_filter=symbol,
    )
    console.print(Panel(result, title="Retrieved Context"))


@app.command()
def run(
    repo: Path = typer.Option(..., "--repo", "-r", help="Target repository path."),
    task: str = typer.Option(..., "--task", "-t", help="Natural language coding task."),
    test_command: str | None = typer.Option(None, "--test-command", help="Allowlisted test command, e.g. pytest -q."),
) -> None:
    """Run the full LangGraph coding agent."""
    config = AgentConfig.from_env(PROJECT_ROOT)
    final_state = run_agent(repo, task, test_command, config=config, project_root=PROJECT_ROOT)
    console.print(Panel(final_state.get("plan", "No plan generated."), title="Plan"))
    console.print(Panel(final_state.get("review", "No review generated."), title="Review Subagent"))
    console.print(Panel(final_state.get("final_answer", "No final summary."), title="Final Summary"))
    console.print(f"[bold]Trace:[/bold] {final_state.get('trace_path')}")


@app.command()
def stats(
    path: Path = typer.Option(PROJECT_ROOT / "traces", "--path", "-p", help="Trace file or trace directory."),
) -> None:
    """Summarize JSONL traces into simple benchmark-style metrics."""
    trace_stats = compute_trace_stats(path)
    console.print(Panel(trace_stats.render(), title="Trace Stats"))


@app.command("eval-retrieval")
def eval_retrieval(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Retrieval evaluation JSONL."),
    ks: str = typer.Option("1,3,5", "--ks", help="Comma-separated K values."),
) -> None:
    """Evaluate repository retrieval with Recall@K, HitRate@K, and MRR."""
    parsed_ks = tuple(int(value.strip()) for value in ks.split(",") if value.strip())
    metrics = evaluate_retrieval_dataset(dataset, ks=parsed_ks)
    console.print(Panel(metrics.render(), title="Retrieval Evaluation"))


@app.command("build-mbpp")
def build_mbpp_command(
    output_dir: Path = typer.Option(PROJECT_ROOT / "data", "--output-dir", "-o", help="Output data directory."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum records to build."),
    split: str = typer.Option("test", "--split", help="Dataset split."),
) -> None:
    """Download MBPP and build runnable mini repos plus SFT JSONL."""
    result = build_mbpp(output_dir=output_dir, limit=limit, split=split)
    console.print(Panel(result.render(), title="MBPP Build Result"))


@app.command("build-humaneval")
def build_humaneval_command(
    output_dir: Path = typer.Option(PROJECT_ROOT / "data", "--output-dir", "-o", help="Output data directory."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum records to build."),
    split: str = typer.Option("test", "--split", help="Dataset split."),
) -> None:
    """Download HumanEval and build runnable mini repos plus SFT JSONL."""
    result = build_humaneval(output_dir=output_dir, limit=limit, split=split)
    console.print(Panel(result.render(), title="HumanEval Build Result"))


@app.command("build-swebench-lite")
def build_swebench_lite_command(
    output_dir: Path = typer.Option(PROJECT_ROOT / "data", "--output-dir", "-o", help="Output data directory."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum records to build."),
    split: str = typer.Option("test", "--split", help="Dataset split."),
) -> None:
    """Download SWE-bench Lite metadata and build task manifest."""
    result = build_swebench_lite(output_dir=output_dir, limit=limit, split=split)
    console.print(Panel(result.render(), title="SWE-bench Lite Build Result"))


@app.command("traces-to-sft")
def traces_to_sft_command(
    trace_path: Path = typer.Option(PROJECT_ROOT / "traces", "--trace-path", "-t", help="Trace file or directory."),
    output_path: Path = typer.Option(PROJECT_ROOT / "data" / "sft" / "agent_traces_sft.jsonl", "--output-path", "-o"),
) -> None:
    """Convert successful tool calls in JSONL traces into SFT samples."""
    count = traces_to_sft(trace_path=trace_path, output_path=output_path)
    console.print(Panel(f"Wrote {count} SFT samples to {output_path}", title="Trace SFT Result"))


@app.command("swebench-to-sft")
def swebench_to_sft_command(
    input_path: Path = typer.Option(PROJECT_ROOT / "data" / "tasks" / "swebench_lite_tasks.jsonl", "--input", "-i"),
    output_path: Path = typer.Option(PROJECT_ROOT / "data" / "sft" / "swebench_lite_patch_sft.jsonl", "--output", "-o"),
    mode: str = typer.Option("patch", "--mode", help="patch or plan"),
) -> None:
    """Convert SWE-bench Lite task manifest into patch/plan SFT data."""
    count = swebench_to_sft(input_path=input_path, output_path=output_path, mode=mode)
    console.print(Panel(f"Wrote {count} SWE-bench SFT samples to {output_path}", title="SWE-bench SFT Result"))


@app.command("local-tasks-to-sft")
def local_tasks_to_sft_command(
    input_path: Path = typer.Option(..., "--input", "-i", help="Local task manifest JSONL."),
    output_path: Path = typer.Option(PROJECT_ROOT / "data" / "sft" / "local_tasks_strategy_sft.jsonl", "--output", "-o"),
) -> None:
    """Convert local runnable tasks into high-level strategy SFT samples."""
    count = local_tasks_to_sft(input_path=input_path, output_path=output_path)
    console.print(Panel(f"Wrote {count} local-task SFT samples to {output_path}", title="Local Task SFT Result"))


@app.command("run-tasks")
def run_tasks_command(
    manifest_path: Path = typer.Option(..., "--manifest", "-m", help="Task manifest JSONL."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum runnable local tasks."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only validate runnable tasks; do not call the Agent."),
    output_path: Path | None = typer.Option(None, "--output", "-o", help="Run result JSONL path."),
) -> None:
    """Run local task manifests through the Agent, or dry-run validate them."""
    config = AgentConfig.from_env(PROJECT_ROOT)
    result = run_task_manifest(
        manifest_path=manifest_path,
        config=config,
        project_root=PROJECT_ROOT,
        limit=limit,
        dry_run=dry_run,
        output_path=output_path,
    )
    console.print(Panel(result.render(), title="Task Run Result"))


if __name__ == "__main__":
    app()

"""Production CLI for indexing, searching, and chatting with a repository."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.text import Text

from coding_agent.agent.executor import AgentExecutor
from coding_agent.agent.prompt import PromptBuilder
from coding_agent.agent.team import (
    RESEARCH_PROMPT,
    SYNTHESIS_PROMPT,
    MultiAgentCoordinator,
)
from coding_agent.agent.workflow import LangGraphAgent
from coding_agent.embedding.models import EmbeddingConfig
from coding_agent.embedding.sentence_transformers import SentenceTransformerEmbedder
from coding_agent.indexing.manifest import IndexManifestStore
from coding_agent.indexing.service import RepositoryIndexer
from coding_agent.llm.client import GroqClient
from coding_agent.memory.store import ConversationManager, ConversationStore
from coding_agent.navigation import NavigationIndex
from coding_agent.observability import get_metrics_collector
from coding_agent.planner import SimplePlanner
from coding_agent.scanner.ignore import IGNORE_DIRECTORIES
from coding_agent.scanner.scanner import RepositoryScanner
from coding_agent.search.hybrid import HybridSearcher
from coding_agent.search.keyword import BM25Searcher
from coding_agent.search.semantic import SemanticSearcher
from coding_agent.tools.models import ToolParameter
from coding_agent.tools.registry import ToolRegistry
from coding_agent.vectordb.client import LanceDBClient

app = typer.Typer(no_args_is_help=True, help="Index and chat with a local code repository.")
logger = logging.getLogger(__name__)
console = Console(width=160)
DEFAULT_DB_PATH = Path(".mini-agent/lancedb")
SYSTEM_PROMPT = """You are a careful coding assistant for the current repository.
Use retrieved code context and available tools to ground answers. Never claim to
have changed files; this agent currently provides read-only repository tools."""
INDEXABLE_SUFFIXES = {".py", ".java", ".js", ".ts"}


def _make_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(EmbeddingConfig())


def _make_db_client(path: Path, dimension: int) -> LanceDBClient:
    return LanceDBClient(uri=path, dimension=dimension)


def _repository_inventory(repo: Path):
    """Scan local file metadata once for inspect/stats presentation."""
    return RepositoryScanner(repo).scan(repo)


def _show_index_status(repo: Path, db_path: Path) -> None:
    manifest = IndexManifestStore(db_path / "index_manifest.json").load()
    if manifest is None:
        console.print("[yellow]Index manifest: not found[/yellow]")
    elif manifest.repository_root != str(repo.resolve()):
        console.print(
            f"[yellow]Index manifest belongs to {manifest.repository_root}[/yellow]"
        )
    else:
        console.print(f"Index manifest: {len(manifest.file_hashes)} files tracked")


def _make_searcher(
    repo: Path, embedder: SentenceTransformerEmbedder, db_client: LanceDBClient
) -> HybridSearcher:
    chunks, _, _, _ = RepositoryIndexer(embedder, db_client).collect_chunks(repo)
    return HybridSearcher(
        BM25Searcher(chunks),
        SemanticSearcher(embedder, db_client),
    )


def _make_navigation_index(repo: Path) -> NavigationIndex:
    return NavigationIndex.from_repository(repo)


def _format_definitions(definitions) -> str:
    if not definitions:
        return "No matching definitions found."
    return "\n\n".join(
        f"{item.qualified_name} ({item.kind}) — "
        f"{item.path}:{item.start_line}-{item.end_line}\n{item.code}"
        for item in definitions
    )


def _format_references(references) -> str:
    if not references:
        return "No matching references found."
    return "\n".join(
        f"{item.path}:{item.line}: {item.code}"
        + (" [definition]" if item.is_definition else "")
        for item in references
    )


def create_coding_tools(
    repo: Path,
    retriever: HybridSearcher,
    navigation: NavigationIndex | None = None,
) -> ToolRegistry:
    """Create read-only repository tools constrained to the selected root."""
    root = repo.resolve()
    navigation = navigation or NavigationIndex(root, [])
    registry = ToolRegistry()

    def read_file(path: str) -> str:
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("Requested path is outside the repository") from error
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def search_code(query: str, top_k: int = 5) -> str:
        results = retriever.search(query, top_k=max(1, min(top_k, 20)))
        if not results:
            return "No matching code chunks found."
        return "\n\n".join(
            f"[{result.chunk.path}:{result.chunk.start_line}-"
            f"{result.chunk.end_line} {result.chunk.display_name}]\n{result.chunk.code}"
            for result in results
        )

    def find_definition(symbol: str) -> str:
        return _format_definitions(navigation.find_definition(symbol))

    def find_references(symbol: str, limit: int = 100) -> str:
        return _format_references(navigation.find_references(symbol, limit=limit))

    def search_symbols(query: str, limit: int = 20) -> str:
        return _format_definitions(navigation.search_symbols(query, limit=limit))

    registry.register(
        "read_file",
        read_file,
        "Read a UTF-8 file from the repository.",
        [ToolParameter("path", "Repository-relative file path", "string")],
    )
    registry.register(
        "search_code",
        search_code,
        "Search indexed repository code using keyword and semantic retrieval.",
        [
            ToolParameter("query", "What code to find", "string"),
            ToolParameter("top_k", "Maximum results (1-20)", "integer", required=False, default=5),
        ],
    )
    registry.register(
        "find_definition",
        find_definition,
        "Find AST-indexed definitions by exact symbol or qualified name.",
        [ToolParameter("symbol", "Symbol name, optionally qualified", "string")],
    )
    registry.register(
        "find_references",
        find_references,
        "Find whole-identifier source-line occurrences (lexical, not compiler-accurate).",
        [
            ToolParameter("symbol", "Symbol identifier", "string"),
            ToolParameter(
                "limit", "Maximum matches (1-500)", "integer", required=False, default=100
            ),
        ],
    )
    registry.register(
        "search_symbols",
        search_symbols,
        "Search AST-indexed functions, methods, classes, and interfaces.",
        [
            ToolParameter("query", "Symbol name or identifier terms", "string"),
            ToolParameter(
                "limit", "Maximum definitions", "integer", required=False, default=20
            ),
        ],
    )
    return registry


def _watch_filter(_change: object, path: str) -> bool:
    """Ignore generated directories and non-source files in watch mode."""
    changed_path = Path(path)
    return (
        changed_path.suffix.lower() in INDEXABLE_SUFFIXES
        and not any(part in IGNORE_DIRECTORIES for part in changed_path.parts)
    )


@app.command()
def index(
    repo_argument: Annotated[
        Path | None, typer.Argument(exists=True, file_okay=False)
    ] = None,
    repo_option: Annotated[
        Path | None, typer.Option("--repo", exists=True, file_okay=False)
    ] = None,
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing vectors.")
    ] = False,
    watch: Annotated[bool, typer.Option("--watch", help="Keep indexing source changes.")] = False,
) -> None:
    """Incrementally index a repository; pass --watch to follow file changes."""
    repo = (repo_option or repo_argument or Path(".")).resolve()
    embedder = _make_embedder()
    db_client = _make_db_client(db_path, embedder.embedding_dimension)
    indexer = RepositoryIndexer(embedder, db_client)

    def run_index(rebuild: bool) -> None:
        metrics = get_metrics_collector()
        before = metrics.snapshot()
        summary = indexer.index(repo, overwrite=rebuild)
        after = metrics.snapshot()
        typer.echo(
            f"Scanned {summary.scanned_files} files; indexed {summary.chunks} changed chunks "
            f"from {summary.parsed_files} files; unchanged {summary.unchanged_files}; "
            f"deleted {summary.deleted_files}; failed {summary.failed_files}."
        )
        timings = []
        for name in (
            "scan",
            "parse",
            "chunk",
            "embedding",
            "index.vector_write",
            "vector.upsert",
        ):
            total_after = after.timings.get(name)
            total_before = before.timings.get(name)
            elapsed = (total_after.total_ms if total_after else 0.0) - (
                total_before.total_ms if total_before else 0.0
            )
            if elapsed or (total_after and not total_before):
                timings.append(f"{name}={elapsed:.1f} ms")
        if timings:
            typer.echo("Timing: " + ", ".join(timings))

    run_index(overwrite)
    if watch:
        from watchfiles import watch as watch_files

        typer.echo("Watching for source changes. Press Ctrl+C to stop.")
        try:
            for _changes in watch_files(repo, watch_filter=_watch_filter):
                run_index(rebuild=False)
        except KeyboardInterrupt:
            typer.echo("Stopped watching.")


@app.command()
def reindex(
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("."),
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """Discard the current vector table and perform a complete rebuild."""
    embedder = _make_embedder()
    db_client = _make_db_client(db_path, embedder.embedding_dimension)
    summary = RepositoryIndexer(embedder, db_client).index(repo, overwrite=True)
    typer.echo(
        f"Rebuilt index: scanned {summary.scanned_files} files; created "
        f"{summary.chunks} chunks from {summary.parsed_files} files; "
        f"failed {summary.failed_files}."
    )


@app.command()
def inspect(
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("."),
    limit: Annotated[int, typer.Option(min=1, max=200, help="Maximum files to list.")] = 30,
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """Inspect repository files and show whether the local index matches it."""
    repo = repo.resolve()
    inventory = _repository_inventory(repo)
    console.print(Text(f"Repository {inventory.root}", no_wrap=True))
    console.print(
        f"Files: {inventory.indexed_files} | Directories: {inventory.directories} | "
        f"Ignored: {inventory.ignored_files}"
    )
    _show_index_status(repo, db_path)

    table = Table(title=f"Files (showing up to {limit})")
    table.add_column("Path", overflow="fold")
    table.add_column("Language")
    table.add_column("Size", justify="right")
    for file in inventory.files[:limit]:
        table.add_row(file.path.as_posix(), file.language.value, f"{file.size:,} B")
    console.print(table)


@app.command()
def stats(
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("."),
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """Show repository language counts and local vector/manifest statistics."""
    repo = repo.resolve()
    inventory = _repository_inventory(repo)
    language_counts: dict[str, int] = {}
    for file in inventory.files:
        language = file.language.value
        language_counts[language] = language_counts.get(language, 0) + 1

    table = Table(title="Repository statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Root", str(inventory.root))
    table.add_row("Files scanned", f"{inventory.indexed_files:,}")
    table.add_row("Directories", f"{inventory.directories:,}")
    table.add_row("Ignored entries", f"{inventory.ignored_files:,}")
    table.add_section()
    for language, count in sorted(language_counts.items()):
        table.add_row(f"{language} files", f"{count:,}")

    db_client = _make_db_client(db_path, dimension=384)
    vector_count = db_client.count()
    table.add_section()
    table.add_row("Vector records", f"{vector_count:,}")
    manifest = IndexManifestStore(db_path / "index_manifest.json").load()
    if manifest and manifest.repository_root == str(repo):
        tracked_count = len(manifest.file_hashes)
    else:
        tracked_count = 0
    table.add_row("Manifest tracked files", f"{tracked_count:,}")
    if manifest and manifest.repository_root != str(repo):
        table.add_row("Manifest repository", manifest.repository_root)
    metrics_snapshot = get_metrics_collector().snapshot()
    table.add_section()
    console.print("[dim]Runtime telemetry for this CLI process[/dim]")
    for metric_name, summary in sorted(metrics_snapshot.timings.items()):
        table.add_row(
            f"Runtime {metric_name} (total / calls)",
            f"{summary.total_ms:.1f} ms / {summary.count}",
        )
    for metric_name, value in sorted(metrics_snapshot.counters.items()):
        table.add_row(f"Runtime {metric_name}", f"{value:,}")
    for metric_name, samples in sorted(metrics_snapshot.values.items()):
        if samples:
            table.add_row(
                f"Runtime {metric_name} (n / avg / max)",
                f"{len(samples)} / {sum(samples) / len(samples):.1f} / {max(samples):.1f}",
            )
    console.print(table)


@app.command()
def search(
    query: str,
    repo: Annotated[Path, typer.Option("--repo", exists=True, file_okay=False)] = Path("."),
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    top_k: Annotated[int, typer.Option(min=1, max=50, help="Maximum results to display.")] = 5,
) -> None:
    """Search repository code with BM25 and semantic retrieval."""
    embedder = _make_embedder()
    db_client = _make_db_client(db_path, embedder.embedding_dimension)
    retriever = _make_searcher(repo, embedder, db_client)
    results = retriever.search(query, top_k=top_k)
    if not results:
        typer.echo("No matching code found.")
        return
    for result in results:
        typer.echo(
            f"\n{result.chunk.path}:{result.chunk.start_line}-"
            f"{result.chunk.end_line}  score={result.score:.3f}"
        )
        typer.echo(result.chunk.code)


@app.command("find-definition")
def find_definition_command(
    symbol: str,
    repo: Annotated[Path, typer.Option("--repo", exists=True, file_okay=False)] = Path("."),
) -> None:
    """Find AST-indexed definitions for an exact symbol or qualified name."""
    navigation = _make_navigation_index(repo)
    typer.echo(_format_definitions(navigation.find_definition(symbol)))


@app.command("find-references")
def find_references_command(
    symbol: str,
    repo: Annotated[Path, typer.Option("--repo", exists=True, file_okay=False)] = Path("."),
    limit: Annotated[int, typer.Option(min=1, max=500)] = 100,
) -> None:
    """Find lexical whole-identifier references in supported repository files."""
    navigation = _make_navigation_index(repo)
    typer.echo(_format_references(navigation.find_references(symbol, limit=limit)))


@app.command("search-symbols")
def search_symbols_command(
    query: str,
    repo: Annotated[Path, typer.Option("--repo", exists=True, file_okay=False)] = Path("."),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
) -> None:
    """Search AST-indexed symbol definitions by name or identifier terms."""
    navigation = _make_navigation_index(repo)
    typer.echo(_format_definitions(navigation.search_symbols(query, limit=limit)))


@app.command()
def chat(
    repo: Annotated[Path, typer.Option("--repo", exists=True, file_okay=False)] = Path("."),
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    memory_path: Annotated[
        Path, typer.Option("--memory-path", help="Directory for saved conversations.")
    ] = Path(".mini-agent/conversations"),
    conversation_id: Annotated[
        str | None, typer.Option("--conversation-id", help="Resume a saved conversation.")
    ] = None,
    multi_agent: Annotated[
        bool, typer.Option("--multi-agent", help="Run researcher then synthesizer agents.")
    ] = False,
) -> None:
    """Start an interactive coding-agent session (use /exit to stop)."""
    load_dotenv()
    try:
        llm_client = GroqClient()
        embedder = _make_embedder()
        db_client = _make_db_client(db_path, embedder.embedding_dimension)
        retriever = _make_searcher(repo, embedder, db_client)
        navigation = _make_navigation_index(repo)
        tools = create_coding_tools(repo, retriever, navigation)
        conversation_manager = ConversationManager(ConversationStore(memory_path))
        if multi_agent:
            researcher_executor = AgentExecutor(
                llm_client=llm_client,
                tool_registry=tools,
                prompt_builder=PromptBuilder(RESEARCH_PROMPT),
                max_steps=8,
                retriever=retriever,
                planner=SimplePlanner(),
            )
            synthesizer_executor = AgentExecutor(
                llm_client=llm_client,
                tool_registry=ToolRegistry(),
                prompt_builder=PromptBuilder(SYNTHESIS_PROMPT),
                conversation_manager=conversation_manager,
                planner=SimplePlanner(),
            )
            researcher = LangGraphAgent(researcher_executor)
            synthesizer = LangGraphAgent(synthesizer_executor)
            runner: LangGraphAgent | MultiAgentCoordinator = MultiAgentCoordinator(
                researcher, synthesizer
            )
        else:
            executor = AgentExecutor(
                llm_client=llm_client,
                tool_registry=tools,
                prompt_builder=PromptBuilder(SYSTEM_PROMPT),
                conversation_manager=conversation_manager,
                retriever=retriever,
                planner=SimplePlanner(),
            )
            runner = LangGraphAgent(executor)
    except Exception as error:
        typer.echo(f"Unable to start chat: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Chat ready. Enter /exit to quit.")
    while True:
        try:
            goal = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye.")
            break
        if goal.strip().lower() in {"/exit", "/quit"}:
            break
        if not goal.strip():
            continue

        if isinstance(runner, MultiAgentCoordinator):
            team_result = runner.execute(goal, conversation_id=conversation_id)
            answer = team_result.answer
            status = team_result.status
            active_manager = runner.synthesizer.conversation_manager
        else:
            state = runner.execute(goal, conversation_id=conversation_id)
            answer = next(
                (
                    str(message["content"])
                    for message in reversed(state["messages"])
                    if message.get("role") == "assistant" and message.get("content")
                ),
                "",
            )
            status = state["status"]
            active_manager = runner.conversation_manager
        active_conversation = (
            active_manager.get_active_conversation() if active_manager else None
        )
        if conversation_id is None and active_conversation:
            conversation_id = active_conversation.conversation_id
            typer.echo(f"Saved conversation ID: {conversation_id}")
        if answer:
            typer.echo(f"\nAssistant: {answer}\n")
        if status != "SUCCESS":
            typer.echo(f"Agent status: {status}", err=True)


def main() -> None:
    """Entry point used by the installed `mini-agent` command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app()


if __name__ == "__main__":
    main()

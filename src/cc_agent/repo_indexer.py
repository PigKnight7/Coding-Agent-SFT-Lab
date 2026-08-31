from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cc_agent.rag import (
    DenseRepoRetriever,
    HybridRepoRetriever,
    RAGConfig,
    RetrievalFilters,
    RetrievalResult,
    render_results,
)

if TYPE_CHECKING:
    from cc_agent.rag import EmbeddingProvider

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".css",
    ".html",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".rb",
}


@dataclass(frozen=True)
class SymbolRecord:
    kind: str
    name: str
    path: str
    line: int

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} {self.name}"


@dataclass(frozen=True)
class RepoSnapshot:
    tree: str
    file_summaries: str
    project_rules: str
    symbols: str
    retrieval_notes: str

    def as_context(self) -> str:
        return (
            "# Repository tree\n"
            f"{self.tree}\n\n"
            "# Symbol index\n"
            f"{self.symbols}\n\n"
            "# Retrieval notes\n"
            f"{self.retrieval_notes}\n\n"
            "# Important file previews\n"
            f"{self.file_summaries}\n"
        )


class RepoIndexer:
    def __init__(
        self,
        repo_path: str | Path,
        max_files: int = 80,
        preview_chars: int = 1200,
        rag_config: RAGConfig | None = None,
        embedder: "EmbeddingProvider | None" = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.max_files = max_files
        self.preview_chars = preview_chars
        self.rag_config = rag_config or RAGConfig.from_env()
        self.embedder = embedder
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {self.repo_path}")

    def snapshot(self, query: str | None = None) -> RepoSnapshot:
        files = self._collect_files()
        tree = self._build_tree(files)
        summaries = self._build_summaries(files[: self.max_files])
        rules = self._read_project_rules()
        symbols = self._build_symbol_index(files)
        retrieval_notes = (
            self.retrieve(query or "", top_k=8)
            if query
            else "No task-specific retrieval query provided."
        )
        return RepoSnapshot(
            tree=tree,
            file_summaries=summaries,
            project_rules=rules,
            symbols=symbols,
            retrieval_notes=retrieval_notes,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        chars_per_file: int = 1400,
        path_filter: str | None = None,
        language_filter: str | None = None,
        symbol_filter: str | None = None,
    ) -> str:
        """Retrieve repository context using hybrid, dense, or legacy lexical mode."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.rag_config.validate()
        if self.rag_config.mode in {"hybrid", "dense"}:
            results = self.search_chunks(
                query=query,
                top_k=top_k,
                path_filter=path_filter,
                language_filter=language_filter,
                symbol_filter=symbol_filter,
            )
            return render_results(results, max_tokens=self.rag_config.context_max_tokens)
        return self._retrieve_lexical(query=query, top_k=top_k, chars_per_file=chars_per_file)

    def search_chunks(
        self,
        query: str,
        top_k: int = 8,
        path_filter: str | None = None,
        language_filter: str | None = None,
        symbol_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Return structured retrieval results for evaluation and programmatic use."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.rag_config.validate()
        if self.rag_config.mode == "lexical":
            raise ValueError("Structured chunk results require hybrid or dense retrieval mode")
        files = [path for path in self._collect_files() if path.suffix.lower() in TEXT_EXTENSIONS]
        retriever_class = HybridRepoRetriever if self.rag_config.mode == "hybrid" else DenseRepoRetriever
        retriever = retriever_class(self.repo_path, self.rag_config, self.embedder)
        filters = RetrievalFilters(
            path=path_filter,
            language=language_filter,
            symbol=symbol_filter,
        )
        return retriever.retrieve(query=query, files=files, top_k=top_k, filters=filters)

    def _retrieve_lexical(self, query: str, top_k: int = 8, chars_per_file: int = 1400) -> str:
        tokens = _tokenize(query)
        if not tokens:
            return "No retrieval query terms available."

        scored: list[tuple[float, Path, str]] = []
        for path in self._collect_files():
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            score = _score_text(tokens, path.relative_to(self.repo_path).as_posix(), text)
            if score <= 0:
                continue
            scored.append((score, path, text))

        if not scored:
            return "No relevant files found by lexical retrieval. Use list_files or grep next."

        chunks: list[str] = []
        for score, path, text in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]:
            rel = path.relative_to(self.repo_path).as_posix()
            excerpt = _best_excerpt(tokens, text, chars_per_file)
            chunks.append(f"## {rel} score={score:.2f}\n```\n{excerpt}\n```")
        return "\n\n".join(chunks)

    def _collect_files(self) -> list[Path]:
        result: list[Path] = []
        for path in sorted(self.repo_path.rglob("*")):
            rel = path.relative_to(self.repo_path)
            if any(part in IGNORED_DIRS for part in rel.parts):
                continue
            if path.is_file():
                result.append(path)
        return result

    def _build_tree(self, files: list[Path]) -> str:
        lines: list[str] = []
        for path in files[: self.max_files]:
            rel = path.relative_to(self.repo_path)
            depth = len(rel.parts) - 1
            prefix = "  " * depth + "- "
            lines.append(f"{prefix}{rel.as_posix()}")
        if len(files) > self.max_files:
            lines.append(f"... {len(files) - self.max_files} more files omitted")
        return "\n".join(lines) or "(empty repository)"

    def _build_summaries(self, files: list[Path]) -> str:
        chunks: list[str] = []
        for path in files:
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            rel = path.relative_to(self.repo_path).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            preview = text[: self.preview_chars].strip()
            if not preview:
                continue
            chunks.append(f"## {rel}\n```\n{preview}\n```")
        return "\n\n".join(chunks) or "No text file previews available."

    def _build_symbol_index(self, files: list[Path]) -> str:
        records: list[SymbolRecord] = []
        for path in files:
            if path.suffix != ".py":
                continue
            rel = path.relative_to(self.repo_path).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    records.append(SymbolRecord("class", node.name, rel, node.lineno))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    records.append(SymbolRecord("function", node.name, rel, node.lineno))
        if not records:
            return "No Python symbols found."
        return "\n".join(record.render() for record in records[:120])

    def _read_project_rules(self) -> str:
        candidates = [self.repo_path / "AGENT.md", self.repo_path / "CLAUDE.md"]
        for path in candidates:
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")
        return "No project-specific AGENT.md or CLAUDE.md found. Follow safe minimal-change defaults."


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+", text)]


def _score_text(tokens: list[str], path: str, text: str) -> float:
    haystack = f"{path}\n{text}".lower()
    score = 0.0
    for token in tokens:
        count = haystack.count(token)
        if count:
            score += 1.0 + math.log1p(count)
            if token in path.lower():
                score += 2.0
    return score


def _best_excerpt(tokens: list[str], text: str, limit: int) -> str:
    lowered = text.lower()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    if not positions:
        return text[:limit].strip()
    center = min(positions)
    start = max(0, center - limit // 3)
    return text[start : start + limit].strip()

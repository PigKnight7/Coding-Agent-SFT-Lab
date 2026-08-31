from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for dense and hybrid repository retrieval."""

    mode: str = "hybrid"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str | None = None
    index_dir: Path = Path(".rag_indexes")
    batch_size: int = 64
    chunk_chars: int = 1800
    chunk_overlap_lines: int = 5
    candidate_multiplier: int = 4
    rrf_k: int = 60
    identifier_boost: float = 0.03
    context_max_tokens: int = 2000
    reranker: str = "heuristic"
    rerank_weight: float = 0.02
    git_change_boost: float = 0.02

    @classmethod
    def from_env(cls) -> "RAGConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        default_index_dir = Path.home() / ".cache" / "cc-agent" / "rag"
        return cls(
            mode=os.getenv("CC_AGENT_RETRIEVAL_MODE", "hybrid").strip().lower(),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "openai").strip().lower(),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            api_key=os.getenv("RAG_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            base_url=os.getenv("RAG_EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL")) or None,
            index_dir=Path(os.getenv("RAG_INDEX_DIR", str(default_index_dir))).expanduser(),
            batch_size=max(1, int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))),
            chunk_chars=max(400, int(os.getenv("RAG_CHUNK_CHARS", "1800"))),
            chunk_overlap_lines=max(0, int(os.getenv("RAG_CHUNK_OVERLAP_LINES", "5"))),
            candidate_multiplier=max(1, int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "4"))),
            rrf_k=max(1, int(os.getenv("RAG_RRF_K", "60"))),
            identifier_boost=max(0.0, float(os.getenv("RAG_IDENTIFIER_BOOST", "0.03"))),
            context_max_tokens=max(256, int(os.getenv("RAG_CONTEXT_MAX_TOKENS", "2000"))),
            reranker=os.getenv("RAG_RERANKER", "heuristic").strip().lower(),
            rerank_weight=max(0.0, float(os.getenv("RAG_RERANK_WEIGHT", "0.02"))),
            git_change_boost=max(0.0, float(os.getenv("RAG_GIT_CHANGE_BOOST", "0.02"))),
        )

    def validate(self) -> None:
        if self.mode not in {"hybrid", "dense", "lexical"}:
            raise ValueError("CC_AGENT_RETRIEVAL_MODE must be 'hybrid', 'dense', or 'lexical'")
        if self.mode in {"hybrid", "dense"} and self.embedding_provider != "openai":
            raise ValueError("The current RAG implementation supports RAG_EMBEDDING_PROVIDER=openai")
        if self.reranker not in {"heuristic", "none"}:
            raise ValueError("RAG_RERANKER must be 'heuristic' or 'none'")


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    path: str
    language: str
    symbol: str | None
    symbol_type: str | None
    start_line: int
    end_line: int
    content: str

    @classmethod
    def create(
        cls,
        *,
        path: str,
        language: str,
        symbol: str | None,
        symbol_type: str | None,
        start_line: int,
        end_line: int,
        content: str,
    ) -> "CodeChunk":
        identity = f"{path}:{start_line}:{end_line}:{content}".encode("utf-8")
        return cls(
            chunk_id=hashlib.sha256(identity).hexdigest(),
            path=path,
            language=language,
            symbol=symbol,
            symbol_type=symbol_type,
            start_line=start_line,
            end_line=end_line,
            content=content,
        )

    def embedding_text(self) -> str:
        metadata = f"file: {self.path}\nlanguage: {self.language}"
        if self.symbol:
            metadata += f"\nsymbol: {self.symbol_type} {self.symbol}"
        return f"{metadata}\n\n{self.content}"


@dataclass(frozen=True)
class RetrievalResult:
    chunk: CodeChunk
    score: float
    retrieval_method: str = "dense"


@dataclass(frozen=True)
class RetrievalFilters:
    path: str | None = None
    language: str | None = None
    symbol: str | None = None

    def matches(self, chunk: CodeChunk) -> bool:
        if self.path and self.path.lower() not in chunk.path.lower():
            return False
        if self.language and self.language.lower() != chunk.language.lower():
            return False
        if self.symbol and self.symbol.lower() not in (chunk.symbol or "").lower():
            return False
        return True


@dataclass(frozen=True)
class IndexBuildStats:
    total_chunks: int
    embedded_chunks: int
    reused_chunks: int
    rebuilt: bool


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int,
        changed_paths: set[str],
    ) -> list[RetrievalResult]: ...


class CodeAwareReranker:
    """Lightweight reranker using query coverage, symbol matches, and Git changes."""

    def __init__(self, relevance_weight: float = 0.02, git_change_boost: float = 0.02):
        self.relevance_weight = relevance_weight
        self.git_change_boost = git_change_boost

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int,
        changed_paths: set[str],
    ) -> list[RetrievalResult]:
        query_tokens = set(_search_tokens(query))
        reranked: list[RetrievalResult] = []
        for result in results:
            chunk_tokens = set(_search_tokens(_chunk_search_text(result.chunk)))
            coverage = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            score = result.score + self.relevance_weight * coverage
            if result.chunk.path in changed_paths:
                score += self.git_change_boost
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=score,
                    retrieval_method="hybrid+rerank",
                )
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True)[: max(1, top_k)]


class EmbeddingProvider(Protocol):
    @property
    def identifier(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, config: RAGConfig):
        from openai import OpenAI

        config.validate()
        if not config.api_key:
            raise RuntimeError(
                "Dense retrieval requires RAG_EMBEDDING_API_KEY or OPENAI_API_KEY. "
                "Set CC_AGENT_RETRIEVAL_MODE=lexical to use the legacy retriever."
            )
        kwargs: dict[str, str] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.model = config.embedding_model
        self.batch_size = config.batch_size

    @property
    def identifier(self) -> str:
        return f"openai:{self.model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


class PersistentVectorIndex:
    """Small-repository vector store persisted as JSON and searched with cosine similarity."""

    SCHEMA_VERSION = 2

    def __init__(self, path: Path):
        self.path = path
        self.fingerprint = ""
        self.embedding_provider = ""
        self.chunks: list[CodeChunk] = []
        self.vectors: list[list[float]] = []

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != self.SCHEMA_VERSION:
                return False
            chunks = [CodeChunk(**item) for item in payload["chunks"]]
            vectors = payload["vectors"]
            if len(chunks) != len(vectors):
                return False
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        self.fingerprint = str(payload.get("fingerprint", ""))
        self.embedding_provider = str(payload.get("embedding_provider", ""))
        self.chunks = chunks
        self.vectors = vectors
        return True

    def save(
        self,
        *,
        fingerprint: str,
        embedding_provider: str,
        chunks: list[CodeChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each code chunk must have one embedding vector")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "embedding_provider": embedding_provider,
            "chunks": [asdict(chunk) for chunk in chunks],
            "vectors": vectors,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary_path.replace(self.path)
        self.fingerprint = fingerprint
        self.embedding_provider = embedding_provider
        self.chunks = chunks
        self.vectors = vectors

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        predicate: Callable[[CodeChunk], bool] | None = None,
    ) -> list[RetrievalResult]:
        scored = [
            RetrievalResult(chunk=chunk, score=_cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
            if predicate is None or predicate(chunk)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[: max(1, top_k)]


class DenseRepoRetriever:
    def __init__(
        self,
        repo_path: str | Path,
        config: RAGConfig,
        embedder: EmbeddingProvider | None = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.config.validate()
        self.embedder = embedder or OpenAIEmbeddingProvider(config)
        repo_key = hashlib.sha256(str(self.repo_path).encode("utf-8")).hexdigest()[:20]
        self.index = PersistentVectorIndex(config.index_dir / f"{repo_key}.json")
        self.last_index_stats = IndexBuildStats(0, 0, 0, False)

    def prepare(self, files: Sequence[Path]) -> None:
        fingerprint = _repository_fingerprint(self.repo_path, files, self.config, self.embedder.identifier)
        index_loaded = self.index.load()
        provider_is_compatible = index_loaded and self.index.embedding_provider == self.embedder.identifier
        index_is_current = (
            index_loaded
            and self.index.fingerprint == fingerprint
            and provider_is_compatible
        )
        if index_is_current:
            self.last_index_stats = IndexBuildStats(len(self.index.chunks), 0, len(self.index.chunks), False)
            return

        chunks = chunk_repository(
            self.repo_path,
            files,
            max_chars=self.config.chunk_chars,
            overlap_lines=self.config.chunk_overlap_lines,
        )
        previous_vectors = (
            {chunk.chunk_id: vector for chunk, vector in zip(self.index.chunks, self.index.vectors)}
            if provider_is_compatible
            else {}
        )
        missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in previous_vectors]
        missing_vectors = (
            self.embedder.embed_documents([chunk.embedding_text() for chunk in missing_chunks])
            if missing_chunks
            else []
        )
        new_vectors = dict(zip((chunk.chunk_id for chunk in missing_chunks), missing_vectors))
        vectors = [previous_vectors.get(chunk.chunk_id) or new_vectors[chunk.chunk_id] for chunk in chunks]
        reused_chunks = len(chunks) - len(missing_chunks)
        self.index.save(
            fingerprint=fingerprint,
            embedding_provider=self.embedder.identifier,
            chunks=chunks,
            vectors=vectors,
        )
        self.last_index_stats = IndexBuildStats(
            total_chunks=len(chunks),
            embedded_chunks=len(missing_chunks),
            reused_chunks=reused_chunks,
            rebuilt=True,
        )

    def retrieve(
        self,
        query: str,
        files: Sequence[Path],
        top_k: int = 8,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []
        self.prepare(files)
        query_vector = self.embedder.embed_query(query)
        predicate = filters.matches if filters else None
        return self.index.search(query_vector, top_k, predicate)


class HybridRepoRetriever(DenseRepoRetriever):
    """Fuse dense and BM25 chunk rankings with reciprocal rank fusion."""

    def __init__(
        self,
        repo_path: str | Path,
        config: RAGConfig,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
    ):
        super().__init__(repo_path, config, embedder)
        self.reranker = reranker or CodeAwareReranker(config.rerank_weight, config.git_change_boost)

    def retrieve(
        self,
        query: str,
        files: Sequence[Path],
        top_k: int = 8,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []
        self.prepare(files)
        candidate_count = max(top_k, top_k * self.config.candidate_multiplier)
        predicate = filters.matches if filters else None
        query_vector = self.embedder.embed_query(query)
        dense_results = self.index.search(query_vector, candidate_count, predicate)
        lexical_results = bm25_search(
            self.index.chunks,
            query,
            top_k=candidate_count,
            filters=filters,
        )
        fused_results = _reciprocal_rank_fusion(
            query,
            dense_results,
            lexical_results,
            top_k=candidate_count,
            rrf_k=self.config.rrf_k,
            identifier_boost=self.config.identifier_boost,
        )
        if self.config.reranker == "none":
            return fused_results[:top_k]
        changed_paths = _git_changed_paths(self.repo_path)
        return self.reranker.rerank(
            query,
            fused_results,
            top_k=top_k,
            changed_paths=_expand_paths_with_python_dependencies(self.repo_path, changed_paths),
        )


def chunk_repository(
    repo_path: Path,
    files: Sequence[Path],
    *,
    max_chars: int = 1800,
    overlap_lines: int = 5,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        relative_path = file_path.relative_to(repo_path).as_posix()
        if file_path.suffix.lower() == ".py":
            python_chunks = _chunk_python(relative_path, text, max_chars, overlap_lines)
            if python_chunks:
                chunks.extend(python_chunks)
                continue
        structured_chunks = _chunk_with_tree_sitter(
            relative_path,
            file_path.suffix.lower(),
            text,
            max_chars,
            overlap_lines,
        )
        if structured_chunks:
            chunks.extend(structured_chunks)
            continue
        chunks.extend(
            _chunk_lines(
                relative_path,
                _language_for_path(file_path),
                text,
                max_chars=max_chars,
                overlap_lines=overlap_lines,
            )
        )
    return chunks


def bm25_search(
    chunks: Sequence[CodeChunk],
    query: str,
    *,
    top_k: int,
    filters: RetrievalFilters | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[RetrievalResult]:
    """Rank code chunks with a dependency-free BM25 implementation."""
    query_tokens = _search_tokens(query)
    candidates = [chunk for chunk in chunks if filters is None or filters.matches(chunk)]
    if not query_tokens or not candidates:
        return []

    tokenized_documents = [_search_tokens(_chunk_search_text(chunk)) for chunk in candidates]
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequencies.update(set(tokens))
    average_length = sum(len(tokens) for tokens in tokenized_documents) / len(tokenized_documents)

    results: list[RetrievalResult] = []
    for chunk, tokens in zip(candidates, tokenized_documents):
        frequencies = Counter(tokens)
        document_length = max(1, len(tokens))
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            document_frequency = document_frequencies[token]
            inverse_document_frequency = math.log(
                1.0 + (len(candidates) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * document_length / max(1.0, average_length))
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        if score > 0.0:
            results.append(RetrievalResult(chunk=chunk, score=score, retrieval_method="bm25"))
    return sorted(results, key=lambda item: item.score, reverse=True)[: max(1, top_k)]


def render_results(results: Sequence[RetrievalResult], max_tokens: int = 2000) -> str:
    if not results:
        return "No relevant code chunks found by retrieval. Use list_files or grep next."
    rendered: list[str] = []
    selected: list[CodeChunk] = []
    used_tokens = 0
    for result in results:
        chunk = result.chunk
        if any(_chunks_are_near_duplicates(chunk, existing) for existing in selected):
            continue
        symbol = f" symbol={chunk.symbol_type}:{chunk.symbol}" if chunk.symbol else ""
        block = (
            f"## {chunk.path}:{chunk.start_line}-{chunk.end_line} "
            f"score={result.score:.4f} method={result.retrieval_method}{symbol}\n"
            f"```{chunk.language}\n{chunk.content}\n```"
        )
        block_tokens = _estimate_tokens(block)
        remaining_tokens = max_tokens - used_tokens
        if remaining_tokens <= 0:
            break
        if block_tokens > remaining_tokens:
            block = _truncate_block(chunk, result, symbol, remaining_tokens)
            if not block:
                break
            block_tokens = _estimate_tokens(block)
        rendered.append(block)
        selected.append(chunk)
        used_tokens += block_tokens
    return "\n\n".join(rendered)


def _reciprocal_rank_fusion(
    query: str,
    dense_results: Sequence[RetrievalResult],
    lexical_results: Sequence[RetrievalResult],
    *,
    top_k: int,
    rrf_k: int,
    identifier_boost: float,
) -> list[RetrievalResult]:
    chunks: dict[str, CodeChunk] = {}
    scores: Counter[str] = Counter()
    for ranking in (dense_results, lexical_results):
        for rank, result in enumerate(ranking, 1):
            chunks[result.chunk.chunk_id] = result.chunk
            scores[result.chunk.chunk_id] += 1.0 / (rrf_k + rank)
    for chunk_id, chunk in chunks.items():
        scores[chunk_id] += identifier_boost * _identifier_match_count(query, chunk)
    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[: max(1, top_k)]
    return [
        RetrievalResult(chunk=chunks[chunk_id], score=scores[chunk_id], retrieval_method="hybrid")
        for chunk_id in ranked_ids
    ]


def _search_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]", text):
        lowered = raw_token.lower()
        tokens.append(lowered)
        identifier_parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw_token).replace("_", " ").split()
        tokens.extend(part.lower() for part in identifier_parts if part.lower() != lowered)
    return tokens


def _chunk_search_text(chunk: CodeChunk) -> str:
    return f"{chunk.path}\n{chunk.symbol or ''}\n{chunk.content}"


def _identifier_match_count(query: str, chunk: CodeChunk) -> int:
    query_identifiers = set(_search_tokens(query))
    metadata_identifiers = set(_search_tokens(f"{chunk.path} {chunk.symbol or ''}"))
    return len({token for token in query_identifiers & metadata_identifiers if len(token) >= 3})


def _chunks_are_near_duplicates(left: CodeChunk, right: CodeChunk) -> bool:
    if left.chunk_id == right.chunk_id:
        return True
    if left.path != right.path or left.symbol != right.symbol:
        return False
    overlap = max(0, min(left.end_line, right.end_line) - max(left.start_line, right.start_line) + 1)
    shorter_length = min(left.end_line - left.start_line + 1, right.end_line - right.start_line + 1)
    return overlap / max(1, shorter_length) >= 0.8


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _truncate_block(
    chunk: CodeChunk,
    result: RetrievalResult,
    symbol: str,
    max_tokens: int,
) -> str:
    header = (
        f"## {chunk.path}:{chunk.start_line}-{chunk.end_line} "
        f"score={result.score:.4f} method={result.retrieval_method}{symbol}\n"
        f"```{chunk.language}\n"
    )
    footer = "\n... context truncated\n```"
    available_chars = max_tokens * 4 - len(header) - len(footer)
    if available_chars <= 0:
        return ""
    content = chunk.content[:available_chars].rsplit("\n", 1)[0] or chunk.content[:available_chars]
    return f"{header}{content}{footer}"


def _chunk_python(path: str, text: str, max_chars: int, overlap_lines: int) -> list[CodeChunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    if not nodes:
        return []

    chunks: list[CodeChunk] = []
    covered_lines: set[int] = set()
    for node in nodes:
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)
        covered_lines.update(range(start_line, end_line + 1))
        node_text = "\n".join(lines[start_line - 1 : end_line])
        symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks.extend(
            _chunk_lines(
                path,
                "python",
                node_text,
                max_chars=max_chars,
                overlap_lines=overlap_lines,
                line_offset=start_line - 1,
                symbol=node.name,
                symbol_type=symbol_type,
            )
        )

    module_lines = [line if number not in covered_lines else "" for number, line in enumerate(lines, 1)]
    module_text = "\n".join(module_lines)
    if module_text.strip():
        chunks.extend(
            _chunk_lines(
                path,
                "python",
                module_text,
                max_chars=max_chars,
                overlap_lines=overlap_lines,
                symbol="<module>",
                symbol_type="module",
                skip_empty_windows=True,
            )
        )
    return chunks


def _chunk_with_tree_sitter(
    path: str,
    suffix: str,
    text: str,
    max_chars: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    language = {
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".rb": "ruby",
    }.get(suffix)
    if language is None:
        return []
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(language)
        source = text.encode("utf-8")
        tree = parser.parse(source)
    except (ImportError, LookupError, OSError, RuntimeError, TypeError, ValueError):
        return []

    declaration_types = {
        "class_declaration": "class",
        "class_definition": "class",
        "function_declaration": "function",
        "function_definition": "function",
        "method_declaration": "method",
        "method_definition": "method",
        "interface_declaration": "interface",
        "struct_item": "struct",
        "function_item": "function",
        "impl_item": "implementation",
        "type_declaration": "type",
    }
    declarations: list[tuple[object, str]] = []

    def visit(node: object) -> None:
        node_type = getattr(node, "type", "")
        if node_type in declaration_types:
            declarations.append((node, declaration_types[node_type]))
            return
        for child in getattr(node, "named_children", []):
            visit(child)

    visit(tree.root_node)
    chunks: list[CodeChunk] = []
    covered_lines: set[int] = set()
    for node, symbol_type in declarations:
        start_byte = int(getattr(node, "start_byte"))
        end_byte = int(getattr(node, "end_byte"))
        start_point = getattr(node, "start_point")
        end_point = getattr(node, "end_point")
        start_line = _point_row(start_point) + 1
        end_line = _point_row(end_point) + 1
        covered_lines.update(range(start_line, end_line + 1))
        name_node = node.child_by_field_name("name")
        symbol = source[name_node.start_byte : name_node.end_byte].decode("utf-8") if name_node else None
        node_text = source[start_byte:end_byte].decode("utf-8", errors="replace")
        chunks.extend(
            _chunk_lines(
                path,
                _language_for_suffix(suffix),
                node_text,
                max_chars=max_chars,
                overlap_lines=overlap_lines,
                line_offset=start_line - 1,
                symbol=symbol,
                symbol_type=symbol_type,
            )
        )
    lines = text.splitlines()
    module_text = "\n".join(
        line if line_number not in covered_lines else ""
        for line_number, line in enumerate(lines, 1)
    )
    if module_text.strip():
        chunks.extend(
            _chunk_lines(
                path,
                _language_for_suffix(suffix),
                module_text,
                max_chars=max_chars,
                overlap_lines=overlap_lines,
                symbol="<module>",
                symbol_type="module",
                skip_empty_windows=True,
            )
        )
    return chunks


def _point_row(point: object) -> int:
    if hasattr(point, "row"):
        return int(getattr(point, "row"))
    return int(point[0])  # type: ignore[index]


def _chunk_lines(
    path: str,
    language: str,
    text: str,
    *,
    max_chars: int,
    overlap_lines: int,
    line_offset: int = 0,
    symbol: str | None = None,
    symbol_type: str | None = None,
    skip_empty_windows: bool = False,
) -> list[CodeChunk]:
    lines = text.splitlines()
    chunks: list[CodeChunk] = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines) and (size + len(lines[end]) + 1 <= max_chars or end == start):
            size += len(lines[end]) + 1
            end += 1
        content = "\n".join(lines[start:end]).strip()
        if content or not skip_empty_windows:
            chunks.append(
                CodeChunk.create(
                    path=path,
                    language=language,
                    symbol=symbol,
                    symbol_type=symbol_type,
                    start_line=line_offset + start + 1,
                    end_line=line_offset + end,
                    content=content,
                )
            )
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def _repository_fingerprint(
    repo_path: Path,
    files: Sequence[Path],
    config: RAGConfig,
    embedding_provider: str,
) -> str:
    digest = hashlib.sha256()
    chunker = "tree-sitter" if _tree_sitter_is_available() else "fallback"
    digest.update(
        f"chunking-v2:{chunker}:{embedding_provider}:{config.chunk_chars}:"
        f"{config.chunk_overlap_lines}".encode("utf-8")
    )
    for path in sorted(files):
        try:
            content = path.read_bytes()
        except OSError:
            continue
        digest.update(path.relative_to(repo_path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _tree_sitter_is_available() -> bool:
    try:
        return importlib.util.find_spec("tree_sitter_language_pack") is not None
    except (ImportError, ValueError):
        return False


def _git_changed_paths(repo_path: Path) -> set[str]:
    try:
        root_result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if root_result.returncode != 0:
            return set()
        git_root = Path(root_result.stdout.strip()).resolve()
        relative_repo = repo_path.resolve().relative_to(git_root).as_posix()
        prefix = "" if relative_repo == "." else f"{relative_repo}/"
        status_result = subprocess.run(
            ["git", "-C", str(git_root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if status_result.returncode != 0:
            return set()
    except (OSError, subprocess.SubprocessError, ValueError):
        return set()

    changed: set[str] = set()
    for line in status_result.stdout.splitlines():
        raw_path = line[3:].strip().strip('"')
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        if prefix and not raw_path.startswith(prefix):
            continue
        changed.add(raw_path[len(prefix) :] if prefix else raw_path)
    return changed


def _expand_paths_with_python_dependencies(repo_path: Path, seed_paths: set[str]) -> set[str]:
    if not seed_paths:
        return set()
    ignored_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}
    python_files = [
        path
        for path in repo_path.rglob("*.py")
        if not any(part in ignored_dirs for part in path.relative_to(repo_path).parts)
    ]
    module_to_path: dict[str, str] = {}
    for path in python_files:
        relative = path.relative_to(repo_path)
        module_parts = relative.with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module = ".".join(module_parts)
        if module:
            module_to_path[module] = relative.as_posix()

    graph: dict[str, set[str]] = {path: set() for path in module_to_path.values()}
    for path in python_files:
        relative_path = path.relative_to(repo_path).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        for imported_module in imported_modules:
            dependency = _resolve_module_path(imported_module, module_to_path)
            if dependency and dependency != relative_path:
                graph.setdefault(relative_path, set()).add(dependency)
                graph.setdefault(dependency, set()).add(relative_path)

    expanded = set(seed_paths)
    for path in seed_paths:
        expanded.update(graph.get(path, set()))
    return expanded


def _resolve_module_path(module: str, module_to_path: dict[str, str]) -> str | None:
    candidate = module
    while candidate:
        if candidate in module_to_path:
            return module_to_path[candidate]
        candidate = candidate.rpartition(".")[0]
    return None


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _language_for_path(path: Path) -> str:
    return _language_for_suffix(path.suffix.lower())


def _language_for_suffix(suffix: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".rb": "ruby",
    }.get(suffix, "text")

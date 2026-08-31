from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

from cc_agent.rag import (
    CodeAwareReranker,
    CodeChunk,
    DenseRepoRetriever,
    RAGConfig,
    RetrievalResult,
    _expand_paths_with_python_dependencies,
    bm25_search,
    chunk_repository,
    render_results,
)
from cc_agent.repo_indexer import RepoIndexer
from cc_agent.retrieval_eval import RetrievalExample, evaluate_rankings


class KeywordEmbeddingProvider:
    def __init__(self) -> None:
        self.document_calls = 0
        self.embedded_document_count = 0
        self.query_calls = 0

    @property
    def identifier(self) -> str:
        return "test:keyword-v1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        self.embedded_document_count += len(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(any(term in lowered for term in ("subtract", "difference", "minus"))),
            float(any(term in lowered for term in ("add", "sum", "plus"))),
            float(any(term in lowered for term in ("token", "authentication", "login"))),
            0.1,
        ]


def _config(index_dir: Path, mode: str = "dense") -> RAGConfig:
    return RAGConfig(
        mode=mode,
        embedding_provider="openai",
        embedding_model="test-model",
        index_dir=index_dir,
        chunk_chars=400,
        chunk_overlap_lines=2,
        context_max_tokens=500,
    )


class RAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_python_chunking_preserves_symbol_and_line_metadata(self) -> None:
        source = self.tmp_path / "calculator.py"
        source.write_text(
            "import math\n\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def subtract(a, b):\n"
            "    return a - b\n",
            encoding="utf-8",
        )

        chunks = chunk_repository(self.tmp_path, [source], max_chars=400, overlap_lines=2)

        subtract_chunk = next(chunk for chunk in chunks if chunk.symbol == "subtract")
        self.assertEqual(subtract_chunk.path, "calculator.py")
        self.assertEqual(subtract_chunk.symbol_type, "function")
        self.assertEqual(subtract_chunk.start_line, 6)
        self.assertEqual(subtract_chunk.end_line, 7)
        self.assertIn("return a - b", subtract_chunk.content)

    def test_tree_sitter_chunking_preserves_javascript_symbol(self) -> None:
        source = self.tmp_path / "service.js"
        source_text = "function greet(name) {\n  return `hello ${name}`;\n}\n"
        source.write_text(source_text, encoding="utf-8")
        source_bytes = source_text.encode("utf-8")

        class FakeNode:
            def __init__(self, node_type: str, start: int, end: int, name: object = None):
                self.type = node_type
                self.start_byte = start
                self.end_byte = end
                self.start_point = (0, 0)
                self.end_point = (2, 1)
                self.named_children: list[object] = []
                self._name = name

            def child_by_field_name(self, field: str) -> object:
                return self._name if field == "name" else None

        name_start = source_bytes.index(b"greet")
        name_node = FakeNode("identifier", name_start, name_start + len(b"greet"))
        declaration = FakeNode("function_declaration", 0, len(source_bytes), name_node)
        root = FakeNode("program", 0, len(source_bytes))
        root.named_children = [declaration]
        fake_parser = types.SimpleNamespace(parse=lambda _: types.SimpleNamespace(root_node=root))
        fake_module = types.ModuleType("tree_sitter_language_pack")
        fake_module.get_parser = lambda _: fake_parser

        with patch.dict("sys.modules", {"tree_sitter_language_pack": fake_module}):
            chunks = chunk_repository(self.tmp_path, [source], max_chars=400, overlap_lines=2)

        self.assertEqual(chunks[0].symbol, "greet")
        self.assertEqual(chunks[0].symbol_type, "function")
        self.assertEqual(chunks[0].language, "javascript")

    def test_dense_retrieval_returns_semantically_matching_chunk(self) -> None:
        repo = self.tmp_path / "repo"
        repo.mkdir()
        (repo / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n\n"
            "def subtract(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
        embedder = KeywordEmbeddingProvider()
        indexer = RepoIndexer(repo, rag_config=_config(self.tmp_path / "index"), embedder=embedder)

        result = indexer.retrieve("calculate the difference between two numbers", top_k=1)

        self.assertIn("calculator.py:4-5", result)
        self.assertIn("symbol=function:subtract", result)
        self.assertIn("return a - b", result)

    def test_bm25_ranks_exact_code_identifier_first(self) -> None:
        target = CodeChunk.create(
            path="handlers.py",
            language="python",
            symbol="validate_access_token",
            symbol_type="function",
            start_line=10,
            end_line=12,
            content="def validate_access_token(token):\n    return token.is_valid()",
        )
        unrelated = CodeChunk.create(
            path="users.py",
            language="python",
            symbol="create_user",
            symbol_type="function",
            start_line=1,
            end_line=2,
            content="def create_user(name):\n    return User(name)",
        )

        results = bm25_search([unrelated, target], "validate_access_token", top_k=2)

        self.assertEqual(results[0].chunk.chunk_id, target.chunk_id)
        self.assertEqual(results[0].retrieval_method, "bm25")

    def test_hybrid_retrieval_combines_dense_and_identifier_matching(self) -> None:
        repo = self.tmp_path / "repo"
        repo.mkdir()
        (repo / "handlers.py").write_text(
            "def generic_handler(payload):\n    return payload\n\n"
            "def target_handler(payload):\n    return payload.get('target')\n",
            encoding="utf-8",
        )
        embedder = KeywordEmbeddingProvider()
        indexer = RepoIndexer(
            repo,
            rag_config=_config(self.tmp_path / "index", mode="hybrid"),
            embedder=embedder,
        )

        result = indexer.retrieve("fix target_handler", top_k=1)

        self.assertIn("symbol=function:target_handler", result)
        self.assertIn("method=hybrid", result)

    def test_hybrid_retrieval_applies_language_and_path_filters(self) -> None:
        repo = self.tmp_path / "repo"
        repo.mkdir()
        (repo / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (repo / "web.ts").write_text("export const add = (a, b) => a + b;\n", encoding="utf-8")
        embedder = KeywordEmbeddingProvider()
        indexer = RepoIndexer(
            repo,
            rag_config=_config(self.tmp_path / "index", mode="hybrid"),
            embedder=embedder,
        )

        result = indexer.retrieve(
            "add two values",
            top_k=3,
            path_filter="web.ts",
            language_filter="typescript",
        )

        self.assertIn("web.ts", result)
        self.assertNotIn("calculator.py", result)

        symbol_result = indexer.retrieve("add two values", top_k=3, symbol_filter="add")
        self.assertIn("calculator.py", symbol_result)
        self.assertNotIn("web.ts", symbol_result)

    def test_context_builder_deduplicates_overlap_and_honors_budget(self) -> None:
        first = CodeChunk.create(
            path="large.py",
            language="python",
            symbol="large_function",
            symbol_type="function",
            start_line=1,
            end_line=100,
            content="line = 'value'\n" * 100,
        )
        overlapping = CodeChunk.create(
            path="large.py",
            language="python",
            symbol="large_function",
            symbol_type="function",
            start_line=5,
            end_line=95,
            content="line = 'value'\n" * 90,
        )

        context = render_results(
            [
                RetrievalResult(first, 1.0, "hybrid"),
                RetrievalResult(overlapping, 0.9, "hybrid"),
            ],
            max_tokens=120,
        )

        self.assertEqual(context.count("## large.py"), 1)
        self.assertIn("context truncated", context)
        self.assertLessEqual(len(context), 120 * 4)

    def test_code_aware_reranker_boosts_git_changed_file(self) -> None:
        unchanged = CodeChunk.create(
            path="old.py",
            language="python",
            symbol="handle",
            symbol_type="function",
            start_line=1,
            end_line=2,
            content="def handle(value):\n    return value",
        )
        changed = CodeChunk.create(
            path="changed.py",
            language="python",
            symbol="handle",
            symbol_type="function",
            start_line=1,
            end_line=2,
            content="def handle(value):\n    return value",
        )
        reranker = CodeAwareReranker(relevance_weight=0.0, git_change_boost=0.1)

        results = reranker.rerank(
            "handle value",
            [RetrievalResult(unchanged, 0.5), RetrievalResult(changed, 0.5)],
            top_k=2,
            changed_paths={"changed.py"},
        )

        self.assertEqual(results[0].chunk.path, "changed.py")
        self.assertEqual(results[0].retrieval_method, "hybrid+rerank")

    def test_python_dependency_graph_expands_changed_paths(self) -> None:
        package = self.tmp_path / "package"
        package.mkdir()
        (package / "service.py").write_text(
            "from package.utils import normalize\n\ndef run(value):\n    return normalize(value)\n",
            encoding="utf-8",
        )
        (package / "utils.py").write_text(
            "def normalize(value):\n    return value.strip()\n",
            encoding="utf-8",
        )

        expanded = _expand_paths_with_python_dependencies(
            self.tmp_path,
            {"package/utils.py"},
        )

        self.assertEqual(expanded, {"package/utils.py", "package/service.py"})

    def test_dense_index_is_reused_when_repository_is_unchanged(self) -> None:
        repo = self.tmp_path / "repo"
        repo.mkdir()
        (repo / "auth.py").write_text(
            "def validate_token(token):\n    return bool(token)\n",
            encoding="utf-8",
        )
        embedder = KeywordEmbeddingProvider()
        config = _config(self.tmp_path / "index")

        RepoIndexer(repo, rag_config=config, embedder=embedder).retrieve("authentication", top_k=1)
        RepoIndexer(repo, rag_config=config, embedder=embedder).retrieve("login token", top_k=1)

        self.assertEqual(embedder.document_calls, 1)
        self.assertEqual(embedder.query_calls, 2)
        self.assertTrue(list((self.tmp_path / "index").glob("*.json")))

    def test_dense_index_is_rebuilt_after_repository_change(self) -> None:
        repo = self.tmp_path / "repo"
        repo.mkdir()
        source = repo / "service.py"
        source.write_text("def login():\n    return True\n", encoding="utf-8")
        embedder = KeywordEmbeddingProvider()
        config = _config(self.tmp_path / "index")

        RepoIndexer(repo, rag_config=config, embedder=embedder).retrieve("login", top_k=1)
        source.write_text("def validate_token(token):\n    return bool(token)\n", encoding="utf-8")
        RepoIndexer(repo, rag_config=config, embedder=embedder).retrieve("authentication", top_k=1)

        self.assertEqual(embedder.document_calls, 2)

    def test_incremental_index_only_embeds_changed_chunks(self) -> None:
        repo = self.tmp_path / "repo"
        repo.mkdir()
        source = repo / "service.py"
        source.write_text(
            "def stable():\n    return 'stable'\n\n"
            "def changing():\n    return 'before'\n",
            encoding="utf-8",
        )
        embedder = KeywordEmbeddingProvider()
        config = _config(self.tmp_path / "index")
        first = DenseRepoRetriever(repo, config, embedder)
        first.retrieve("stable", [source], top_k=1)

        source.write_text(
            "def stable():\n    return 'stable'\n\n"
            "def changing():\n    return 'after'\n",
            encoding="utf-8",
        )
        second = DenseRepoRetriever(repo, config, embedder)
        second.retrieve("changing", [source], top_k=1)

        self.assertEqual(embedder.embedded_document_count, 3)
        self.assertEqual(second.last_index_stats.embedded_chunks, 1)
        self.assertEqual(second.last_index_stats.reused_chunks, 1)

    def test_retrieval_metrics_compute_recall_and_mrr(self) -> None:
        examples = [
            RetrievalExample("one", "repo", "query", ("target.py",)),
            RetrievalExample("two", "repo", "query", ("target.py",)),
        ]
        target = CodeChunk.create(
            path="target.py",
            language="python",
            symbol="target",
            symbol_type="function",
            start_line=1,
            end_line=2,
            content="def target():\n    pass",
        )
        other = CodeChunk.create(
            path="other.py",
            language="python",
            symbol="other",
            symbol_type="function",
            start_line=1,
            end_line=2,
            content="def other():\n    pass",
        )

        metrics = evaluate_rankings(
            examples,
            [
                [RetrievalResult(target, 1.0), RetrievalResult(other, 0.5)],
                [RetrievalResult(other, 1.0), RetrievalResult(target, 0.5)],
            ],
            ks=(1, 2),
        )

        self.assertEqual(metrics.recall_at_k[1], 0.5)
        self.assertEqual(metrics.recall_at_k[2], 1.0)
        self.assertEqual(metrics.hit_rate_at_k[1], 0.5)
        self.assertEqual(metrics.mrr, 0.75)

    def test_legacy_lexical_mode_remains_available(self) -> None:
        (self.tmp_path / "service.py").write_text("def unique_symbol():\n    return 1\n", encoding="utf-8")
        indexer = RepoIndexer(
            self.tmp_path,
            rag_config=_config(self.tmp_path / "index", mode="lexical"),
        )

        result = indexer.retrieve("unique_symbol", top_k=1)

        self.assertIn("service.py", result)
        self.assertIn("def unique_symbol", result)


if __name__ == "__main__":
    unittest.main()

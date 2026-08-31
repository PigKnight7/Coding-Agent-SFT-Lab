from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cc_agent.rag import RAGConfig, RetrievalResult
from cc_agent.repo_indexer import RepoIndexer


@dataclass(frozen=True)
class RetrievalExample:
    id: str
    repo: str
    query: str
    relevant_paths: tuple[str, ...]
    relevant_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalMetrics:
    examples: int
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float

    def render(self) -> str:
        lines = [f"Examples: {self.examples}", f"MRR: {self.mrr:.4f}"]
        for k in sorted(self.recall_at_k):
            lines.append(
                f"Recall@{k}: {self.recall_at_k[k]:.4f} | "
                f"HitRate@{k}: {self.hit_rate_at_k[k]:.4f}"
            )
        return "\n".join(lines)


def load_retrieval_examples(path: str | Path) -> list[RetrievalExample]:
    dataset_path = Path(path)
    examples: list[RetrievalExample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            relevant_paths = tuple(str(item) for item in payload["relevant_paths"])
            if not relevant_paths:
                raise ValueError("relevant_paths must not be empty")
            examples.append(
                RetrievalExample(
                    id=str(payload.get("id", f"line_{line_number}")),
                    repo=str(payload["repo"]),
                    query=str(payload["query"]),
                    relevant_paths=relevant_paths,
                    relevant_symbols=tuple(str(item) for item in payload.get("relevant_symbols", [])),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid retrieval example at line {line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"No retrieval examples found in {dataset_path}")
    return examples


def evaluate_rankings(
    examples: Sequence[RetrievalExample],
    rankings: Sequence[Sequence[RetrievalResult]],
    ks: Sequence[int] = (1, 3, 5),
) -> RetrievalMetrics:
    if len(examples) != len(rankings):
        raise ValueError("Each retrieval example must have one ranking")
    normalized_ks = sorted({int(k) for k in ks if int(k) > 0})
    if not normalized_ks:
        raise ValueError("At least one positive K value is required")

    recall_totals = {k: 0.0 for k in normalized_ks}
    hit_totals = {k: 0 for k in normalized_ks}
    reciprocal_rank_total = 0.0
    for example, ranking in zip(examples, rankings):
        relevant_paths = {_normalize_path(path) for path in example.relevant_paths}
        relevant_symbols = {symbol.lower() for symbol in example.relevant_symbols}
        relevant_ranks: list[int] = []
        for rank, result in enumerate(ranking, 1):
            path_matches = _normalize_path(result.chunk.path) in relevant_paths
            symbol_matches = not relevant_symbols or (result.chunk.symbol or "").lower() in relevant_symbols
            if path_matches and symbol_matches:
                relevant_ranks.append(rank)
        if relevant_ranks:
            reciprocal_rank_total += 1.0 / min(relevant_ranks)
        for k in normalized_ks:
            retrieved_paths = {
                _normalize_path(result.chunk.path)
                for result in ranking[:k]
                if not relevant_symbols or (result.chunk.symbol or "").lower() in relevant_symbols
            }
            relevant_retrieved = len(relevant_paths & retrieved_paths)
            recall_totals[k] += relevant_retrieved / len(relevant_paths)
            hit_totals[k] += int(relevant_retrieved > 0)

    count = len(examples)
    return RetrievalMetrics(
        examples=count,
        recall_at_k={k: recall_totals[k] / count for k in normalized_ks},
        hit_rate_at_k={k: hit_totals[k] / count for k in normalized_ks},
        mrr=reciprocal_rank_total / count,
    )


def evaluate_retrieval_dataset(
    dataset_path: str | Path,
    *,
    ks: Sequence[int] = (1, 3, 5),
    config: RAGConfig | None = None,
) -> RetrievalMetrics:
    path = Path(dataset_path).resolve()
    examples = load_retrieval_examples(path)
    normalized_ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not normalized_ks:
        raise ValueError("At least one positive K value is required")
    max_k = max(normalized_ks)
    rankings: list[list[RetrievalResult]] = []
    for example in examples:
        repo_path = Path(example.repo)
        if not repo_path.is_absolute():
            repo_path = path.parent / repo_path
        rankings.append(
            RepoIndexer(repo_path, rag_config=config).search_chunks(
                query=example.query,
                top_k=max_k,
            )
        )
    return evaluate_rankings(examples, rankings, normalized_ks)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")

"""FAISS-backed retrieval of evidence from the store policy PDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from src.rag.embeddings import OpenAIEmbedder, TextEmbedder
from src.rag.ingest import (
    POLICY_PDF_PATH,
    PolicyChunk,
    extract_policy_chunks,
)


class PolicyRetrievalError(ValueError):
    """Raised when embeddings cannot be represented by the policy index."""


@dataclass(frozen=True, slots=True)
class PolicySearchResult:
    """Retrieved policy evidence and its source metadata; no answer is generated."""

    chunk_id: str
    text: str
    section: str | None
    page_start: int
    page_end: int
    pages: tuple[int, ...]
    source: str
    score: float


def _embedding_matrix(
    embedder: TextEmbedder,
    texts: Sequence[str],
) -> np.ndarray:
    try:
        vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PolicyRetrievalError(f"Could not create policy embeddings: {exc}") from exc

    if vectors.ndim == 1 and len(texts) == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise PolicyRetrievalError(
            "The embedder must return one non-empty vector for each input text."
        )
    if not np.isfinite(vectors).all():
        raise PolicyRetrievalError("The embedder returned a non-finite vector value.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise PolicyRetrievalError("The embedder returned an empty (zero-length) vector.")
    faiss.normalize_L2(vectors)
    return vectors


class PolicyRetriever:
    """Build an in-memory FAISS index from section-aware policy chunks."""

    def __init__(self, chunks: Sequence[PolicyChunk], embedder: TextEmbedder) -> None:
        if not chunks:
            raise PolicyRetrievalError("Cannot build a policy index without chunks.")
        self.chunks = tuple(chunks)
        self.embedder = embedder
        document_vectors = _embedding_matrix(
            embedder,
            [chunk.text for chunk in self.chunks],
        )
        self.index = faiss.IndexFlatIP(document_vectors.shape[1])
        self.index.add(document_vectors)

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str | Path,
        embedder: TextEmbedder,
    ) -> PolicyRetriever:
        return cls(extract_policy_chunks(pdf_path), embedder)

    def search(self, query: str, k: int = 3) -> list[PolicySearchResult]:
        _validate_search(query, k)
        query_vector = _embedding_matrix(self.embedder, [query])
        if query_vector.shape[1] != self.index.d:
            raise PolicyRetrievalError(
                "The query embedding dimension does not match the policy index."
            )

        result_count = min(k, self.index.ntotal)
        scores, positions = self.index.search(query_vector, result_count)
        results: list[PolicySearchResult] = []
        for score, position in zip(scores[0], positions[0]):
            chunk = self.chunks[int(position)]
            # Retrieved policy text is evidence only and must be treated as
            # untrusted context by any future agent; this layer never executes it.
            results.append(
                PolicySearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    pages=chunk.pages,
                    source=chunk.source,
                    score=float(score),
                )
            )
        return results


def _validate_search(query: str, k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")


_DEFAULT_EMBEDDER: OpenAIEmbedder | None = None
_DEFAULT_RETRIEVER: PolicyRetriever | None = None
_DEFAULT_RETRIEVER_KEY: tuple[str, int, int] | None = None


def _default_retriever(pdf_path: str | Path) -> PolicyRetriever:
    global _DEFAULT_EMBEDDER, _DEFAULT_RETRIEVER, _DEFAULT_RETRIEVER_KEY

    path = Path(pdf_path)
    if not path.is_file():
        # Keep PDF errors consistent with the extraction layer.
        extract_policy_chunks(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if _DEFAULT_RETRIEVER is not None and _DEFAULT_RETRIEVER_KEY == key:
        return _DEFAULT_RETRIEVER

    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = OpenAIEmbedder.from_environment()
    _DEFAULT_RETRIEVER = PolicyRetriever.from_pdf(path, _DEFAULT_EMBEDDER)
    _DEFAULT_RETRIEVER_KEY = key
    return _DEFAULT_RETRIEVER


def search_policy(
    query: str,
    k: int = 3,
    *,
    pdf_path: str | Path = POLICY_PDF_PATH,
    embedder: TextEmbedder | None = None,
) -> list[PolicySearchResult]:
    """Return top-k policy evidence only; this function does not generate answers."""
    _validate_search(query, k)
    retriever = (
        _default_retriever(pdf_path)
        if embedder is None
        else PolicyRetriever.from_pdf(pdf_path, embedder)
    )
    return retriever.search(query, k)

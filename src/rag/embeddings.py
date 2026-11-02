"""Embedding client abstraction for policy-vector indexing and search."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol, Sequence

from dotenv import load_dotenv


class EmbeddingConfigurationError(ValueError):
    """Raised when embedding credentials or model configuration are unavailable."""


class TextEmbedder(Protocol):
    """Small interface that allows local deterministic embedders in tests."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector for each input text, preserving input order."""


@dataclass(slots=True)
class OpenAIEmbedder:
    """OpenAI-compatible embeddings client, configurable through environment vars."""

    client: object
    model: str
    base_url: str | None = None

    @classmethod
    def from_environment(cls) -> OpenAIEmbedder:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        if not api_key:
            raise EmbeddingConfigurationError(
                "OPENAI_API_KEY is required to create policy embeddings."
            )
        if not model:
            raise EmbeddingConfigurationError(
                "OPENAI_EMBEDDING_MODEL is required to create policy embeddings."
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "The openai package is required for policy embeddings."
            ) from exc
        return cls(client=client, model=model, base_url=base_url)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input texts must not be empty.")

        response = self.client.embeddings.create(  # type: ignore[attr-defined]
            model=self.model,
            input=list(texts),
        )
        items = list(response.data)
        indices = [getattr(item, "index", None) for item in items]
        if len(items) != len(texts):
            raise ValueError("The embeddings provider returned an unexpected result count.")
        if all(index is None for index in indices):
            # Some OpenAI-compatible endpoints omit indexes but preserve request order.
            ordered = items
        elif all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in indices
        ) and sorted(indices) == list(range(len(texts))):
            ordered = [item for _, item in sorted(zip(indices, items))]
        elif any(index is None for index in indices) and all(
            index is None or index == position
            for position, index in enumerate(indices)
        ):
            # Preserve the response array when every supplied index confirms its
            # position and only some index fields are absent.
            ordered = items
        else:
            raise ValueError(
                "The embeddings provider returned missing or invalid response indices."
            )
        return [list(item.embedding) for item in ordered]

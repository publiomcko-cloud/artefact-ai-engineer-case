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
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise ValueError("The embeddings provider returned an unexpected result count.")
        return [list(item.embedding) for item in ordered]

from types import SimpleNamespace

import pytest

from src.rag.embeddings import OpenAIEmbedder


def _embedder_with_response(data):
    embeddings = SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(data=data)
    )
    return OpenAIEmbedder(
        client=SimpleNamespace(embeddings=embeddings),
        model="local-test-model",
    )


def test_embedder_preserves_response_order_when_provider_omits_indices() -> None:
    embedder = _embedder_with_response(
        [
            SimpleNamespace(index=None, embedding=[1.0, 0.0]),
            SimpleNamespace(index=None, embedding=[0.0, 1.0]),
        ]
    )

    assert embedder.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embedder_sorts_responses_by_complete_indices() -> None:
    embedder = _embedder_with_response(
        [
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
    )

    assert embedder.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embedder_preserves_order_when_missing_indices_align_with_array_positions() -> None:
    embedder = _embedder_with_response(
        [
            SimpleNamespace(index=None, embedding=[1.0, 0.0]),
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
        ]
    )

    assert embedder.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embedder_rejects_mixed_response_index_metadata() -> None:
    embedder = _embedder_with_response(
        [
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=None, embedding=[1.0, 0.0]),
        ]
    )

    with pytest.raises(ValueError, match="missing or invalid response indices"):
        embedder.embed(["first", "second"])

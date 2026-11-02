from hashlib import sha256
import re
import unicodedata

import pytest
import pymupdf

from src.rag.ingest import (
    POLICY_PDF_PATH,
    POLICY_SOURCE,
    PolicyDocumentError,
    chunk_policy_pages,
    extract_policy_chunks,
    extract_policy_pages,
)
from src.rag.retriever import PolicySearchResult, search_policy


class DeterministicHashEmbedder:
    """Local feature-hash embedder used to exercise retrieval without an API."""

    dimensions = 4096

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = unicodedata.normalize("NFKD", text.casefold())
            normalized = "".join(
                char for char in normalized if not unicodedata.combining(char)
            )
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9]+", normalized):
                slot = int.from_bytes(sha256(token.encode("utf-8")).digest()[:4], "big")
                vector[slot % self.dimensions] += 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture(scope="module")
def deterministic_embedder() -> DeterministicHashEmbedder:
    return DeterministicHashEmbedder()


@pytest.fixture(scope="module")
def policy_pages():
    return extract_policy_pages(POLICY_PDF_PATH)


@pytest.fixture(scope="module")
def policy_chunks():
    return extract_policy_chunks(POLICY_PDF_PATH)


def test_real_policy_pdf_extracts_usable_text_and_page_numbers(policy_pages) -> None:
    assert len(policy_pages) == 8
    assert all(page.page_number == number for number, page in enumerate(policy_pages, 1))
    assert all(page.text.strip() for page in policy_pages)
    assert "MANUAL DE POLÍTICAS E PROCEDIMENTOS" in policy_pages[0].text
    assert "10. Disposições Finais" in policy_pages[7].text


def test_missing_policy_pdf_fails_clearly(tmp_path) -> None:
    with pytest.raises(PolicyDocumentError, match="Policy PDF not found"):
        extract_policy_pages(tmp_path / "missing.pdf")


def test_policy_pdf_without_text_fails_clearly(tmp_path) -> None:
    path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()

    with pytest.raises(PolicyDocumentError, match="contains no usable text"):
        extract_policy_pages(path)


def test_real_major_sections_are_discoverable(policy_chunks) -> None:
    expected_sections = {
        "1. Sobre a Empório da Música",
        "2. Horário de Funcionamento",
        "3. Formas de Pagamento",
        "4. Política de Trocas e Devoluções",
        "5. Política de Frete e Entregas",
        "6. Promoções e Descontos",
        "7. Atendimento via WhatsApp",
        "8. Garantia",
        "9. Privacidade e Proteção de Dados",
        "10. Disposições Finais",
    }

    assert expected_sections <= {chunk.section for chunk in policy_chunks}


def test_chunks_are_nonempty_and_retain_source_section_and_page_metadata(
    policy_chunks,
) -> None:
    assert policy_chunks
    assert all(chunk.text.strip() for chunk in policy_chunks)
    assert all(chunk.source == POLICY_SOURCE for chunk in policy_chunks)
    assert all(chunk.page_start <= chunk.page_end for chunk in policy_chunks)
    assert all(chunk.pages for chunk in policy_chunks)
    assert all(chunk.section is not None for chunk in policy_chunks if chunk.page_start > 1)

    shipping = next(
        chunk for chunk in policy_chunks
        if chunk.section == "5. Política de Frete e Entregas"
    )
    assert shipping.pages == (4, 5)
    assert shipping.page_start == 4
    assert shipping.page_end == 5


def test_chunk_ids_are_deterministic_for_the_same_pdf(policy_pages) -> None:
    first = chunk_policy_pages(policy_pages, source=POLICY_SOURCE)
    second = chunk_policy_pages(policy_pages, source=POLICY_SOURCE)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_empty_query_fails_clearly_without_creating_embeddings(
    deterministic_embedder,
) -> None:
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        search_policy("  ", embedder=deterministic_embedder)


def test_nonpositive_k_fails_clearly(deterministic_embedder) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        search_policy("formas de pagamento", k=0, embedder=deterministic_embedder)


def test_k_is_respected_and_results_are_structured_evidence(
    deterministic_embedder,
) -> None:
    results = search_policy(
        "formas pagamento PIX cartao boleto parcelamento",
        k=2,
        embedder=deterministic_embedder,
    )

    assert len(results) == 2
    assert all(isinstance(result, PolicySearchResult) for result in results)
    assert all(result.text.strip() for result in results)
    assert all(result.source == POLICY_SOURCE for result in results)
    assert all(result.pages and result.page_start > 0 for result in results)
    assert all(result.section is not None for result in results)
    assert all(isinstance(result.score, float) for result in results)


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        (
            "trocas devolucoes direito arrependimento devolucao online",
            "4. Política de Trocas e Devoluções",
        ),
        (
            "formas pagamento PIX cartao debito credito boleto parcelamento",
            "3. Formas de Pagamento",
        ),
        (
            "frete entregas Correios PAC SEDEX Jadlog rastreamento prazo",
            "5. Política de Frete e Entregas",
        ),
        (
            "garantia legal fabricante defeitos prazo certificado",
            "8. Garantia",
        ),
        (
            "privacidade dados pessoais LGPD exclusao consentimento",
            "9. Privacidade e Proteção de Dados",
        ),
    ],
)
def test_policy_queries_retrieve_the_relevant_section(
    deterministic_embedder,
    query: str,
    expected_section: str,
) -> None:
    results = search_policy(query, embedder=deterministic_embedder)

    assert results
    assert results[0].section == expected_section
    assert results[0].text
    assert results[0].source == POLICY_SOURCE
    assert results[0].page_start > 0

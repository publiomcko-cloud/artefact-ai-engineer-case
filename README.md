# Artefact AI Engineer Technical Case

This repository contains the Artefact AI Engineer technical case MVP for Empório da Música, backed by the supplied store datasets and policy manual.

## Current scope

Phase 1 implements deterministic product and promotion behavior. Phase 2 adds section-aware extraction and FAISS retrieval for the store-policy PDF. Phase 3 adds deterministic, customer-verified order lookup. Policy retrieval returns evidence only; it does not generate answers. The provided source data remains unchanged.

## Environment

Use Python 3.11 or 3.12. Create and activate a virtual environment, then install the bootstrap dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `OPENAI_EMBEDDING_MODEL` before using live policy retrieval. Set `OPENAI_BASE_URL` only when using an OpenAI-compatible endpoint. The tests use a deterministic local embedder and require no API key or network access. Do not commit `.env` or API keys.

## Tests

Run the complete automated suite with:

```bash
pytest -q
```

## Project structure

- `data/` — six source CSV datasets and the store policies PDF.
- `src/` — validated store data, deterministic product/promotion/order tools, and policy retrieval; agent modules remain future work.
- `tests/` — focused deterministic product, promotion, order privacy, and policy retrieval tests.
- `evals/` — placeholder for the optional behavioral evaluation suite.
- `examples/` — placeholder for conversations to be captured after the application exists.
- `docs/MVP_SPEC.md` — scope, architecture, repository layout, and implementation plan.

## Next implementation phase

The next phase is single-agent LangGraph integration described in [docs/MVP_SPEC.md](docs/MVP_SPEC.md). It is not part of this implementation.

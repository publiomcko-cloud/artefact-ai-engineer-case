# Artefact AI Engineer Technical Case

This repository contains the Artefact AI Engineer technical case MVP for Empório da Música, backed by the supplied store datasets and policy manual.

## Current scope

Phase 1 implements the deterministic data loader, product search and lookup, sellability checks, promotion lookup, and offer pricing. Phase 2 adds section-aware extraction and FAISS retrieval for the store-policy PDF. Policy retrieval returns evidence only; it does not generate answers. LLM integration, order behavior, and the chat interface are later phases and are not implemented here. The provided source data remains unchanged.

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
- `src/` — placeholders for the planned data, agent, tool, and retrieval modules.
- `tests/` — placeholders for the planned deterministic business-rule tests.
- `evals/` — placeholder for the optional behavioral evaluation suite.
- `examples/` — placeholder for conversations to be captured after the application exists.
- `docs/MVP_SPEC.md` — scope, architecture, repository layout, and implementation plan.

## Next implementation phase

The next phase is the single-agent integration described in [docs/MVP_SPEC.md](docs/MVP_SPEC.md). It is not part of this implementation.

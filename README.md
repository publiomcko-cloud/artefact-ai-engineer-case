# Artefact AI Engineer Technical Case

This repository contains the Artefact AI Engineer technical case MVP for Empório da Música, backed by the supplied store datasets and policy manual.

## Current scope

Phase 1 implements the deterministic product and promotion layer. Phase 2 adds policy retrieval from the store-policy PDF. Phase 3 adds verified deterministic order lookup. Phase 4 adds single-agent LangGraph orchestration over those tools. Policy retrieval returns evidence only; it does not generate answers. The provided source data remains unchanged.

## Environment

Use Python 3.11 or 3.12. Create and activate a virtual environment, then install the bootstrap dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_EMBEDDING_MODEL` for the chat agent and live policy retrieval. Set `OPENAI_BASE_URL` only when using an OpenAI-compatible endpoint. The tests use injected local models/retrievers and require no API key or network access. Do not commit `.env` or API keys.

## Tests

Run the complete automated suite with:

```bash
pytest -q
```

## Project structure

- `data/` — six source CSV datasets and the store policies PDF.
- `src/` — validated store data, deterministic tools, policy retrieval, and single-agent orchestration.
- `tests/` — focused deterministic product, promotion, order privacy, and policy retrieval tests.
- `evals/` — placeholder for the optional behavioral evaluation suite.
- `examples/` — placeholder for conversations to be captured after the application exists.
- `docs/MVP_SPEC.md` — scope, architecture, repository layout, and implementation plan.

## Next implementation phase

The Streamlit chat interface remains the next phase. It is not part of this implementation.

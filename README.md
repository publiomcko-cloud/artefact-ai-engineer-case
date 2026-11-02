# Artefact AI Engineer Technical Case

This repository contains the Artefact AI Engineer technical case MVP for Empório da Música, backed by the supplied store datasets and policy manual.

## Current scope

Phase 1 implements the deterministic data loader, product search and lookup, sellability checks, promotion lookup, and offer pricing. The provided source data remains unchanged. LLM integration, policy retrieval, order behavior, and the chat interface are planned later phases and are not implemented here.

## Environment

Use Python 3.11 or 3.12. Create and activate a virtual environment, then install the bootstrap dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` when a later implementation requires model configuration. Do not commit `.env` or API keys.

## Project structure

- `data/` — six source CSV datasets and the store policies PDF.
- `src/` — placeholders for the planned data, agent, tool, and retrieval modules.
- `tests/` — placeholders for the planned deterministic business-rule tests.
- `evals/` — placeholder for the optional behavioral evaluation suite.
- `examples/` — placeholder for conversations to be captured after the application exists.
- `docs/MVP_SPEC.md` — scope, architecture, repository layout, and implementation plan.

## Next implementation phase

The next phase is policy retrieval from the provided store manual. Continue to follow the implementation order and scope in [docs/MVP_SPEC.md](docs/MVP_SPEC.md).

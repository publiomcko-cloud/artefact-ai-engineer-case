# Artefact AI Engineer Technical Case

This repository contains the initial project bootstrap for the Artefact AI Engineer technical case. The case describes a customer-service assistant for Empório da Música, backed by the provided store datasets and policy manual.

## Current scope

This initial commit establishes the repository structure, keeps the supplied runtime data unchanged, and records the planned MVP in [docs/MVP_SPEC.md](docs/MVP_SPEC.md). Application features have not been implemented yet.

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

Begin with deterministic loading and validation of the structured data, followed by product, promotion, and order rules. Keep LLM integration, policy retrieval, and the user interface for their later phases as described in the specification.

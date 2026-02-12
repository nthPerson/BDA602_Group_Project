# CLAUDE.md — Project Context for AI Assistants

> This file provides context for Claude (and other AI assistants) when working on this codebase. Update it as each development stage is completed.

---

## Project Summary

**Agentic Academic Paper Citation Recommendation System** — a multi-agent RAG pipeline that, given a paragraph from an AI/ML research paper, recommends relevant citation candidates with justifications.

- **Course:** BDA602 — Group Project
- **Timeline:** ~8 weeks
- **Repository:** `nthPerson/BDA602_Group_Project`, branch `main`

## Key Documents

| Document | Purpose |
|---|---|
| `SYSTEM_OVERVIEW.md` | Authoritative technical design — architecture, data models, agent specs, dependency stack |
| `DEVELOPMENT_ROADMAP.md` | 10-stage incremental plan with acceptance criteria and test inventories |
| `docs/SETUP_GUIDE.md` | Teammate-friendly setup instructions (update as necessary) |
| `docs/stages/STAGE_*.md` | Per-stage implementation details, test lists, and teammate explanations |

**Always read `SYSTEM_OVERVIEW.md` before making architectural decisions.** It is the source of truth for data models, agent I/O contracts, and technology choices.

---

## Architecture Overview

```
User text → Agent 1 (Intent) → Agent 2 (Dense Retrieval) → Agent 3 (Citation Expansion)
         → Agent 4 (Rerank + Ground) → Agent 5 (Synthesis) → Ranked Recommendations
```

- **Orchestration:** LangGraph `StateGraph` with linear edges and graceful degradation
- **State:** Shared `AgentState` TypedDict passed between all nodes
- **Pattern:** Retrieve → Expand → Rerank → Ground → Synthesize

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python ≥3.11 |
| Orchestration | LangGraph ≥0.2.0 |
| LLM | OpenAI gpt-4o-mini via `instructor` for structured output |
| Embeddings | `sentence-transformers` — BAAI/bge-base-en-v1.5 (768-dim) |
| Reranking | BAAI/bge-reranker-base (cross-encoder) |
| Vector store | Qdrant v1.12+ (Docker) |
| Metadata store | SQLite |
| Data source | OpenAlex via `pyalex` (CS.AI/LG/CL/CV, 2018–2025) |
| UI | Streamlit ≥1.39 |
| Testing | pytest, pytest-asyncio |
| Linting | ruff |
| Validation | Pydantic v2, pydantic-settings |

## Project Structure

```
src/
├── config.py                    # Pydantic Settings (loads from .env)
├── data/
│   ├── models.py                # Paper, CitationEdge, ScoredPaper, RankedPaper, GroundedPaper, Recommendation, QueryAnalysis, EvalSample
│   ├── db.py                    # SQLite CRUD — papers, citation edges, queries
│   ├── openalex_client.py       # OpenAlex API wrapper with pagination + rate limiting
│   └── corpus_builder.py        # Orchestrates corpus collection
├── indexing/
│   ├── embedder.py              # BGE embedding pipeline (instruction-prefixed)
│   └── qdrant_store.py          # Qdrant collection management + search
├── agents/
│   ├── query_agent.py           # Agent 1: intent classification + keyword extraction (LLM)
│   ├── retrieval_agent.py       # Agent 2: dense retrieval from Qdrant
│   ├── expansion_agent.py       # Agent 3: citation graph BFS (depth-1, SQLite)
│   ├── reranking_agent.py       # Agent 4: cross-encoder rerank + LLM grounding
│   └── synthesis_agent.py       # Agent 5: confidence filter, composite scoring, formatting
├── orchestration/
│   ├── state.py                 # AgentState TypedDict
│   └── graph.py                 # LangGraph StateGraph definition
└── evaluation/
    ├── metrics.py               # Recall@K, MRR, MAP
    ├── dataset.py               # Eval dataset construction from corpus
    ├── baselines.py             # BM25 baseline
    └── runner.py                # Evaluation loop

app/
└── streamlit_app.py             # Streamlit UI

scripts/
├── build_corpus.py              # CLI: collect papers from OpenAlex
├── index_corpus.py              # CLI: embed + index into Qdrant
├── build_eval_dataset.py        # CLI: construct eval dataset
└── run_evaluation.py            # CLI: run evaluation suite

tests/
├── conftest.py                  # Shared fixtures (tmp SQLite, mock clients)
├── test_data/                   # Models + DB tests
├── test_agents/                 # Per-agent unit tests (mocked dependencies)
├── test_indexing/               # Qdrant integration tests
├── test_orchestration/          # LangGraph pipeline tests
├── test_evaluation/             # Metric + dataset tests
├── test_integration/            # End-to-end pipeline tests
└── test_app/                    # Streamlit smoke tests
```

---

## Development Stage Tracker

Update this section as stages are completed. Check the box and fill in the date.

- [x] **Stage 0** — Project Scaffolding (pyproject.toml, docker-compose, project dirs) — ✅ 2026-02-10
- [x] **Stage 1** — Data Models & Database Layer (models.py, db.py, config.py) — ✅ 2026-02-10
- [x] **Stage 2** — Data Acquisition (openalex_client.py, corpus_builder.py) — ✅ 2026-02-10
- [x] **Stage 3** — Embedding & Indexing (embedder.py, qdrant_store.py) — ✅ 2026-02-11
- [x] **Stage 4** — Agents 1 & 2 (query_agent.py, retrieval_agent.py) — ✅ 2026-02-11
- [ ] **Stage 5** — Agent 3: Citation Expansion (expansion_agent.py)
- [ ] **Stage 6** — Agent 4: Reranking & Grounding (reranking_agent.py)
- [ ] **Stage 7** — Agent 5 + Orchestration (synthesis_agent.py, graph.py)
- [ ] **Stage 8** — Evaluation Framework (metrics.py, dataset.py, runner.py)
- [ ] **Stage 9** — Streamlit UI (streamlit_app.py)

---

## Coding Conventions

### Style

- **Formatter/Linter:** `ruff` — run `ruff check src/ tests/` and `ruff format src/ tests/` before committing
- **Type hints:** Required on all function signatures. Use `| None` instead of `Optional[]`.
- **Docstrings:** Google style. Required on public functions and classes.
- **Imports:** Group as stdlib → third-party → local, separated by blank lines. Use absolute imports from `src.`.

### Data Models

- All domain objects are **dataclasses** defined in `src/data/models.py`.
- Never use raw dicts for structured data passed between agents. Always use the typed models.
- Agent I/O contracts are defined in `SYSTEM_OVERVIEW.md` §5 — respect them.

### Agent Implementation Pattern

Each agent follows this pattern:

```python
# src/agents/<agent_name>.py
from dataclasses import dataclass

@dataclass
class AgentNameConfig:
    """Configuration for this agent."""
    param: type = default

class AgentName:
    """One-line description."""

    def __init__(self, config: AgentNameConfig, ...dependencies):
        ...

    def run(self, state: AgentState) -> AgentState:
        """Execute this agent's step. Reads from and writes to state."""
        ...
```

- Agents receive the full `AgentState` and return an updated copy.
- Agents must not have side effects beyond their designated state keys.
- Dependencies (DB, Qdrant client, LLM client) are injected via `__init__`, never imported globally.

### Testing

- **Test markers:** Use `@pytest.mark.integration` for tests requiring external services (Qdrant, OpenAI). Use `@pytest.mark.slow` for tests taking >10 seconds.
- **Mocking:** Unit tests mock all external dependencies. Use `unittest.mock.patch` or `pytest` fixtures.
- **Fixtures:** Shared fixtures go in `tests/conftest.py` (e.g., `tmp_db`, `sample_papers`, `mock_qdrant_client`).
- **Run fast tests:** `pytest -m "not integration and not slow"`
- **Run all tests:** `pytest -v`

### Git Workflow

- Work on feature branches: `stage-N/description`
- Write descriptive commit messages: `stage 1: implement Paper and CitationEdge dataclasses`
- PR into `main` when the stage's acceptance criteria are met

---

## Environment Setup (Quick Reference)

```bash
# Clone and setup
git clone https://github.com/nthPerson/BDA602_Group_Project.git
cd BDA602_Group_Project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Environment variables
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum

# Start Qdrant
docker compose up -d

# Verify
pytest
ruff check src/
```

See `docs/SETUP_GUIDE.md` for the full guide.

---

## Common Operations

```bash
# Run all fast tests
pytest -m "not integration and not slow" -v

# Run a specific stage's tests
pytest tests/test_data/ -v                    # Stage 1
pytest tests/test_agents/test_query_agent.py -v  # Stage 4

# Format and lint
ruff format src/ tests/
ruff check src/ tests/ --fix

# Build corpus (Stage 2)
python scripts/build_corpus.py

# Index into Qdrant (Stage 3)
python scripts/index_corpus.py

# Run evaluation (Stage 8)
python scripts/run_evaluation.py --mode full

# Launch UI (Stage 9)
streamlit run app/streamlit_app.py
```

---

## Known Constraints & Gotchas

- **OpenAI API key required** for Agents 1 and 4 (intent analysis and grounding). Unit tests mock this, but integration tests need a real key in `.env`.
- **Qdrant must be running** (`docker compose up -d`) for any indexing or retrieval tests.
- **First embedding run is slow** — the BGE model (~400 MB) and cross-encoder (~400 MB) download on first use.
- **SQLite is single-writer** — don't run parallel corpus builds. Sequential is fine.
- **Corpus build is a one-time operation** — takes ~30–60 min due to OpenAlex rate limits. Once built, `data/papers.db` is reused.
- **The `data/` directory is gitignored.** Each developer builds their own local corpus.
- **Evaluation costs money** — ~$1–2 for 200 samples via OpenAI. Run BM25 baseline (free) first to verify the framework works.

---

## Debugging Tips

- **Inspect agent state:** The `AgentState` dict is fully readable at any pipeline step. Print or log `state.keys()` to see what each agent produced.
- **Test agents in isolation:** Each agent has a standalone `run()` method. You can invoke any agent directly with a hand-crafted `AgentState` dict.
- **Qdrant dashboard:** Visit `http://localhost:6333/dashboard` to inspect collections, point counts, and run manual queries.
- **SQLite inspection:** `sqlite3 data/papers.db ".tables"` and `sqlite3 data/papers.db "SELECT COUNT(*) FROM papers"` for quick checks.
- **LangGraph tracing:** LangGraph supports LangSmith tracing. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to get full pipeline traces (optional, not required).

---

## What NOT to Do

- **Don't skip the typed models.** Raw dicts between agents create untraceable bugs.
- **Don't use `gpt-4o` or `gpt-4-turbo`.** We use `gpt-4o-mini` — it's 10x cheaper and sufficient for this task.
- **Don't recurse the citation graph.** Depth-1 BFS only. Deeper traversal explodes the candidate set.
- **Don't add new dependencies without updating `pyproject.toml`.** Run `pip install -e ".[dev]"` after adding.
- **Don't commit data files.** The `data/` directory is gitignored. Corpus and eval results are built locally.
- **Don't run the full evaluation in a loop.** Each run costs ~$1–2 in API calls.

---

*Last updated: 2026-02-10 — Initial version (pre-implementation). Update this file as each stage is completed with any new conventions, gotchas, or patterns discovered during development.*

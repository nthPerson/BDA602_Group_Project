# Development Roadmap

> **How to use this document:** This is the master development plan for the Agentic Academic Paper Citation Recommendation System. It breaks the project into 10 incremental stages, each with clear deliverables, acceptance criteria, and tests. Each stage has a dedicated document in [`docs/stages/`](docs/stages/) with implementation details, test instructions, and notes for teammates. Before starting any work, read the [Setup Guide](docs/SETUP_GUIDE.md) to get the project running on your machine.

---

## Roadmap Summary

| Stage | Name | Depends On | Key Deliverable | Testable? |
|---|---|---|---|---|
| **0** | [Project Scaffolding](#stage-0-project-scaffolding) | — | Installable Python package, Docker services, CI skeleton | Yes (`pip install -e .` succeeds, `pytest` runs) |
| **1** | [Data Models & Database Layer](#stage-1-data-models--database-layer) | 0 | `Paper`/`CitationEdge` dataclasses, SQLite CRUD module | Yes (14+ unit tests) |
| **2** | [Data Acquisition](#stage-2-data-acquisition) | 1 | ~15k–25k papers in SQLite with citation edges | Yes (corpus integrity checks) |
| **3** | [Embedding & Indexing](#stage-3-embedding--indexing) | 1, 2 | Papers embedded and indexed in Qdrant | Yes (similarity search smoke tests) |
| **4** | [Agents 1 & 2 — Query Analysis + Retrieval](#stage-4-agents-1--2--query-analysis--primary-retrieval) | 3 | First working "text in → papers out" loop | Yes (retrieval sanity tests) |
| **5** | [Agent 3 — Citation Expansion](#stage-5-agent-3--citation-expansion) | 4 | Citation graph traversal augmenting retrieval | Yes (expansion logic unit tests) |
| **6** | [Agent 4 — Reranking & Grounding](#stage-6-agent-4--reranking--grounding) | 5 | Cross-encoder reranking + LLM justifications | Yes (reranking + grounding tests) |
| **7** | [Agent 5 + Orchestration](#stage-7-agent-5--orchestration) | 6 | Full LangGraph pipeline, end-to-end | Yes (pipeline integration test) |
| **8** | [Evaluation Framework](#stage-8-evaluation-framework) | 7 | Evaluation dataset, metrics, baseline + ablation results | Yes (metric unit tests, evaluation run) |
| **9** | [Streamlit UI](#stage-9-streamlit-ui) | 7 | Interactive web app for the pipeline | Partially (manual + smoke test) |

A final **polish/report phase** follows after Stage 9 but is not a formal stage — it covers documentation cleanup, demo preparation, and the written report.

### Dependency Graph

```
Stage 0: Scaffolding
   │
   ▼
Stage 1: Data Models + DB
   │
   ├──────────────────┐
   ▼                  ▼
Stage 2: Acquisition  (parallel-ready with Stage 1 tests)
   │
   ▼
Stage 3: Embedding + Indexing
   │
   ▼
Stage 4: Agents 1 & 2 (Query + Retrieval)
   │
   ▼
Stage 5: Agent 3 (Expansion)
   │
   ▼
Stage 6: Agent 4 (Reranking + Grounding)
   │
   ▼
Stage 7: Agent 5 + LangGraph Orchestration
   │
   ├─────────────────────┐
   ▼                     ▼
Stage 8: Evaluation    Stage 9: Streamlit UI
                         (these two can run in parallel)
```

---

## Stage Details

### Stage 0: Project Scaffolding

**Goal:** A clean, installable project skeleton that every team member can clone and run within 15 minutes.

**Deliverables:**
- Directory structure matching SYSTEM_OVERVIEW.md §9
- `pyproject.toml` with all dependencies (editable install via `pip install -e ".[dev]"`)
- `docker-compose.yml` for Qdrant
- `.env.example` with required environment variables
- `.gitignore` covering Python, data files, IDE configs
- Empty `conftest.py` and a single passing `test_smoke.py`
- `ruff` configuration for consistent formatting
- `docs/SETUP_GUIDE.md` completed

**Acceptance Criteria:**
1. Fresh clone → follow `SETUP_GUIDE.md` → `pip install -e ".[dev]"` succeeds
2. `docker compose up -d` starts Qdrant without errors
3. `pytest` runs and passes (at least the smoke test)
4. `ruff check src/` passes with no violations

**Tests:**
- `tests/test_smoke.py` — verifies imports work, config loads from env

**Stage document:** [`docs/stages/STAGE_0_PROJECT_SCAFFOLDING.md`](docs/stages/STAGE_0_PROJECT_SCAFFOLDING.md)

---

### Stage 1: Data Models & Database Layer

**Goal:** All data types and the SQLite persistence layer are implemented and fully tested.

**Deliverables:**
- `src/data/models.py` — `Paper`, `CitationEdge`, `EvalSample`, `ScoredPaper`, `RankedPaper`, `GroundedPaper`, `Recommendation` dataclasses
- `src/data/db.py` — SQLite module: create tables, insert/upsert papers, insert citation edges, query papers by ID/filter, query citation graph (references, cited-by)
- `src/config.py` — Pydantic `Settings` class (DB path, Qdrant host/port, OpenAI key, model names, etc.)

**Acceptance Criteria:**
1. All data model classes are importable and have correct type hints
2. SQLite module can create a fresh database, insert 100 test papers, and query them back
3. Citation graph queries return correct results for known test edges
4. Config loads from `.env` file and environment variables

**Tests (14+ cases):**
- `tests/test_data/test_models.py` — validate dataclass construction, serialization, edge cases (missing optional fields)
- `tests/test_data/test_db.py` — insert papers, query by ID, query by year range, insert citation edges, get references for a paper, get cited-by for a paper, handle duplicates, handle missing papers, test index performance on 1k records
- `tests/test_config.py` — config loads defaults, config loads from env, required fields raise errors when missing

**Stage document:** [`docs/stages/STAGE_1_DATA_MODELS_AND_DB.md`](docs/stages/STAGE_1_DATA_MODELS_AND_DB.md)

---

### Stage 2: Data Acquisition

**Goal:** A complete, static paper corpus is collected from OpenAlex and stored in SQLite.

**Deliverables:**
- `src/data/openalex_client.py` — wrapper around `pyalex` for querying AI/ML papers with pagination, rate-limit handling, and local caching of raw API responses
- `src/data/corpus_builder.py` — orchestrator that calls the client, transforms results into `Paper` objects, extracts citation edges, and inserts everything into SQLite
- `scripts/build_corpus.py` — CLI entry point for corpus building (with progress reporting)
- A populated SQLite database at `data/papers.db` (gitignored, but reproducible by re-running the script)

**Acceptance Criteria:**
1. `scripts/build_corpus.py` completes without errors and reports paper count
2. SQLite database contains ≥10,000 papers (target: 15k–25k)
3. ≥80% of papers in the corpus have at least 1 reference edge in `citation_edges`
4. Paper records have non-empty titles and abstracts

**Tests:**
- `tests/test_data/test_openalex_client.py` — test API query construction, test response parsing with a saved fixture (no live API call in CI), test pagination logic
- `scripts/validate_corpus.py` — standalone script that prints corpus statistics and runs integrity checks (paper count, edge count, null check, year distribution)

**Stage document:** [`docs/stages/STAGE_2_DATA_ACQUISITION.md`](docs/stages/STAGE_2_DATA_ACQUISITION.md)

---

### Stage 3: Embedding & Indexing

**Goal:** All papers are embedded and indexed in Qdrant, and basic similarity search works.

**Deliverables:**
- `src/indexing/embedder.py` — embedding pipeline using `sentence-transformers` (BGE-base), batch encoding, progress bar
- `src/indexing/qdrant_store.py` — Qdrant collection management: create collection, upsert points with payloads, search, delete collection
- `scripts/index_corpus.py` — CLI entry point: reads papers from SQLite, embeds them, upserts into Qdrant

**Acceptance Criteria:**
1. `scripts/index_corpus.py` completes and reports the number of vectors indexed
2. Qdrant collection contains the same number of points as papers in SQLite
3. A similarity search for "transformer attention mechanism" returns papers about transformers in the top 10
4. Payload-filtered search (e.g., `year >= 2020`) returns only papers matching the filter

**Tests:**
- `tests/test_indexing/test_embedder.py` — verify embedding dimensions (768), verify normalization (L2 norm ≈ 1.0), verify batch vs. single produces same results
- `tests/test_indexing/test_qdrant_store.py` — create collection, upsert 10 test vectors, search, verify top result, filtered search, delete collection (runs against a live Qdrant in Docker — requires `docker compose up`)
- A manual "sanity search" script (`scripts/test_search.py`) that lets you type a query and see top-5 results interactively

**Stage document:** [`docs/stages/STAGE_3_EMBEDDING_AND_INDEXING.md`](docs/stages/STAGE_3_EMBEDDING_AND_INDEXING.md)

---

### Stage 4: Agents 1 & 2 — Query Analysis + Primary Retrieval

**Goal:** The first two agents are implemented and wired together, producing the system's first working "text in → papers out" capability.

**Deliverables:**
- `src/agents/query_agent.py` — Agent 1: LLM-based intent analysis with `instructor`, fallback to keyword extraction
- `src/agents/retrieval_agent.py` — Agent 2: dense retrieval from Qdrant with optional metadata filters
- `src/orchestration/state.py` — `PipelineState` / `AgentState` TypedDict definitions

**Acceptance Criteria:**
1. Agent 1 accepts a paragraph of text and returns a valid `QueryAnalysis` (topic keywords, citation intent, expanded query)
2. Agent 1 fallback produces reasonable keywords when OpenAI is unreachable (mock test)
3. Agent 2 accepts a `QueryAnalysis` and returns ≤30 `ScoredPaper` results from Qdrant
4. Manual test: paste an abstract about "federated learning" → Agent 1 extracts reasonable keywords → Agent 2 returns papers about federated learning

**Tests:**
- `tests/test_agents/test_query_agent.py`:
  - Test with mocked OpenAI returning valid structured JSON → verify parsing
  - Test with mocked OpenAI returning invalid JSON → verify retry and fallback
  - Test fallback keyword extraction produces non-empty keywords
  - Test `CitationIntent` enum values are handled correctly
- `tests/test_agents/test_retrieval_agent.py`:
  - Test with mocked Qdrant returning known results → verify `ScoredPaper` mapping
  - Test metadata filter construction (year range)
  - Test empty result handling
  - Integration test with live Qdrant + indexed corpus: query about "graph neural networks" → verify top results are relevant (title keyword check)

**Stage document:** [`docs/stages/STAGE_4_AGENTS_1_AND_2.md`](docs/stages/STAGE_4_AGENTS_1_AND_2.md)

---

### Stage 5: Agent 3 — Citation Expansion

**Goal:** Citation graph traversal works, expanding the candidate set beyond what dense retrieval alone provides.

**Deliverables:**
- `src/agents/expansion_agent.py` — Agent 3: depth-1 BFS over citation graph, deduplication, prioritization heuristic, size limiting

**Acceptance Criteria:**
1. Given 10 seed papers, Agent 3 returns the originals PLUS additional papers from the citation graph
2. No duplicates in the expanded set
3. Expansion respects the `max_expansion` limit
4. Papers appearing in multiple seed papers' citation lists are prioritized

**Tests:**
- `tests/test_agents/test_expansion_agent.py`:
  - Test with a synthetic citation graph (10 papers, known edges): verify expansion finds expected neighbors
  - Test deduplication: seed papers do not appear twice in output
  - Test `max_expansion` limit is enforced
  - Test with a paper that has no citation edges → expansion returns only the originals
  - Test prioritization: a paper cited by 3 seed papers appears before one cited by 1
  - Integration test: run Agents 1→2→3 in sequence, verify expanded set is larger than retrieval set

**Stage document:** [`docs/stages/STAGE_5_CITATION_EXPANSION.md`](docs/stages/STAGE_5_CITATION_EXPANSION.md)

---

### Stage 6: Agent 4 — Reranking & Grounding

**Goal:** The combined pool of dense retrieval + expansion candidates is reranked by a cross-encoder, and the top results receive LLM-generated justifications.

**Deliverables:**
- `src/agents/reranking_agent.py` — contains both `RerankingStage` (cross-encoder) and `GroundingStage` (LLM justification), composed together as Agent 4

**Acceptance Criteria:**
1. Cross-encoder reranking re-orders candidates (the output order differs from the cosine similarity order)
2. Top-5 papers receive grounding output: a supporting snippet and a justification
3. The grounding justification references only content from the candidate paper's abstract (not hallucinated content)
4. The full Agent 4 pipeline completes in <10 seconds for 50 candidates on CPU

**Tests:**
- `tests/test_agents/test_reranking_agent.py`:
  - **Reranking tests:**
    - Test with mocked cross-encoder scores → verify output is sorted by rerank score
    - Test `top_k` parameter is respected
    - Test with all-identical scores → verify stable ordering
  - **Grounding tests:**
    - Test with mocked LLM → verify `GroundedPaper` fields are populated
    - Test with mocked LLM returning invalid response → verify retry behavior
    - Test that grounding prompt includes user text AND paper abstract
  - **Integration test:**
    - Run Agents 1→2→3→4 with a known query → verify that the top reranked result is more specific to the query than the 30th result from Agent 2

**Stage document:** [`docs/stages/STAGE_6_RERANKING_AND_GROUNDING.md`](docs/stages/STAGE_6_RERANKING_AND_GROUNDING.md)

---

### Stage 7: Agent 5 + Orchestration

**Goal:** All five agents are wired into a LangGraph state machine, producing a complete end-to-end pipeline.

**Deliverables:**
- `src/agents/synthesis_agent.py` — Agent 5: confidence filtering, composite scoring, final `Recommendation` formatting
- `src/orchestration/graph.py` — LangGraph `StateGraph` definition with all 5 nodes, linear edges, error handling, retry hooks, and observability logging
- `src/orchestration/state.py` — finalized `AgentState` TypedDict

**Acceptance Criteria:**
1. Calling `app.invoke({"user_text": "..."})` runs all 5 agents in sequence
2. The final state contains a non-empty `final_recommendations` list
3. Each `Recommendation` has all required fields (rank, paper_id, title, justification, snippet, confidence)
4. If any agent fails, the pipeline logs the error and degrades gracefully (e.g., no expansion → pipeline continues without expansion)
5. `state["metadata"]` contains timing information for each agent

**Tests:**
- `tests/test_agents/test_synthesis_agent.py`:
  - Test confidence filtering (papers below threshold are excluded)
  - Test composite score sorting (rerank_score * confidence)
  - Test output formatting (rank numbering, required fields)
- `tests/test_orchestration/test_graph.py`:
  - Test full pipeline with mocked agents → verify state transitions
  - Test error injection: mock Agent 3 to raise an exception → verify pipeline still produces results (from Agents 1+2 only)
  - Test metadata/timing is recorded
- **End-to-end integration test** (requires Qdrant + OpenAI API key):
  - `tests/test_integration/test_pipeline_e2e.py`: run the full pipeline with a real query, verify structured output

**Stage document:** [`docs/stages/STAGE_7_SYNTHESIS_AND_ORCHESTRATION.md`](docs/stages/STAGE_7_SYNTHESIS_AND_ORCHESTRATION.md)

---

### Stage 8: Evaluation Framework

**Goal:** Rigorous, reproducible evaluation with quantitative metrics, baselines, and ablation results.

**Deliverables:**
- `src/evaluation/metrics.py` — `compute_recall_at_k()`, `compute_mrr()`, `compute_map()` implementations
- `src/evaluation/dataset.py` — `build_eval_dataset()`: selects papers with ≥5 in-corpus references, constructs `EvalSample` objects
- `src/evaluation/runner.py` — `evaluate_pipeline()`: runs the pipeline on each eval sample, computes aggregate metrics
- `scripts/build_eval_dataset.py` — CLI to build and save the eval dataset
- `scripts/run_evaluation.py` — CLI to run full evaluation and print results table
- BM25 baseline implementation (in `src/evaluation/baselines.py` or inline)

**Acceptance Criteria:**
1. Evaluation dataset contains ≥100 samples (target: 200–500)
2. Metric implementations are verified against hand-calculated examples
3. BM25 baseline evaluation completes and produces Recall@5/10/20, MRR, MAP
4. Full pipeline evaluation completes and produces the same metrics
5. Results are saved to `data/evaluation_results.json`

**Tests:**
- `tests/test_evaluation/test_metrics.py`:
  - Test `compute_recall_at_k` with known inputs:
    - Perfect recall → 1.0
    - No recall → 0.0
    - Partial recall → correct fraction
  - Test `compute_mrr` with known rankings
  - Test `compute_map` with known rankings
  - Test edge cases: empty ground truth, empty recommendations
- `tests/test_evaluation/test_dataset.py`:
  - Test dataset builder selects papers with sufficient in-corpus references
  - Test that query paper is excluded from ground truth

**Stage document:** [`docs/stages/STAGE_8_EVALUATION.md`](docs/stages/STAGE_8_EVALUATION.md)

---

### Stage 9: Streamlit UI

**Goal:** An interactive web application that exposes the full pipeline to users.

**Deliverables:**
- `app/streamlit_app.py` — complete Streamlit app with:
  - Text input area
  - "Find Citations" button
  - Pipeline status indicators (per-agent progress)
  - Ranked recommendation cards with expandable justifications/snippets
  - Sidebar with advanced options (year range, result count)
  - Debug panel (collapsed) showing full PipelineState, token usage, latency

**Acceptance Criteria:**
1. `streamlit run app/streamlit_app.py` launches without errors
2. Pasting a paragraph and clicking "Find Citations" produces visible recommendations within 30 seconds
3. Each recommendation card shows: title, authors, year, citation count, confidence score
4. Expanding a recommendation reveals justification and supporting snippet
5. Pipeline status updates are visible during execution (not just after)

**Tests:**
- **Manual test protocol** (documented in the stage document):
  - Test case 1: Paste abstract about "attention mechanisms in NLP" → verify transformer papers in results
  - Test case 2: Paste abstract about "reinforcement learning for robotics" → verify RL papers in results
  - Test case 3: Submit empty input → verify graceful error message
  - Test case 4: Toggle advanced options and verify they affect results
- **Smoke test** (`tests/test_app/test_streamlit_smoke.py`):
  - Import the app module without errors
  - Verify core functions are callable

**Stage document:** [`docs/stages/STAGE_9_STREAMLIT_UI.md`](docs/stages/STAGE_9_STREAMLIT_UI.md)

---

## After the Stages: Polish & Reporting

Once all 10 stages are complete, the remaining work is:

1. **Ablation studies** — run the evaluation with components disabled (see SYSTEM_OVERVIEW.md §8.5)
2. **Results analysis** — create figures and tables for the final report
3. **README finalization** — ensure the project README is comprehensive and demo-ready
4. **Code cleanup** — final `ruff` pass, remove dead code, ensure all docstrings are present
5. **Final report** — write the academic paper/report documenting the project

This is not a formal stage because it doesn't produce new testable code — it produces analysis and documentation.

---

## Running All Tests

Every stage's tests accumulate. At any point in development, running all tests shows the health of all completed stages:

```bash
# Run all unit tests (fast, no external services needed)
pytest tests/ -m "not integration" -v

# Run all tests including integration tests (requires Qdrant running)
pytest tests/ -v

# Run tests for a specific stage/module
pytest tests/test_data/ -v          # Stage 1
pytest tests/test_indexing/ -v      # Stage 3
pytest tests/test_agents/ -v        # Stages 4-7
pytest tests/test_evaluation/ -v    # Stage 8

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Markers

Tests are marked by category so you can run subsets:

```python
# In conftest.py:
# @pytest.mark.integration  — requires external services (Qdrant, OpenAI)
# @pytest.mark.slow         — takes >10 seconds
# (unmarked tests are pure unit tests: fast, no external dependencies)
```

---

## For Teammates: Quick Reference

| I want to... | Do this |
|---|---|
| Set up the project for the first time | Follow [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) |
| Understand what a stage built | Read `docs/stages/STAGE_N_*.md` |
| Run tests for a stage | See the "Tests" section in that stage's document |
| Run the full test suite | `pytest tests/ -v` |
| Check the overall system design | Read [`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md) |
| Start the UI | `streamlit run app/streamlit_app.py` (after Stage 9) |
| Rebuild the corpus | `python scripts/build_corpus.py` (after Stage 2) |
| Re-index into Qdrant | `python scripts/index_corpus.py` (after Stage 3) |
| Run evaluation | `python scripts/run_evaluation.py` (after Stage 8) |

---

*This roadmap was generated on 2026-02-10 and is a living document. Update it as stages are completed.*

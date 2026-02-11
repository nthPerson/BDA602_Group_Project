# Stage 1 — Data Models & Database Layer

> **Status:** Not Started
> **Depends on:** Stage 0 (Project Scaffolding)
> **Estimated effort:** 3–5 hours

---

## What This Stage Builds

This stage implements the project's **core data types** and the **SQLite persistence layer**. These are the foundational building blocks that every other stage depends on. After this stage, we can store, query, and traverse paper metadata and citation relationships.

### Components Created

| File | Purpose |
|---|---|
| `src/data/models.py` | All data model classes (Paper, CitationEdge, ScoredPaper, etc.) |
| `src/data/db.py` | SQLite database operations (create, insert, query, citation graph) |
| `src/config.py` | Pydantic `Settings` class — expanded with all config fields |
| `tests/test_data/test_models.py` | Unit tests for data models |
| `tests/test_data/test_db.py` | Unit tests for database operations |
| `tests/test_config.py` | Unit tests for configuration loading |

### Data Models Implemented

| Class | Fields | Used By |
|---|---|---|
| `Paper` | paper_id, title, abstract, year, citation_count, doi, arxiv_id, authors, concepts, source, references, cited_by_count, chunk_texts | Corpus storage, all agents |
| `CitationEdge` | source_id, target_id | Citation graph traversal (Agent 3) |
| `ScoredPaper` | paper_id, title, abstract, year, citation_count, similarity_score, concepts | Agent 2 output |
| `RankedPaper` | paper (ScoredPaper fields), rerank_score | Agent 4 Stage A output |
| `GroundedPaper` | paper_id, title, rerank_score, supporting_snippet, justification, confidence | Agent 4 Stage B output |
| `Recommendation` | rank, paper_id, title, authors, year, citation_count, justification, supporting_snippet, confidence, citation_intent_match | Agent 5 / final output |
| `QueryAnalysis` | topic_keywords, citation_intent, expanded_query, confidence | Agent 1 output |
| `CitationIntent` | Enum: BACKGROUND, METHOD, COMPARISON, BENCHMARK | Part of QueryAnalysis |
| `EvalSample` | query_paper_id, query_text, ground_truth_ids, ground_truth_count | Evaluation framework |

### Database Operations Implemented

| Function | Description |
|---|---|
| `create_tables(db)` | Creates `papers` and `citation_edges` tables with indexes |
| `insert_paper(db, paper)` | Inserts or updates a paper record |
| `insert_papers_batch(db, papers)` | Batch insert for performance |
| `insert_citation_edge(db, edge)` | Inserts a citation relationship |
| `insert_citation_edges_batch(db, edges)` | Batch insert for citation edges |
| `get_paper_by_id(db, paper_id)` | Fetch a single paper by ID |
| `get_papers_by_ids(db, paper_ids)` | Fetch multiple papers by ID list |
| `get_papers_by_filter(db, year_min, year_max, min_citations)` | Filtered query |
| `get_references(db, paper_id)` | Get all papers cited BY this paper |
| `get_cited_by(db, paper_id)` | Get all papers that CITE this paper |
| `get_paper_count(db)` | Total paper count |
| `get_edge_count(db)` | Total citation edge count |
| `paper_exists(db, paper_id)` | Check if a paper ID is in the corpus |

---

## Acceptance Criteria

- [ ] All data model classes are importable: `from src.data.models import Paper, CitationEdge, ...`
- [ ] Models have correct type hints and support optional fields
- [ ] `create_tables()` creates a fresh SQLite database with the correct schema
- [ ] Inserting 100 papers and querying them back returns identical data
- [ ] Citation graph queries return correct results for known test edges
- [ ] Batch insert of 1,000 papers completes in <1 second
- [ ] Config loads from `.env` and environment variables; missing required fields raise clear errors
- [ ] All 14+ tests pass

---

## How to Test This Stage

```bash
# Run all Stage 1 tests
pytest tests/test_data/ tests/test_config.py -v

# Run just the model tests
pytest tests/test_data/test_models.py -v

# Run just the database tests
pytest tests/test_data/test_db.py -v

# Run just the config tests
pytest tests/test_config.py -v
```

### Test Inventory

#### `tests/test_data/test_models.py`

| Test | What It Checks |
|---|---|
| `test_paper_creation` | Construct a `Paper` with all fields |
| `test_paper_optional_fields` | `doi` and `arxiv_id` can be `None` |
| `test_citation_edge_creation` | Construct a `CitationEdge` |
| `test_query_analysis_creation` | Construct a `QueryAnalysis` with valid intent |
| `test_citation_intent_values` | All 4 enum values exist |
| `test_scored_paper_creation` | Construct a `ScoredPaper` |
| `test_recommendation_creation` | Construct a `Recommendation` with all fields |

#### `tests/test_data/test_db.py`

| Test | What It Checks |
|---|---|
| `test_create_tables` | Tables are created without errors |
| `test_insert_and_query_paper` | Insert a paper, query by ID, verify all fields match |
| `test_insert_papers_batch` | Batch insert 50 papers, verify count |
| `test_insert_duplicate_paper` | Upserting an existing paper updates it |
| `test_insert_citation_edge` | Insert an edge, verify it exists |
| `test_get_references` | Insert known edges, verify forward traversal |
| `test_get_cited_by` | Insert known edges, verify backward traversal |
| `test_get_papers_by_filter` | Filter by year range returns correct subset |
| `test_paper_not_found` | Querying a non-existent ID returns `None` |
| `test_get_paper_count` | Count matches number of inserted papers |
| `test_paper_exists` | Returns `True` for existing, `False` for missing |

#### `tests/test_config.py`

| Test | What It Checks |
|---|---|
| `test_config_loads_defaults` | Config initializes with default values when env vars missing |
| `test_config_loads_from_env` | Config reads from environment variables |
| `test_config_openai_key_required` | Missing `OPENAI_API_KEY` raises an informative error |

---

## Key Concepts for Teammates

### What are dataclasses?

Dataclasses are Python's way of creating structured data containers. Think of them like a template for a "record" or "row":

```python
from dataclasses import dataclass

@dataclass
class Paper:
    paper_id: str
    title: str
    year: int
```

You create one like: `p = Paper(paper_id="W123", title="Attention Is All You Need", year=2017)`

### What is SQLite?

SQLite is a lightweight database stored in a single file (e.g., `data/papers.db`). It's built into Python — no server needed. We use it to store paper metadata and citation relationships. Think of it like a spreadsheet with two sheets: one for papers and one for "paper A cites paper B" relationships.

### What is the citation graph?

The citation graph is the network of which papers cite which other papers:
- **References (forward):** Paper A's bibliography — the papers A cites
- **Cited-by (backward):** Papers that cite A — A's "impact"

Agent 3 uses this graph to find related papers that simple text search might miss.

---

## Notes for Teammates

- **No external services needed** for this stage. All tests use in-memory SQLite databases.
- **No API keys needed.** The config test mocks the OpenAI key.
- **If you're new to pytest:** just run `pytest tests/test_data/ -v` and look at the output. Green = pass, red = fail. The `-v` flag shows each test name.

---

*Completed by: [name] on [date]*
*Reviewed by: [name] on [date]*

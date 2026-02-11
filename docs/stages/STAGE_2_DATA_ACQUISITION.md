# Stage 2 — Data Acquisition

> **Status:** Not Started
> **Depends on:** Stage 1 (Data Models & Database Layer)
> **Estimated effort:** 4–6 hours (mostly waiting for API responses)

---

## What This Stage Builds

This stage collects the actual paper corpus from **OpenAlex** — a free, open academic metadata API. By the end of this stage, we have a SQLite database populated with ~15,000–25,000 AI/ML papers and their citation relationships.

This is a **data pipeline** stage, not a code-heavy stage. The main work is writing a robust API client, running it, and validating the resulting dataset.

### Components Created

| File | Purpose |
|---|---|
| `src/data/openalex_client.py` | Wrapper around the `pyalex` library: queries AI/ML papers, handles pagination, rate limiting, and caching |
| `src/data/corpus_builder.py` | Orchestrates the full corpus build: calls the client, transforms API responses into `Paper` objects, extracts citation edges, inserts into SQLite |
| `scripts/build_corpus.py` | CLI entry point to build the corpus (run once, takes 30–60 minutes) |
| `scripts/validate_corpus.py` | CLI script to print corpus statistics and run integrity checks |
| `tests/test_data/test_openalex_client.py` | Unit tests for the OpenAlex client (uses saved fixtures, no live API) |
| `data/papers.db` | The populated SQLite database (gitignored — too large for Git) |

### What Gets Collected

For each paper, we store:
- **Metadata:** paper_id (OpenAlex work ID), title, abstract, year, citation count, DOI, arXiv ID, authors, concepts (topic labels), venue/source
- **Citation edges:** For each paper, we record which other OpenAlex work IDs it references

### Corpus Scope

| Parameter | Value | Rationale |
|---|---|---|
| Domains | CS.AI, CS.LG, CS.CL, CS.CV | Core AI/ML subfields |
| Years | 2018–2025 | Modern, relevant papers |
| Filter | Must have abstract | Can't embed without text |
| Sort | By citation count (descending) | Prioritize high-impact papers |
| Target size | 15,000–25,000 papers | Large enough for meaningful retrieval, small enough to manage |

---

## Acceptance Criteria

- [ ] `scripts/build_corpus.py` completes without uncaught errors
- [ ] SQLite database contains ≥10,000 papers
- [ ] ≥80% of papers have at least 1 citation edge in the database
- [ ] All papers have non-empty titles and abstracts
- [ ] `scripts/validate_corpus.py` prints statistics and passes all integrity checks
- [ ] Raw OpenAlex API responses are cached locally (so the API doesn't need to be called again)
- [ ] Unit tests pass

---

## How to Run the Corpus Builder

> **Important:** This script takes **30–60 minutes** to complete because it makes thousands of API calls. Run it once and cache the results.

```bash
# Make sure the virtual environment is activated
source .venv/bin/activate

# Run the corpus builder
python scripts/build_corpus.py

# The script will print progress like:
# [1/100] Fetching page 1... 200 papers
# [2/100] Fetching page 2... 200 papers
# ...
# Done! 18,542 papers and 127,891 citation edges stored.
```

### Validate the corpus

```bash
python scripts/validate_corpus.py
```

**Expected output (approximate):**
```
=== Corpus Statistics ===
Total papers:         18,542
Total citation edges: 127,891
Papers with ≥1 edge:  15,230 (82.1%)
Year range:           2018–2025
Unique concepts:      847

Papers by year:
  2018: 1,892
  2019: 2,341
  2020: 2,876
  ...

Top 10 cited papers:
  1. Attention Is All You Need (2017) — 90,231 citations
  2. BERT: Pre-training of Deep... (2018) — 75,123 citations
  ...

=== Integrity Checks ===
✓ No papers with empty titles
✓ No papers with empty abstracts
✓ All citation edges reference valid paper IDs
✓ No duplicate paper IDs
```

---

## How to Test This Stage

```bash
# Run unit tests (no live API calls needed)
pytest tests/test_data/test_openalex_client.py -v

# Run the validation script (requires corpus to be built first)
python scripts/validate_corpus.py
```

### Test Inventory

#### `tests/test_data/test_openalex_client.py`

| Test | What It Checks |
|---|---|
| `test_query_construction` | OpenAlex filter parameters are set correctly (concepts, date range, has_abstract) |
| `test_response_parsing` | A saved API response fixture is correctly parsed into `Paper` objects |
| `test_citation_edge_extraction` | References from API response are correctly converted to `CitationEdge` objects |
| `test_empty_response_handling` | Empty API response doesn't crash |
| `test_missing_abstract_skipped` | Papers without abstracts are filtered out |
| `test_pagination_logic` | Client correctly paginates through multiple pages |

#### `scripts/validate_corpus.py` (acts as an integration test)

| Check | What It Verifies |
|---|---|
| Paper count ≥ 10,000 | Enough data was collected |
| Edge coverage ≥ 80% | Citation graph is well-populated |
| No empty titles | Data quality |
| No empty abstracts | Data quality |
| All edge IDs exist in papers | Referential integrity |
| No duplicate IDs | Data consistency |

---

## Key Concepts for Teammates

### What is OpenAlex?

[OpenAlex](https://openalex.org/) is a free, open index of academic papers — think of it as an open-source version of Google Scholar's database. It has metadata for over 200 million papers, including titles, abstracts, authors, citation counts, and citation relationships.

We query it via its REST API using the `pyalex` Python library.

### What is a "citation edge"?

If Paper A cites Paper B in its bibliography, we store that as a citation edge: `(A, B)`. This lets us later traverse the graph — "what does Paper X cite?" and "what cites Paper X?" — which is how Agent 3 (Citation Expansion) finds additional relevant papers.

### Why is the corpus static?

We build the corpus once and use it for the rest of the project. We don't continually fetch new papers because:
1. It adds complexity (scheduled jobs, incremental updates)
2. The evaluation needs a fixed corpus to produce reproducible metrics
3. 8 weeks isn't long enough for the corpus to become meaningfully outdated

### How long does it take?

About 30–60 minutes for the first run. OpenAlex has generous rate limits (especially with a polite email), but we're fetching 15k+ papers with all their metadata and references. The results are cached locally, so you won't need to re-run it unless you want to change the corpus scope.

---

## Notes for Teammates

- **You don't need an API key** for OpenAlex. It's fully open. Adding a polite email in the config speeds up rate limits.
- **Don't try to run `build_corpus.py` on slow WiFi.** It makes thousands of HTTP requests. Use a stable connection.
- **If the script fails mid-way,** it can be re-run safely — it uses upsert logic, so it won't create duplicates.
- **The `data/` directory is gitignored.** You need to build the corpus yourself, or get `papers.db` from a teammate (it's ~50–100 MB).

---

*Completed by: [name] on [date]*
*Reviewed by: [name] on [date]*

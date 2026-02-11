# Stage 3 — Embedding & Indexing

> **Status:** Not Started
> **Depends on:** Stage 1 (Data Models), Stage 2 (Data Acquisition — corpus must be built)
> **Estimated effort:** 3–5 hours

---

## What This Stage Builds

This stage takes the paper corpus from SQLite and **embeds** each paper into a 768-dimensional vector, then indexes those vectors in **Qdrant** (a vector database) for fast similarity search. After this stage, we can type a query and find the most semantically similar papers in milliseconds.

### Components Created

| File | Purpose |
|---|---|
| `src/indexing/embedder.py` | Embedding pipeline: loads the sentence-transformer model, encodes text into vectors |
| `src/indexing/qdrant_store.py` | Qdrant collection management: create/delete collections, upsert vectors with metadata payloads, similarity search |
| `scripts/index_corpus.py` | CLI entry point: reads papers from SQLite → embeds → upserts into Qdrant |
| `scripts/test_search.py` | Interactive script: type a query, see top-5 most similar papers |
| `tests/test_indexing/test_embedder.py` | Unit tests for embedding pipeline |
| `tests/test_indexing/test_qdrant_store.py` | Integration tests for Qdrant operations (requires Docker) |

### How It Works

```
SQLite (papers.db)
       │
       │  Read papers
       ▼
┌─────────────────┐
│ Embedding Model  │  BAAI/bge-base-en-v1.5 (768 dimensions)
│ (sentence-       │  Input: "Title\n\nAbstract"
│  transformers)   │  Output: normalized 768-dim vector
└────────┬────────┘
         │
         │  Batch upload vectors + metadata payloads
         ▼
┌─────────────────┐
│ Qdrant (Docker)  │  Collection: "papers"
│                  │  Each point: vector + {paper_id, title, abstract, year, ...}
│                  │  Payload indexes on: year, citation_count, concepts
└─────────────────┘
```

---

## Acceptance Criteria

- [ ] `scripts/index_corpus.py` completes and reports the number of indexed vectors
- [ ] Qdrant collection point count matches paper count in SQLite
- [ ] Similarity search for "transformer attention mechanism" returns transformer-related papers in top 10
- [ ] Payload-filtered search (e.g., `year >= 2022`) returns only matching papers
- [ ] Embeddings have correct dimensions (768) and are normalized (L2 norm ≈ 1.0)
- [ ] All tests pass

---

## How to Run the Indexing Pipeline

> **Prerequisites:**
> 1. Corpus must be built (Stage 2): `data/papers.db` exists
> 2. Qdrant must be running: `docker compose up -d`

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Qdrant
docker compose up -d

# Run the indexing script
python scripts/index_corpus.py

# Expected output:
# Loading embedding model (BAAI/bge-base-en-v1.5)...
# Reading 18,542 papers from SQLite...
# Embedding papers (batch_size=64)...
# [████████████████████████████████████████] 100% (18,542/18,542)
# Upserting into Qdrant...
# Done! 18,542 vectors indexed in collection 'papers'.
```

### Try it out

```bash
# Interactive search test
python scripts/test_search.py

# Enter a query: graph neural networks for molecular property prediction
# 
# Top 5 results:
# 1. [0.89] "Neural Message Passing for Quantum Chemistry" (2017)
# 2. [0.87] "SchNet: A continuous-filter convolutional neural..." (2018)
# 3. [0.85] "How Powerful are Graph Neural Networks?" (2019)
# ...
```

---

## How to Test This Stage

```bash
# Unit tests (no Docker needed)
pytest tests/test_indexing/test_embedder.py -v

# Integration tests (requires Qdrant running in Docker)
pytest tests/test_indexing/test_qdrant_store.py -v

# All indexing tests
pytest tests/test_indexing/ -v
```

### Test Inventory

#### `tests/test_indexing/test_embedder.py` (no Docker required)

| Test | What It Checks |
|---|---|
| `test_embedding_dimensions` | Output vectors have 768 dimensions |
| `test_embedding_normalization` | Output vectors have L2 norm ≈ 1.0 (within 1e-5 tolerance) |
| `test_batch_equals_single` | Batch encoding and single encoding produce the same vectors |
| `test_empty_text_handling` | Empty or whitespace-only input is handled gracefully |
| `test_different_texts_different_vectors` | Distinct texts produce distinct embeddings |
| `test_make_paper_text` | Title + abstract concatenation format is correct |

#### `tests/test_indexing/test_qdrant_store.py` (requires Docker — marked `@pytest.mark.integration`)

| Test | What It Checks |
|---|---|
| `test_create_collection` | Collection is created with correct vector config |
| `test_upsert_and_count` | Upserting 10 vectors, point count is 10 |
| `test_search_returns_results` | Searching with a known vector returns results |
| `test_search_top_result_most_similar` | The most similar vector is ranked first |
| `test_filtered_search_by_year` | Year filter excludes non-matching papers |
| `test_delete_collection` | Collection is cleanly deleted |
| `test_upsert_overwrites` | Re-upserting a point updates (not duplicates) it |

---

## Key Concepts for Teammates

### What is an embedding?

An embedding is a way to represent text as a list of numbers (a "vector"). Texts with similar meaning produce similar vectors. For example:
- "transformer attention mechanism" → [0.12, -0.34, 0.56, ...]
- "self-attention in neural networks" → [0.13, -0.32, 0.55, ...]  (very similar!)
- "cooking recipes for pasta" → [0.87, 0.23, -0.91, ...]  (very different)

We use the `BAAI/bge-base-en-v1.5` model, which produces 768-dimensional vectors.

### What is Qdrant?

Qdrant is a **vector database** — it's optimized for storing millions of vectors and finding the ones most similar to a query vector. Think of it like a search engine, but instead of matching keywords, it matches **meaning**.

It runs as a Docker container on your machine. You don't interact with it directly — our code talks to it via a Python client.

### What is a "payload"?

When we store a vector in Qdrant, we attach a **payload** — extra metadata like the paper's title, year, and citation count. This lets us filter search results (e.g., "only papers from 2022 or later") without having to look up each paper in SQLite.

### Why does indexing take a while?

The embedding model processes each paper's title + abstract to produce a vector. For ~20k papers, this takes:
- **10–20 minutes** with a GPU
- **1–2 hours** on CPU only

This is done once. After that, searching is milliseconds.

---

## Notes for Teammates

- **You MUST have Docker running** for this stage. Run `docker compose up -d` first.
- **If you have a GPU**, sentence-transformers will use it automatically for faster embedding. If not, it falls back to CPU — just takes longer.
- **If indexing is interrupted,** you can re-run `scripts/index_corpus.py` safely. It recreates the collection from scratch.
- **The Qdrant data is stored in `data/qdrant_storage/`** (gitignored). If you delete this directory, you'll need to re-index.

---

*Completed by: [name] on [date]*
*Reviewed by: [name] on [date]*

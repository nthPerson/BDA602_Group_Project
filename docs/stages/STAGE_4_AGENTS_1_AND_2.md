# Stage 4 — Agents 1 & 2: Query Analysis + Primary Retrieval

> **Status:** Not Started
> **Depends on:** Stage 3 (Embedding & Indexing — Qdrant must be populated)
> **Estimated effort:** 4–6 hours

---

## What This Stage Builds

This stage implements the first two agents and wires them together, producing the system's first working capability: **"text in → relevant papers out."** After this stage, you can paste a paragraph of text and get back a ranked list of the most relevant papers from the corpus.

### Components Created

| File | Purpose |
|---|---|
| `src/agents/query_agent.py` | **Agent 1:** Takes raw user text → produces structured query (keywords, intent, expanded query) via LLM |
| `src/agents/retrieval_agent.py` | **Agent 2:** Takes structured query → searches Qdrant → returns top-N similar papers |
| `src/orchestration/state.py` | `PipelineState` / `AgentState` TypedDict definitions |
| `tests/test_agents/test_query_agent.py` | Unit tests for Agent 1 |
| `tests/test_agents/test_retrieval_agent.py` | Unit + integration tests for Agent 2 |

### How It Works

```
User text: "Recent advances in transformer architectures have shown
            that scaling model parameters improves few-shot learning."
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Agent 1: Query & Intent Analysis                              │
│                                                                │
│ LLM (gpt-4o-mini) extracts:                                  │
│   keywords: ["transformer", "scaling", "few-shot learning"]   │
│   intent: BACKGROUND                                          │
│   expanded_query: "transformer scaling laws few-shot          │
│                    learning large language models"             │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Agent 2: Primary Retrieval                                    │
│                                                                │
│ 1. Embed the expanded_query using BGE-base                    │
│ 2. Search Qdrant for top 30 most similar paper vectors        │
│ 3. Return ranked list of ScoredPaper objects                  │
│                                                                │
│ Results:                                                       │
│   1. [0.91] "Language Models are Few-Shot Learners" (2020)    │
│   2. [0.88] "Scaling Laws for Neural Language Models" (2020)  │
│   3. [0.85] "A Survey of Large Language Models" (2023)        │
│   ...                                                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Acceptance Criteria

- [ ] Agent 1 accepts a paragraph and returns a valid `QueryAnalysis` (keywords, intent, expanded query)
- [ ] Agent 1 fallback produces reasonable keywords when OpenAI is unreachable
- [ ] Agent 2 accepts a `QueryAnalysis` and returns ≤30 `ScoredPaper` results
- [ ] Agent 2 results are sorted by similarity score (highest first)
- [ ] Metadata filters (year range) work correctly
- [ ] Manual end-to-end test: paste an abstract → get back relevant papers
- [ ] All unit and integration tests pass

---

## How to Test This Stage

```bash
# Unit tests only (mocked LLM and Qdrant — no services needed)
pytest tests/test_agents/test_query_agent.py tests/test_agents/test_retrieval_agent.py -v -m "not integration"

# Integration tests (requires Qdrant indexed + OpenAI API key)
pytest tests/test_agents/test_query_agent.py tests/test_agents/test_retrieval_agent.py -v
```

### Test Inventory

#### `tests/test_agents/test_query_agent.py`

| Test | What It Checks | Requires API? |
|---|---|---|
| `test_valid_structured_output` | Mocked LLM returns valid JSON → `QueryAnalysis` is correctly parsed | No |
| `test_keywords_extracted` | Output contains non-empty topic_keywords list | No |
| `test_citation_intent_valid` | Citation intent is one of the 4 valid enum values | No |
| `test_expanded_query_nonempty` | Expanded query is not empty | No |
| `test_retry_on_invalid_json` | Mocked LLM returns junk first, valid JSON second → retry succeeds | No |
| `test_fallback_on_api_failure` | LLM completely fails → fallback keyword extraction produces keywords | No |
| `test_fallback_uses_raw_text` | In fallback mode, expanded_query equals the raw input text | No |
| `test_live_query_analysis` | Real LLM call with a known abstract → output is sensible | **Yes** (integration) |

#### `tests/test_agents/test_retrieval_agent.py`

| Test | What It Checks | Requires Services? |
|---|---|---|
| `test_scored_paper_mapping` | Mocked Qdrant results → correctly mapped to `ScoredPaper` objects | No |
| `test_results_sorted_by_score` | Output is sorted by similarity_score descending | No |
| `test_top_n_respected` | No more than `top_n` results returned | No |
| `test_year_filter_construction` | Year range parameters generate correct Qdrant filter | No |
| `test_empty_results` | Qdrant returns nothing → empty list (no crash) | No |
| `test_live_retrieval_transformers` | Query "transformer attention" → top 10 includes "Attention Is All You Need" | **Yes** (Qdrant) |
| `test_live_retrieval_year_filter` | Query with `year_min=2022` → all results have year ≥ 2022 | **Yes** (Qdrant) |

### Manual End-to-End Test

After both agents are implemented, run them together:

```python
# scripts/test_agents_1_2.py (or in a notebook)
from src.agents.query_agent import analyze_query
from src.agents.retrieval_agent import PrimaryRetrievalAgent
from src.indexing.qdrant_store import get_qdrant_client
from src.indexing.embedder import load_embedder

# Agent 1
query = "Few-shot learning has emerged as a critical capability of large language models."
analysis = analyze_query(query)
print(f"Intent: {analysis.citation_intent}")
print(f"Keywords: {analysis.topic_keywords}")
print(f"Expanded query: {analysis.expanded_query}")

# Agent 2
agent2 = PrimaryRetrievalAgent(get_qdrant_client(), load_embedder())
results = agent2.retrieve(analysis, top_n=10)
for i, paper in enumerate(results):
    print(f"  {i+1}. [{paper.similarity_score:.3f}] {paper.title} ({paper.year})")
```

---

## Key Concepts for Teammates

### What does Agent 1 actually do?

Agent 1 takes raw text (like a paragraph from a paper) and asks an LLM (GPT-4o-mini) to:
1. **Extract key technical terms** (e.g., "transformer", "few-shot learning")
2. **Classify the citation intent** — is the user looking for:
   - Background/context papers?
   - Papers describing a method they're using?
   - Papers they're comparing against?
   - Benchmark/dataset papers?
3. **Reformulate the text** into a more effective search query

This is done with **structured output** — the LLM is constrained to return JSON matching our `QueryAnalysis` schema, not free-form text.

### What does Agent 2 do?

Agent 2 is simpler — no LLM involved. It:
1. Takes the expanded query from Agent 1
2. Embeds it into a vector using the same model we used to embed the papers
3. Searches Qdrant for the 30 most similar paper vectors
4. Returns the results with their similarity scores

### What is the "fallback" in Agent 1?

If the OpenAI API is down or unreachable, Agent 1 falls back to a simple keyword extraction approach (TF-IDF or regex-based) instead of crashing. This ensures the pipeline can still produce results (just less polished ones).

### What is `instructor`?

`instructor` is a Python library that wraps the OpenAI API to guarantee structured output. Instead of asking the LLM for free text and hoping it returns valid JSON, `instructor` enforces a Pydantic model schema. If the LLM returns invalid output, it automatically retries with feedback. This makes LLM calls much more reliable.

---

## Notes for Teammates

- **You need an OpenAI API key** to run the live/integration tests for Agent 1. Unit tests use mocks and don't need a key.
- **You need Qdrant running with an indexed corpus** to run Agent 2 integration tests. Run `docker compose up -d` first.
- **The instruction prefix matters.** When embedding a query with BGE, we prefix it with `"Represent this sentence for searching relevant passages: "`. This isn't arbitrary — it was part of the model's training and significantly improves results.
- **Agent 2 is intentionally simple.** No LLM reasoning, no heuristics — just embed and search. The "smart" parts come later (Agents 3, 4, 5).

---

*Completed by: [name] on [date]*
*Reviewed by: [name] on [date]*

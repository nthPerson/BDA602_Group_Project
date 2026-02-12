# Stage 6 — Agent 4: Reranking & Grounding

> **Status:** Complete
> **Depends on:** Stage 5 (Citation Expansion — the full candidate set must be available)
> **Estimated effort:** 5–7 hours

---

## What This Stage Builds

This stage implements the most technically interesting agent in the pipeline. Agent 4 has two sub-stages:

- **Stage A — Cross-encoder reranking:** Re-scores all ~50–70 candidates using a cross-encoder model that is far more accurate than the initial cosine similarity
- **Stage B — LLM grounding:** For the top 5 reranked papers, generates a justification and extracts a supporting snippet, grounding the recommendation in actual paper content

After this stage, the system produces **ranked, explained recommendations** — not just a list of similar papers.

### Components Created

| File | Purpose |
|---|---|
| `src/agents/reranking_agent.py` | Agent 4: cross-encoder reranking (Stage A) + LLM grounding (Stage B) |
| `tests/test_agents/test_reranking_agent.py` | Unit + integration tests for reranking and grounding |

### How It Works

```
Expanded candidates (~50-70 papers)
       │
       ▼
┌────────────────────────────────────────────────────────────┐
│ Stage A: Cross-Encoder Reranking                            │
│                                                              │
│ Model: BAAI/bge-reranker-base                               │
│                                                              │
│ For each candidate:                                          │
│   score = cross_encoder.predict(query, "title\nabstract")   │
│                                                              │
│ Sort by score → take top 10                                  │
│ Time: ~2-3s on CPU for 50 candidates                        │
└──────────────────────┬─────────────────────────────────────┘
                       │ Top 10
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Stage B: LLM Grounding (top 5 only)                         │
│                                                              │
│ Model: gpt-4o-mini (via instructor)                         │
│                                                              │
│ For each of top 5:                                           │
│   → Extract supporting snippet from abstract                 │
│   → Generate relevance justification                         │
│   → Assess confidence score                                  │
│                                                              │
│ 5 calls run in parallel via asyncio.gather                   │
│ Time: ~2s (parallelized)                                     │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
Output: list[GroundedPaper] — ranked papers with justifications
```

### Why Two Stages?

| Aspect | Bi-encoder (Agent 2) | Cross-encoder (Agent 4A) | LLM Grounding (Agent 4B) |
|---|---|---|---|
| **How it compares** | Embeds query and doc separately | Jointly encodes query + doc | Full language understanding |
| **Speed** | ~1ms per paper | ~50ms per paper | ~1-2s per paper |
| **Accuracy** | Good enough for top-30 | Much better ranking | Produces explanations |
| **Scalable?** | Yes (search 25k papers) | For ~50 papers | For ~5 papers |

This is the classic **retrieve → rerank → generate** pattern used in production search and RAG systems.

---

## Acceptance Criteria

- [x] Cross-encoder reranking re-orders candidates (output differs from cosine similarity order)
- [x] Top-K parameter is respected (default: 10)
- [x] Top 5 papers receive grounding output with all required fields
- [x] Justifications reference content from the candidate paper (not hallucinated)
- [x] Parallel grounding completes in <5 seconds for 5 papers
- [x] Full Agent 4 pipeline completes in <10 seconds for 50 candidates on CPU
- [x] All tests pass

---

## How to Test This Stage

```bash
# Unit tests only (mocked cross-encoder + LLM — no services needed)
pytest tests/test_agents/test_reranking_agent.py -v -m "not integration"

# All tests including integration (requires Qdrant + OpenAI API key)
pytest tests/test_agents/test_reranking_agent.py -v
```

### Test Inventory

#### Reranking Tests (no external services needed)

| Test | What It Checks |
|---|---|
| `test_reranking_reorders` | Mocked cross-encoder scores cause a different ordering than input order |
| `test_reranking_top_k` | Output length equals min(top_k, number of candidates) |
| `test_reranking_score_attached` | Each `RankedPaper` has a `rerank_score` field |
| `test_reranking_all_identical_scores` | Stable ordering when all scores are equal |
| `test_reranking_empty_candidates` | Empty input → empty output (no crash) |
| `test_reranking_single_candidate` | One candidate → returned as-is |

#### Grounding Tests (no external services needed for mocked tests)

| Test | What It Checks |
|---|---|
| `test_grounding_output_fields` | Mocked LLM → `GroundedPaper` has snippet, justification, confidence |
| `test_grounding_prompt_contains_context` | The prompt sent to the LLM includes both user text and paper abstract |
| `test_grounding_retry_on_failure` | Mocked LLM fails once, succeeds on retry → correct output |
| `test_grounding_confidence_range` | Confidence is between 0.0 and 1.0 |

#### Integration Tests (require services)

| Test | What It Checks | Requires |
|---|---|---|
| `test_cross_encoder_reranks_real_papers` | Real cross-encoder produces different ranking than cosine similarity for 20 test papers | Qdrant + corpus |
| `test_grounding_real_llm` | Real LLM call produces a non-empty justification that mentions terms from the paper | OpenAI API key |
| `test_full_agent4_pipeline` | Run Agents 1→2→3→4 on a known query → top result is more specifically relevant than the 30th retrieval result | All services |

---

## Key Concepts for Teammates

### What is a cross-encoder?

A **bi-encoder** (used in Agent 2) embeds the query and each document **separately**, then compares the vectors. This is fast but approximate.

A **cross-encoder** feeds the query AND the document **together** into a single model, allowing it to look at how specific words in the query relate to specific words in the document. This is much more accurate but slower.

Example:
- Query: "methods for reducing hallucination in LLMs"
- Paper A title: "Reducing Hallucination in Large Language Models" → cross-encoder gives high score (direct match)
- Paper B title: "Language Model Safety" → cross-encoder gives lower score (related but less specific)

A bi-encoder might rank these similarly (both about LLMs), but the cross-encoder properly ranks A above B.

### What is "grounding"?

Grounding means anchoring the system's output in actual evidence. Instead of just saying "this paper is relevant," the system:
1. **Extracts a specific snippet** from the paper's abstract that supports its relevance
2. **Explains why** the paper is a good citation in natural language

This serves two purposes:
1. The user can quickly verify the recommendation makes sense
2. It reduces hallucination — the LLM is constrained to reference actual paper content

### What is `asyncio.gather`?

When we need to make 5 independent LLM calls (one per paper for grounding), we can run them **in parallel** instead of one after another. `asyncio.gather` is Python's way of doing this:
- Sequential: 5 calls × 1.5s each = 7.5s total
- Parallel: all 5 start at once → ~1.5s total (limited by the slowest single call)

---

## Notes for Teammates

- **The cross-encoder model downloads on first use** (~400 MB). This happens automatically but takes a few minutes on the first run.
- **You need an OpenAI API key** for grounding integration tests, but unit tests use mocks.
- **Grounding costs money** (very little — ~$0.001 per grounding call with gpt-4o-mini), but be aware if running many integration tests.
- **If the cross-encoder is too slow on your machine** (>10s for 50 papers), you can switch to `cross-encoder/ms-marco-MiniLM-L-6-v2` which is ~2.5x faster at a small quality cost.

---

*Completed by: Claude on 2026-02-12*
*Reviewed by: [name] on [date]*

# Stage 5 — Agent 3: Citation Expansion

> **Status:** Not Started
> **Depends on:** Stage 4 (Agents 1 & 2 must be working)
> **Estimated effort:** 3–4 hours

---

## What This Stage Builds

This stage implements **citation graph expansion** — the key differentiator from a vanilla RAG system. Agent 3 takes the top papers from Agent 2 and uses the citation graph to find additional relevant papers that pure text similarity might miss.

For example: if Agent 2 retrieves a paper about "BERT," Agent 3 will also pull in papers that BERT cites (like "Attention Is All You Need") and papers that cite BERT (like "RoBERTa") — even if those weren't in the top retrieval results.

### Components Created

| File | Purpose |
|---|---|
| `src/agents/expansion_agent.py` | Agent 3: citation graph traversal, deduplication, prioritization, size limiting |
| `tests/test_agents/test_expansion_agent.py` | Unit tests with synthetic citation graphs |

### How It Works

```
Agent 2 output: Top 10 papers
       │
       │  For each paper:
       │    1. Look up its references (papers it cites)
       │    2. Look up its cited-by (papers that cite it)
       ▼
┌─────────────────────────────────────────────────┐
│ Citation Graph (SQLite citation_edges table)      │
│                                                   │
│   Paper A ──cites──▶ Paper X                     │
│   Paper A ──cites──▶ Paper Y                     │
│   Paper B ──cites──▶ Paper X  ← X appears twice! │
│   Paper Z ──cites──▶ Paper A  (cited-by)         │
│                                                   │
│   Expansion pool: {X, Y, Z, ...}                 │
│   X is prioritized (cited by 2 seed papers)      │
└─────────────────────────────────────────────────┘
       │
       │  Deduplicate, prioritize, limit to max_expansion
       ▼
Output: Original 30 papers + up to 40 expanded papers = ~50-70 candidates
```

### Agent 3 Behavior

| Parameter | Default | Description |
|---|---|---|
| Seed papers | Top 10 from Agent 2 | Papers whose citation neighborhoods are explored |
| Expansion depth | 1 (direct neighbors only) | No recursive expansion — keeps it simple and fast |
| Max expansion | 40 | Maximum additional papers added |
| Prioritization | Multi-citation count, then citation count | Papers cited by multiple seed papers rank first |

---

## Acceptance Criteria

- [ ] Given 10 seed papers, Agent 3 returns originals PLUS additional papers from the citation graph
- [ ] No duplicates in the expanded set (verified programmatically)
- [ ] `max_expansion` limit is respected (never exceeds configured maximum)
- [ ] Papers appearing in multiple seed papers' citation lists are prioritized
- [ ] A paper with no citation edges is handled gracefully (no crash, no expansion for that paper)
- [ ] The entire expansion step completes in <1 second for typical inputs (10 seed papers)
- [ ] All unit tests pass

---

## How to Test This Stage

```bash
# Unit tests (uses in-memory SQLite — no external services needed)
pytest tests/test_agents/test_expansion_agent.py -v

# Integration test: run Agents 1→2→3 in sequence
pytest tests/test_agents/test_expansion_agent.py -v -m integration
```

### Test Inventory

#### `tests/test_agents/test_expansion_agent.py`

All unit tests use a **synthetic citation graph** — a small, hand-crafted set of papers and edges where we know exactly what the expansion should produce.

**Synthetic test graph:**
```
Papers: A, B, C, D, E, F, G, H, I, J, K, L
Edges:
  A → D, A → E, A → F     (A cites D, E, F)
  B → D, B → G             (B cites D, G)
  C → E, C → H             (C cites E, H)
  I → A, J → A             (I and J cite A — "cited by" for A)
  K → B                    (K cites B)
  L → (nothing — isolated) (L has no edges)
```

| Test | What It Checks |
|---|---|
| `test_expansion_finds_references` | Seed = [A] → expanded includes D, E, F |
| `test_expansion_finds_cited_by` | Seed = [A] → expanded includes I, J |
| `test_expansion_deduplication` | Seed = [A, B] → D appears only once (not twice) |
| `test_expansion_preserves_originals` | Seed = [A, B, C] → all of A, B, C are in the output |
| `test_max_expansion_limit` | Seed = [A, B, C], max_expansion=3 → at most 3 new papers added |
| `test_prioritization_multi_citation` | Seed = [A, B] → D is ranked before G (D is cited by both A and B) |
| `test_isolated_paper_no_crash` | Seed = [L] → output = [L] (no expansion, no error) |
| `test_expansion_excludes_seed_papers` | Seed = [A, I] → A appears once (not duplicated as a cited-by of I) |
| `test_only_corpus_papers_included` | Expansion paper IDs that aren't in the corpus are filtered out |
| `test_integration_agents_1_2_3` | Run Agents 1→2→3 end-to-end with a real query → expanded set is larger than retrieval set |

---

## Key Concepts for Teammates

### Why do we need citation expansion?

Text similarity search is good but has blind spots. Two papers can be highly relevant to each other but use different vocabulary:
- A paper about "masked language modeling" won't be textually similar to "cloze tasks in NLP" even though they're the same concept.
- Citation links capture **topical relatedness** that text similarity misses.

By following citation links, we find papers that are structurally related in the academic knowledge graph, not just textually similar.

### What is "depth-1 BFS"?

BFS = Breadth-First Search. "Depth-1" means we only look one step away in the citation graph:
- Direct references (papers cited by our seed papers)
- Direct citations (papers that cite our seed papers)

We don't go deeper (references of references) because the candidate set explodes and most of the additions are noise. Depth-1 is the sweet spot of recall gain vs. precision cost.

### What is the prioritization heuristic?

When the expansion pool has more papers than `max_expansion`, we need to pick the best ones. The rule is:
1. **Multi-citation papers first:** If Paper X is cited by 3 of our seed papers, it's probably more relevant than a paper cited by only 1 seed paper.
2. **Higher citation count second:** Among papers with the same multi-citation count, prefer more-cited papers (they're more likely to be important).

---

## Notes for Teammates

- **All unit tests run offline** — no Qdrant, no OpenAI, no internet. They use a synthetic in-memory SQLite database.
- **The integration test** requires completed Stages 1–4 (corpus built, Qdrant indexed, API key set).
- **Agent 3 is purely deterministic** — no LLM, no randomness. Given the same inputs and citation graph, it always produces the same output. This makes it very easy to test and debug.
- **Performance:** SQLite lookups on indexed columns are sub-millisecond. Even with 10 seed papers, the entire expansion step takes <500ms.

---

*Completed by: [name] on [date]*
*Reviewed by: [name] on [date]*

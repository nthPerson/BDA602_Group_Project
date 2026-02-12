# Stage 7 — Agent 5 + Orchestration (LangGraph Pipeline)

> **Status:** Complete
> **Depends on:** Stage 6 (all 4 agents must be implemented)
> **Estimated effort:** 5–7 hours

---

## What This Stage Builds

This stage implements the **final agent (synthesis)** and wires **all five agents** into a **LangGraph state machine**. After this stage, the entire pipeline runs end-to-end: text in → structured recommendations out.

This is the stage where the system goes from "individual components" to "a working product."

### Components Created

| File | Purpose |
|---|---|
| `src/agents/synthesis_agent.py` | Agent 5: confidence filtering, composite scoring, final formatting |
| `src/orchestration/state.py` | Finalized `AgentState` TypedDict (may be updated from Stage 4's stub) |
| `src/orchestration/graph.py` | LangGraph `StateGraph` definition: 5 nodes, linear edges, error handling, observability |
| `tests/test_agents/test_synthesis_agent.py` | Unit tests for Agent 5 |
| `tests/test_orchestration/test_graph.py` | Unit + integration tests for the LangGraph pipeline |
| `tests/test_integration/test_pipeline_e2e.py` | Full end-to-end integration test |

### How the Orchestration Works

```
                                    AgentState
                               (TypedDict, shared)
                                       │
┌─────────────────────────────────────────────────────────────────┐
│                         LangGraph StateGraph                      │
│                                                                   │
│   START                                                           │
│     │                                                             │
│     ▼                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌──────────────────┐   │
│  │ query_       │────▶│ primary_    │────▶│ citation_        │   │
│  │ analysis     │     │ retrieval   │     │ expansion        │   │
│  │ (Agent 1)    │     │ (Agent 2)   │     │ (Agent 3)        │   │
│  └─────────────┘     └─────────────┘     └────────┬─────────┘   │
│                                                    │              │
│  ┌─────────────┐     ┌─────────────┐              │              │
│  │ synthesis   │◀────│ grounding   │◀─────────────┘              │
│  │ (Agent 5)   │     │ (Agent 4)   │                             │
│  └──────┬──────┘     └─────────────┘                             │
│         │                                                         │
│         ▼                                                         │
│        END → final_recommendations                                │
│                                                                   │
│   Error handling: If any node fails:                              │
│   → Log error to state["errors"]                                  │
│   → Continue with best available data                             │
│   → (e.g., expansion failure = use retrieval results only)        │
└───────────────────────────────────────────────────────────────────┘
```

### Agent 5 (Synthesis) Behavior

Agent 5 is the thinnest agent — it's mostly formatting and quality control:

1. **Filter** out low-confidence recommendations (below threshold)
2. **Score** remaining by composite score: `rerank_score × confidence`
3. **Format** into `Recommendation` objects with all required fields
4. **Assign ranks** (1, 2, 3, ...)

---

## Acceptance Criteria

- [x] `app = workflow.compile()` succeeds
- [x] `app.invoke({"user_text": "..."})` runs all 5 agents in sequence
- [x] Final state contains non-empty `final_recommendations` list
- [x] Each `Recommendation` has all required fields (rank, paper_id, title, justification, snippet, confidence)
- [x] `state["metadata"]` contains timing for each agent (latency tracking)
- [x] `state["errors"]` is empty for a successful run
- [x] If Agent 3 fails, pipeline still produces results from Agents 1+2 only (graceful degradation)
- [x] If Agent 4 grounding fails, pipeline still produces reranked results without justifications
- [x] All unit and integration tests pass

---

## How to Test This Stage

```bash
# Agent 5 unit tests (no services needed)
pytest tests/test_agents/test_synthesis_agent.py -v

# Orchestration unit tests (mocked agents — no services needed)
pytest tests/test_orchestration/test_graph.py -v -m "not integration"

# Full end-to-end integration test (requires all services)
pytest tests/test_integration/test_pipeline_e2e.py -v
```

### Test Inventory

#### `tests/test_agents/test_synthesis_agent.py`

| Test | What It Checks |
|---|---|
| `test_confidence_filter` | Papers below confidence threshold are excluded |
| `test_composite_score_sorting` | Output sorted by rerank_score × confidence (descending) |
| `test_rank_numbering` | Ranks are 1, 2, 3, ... (sequential, starting at 1) |
| `test_all_fields_present` | Each Recommendation has every required field populated |
| `test_empty_input` | No grounded papers → empty recommendations (no crash) |
| `test_all_below_threshold` | All papers below confidence → empty output |
| `test_formats_multiple_authors` | Authors list is preserved correctly |

#### `tests/test_orchestration/test_graph.py`

| Test | What It Checks |
|---|---|
| `test_graph_compiles` | `StateGraph` compiles without errors |
| `test_graph_runs_all_nodes` | All 5 node functions are called in order (mocked) |
| `test_state_propagates` | State written by Agent 1 is readable by Agent 2, etc. |
| `test_metadata_timing` | Metadata dict contains timing entries for each agent |
| `test_error_in_agent3_graceful` | Agent 3 raises exception → state["errors"] logged, pipeline continues |
| `test_error_in_agent4_graceful` | Agent 4 raises exception → pipeline produces results without grounding |
| `test_all_agents_fail_returns_empty` | All agents fail → empty recommendations, errors logged (no crash) |

#### `tests/test_integration/test_pipeline_e2e.py` (marked `@pytest.mark.integration`)

| Test | What It Checks | Requires |
|---|---|---|
| `test_e2e_produces_recommendations` | Full pipeline run with real query → non-empty recommendations | All services |
| `test_e2e_recommendation_structure` | Output has correct fields and reasonable values | All services |
| `test_e2e_performance` | Full pipeline completes in <30 seconds | All services |

### Manual End-to-End Test

```python
# scripts/test_full_pipeline.py
from src.orchestration.graph import build_pipeline

pipeline = build_pipeline()
result = pipeline.invoke({
    "user_text": "Recent work on diffusion models has shown remarkable "
                 "image generation quality, surpassing GANs on several benchmarks."
})

print(f"Errors: {result['errors']}")
print(f"Recommendations: {len(result['final_recommendations'])}")
for rec in result["final_recommendations"]:
    print(f"  {rec['rank']}. [{rec['confidence']:.0%}] {rec['title']} ({rec['year']})")
    print(f"     {rec['justification'][:100]}...")
```

---

## Key Concepts for Teammates

### What is LangGraph?

LangGraph is a Python library for building AI applications as **state machines** — a graph of steps where:
- Each **node** is a function (in our case, an agent)
- Each **edge** defines what runs next
- A shared **state** object passes between nodes

It's like an assembly line: each station does its job and passes the product to the next station. If one station breaks, we can handle it gracefully instead of the whole line crashing.

### What is "graceful degradation"?

It means the system still works (just with reduced quality) when something goes wrong:
- If citation expansion fails → pipeline uses just the dense retrieval results
- If grounding fails → pipeline shows reranked papers without justifications
- If intent analysis fails → pipeline uses the raw text as the query

This is critical for a demo — you never want the whole thing to crash because one API call timed out.

### What is the AgentState?

It's a Python dictionary (technically a `TypedDict`) that every agent reads from and writes to:

```python
class AgentState(TypedDict):
    user_text: str                                 # Input
    query_analysis: QueryAnalysis | None           # Agent 1 writes
    retrieval_candidates: list[ScoredPaper]        # Agent 2 writes
    expanded_candidates: list[ScoredPaper]         # Agent 3 writes
    reranked_candidates: list[RankedPaper]         # Agent 4A writes
    grounded_candidates: list[GroundedPaper]       # Agent 4B writes
    final_recommendations: list[Recommendation]    # Agent 5 writes
    errors: list[str]                              # Any agent can log errors
    metadata: dict                                 # Timing, token usage, etc.
```

This is fully inspectable at any point — you can see exactly what each agent produced, which is crucial for debugging.

---

## Notes for Teammates

- **This is the biggest integration stage.** Make sure all previous stages pass their tests before starting.
- **Running the full pipeline costs money** (small amounts — ~$0.01 per run from OpenAI calls). Don't run it in a loop.
- **The mocked orchestration tests** are the most important ones — they verify the wiring works without needing any services.
- **If the end-to-end test fails,** check individual agent tests first. The error is usually in one specific agent, not the orchestration.

---

*Completed by: Claude on 2026-02-12*
*Reviewed by: [name] on [date]*

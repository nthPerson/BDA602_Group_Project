# Stage 9 — Streamlit UI

> **Status:** Not Started
> **Depends on:** Stage 7 (full pipeline must be operational)
> **Estimated effort:** 4–6 hours
>
> **Note:** This stage can be developed **in parallel** with Stage 8 (Evaluation), since both depend on Stage 7 and are independent of each other.

---

## What This Stage Builds

This stage creates the **interactive web interface** for the pipeline. After this stage, anyone can open a browser, paste a paragraph of text, and get back citation recommendations with justifications — no Python knowledge required.

### Components Created

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | Complete Streamlit application |
| `tests/test_app/test_streamlit_smoke.py` | Smoke test to verify imports and core functions |

### UI Layout

```
┌────────────────────────────────────────────────────────────────────┐
│  📚 Citation Recommendation System                    [⚙️ Options] │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Paste your paragraph, claim, or draft section:                │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                                                         │  │  │
│  │  │  (text area)                                            │  │  │
│  │  │                                                         │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                               │  │
│  │  [🔍 Find Citations]                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌── Pipeline Status ───────────────────────────────────────────┐  │
│  │  ✅ Intent: METHOD | Keywords: diffusion, image generation   │  │
│  │  ✅ Retrieved 30 candidates (245ms)                          │  │
│  │  ✅ Expanded to 58 candidates (120ms)                        │  │
│  │  ✅ Reranked to top 10 (1.8s)                                │  │
│  │  ✅ Grounded top 5 (2.1s)                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌── Recommendations ──────────────────────────────────────────┐   │
│  │  1. "Denoising Diffusion Probabilistic Models" (2020)   95% │   │
│  │     Ho et al. | Citations: 8,432                             │   │
│  │     ▸ Justification                                          │   │
│  │     ▸ Supporting Snippet                                     │   │
│  │                                                              │   │
│  │  2. "High-Resolution Image Synthesis with..." (2022)    91% │   │
│  │     Rombach et al. | Citations: 5,123                        │   │
│  │     ▸ Justification                                          │   │
│  │     ▸ Supporting Snippet                                     │   │
│  │  ...                                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ▸ Debug Panel (click to expand)                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## UI Features

| Feature | Streamlit Component | Description |
|---|---|---|
| Text input | `st.text_area` | Multi-line input for pasting text |
| Submit button | `st.button` | Triggers the pipeline |
| Pipeline status | `st.status` | Shows real-time progress per agent |
| Recommendation cards | `st.container` + `st.expander` | Ranked results with expandable details |
| Advanced options | `st.sidebar` | Year range slider, result count, debug toggle |
| Debug panel | `st.expander` (collapsed) | Full PipelineState JSON, token usage, latency |
| Error message | `st.error` | Displayed if pipeline fails or input is empty |
| Caching | `st.cache_resource` | Embedding model, Qdrant client, cross-encoder loaded once |

---

## Acceptance Criteria

- [ ] `streamlit run app/streamlit_app.py` launches without errors
- [ ] Pasting a paragraph and clicking "Find Citations" produces visible recommendations
- [ ] Full pipeline completes within 30 seconds (reflected in Pipeline Status)
- [ ] Each recommendation card shows: title, authors, year, citation count, confidence %
- [ ] Expanding a card reveals justification and supporting snippet
- [ ] Pipeline status updates are visible during execution
- [ ] Empty input shows an error message (not a crash)
- [ ] Sidebar options (year range, top-K) affect results
- [ ] Debug panel shows PipelineState JSON and timing

---

## How to Run the UI

```bash
# Prerequisites: virtual env active, Qdrant running, corpus indexed
source .venv/bin/activate
docker compose up -d

# Launch the app
streamlit run app/streamlit_app.py

# This opens your browser to http://localhost:8501
```

---

## How to Test This Stage

This stage is **primarily tested manually** because Streamlit apps are inherently interactive. However, we do have a smoke test.

### Automated Smoke Test

```bash
pytest tests/test_app/test_streamlit_smoke.py -v
```

| Test | What It Checks |
|---|---|
| `test_app_imports` | `app.streamlit_app` imports without errors |
| `test_pipeline_builder_callable` | The pipeline builder function exists and is callable |
| `test_cached_resources_loadable` | Embedding model, Qdrant client load without errors |

### Manual Test Protocol

Run through each test case and check the box:

#### Test Case 1: Transformer Query

- [ ] **Input:** "Recent advances in transformer architectures have shown that scaling model parameters significantly improves few-shot learning capabilities."
- [ ] **Expected:** Top results include papers about transformers, scaling laws, or few-shot learning
- [ ] **Check:** Pipeline Status shows all 5 steps complete
- [ ] **Check:** At least 3 recommendations visible

#### Test Case 2: Reinforcement Learning Query

- [ ] **Input:** "We apply proximal policy optimization to train an agent in a continuous action space for robotic manipulation tasks."
- [ ] **Expected:** Top results include papers about PPO, RL for robotics, or continuous control
- [ ] **Check:** Justifications reference relevant concepts from the papers

#### Test Case 3: Empty Input

- [ ] **Input:** (leave text area empty, click Find Citations)
- [ ] **Expected:** Error message displayed, no crash

#### Test Case 4: Very Short Input

- [ ] **Input:** "attention"
- [ ] **Expected:** Some results returned (may be less accurate due to short context)

#### Test Case 5: Advanced Options

- [ ] **Action:** Set year range to 2022–2025 in sidebar, then submit a query
- [ ] **Expected:** All results have year ≥ 2022

#### Test Case 6: Debug Panel

- [ ] **Action:** Toggle debug panel on, submit a query
- [ ] **Expected:** Full JSON state visible, timing per agent, token usage

---

## Key Concepts for Teammates

### What is Streamlit?

Streamlit is a Python framework that turns Python scripts into web apps. You write Python, and Streamlit handles all the HTML/CSS/JavaScript. No frontend expertise needed.

When you run `streamlit run app.py`, it starts a local web server and opens your browser to it.

### How does `st.cache_resource` work?

The embedding model (~400 MB), cross-encoder (~400 MB), and Qdrant client are loaded **once** when the app starts and shared across all requests. Without caching, every click would reload these models (10+ seconds). With caching, subsequent requests are fast.

### Why real-time status updates?

Users need to know the system is working, not frozen. We use `st.status` to show progress:
- "Analyzing intent..." → ✅ "Intent: METHOD"
- "Retrieving candidates..." → ✅ "30 candidates found (245ms)"
- etc.

This also helps with debugging — if the pipeline hangs, you can see which step it's stuck on.

---

## Notes for Teammates

- **You need ALL prerequisites running** to test the UI: virtual env, Qdrant, indexed corpus, OpenAI API key.
- **Streamlit auto-reloads** when you save changes to `streamlit_app.py`. No need to restart.
- **The first run takes ~30 seconds** because models need to load into memory. Subsequent runs are faster (~5–10s per query).
- **If you see "Connection refused" errors**, make sure Qdrant is running (`docker compose up -d`).
- **UI work is a good task for teammates** with less programming experience — Streamlit is very approachable, and the pipeline is already built. Improvements to layout, styling, or UX are always welcome.

---

*Completed by: [name] on [date]*
*Reviewed by: [name] on [date]*

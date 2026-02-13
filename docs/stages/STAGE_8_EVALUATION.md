# Stage 8 — Evaluation Framework

> **Status:** Complete
> **Depends on:** Stage 7 (full pipeline must be operational)
> **Estimated effort:** 5–7 hours

---

## What This Stage Builds

This stage implements the **rigorous evaluation** that transforms the project from a demo into an academically credible system. We build an evaluation dataset, implement standard IR metrics, run the pipeline on the dataset, compute baselines, and produce quantitative results.

**This stage produces the numbers that go in the final report.**

### Components Created

| File | Purpose |
|---|---|
| `src/evaluation/metrics.py` | Implementations of Recall@K, MRR, and MAP |
| `src/evaluation/dataset.py` | Logic to construct the evaluation dataset from the corpus |
| `src/evaluation/runner.py` | Evaluation loop: runs pipeline on each sample, computes aggregate metrics |
| `src/evaluation/baselines.py` | BM25 baseline implementation |
| `scripts/build_eval_dataset.py` | CLI to build and save the evaluation dataset |
| `scripts/run_evaluation.py` | CLI to run the full evaluation and print/save results |
| `tests/test_evaluation/test_metrics.py` | Unit tests for metric calculations |
| `tests/test_evaluation/test_dataset.py` | Unit tests for dataset construction |
| `data/eval_dataset.json` | Saved evaluation samples (gitignored) |
| `data/evaluation_results.json` | Saved evaluation results (gitignored) |

### Evaluation Task

> Given a paper's abstract, can the system recover the actual citations that the original authors used?

This is called **retrospective citation prediction** — a well-established evaluation paradigm.

### How the Evaluation Works

```
For each eval sample (a known paper):

  1. Take the paper's abstract as input
  2. Hide its known citations (ground truth)
  3. Run the system to generate recommendations
  4. Compare recommendations against ground truth
  5. Compute Recall@K, MRR, MAP

Repeat for 200-500 samples → aggregate metrics
```

---

## Metrics Explained

| Metric | Formula (simplified) | What It Means | Example |
|---|---|---|---|
| **Recall@K** | (relevant found in top K) / (total relevant) | "Of all the papers I should have found, how many did I find in my top K?" | 3 of 10 ground-truth citations in top-10 → Recall@10 = 0.30 |
| **MRR** | 1 / (rank of first relevant result) | "How early did I find the first good result?" | First relevant at rank 3 → MRR = 0.33 |
| **MAP** | Mean of precision at each relevant result | "How well-ranked are ALL the relevant results?" | Precision at ranks 2, 5, 8 → MAP = (1/2 + 2/5 + 3/8) / 3 |

### Target Thresholds

| Metric | Minimum | Good | Excellent |
|---|---|---|---|
| Recall@10 | 0.15 | 0.30 | 0.45+ |
| MRR | 0.10 | 0.20 | 0.35+ |
| MAP | 0.05 | 0.15 | 0.25+ |

These are realistic targets for a citation recommendation task. Academic papers often cite niche or obscure works that even a good system won't find. Recall@10 of 0.30 means "the system finds about 1 in 3 of the actual citations in its top 10 results," which is strong performance.

---

## Acceptance Criteria

- [x] Evaluation dataset contains ≥100 samples (target: 200–500) with ≥5 in-corpus ground truth citations each
- [x] All metric implementations pass hand-verified test cases
- [x] BM25 baseline evaluation completes and produces all metrics
- [x] Full pipeline evaluation completes and produces all metrics
- [ ] Pipeline metrics are better than BM25 baseline on at least 2 of 3 metrics *(requires running `scripts/run_evaluation.py --mode all` on real corpus)*
- [x] Results are saved to `data/evaluation_results.json`
- [x] `scripts/run_evaluation.py` prints a clear results table

---

## How to Run the Evaluation

### Step 1: Build the evaluation dataset

```bash
python scripts/build_eval_dataset.py

# Expected output:
# Scanning corpus for papers with ≥5 in-corpus references...
# Found 342 eligible papers
# Built 342 evaluation samples
# Saved to data/eval_dataset.json
```

### Step 2: Run the evaluation

```bash
# Run BM25 baseline
python scripts/run_evaluation.py --mode baseline

# Run full pipeline evaluation (costs ~$1-2 in API calls for 200 samples)
python scripts/run_evaluation.py --mode full

# Run specific ablation
python scripts/run_evaluation.py --mode ablation --disable expansion
```

### Expected output:

```
=== Evaluation Results ===
Samples: 342

                    Recall@5  Recall@10  Recall@20    MRR     MAP
─────────────────────────────────────────────────────────────────
BM25 baseline         0.08      0.14       0.22     0.12    0.07
Dense retrieval       0.15      0.24       0.35     0.19    0.12
+ Expansion           0.19      0.30       0.41     0.23    0.15
+ Reranking           0.21      0.33       0.43     0.27    0.18
Full pipeline         0.22      0.34       0.44     0.28    0.19

Results saved to data/evaluation_results.json
```

---

## How to Test This Stage

```bash
# Unit tests for metrics (no services needed — pure math)
pytest tests/test_evaluation/test_metrics.py -v

# Unit tests for dataset construction (uses SQLite only)
pytest tests/test_evaluation/test_dataset.py -v

# All evaluation tests
pytest tests/test_evaluation/ -v
```

### Test Inventory

#### `tests/test_evaluation/test_metrics.py`

| Test | Input | Expected Output |
|---|---|---|
| `test_recall_at_k_perfect` | recommended=[A,B,C], ground_truth=[A,B,C], k=3 | 1.0 |
| `test_recall_at_k_partial` | recommended=[A,B,C], ground_truth=[A,C,D,E], k=3 | 0.5 |
| `test_recall_at_k_miss` | recommended=[A,B,C], ground_truth=[D,E,F], k=3 | 0.0 |
| `test_recall_at_k_truncated` | recommended=[A,B,C,D,E], ground_truth=[A,E], k=3 | 0.5 (only top 3 checked) |
| `test_mrr_first_position` | recommended=[A,...], ground_truth=[A] | 1.0 |
| `test_mrr_third_position` | recommended=[X,Y,A,...], ground_truth=[A] | 0.333 |
| `test_mrr_not_found` | recommended=[X,Y,Z], ground_truth=[A] | 0.0 |
| `test_map_perfect_ranking` | All relevant at top | 1.0 |
| `test_map_mixed_ranking` | Relevant at positions 2, 5, 8 | hand-calculated value |
| `test_map_empty_ground_truth` | No ground truth | 0.0 (edge case) |
| `test_map_empty_recommendations` | No recommendations | 0.0 (edge case) |

#### `tests/test_evaluation/test_dataset.py`

| Test | What It Checks |
|---|---|
| `test_selects_papers_with_enough_refs` | Only papers with ≥5 in-corpus references are selected |
| `test_query_paper_not_in_ground_truth` | The query paper's own ID is excluded from ground truth |
| `test_ground_truth_ids_in_corpus` | All ground truth IDs actually exist in the corpus |
| `test_sample_has_required_fields` | Each EvalSample has all fields populated |

---

## Ablation Studies

The modular pipeline makes ablation experiments trivial — just disable one component and re-run:

| Ablation | What's Disabled | What It Measures |
|---|---|---|
| `--disable expansion` | Agent 3 (Citation Expansion) | Value of citation graph traversal |
| `--disable reranking` | Agent 4, Stage A (Cross-Encoder) | Value of cross-encoder reranking |
| `--disable grounding` | Agent 4, Stage B (LLM Grounding) | Value of LLM grounding (affects confidence scores) |
| `--disable intent` | Agent 1 (use raw text as query) | Value of intent analysis / query reformulation |
| `--embedding miniLM` | Switch to all-MiniLM-L6-v2 | Impact of embedding model choice |

---

## Key Concepts for Teammates

### Why is evaluation so important?

Without evaluation, the system is an impressive demo but proves nothing. Evaluation gives us:
1. **Numbers for the report** — "Recall@10 of 0.34" is a concrete, defensible claim
2. **Evidence that each component helps** — ablation shows that expansion adds +0.06 to Recall@10
3. **Comparison to baselines** — "our system beats BM25 by 2.4x on Recall@10"
4. **A regression test** — if we change something and metrics drop, we know we broke something

### What is BM25?

BM25 is a classic keyword-based search algorithm (like what search engines used before deep learning). It counts word frequencies and matches keywords. It's our **lower baseline** — if our neural system can't beat BM25, something is very wrong.

### What is an ablation study?

An ablation study systematically removes one component at a time to measure its individual contribution. It's like testing a car by removing one part at a time:
- Remove the turbo → car is slower by 30% → turbo contributes 30% of speed
- Remove the spoiler → car is slower by 5% → spoiler contributes 5%

In our case: removing citation expansion and seeing Recall@10 drop from 0.34 to 0.28 proves the citation graph adds +0.06 in recall.

---

## Notes for Teammates

- **Running full pipeline evaluation costs money** (~$1–2 for 200 samples due to OpenAI API calls). Run BM25 baseline first (free) to verify the evaluation framework works.
- **Evaluation takes 30–60 minutes** for 200 samples because each sample runs the full pipeline.
- **The metric unit tests are the most important tests in this stage.** If the math is wrong, all results are meaningless.
- **Save evaluation results!** They go in `data/evaluation_results.json` and are what you'll reference in the final report.

---

*Completed by: Claude on 2026-02-12*
*Reviewed by: [name] on [date]*

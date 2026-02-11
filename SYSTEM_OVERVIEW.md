# Agentic Academic Paper Citation Recommendation System — Detailed Technical Overview

> **Purpose of this document:** This is the elaborated technical overview derived from the initial design draft. It is intended to serve as the authoritative reference for constructing the development roadmap. Every architectural choice documented here has been evaluated for single-developer feasibility within an 8-week timeline, and favors battle-tested, well-documented components over novel or experimental alternatives.

---

## Table of Contents

1. [Project Objective & Scope](#1-project-objective--scope)
2. [Core Constraints & Design Drivers](#2-core-constraints--design-drivers)
3. [High-Level System Architecture](#3-high-level-system-architecture)
4. [Data Layer](#4-data-layer)
5. [Agent Specifications](#5-agent-specifications)
6. [Orchestration Layer](#6-orchestration-layer)
7. [User Interface](#7-user-interface)
8. [Evaluation Strategy](#8-evaluation-strategy)
9. [Project Structure](#9-project-structure)
10. [Dependency Stack](#10-dependency-stack)
11. [Testing Strategy](#11-testing-strategy)
12. [Risk Mitigation & Fallback Plans](#12-risk-mitigation--fallback-plans)

---

## 1. Project Objective & Scope

### What the System Does

Build an **agentic Retrieval-Augmented Generation (RAG)** system that, given a paragraph, claim, or draft section from an AI/ML research paper, automatically recommends relevant research papers as citation candidates. For each recommendation, the system provides:

- **Ranked citation candidates** with confidence scores
- **Supporting snippets** extracted from each recommended paper
- **Natural-language justification** explaining why each paper is relevant to the input text

### What Makes It "Agentic"

The term "agentic" here has a precise, defensible meaning: unlike a simple RAG pipeline (embed → retrieve → generate), this system decomposes the citation recommendation task into a **multi-agent pipeline** where each agent has a distinct responsibility, its own inputs/outputs contract, and is orchestrated via an explicit state machine. The pipeline models a real researcher's workflow:

```
understand intent → retrieve candidates → expand via citation graph → rerank → justify → present
```

This is not a single prompt-in/answer-out system. The agents perform **structured reasoning steps** — intent classification, graph traversal, cross-encoder reranking, and grounded justification — each of which is independently testable, observable, and improvable. The orchestration layer (LangGraph) provides deterministic state transitions, retry logic, and full traceability.

### What It Is NOT

- **Not a novel model.** No new architectures are trained. The project uses pre-trained embeddings, cross-encoders, and LLM APIs.
- **Not a general-purpose citation tool.** Scoped to AI/ML papers (CS.AI, CS.LG, CS.CL, CS.CV on arXiv).
- **Not a production SaaS.** It is a demoable, evaluated prototype with real metrics — suitable for an academic project deliverable, not deployment at scale.

### Success Criteria

| Criterion | Threshold |
|---|---|
| Functional end-to-end pipeline | All 5 agents operational, producing structured output |
| Quantitative retrieval metrics | Recall@10 ≥ 0.30, MRR ≥ 0.20 on held-out evaluation set |
| Interactive UI | Working Streamlit app accepting text input and displaying ranked results |
| Engineering narrative | Clear documentation of architecture, ablation results, and design decisions |

---

## 2. Core Constraints & Design Drivers

Every architectural decision in this document is downstream of these four hard constraints:

### 2.1 Time: 8 Weeks

The project must go from zero to a **complete, demoable, evaluated system** in approximately 8 weeks. This means:

- No time for debugging obscure framework issues — use mature, well-documented tools
- No time for hyperparameter sweeps — use known-good defaults
- No time for complex infrastructure — everything runs locally or via simple Docker containers
- The build order must be strictly incremental: each week should produce a working (if limited) system

### 2.2 Staffing: Single Developer

Architecture must be:

- **Modular:** Each component is independently testable and replaceable
- **Debuggable:** Every intermediate result is inspectable (no opaque chains)
- **Incrementally buildable:** The system works at each stage of completion (e.g., a retrieval-only pipeline works before reranking is added)

### 2.3 Risk Tolerance: Low

- **No exotic models.** Stick to widely-used embeddings (sentence-transformers, BGE) and cross-encoders (ms-marco).
- **No bleeding-edge frameworks.** LangGraph is mature enough (v0.2+) and well-documented. Avoid anything in alpha/beta.
- **No custom training.** All models are pre-trained and used off-the-shelf.
- **API-based LLM.** Use OpenAI (gpt-4o-mini) as primary LLM — the most battle-tested API with the best structured-output support. Avoid self-hosted LLMs unless the API becomes a blocker.

### 2.4 Success Is Measured by Engineering, Not Novelty

The project demonstrates:

- Agent orchestration design
- Multi-stage retrieval pipeline engineering
- Reranking and grounding techniques
- Rigorous evaluation methodology
- End-to-end ML system building

This is an ML **engineering** project, not a research contribution.

---

## 3. High-Level System Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User (Streamlit UI)                        │
│                    Inputs: paragraph / claim / draft                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer (LangGraph)                  │
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐  │
│  │  Agent 1:    │     │  Agent 2:    │     │  Agent 3:            │  │
│  │  Query &     │───▶│  Primary      │───▶│  Citation            │  │
│  │  Intent      │     │  Retrieval   │     │  Expansion           │  │
│  └──────────────┘     └──────────────┘     └──────────┬───────────┘  │
│                                                       │              │
│  ┌──────────────────────┐     ┌───────────────────────┘              │
│  │  Agent 5:            │     │                                      │
│  │  Synthesis           │◀───┤                                       |
│  │                      │     │       ┌──────────────────────┐       │
│  └──────────┬───────────┘     └───────│  Agent 4:            │       │
│             │                         │  Reranking +         │       │
│             │                         │  Grounding           │       │
│             │                         └──────────────────────┘       │
└─────────────┼────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Streamlit UI — Results Display                   │
│            Ranked citations / justifications / snippets              │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Infrastructure (Backing Stores)

```
┌─────────────────────────────────┐     ┌───────────────────────────────┐
│         Qdrant (Docker)         │     │    SQLite (Local File)        │
│                                 │     │                               │
│  • Paper embeddings             │     │  • Paper metadata             │
│  • Payload-filtered search      │     │  • Citation graph edges       │
│  • 384-dim or 768-dim vectors   │     │  • references / cited-by      │
│                                 │     │  • Evaluation ground truth    │
└─────────────────────────────────┘     └───────────────────────────────┘
```

### Key Architectural Principle: The Pipeline State Object

The orchestrator maintains a single, typed **`PipelineState`** object that accumulates results from each agent. This is the "single source of truth" for a given run. Every agent reads from and writes to this state. It is fully inspectable at any point — critical for debugging and evaluation.

```python
@dataclass
class PipelineState:
    # Input
    user_text: str

    # Agent 1 output
    query_analysis: QueryAnalysis | None = None

    # Agent 2 output
    retrieval_candidates: list[ScoredPaper] = field(default_factory=list)

    # Agent 3 output
    expanded_candidates: list[ScoredPaper] = field(default_factory=list)

    # Agent 4 output
    reranked_candidates: list[RankedPaper] = field(default_factory=list)
    grounded_candidates: list[GroundedPaper] = field(default_factory=list)

    # Agent 5 output
    final_recommendations: list[Recommendation] = field(default_factory=list)

    # Metadata
    run_id: str = ""
    timestamps: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
```

---

## 4. Data Layer

### 4.1 Paper Corpus

#### Sources

| Source | Role | Why |
|---|---|---|
| **OpenAlex** | Primary source for metadata + citation graph | Free, no auth required, excellent coverage, has `references` and `cited_by` fields, well-structured API |
| **arXiv** | Supplement for full-text access | Free bulk access, covers CS.AI/LG/CL/CV comprehensively |

#### Corpus Scope

- **Domains:** CS.AI, CS.LG, CS.CL, CS.CV (mapped to OpenAlex concepts/topics)
- **Time range:** 2018–2025 (keeps the corpus modern and relevant)
- **Target size:** ~15,000–25,000 papers
- **Content per paper:** title + abstract + (optionally) introduction section

This size is the sweet spot for this project:
- **Large enough** to produce meaningful retrieval behavior and realistic evaluation
- **Small enough** that indexing completes in minutes, iteration is fast, and storage is manageable on a single machine (~2–5 GB for embeddings)
- **Comparable** to similar academic retrieval benchmarks (e.g., TREC-COVID, SCIDOCS)

#### Data Acquisition Strategy

**OpenAlex** is the primary acquisition path because it provides everything needed in a single API:

```python
# Pseudocode for OpenAlex corpus building
# Uses the pyalex library (pip install pyalex)

import pyalex
from pyalex import Works

# Configure polite pool (faster rate limits with email)
pyalex.config.email = "your@email.edu"

# Query for AI/ML papers, 2018-2025, with abstracts
papers = (
    Works()
    .filter(
        concepts={"id": "C154945302|C119857082|C204321447"},  # AI | ML | NLP
        from_publication_date="2018-01-01",
        has_abstract=True,
        type="article",
    )
    .sort(cited_by_count="desc")
    .paginate(per_page=200)
)
```

**arXiv** is used selectively — only when full-text (beyond abstracts) is needed for specific highly-cited papers. The `arxiv` Python package handles this cleanly.

#### Data Freshness

The corpus is built once during the data collection phase and is **static** for the duration of the project. There is no live ingestion pipeline — that would add complexity without proportional value for an 8-week project.

### 4.2 Data Model

#### Paper Record

```python
@dataclass
class Paper:
    paper_id: str            # OpenAlex work ID (e.g., "W2100837269")
    title: str
    abstract: str
    year: int
    citation_count: int
    doi: str | None
    arxiv_id: str | None
    authors: list[str]       # List of author display names
    concepts: list[str]      # OpenAlex concept labels
    source: str | None       # Journal/venue name
    references: list[str]    # List of OpenAlex work IDs this paper cites
    cited_by_count: int      # Total citation count (for filtering/ranking)
    chunk_texts: list[str]   # Chunked text segments (abstract + optional sections)
```

#### Citation Graph Edge

```python
@dataclass
class CitationEdge:
    source_id: str   # The paper that cites
    target_id: str   # The paper that is cited
```

This enables both "references of" (forward) and "cited by" (backward) traversal.

### 4.3 Storage Architecture

#### SQLite — Relational Metadata Store

**Why SQLite:**
- Zero-config, file-based — no server to manage
- Perfect for a single-developer project
- Python's `sqlite3` is in the standard library
- Excellent for relational queries on the citation graph
- Fast enough for 25k records by orders of magnitude

**Schema:**

```sql
CREATE TABLE papers (
    paper_id     TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL,
    year         INTEGER,
    citation_count INTEGER,
    doi          TEXT,
    arxiv_id     TEXT,
    authors      TEXT,          -- JSON array
    concepts     TEXT,          -- JSON array
    source       TEXT
);

CREATE TABLE citation_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES papers(paper_id),
    FOREIGN KEY (target_id) REFERENCES papers(paper_id)
);

CREATE INDEX idx_citations_source ON citation_edges(source_id);
CREATE INDEX idx_citations_target ON citation_edges(target_id);
CREATE INDEX idx_papers_year ON papers(year);
CREATE INDEX idx_papers_citation_count ON papers(citation_count);
```

#### Qdrant — Vector Store

**Why Qdrant (over alternatives):**

| Factor | Qdrant | Chroma | FAISS | pgvector |
|---|---|---|---|---|
| Local deployment | Docker one-liner | In-process | In-process | Requires PostgreSQL |
| Payload filtering | Excellent | Basic | None | SQL-based |
| Python client | Mature, typed | Yes | Lower-level | via SQLAlchemy |
| Persistence | Built-in | SQLite backend | Manual | Built-in |
| Production path | Clear | Limited | Manual | Yes |
| Scalability story | Good | Limited | Manual | Good |

Qdrant is the best balance of **simplicity, capability, and production-readiness** for this project.

**Deployment:**

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.5
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
```

**Collection Configuration:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="papers",
    vectors_config=VectorParams(
        size=768,                    # Matches embedding model output dim
        distance=Distance.COSINE,
    ),
)

# Create payload indexes for filtered search
client.create_payload_index("papers", "year", PayloadSchemaType.INTEGER)
client.create_payload_index("papers", "citation_count", PayloadSchemaType.INTEGER)
client.create_payload_index("papers", "concepts", PayloadSchemaType.KEYWORD)
```

### 4.4 Embedding Strategy

#### Model Selection

| Model | Dims | Speed | Quality (MTEB) | Domain Fit | Choice |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Very fast | Good | General | Fallback |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Very good | General | **Primary** |
| `allenai/specter2` | 768 | Slower | Good | Academic | Consider for ablation |

**Primary choice: `BAAI/bge-base-en-v1.5`**

Rationale:
- Top-tier performance on MTEB retrieval benchmarks
- 768 dimensions — good quality-to-size ratio
- Well-supported by sentence-transformers
- Widely used in production RAG systems
- Instruction-prefixed queries improve retrieval quality ("Represent this sentence for searching relevant passages: ...")

SPECTER2 is tempting because it's trained specifically on scientific text, but BGE-base consistently outperforms it on general retrieval benchmarks and has better tooling support. SPECTER2 can be tested in an ablation study if time permits.

#### Chunking Strategy

For the initial build, **each paper gets a single embedding of its title + abstract concatenation.** This is the simplest approach and is sufficient for abstract-level retrieval.

```python
def make_paper_text(paper: Paper) -> str:
    return f"{paper.title}\n\n{paper.abstract}"
```

If time permits, papers with full text can be chunked into ~512-token segments using a sentence-boundary-aware chunker (e.g., LangChain's `RecursiveCharacterTextSplitter`). Each chunk gets its own embedding and links back to the parent `paper_id`.

#### Embedding Pipeline

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Batch embedding (GPU-accelerated if available)
texts = [make_paper_text(p) for p in papers]
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,  # Required for cosine similarity
)
```

For ~20k papers, embedding takes approximately **10–20 minutes on a GPU** or **1–2 hours on CPU**. This is done once.

---

## 5. Agent Specifications

### 5.1 Agent 1 — Query & Intent Analysis

#### Purpose

Transform raw user text (a paragraph, claim, or draft section) into a structured query representation that downstream agents can consume. This mimics the first step a researcher takes: "What am I actually looking for?"

#### Input/Output Contract

**Input:**
```python
@dataclass
class Agent1Input:
    user_text: str  # The raw paragraph/claim from the user
```

**Output:**
```python
@dataclass
class QueryAnalysis:
    topic_keywords: list[str]       # Key technical terms extracted
    citation_intent: CitationIntent  # Enum: BACKGROUND | METHOD | COMPARISON | BENCHMARK
    expanded_query: str             # Retrieval-optimized reformulation
    confidence: float               # Agent's confidence in its analysis (0-1)
```

```python
class CitationIntent(str, Enum):
    BACKGROUND = "background"    # Citing for general context/motivation
    METHOD = "method"            # Citing a specific technique being used/extended
    COMPARISON = "comparison"    # Citing work being compared against
    BENCHMARK = "benchmark"      # Citing datasets/benchmarks/evaluation standards
```

#### Implementation

**LLM:** `gpt-4o-mini` via OpenAI API
**Structured output:** Use OpenAI's native structured output mode (JSON Schema / function calling) or the `instructor` library for Pydantic model extraction.

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())

def analyze_query(user_text: str) -> QueryAnalysis:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=QueryAnalysis,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI research librarian. Analyze the given text "
                    "from an AI/ML research paper and extract structured information "
                    "to help find relevant citation candidates. "
                    "Identify the key technical concepts, determine what type of "
                    "citation is needed, and reformulate the text into an effective "
                    "retrieval query."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        max_retries=2,
    )
```

**Why `instructor`:**
- Built on top of OpenAI's native function calling
- Automatic retries with validation feedback
- Pydantic model validation ensures type safety
- Well-tested in production systems
- Tiny dependency footprint

**Why `gpt-4o-mini`:**
- Fast (~500ms typical latency)
- Cheap (~$0.15/1M input tokens)
- Excellent at structured extraction tasks
- Native JSON mode support
- Most reliable API in the ecosystem

#### Error Handling

- If the LLM fails to produce valid structured output after 2 retries, fall back to a simplified keyword extraction using TF-IDF over the input text
- If the API is unreachable, log the error, set `citation_intent` to `BACKGROUND` (safest default), and use the raw user text as the `expanded_query`

### 5.2 Agent 2 — Primary Retrieval

#### Purpose

Retrieve the initial set of candidate papers from the vector store using dense embedding similarity, optionally filtered by metadata. This is **pure engineering** — no LLM reasoning involved.

#### Input/Output Contract

**Input:**
```python
@dataclass
class Agent2Input:
    query_analysis: QueryAnalysis  # From Agent 1
    top_n: int = 30                # Number of candidates to retrieve
    year_min: int | None = None    # Optional year filter
    year_max: int | None = None
```

**Output:**
```python
@dataclass
class ScoredPaper:
    paper_id: str
    title: str
    abstract: str
    year: int
    citation_count: int
    similarity_score: float  # Cosine similarity from Qdrant
    concepts: list[str]
```

Returns: `list[ScoredPaper]` of length ≤ `top_n`

#### Implementation

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer

class PrimaryRetrievalAgent:
    def __init__(self, qdrant: QdrantClient, embedder: SentenceTransformer):
        self.qdrant = qdrant
        self.embedder = embedder

    def retrieve(self, input: Agent2Input) -> list[ScoredPaper]:
        # Embed the expanded query (with BGE instruction prefix)
        query_text = f"Represent this sentence for searching relevant passages: {input.query_analysis.expanded_query}"
        query_vector = self.embedder.encode(query_text, normalize_embeddings=True)

        # Build optional metadata filters
        conditions = []
        if input.year_min:
            conditions.append(
                FieldCondition(key="year", range=Range(gte=input.year_min))
            )
        if input.year_max:
            conditions.append(
                FieldCondition(key="year", range=Range(lte=input.year_max))
            )

        search_filter = Filter(must=conditions) if conditions else None

        # Search Qdrant
        results = self.qdrant.search(
            collection_name="papers",
            query_vector=query_vector.tolist(),
            query_filter=search_filter,
            limit=input.top_n,
            with_payload=True,
        )

        return [self._to_scored_paper(hit) for hit in results]
```

#### Design Notes

- The BGE instruction prefix (`"Represent this sentence for searching relevant passages: "`) is **critical** — it was part of the model's training and significantly improves retrieval quality
- No LLM call is needed here — this keeps the agent fast (<100ms typical) and deterministic
- Qdrant's payload filtering is applied **before** the ANN search, so metadata filters don't degrade retrieval speed

### 5.3 Agent 3 — Citation Expansion

#### Purpose

Expand the candidate set by traversing the citation graph. For the top-K retrieved papers, pull their references and cited-by neighbors. This is the **key differentiator** from vanilla RAG — it leverages the structure of the academic citation network.

#### Input/Output Contract

**Input:**
```python
@dataclass
class Agent3Input:
    top_candidates: list[ScoredPaper]  # Top K from Agent 2 (e.g., K=10)
    max_expansion: int = 40            # Max additional papers to add
```

**Output:**
```python
list[ScoredPaper]  # Union of original candidates + expanded candidates (deduplicated)
```

#### Implementation

```python
class CitationExpansionAgent:
    def __init__(self, db: sqlite3.Connection, qdrant: QdrantClient):
        self.db = db
        self.qdrant = qdrant

    def expand(self, input: Agent3Input) -> list[ScoredPaper]:
        seed_ids = {p.paper_id for p in input.top_candidates}
        expansion_ids: set[str] = set()

        for paper in input.top_candidates[:10]:  # Expand from top 10
            # Forward references (papers this one cites)
            refs = self._get_references(paper.paper_id)
            expansion_ids.update(refs)

            # Backward citations (papers that cite this one)
            cited_by = self._get_cited_by(paper.paper_id)
            expansion_ids.update(cited_by)

        # Remove papers already in seed set
        expansion_ids -= seed_ids

        # Filter to only papers in our corpus
        expansion_ids = self._filter_to_corpus(expansion_ids)

        # Limit expansion size
        expansion_list = list(expansion_ids)[:input.max_expansion]

        # Fetch metadata for expanded papers
        expanded_papers = self._fetch_papers(expansion_list)

        # Combine: originals + expanded (with source tag)
        return input.top_candidates + expanded_papers

    def _get_references(self, paper_id: str) -> list[str]:
        cursor = self.db.execute(
            "SELECT target_id FROM citation_edges WHERE source_id = ?",
            (paper_id,)
        )
        return [row[0] for row in cursor.fetchall()]

    def _get_cited_by(self, paper_id: str) -> list[str]:
        cursor = self.db.execute(
            "SELECT source_id FROM citation_edges WHERE target_id = ?",
            (paper_id,)
        )
        return [row[0] for row in cursor.fetchall()]
```

#### Design Notes

- **Depth-1 BFS only.** Going deeper (references of references) explodes the candidate set without proportional quality gain. Keep it simple.
- **Prioritization heuristic:** If the expansion set exceeds `max_expansion`, prioritize by: (1) papers that appear in multiple seed papers' citation lists, (2) higher citation count. This is a simple, effective heuristic.
- **Performance:** SQLite lookups on indexed columns are sub-millisecond. The entire expansion step should complete in <500ms for typical inputs.
- The expanded candidates don't have similarity scores yet — they'll be scored in Agent 4.

### 5.4 Agent 4 — Reranking + Grounding

This is a two-stage agent that is the most technically interesting component.

#### Stage A: Cross-Encoder Reranking

##### Purpose

Re-score all ~50–70 candidates (from Agents 2+3) using a cross-encoder model that jointly encodes the query and each candidate. Cross-encoders are significantly more accurate than bi-encoders for reranking because they can attend to fine-grained interactions between query and document.

##### Model Choice

**`BAAI/bge-reranker-base`** (or `cross-encoder/ms-marco-MiniLM-L-6-v2` as a lighter alternative)

| Model | Quality | Speed | Notes |
|---|---|---|---|
| `BAAI/bge-reranker-base` | Very good | ~50ms/pair | Best quality for the size |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Good | ~20ms/pair | Fastest, good fallback |

##### Implementation

```python
from sentence_transformers import CrossEncoder

class RerankingStage:
    def __init__(self):
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(
        self,
        query: str,
        candidates: list[ScoredPaper],
        top_k: int = 10,
    ) -> list[RankedPaper]:
        # Prepare pairs for cross-encoder
        pairs = [(query, f"{c.title}\n{c.abstract}") for c in candidates]

        # Score all pairs in one batch
        scores = self.cross_encoder.predict(pairs, batch_size=32)

        # Sort by cross-encoder score
        scored = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        return [
            RankedPaper(paper=paper, rerank_score=float(score))
            for paper, score in scored[:top_k]
        ]
```

For ~50 candidates, reranking takes **~2–3 seconds on CPU** or **<0.5s on GPU**. This is acceptable.

#### Stage B: LLM Grounding

##### Purpose

For each of the top-5 reranked papers, use the LLM to: (1) extract the most relevant supporting snippet from the paper's abstract, and (2) generate a natural-language relevance justification. This is the **grounding** step — it forces the system to explain its recommendations with evidence, reducing hallucination.

##### Implementation

```python
@dataclass
class GroundedPaper:
    paper_id: str
    title: str
    rerank_score: float
    supporting_snippet: str    # Extracted from the paper
    justification: str         # LLM-generated explanation
    confidence: float          # LLM's self-assessed confidence

class GroundingStage:
    def ground(self, user_text: str, paper: RankedPaper) -> GroundedPaper:
        return client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=GroundedPaper,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research citation analyst. Given an input text "
                        "and a candidate paper, extract the most relevant snippet "
                        "from the candidate paper's abstract and explain why this "
                        "paper is a good citation for the input text. Be specific "
                        "and grounded — only reference information actually present "
                        "in the candidate paper's abstract."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Input text:\n{user_text}\n\n"
                               f"Candidate paper:\nTitle: {paper.title}\n"
                               f"Abstract: {paper.abstract}",
                },
            ],
        )
```

##### Design Notes

- **5 parallel LLM calls** can be made simultaneously to reduce latency (using `asyncio.gather` with `AsyncOpenAI`)
- Each grounding call takes ~1–2s, so with parallelization the full grounding stage completes in ~2s
- The grounding prompt is deliberately restrictive ("only reference information actually present") to minimize hallucination

### 5.5 Agent 5 — Synthesis

#### Purpose

Assemble the final output: a ranked list of recommended papers with justifications, snippets, and confidence scores. This agent also performs light quality control — filtering out low-confidence recommendations and ensuring the output is well-structured.

#### Output Format

```python
@dataclass
class Recommendation:
    rank: int
    paper_id: str
    title: str
    authors: list[str]
    year: int
    citation_count: int
    justification: str
    supporting_snippet: str
    confidence: float
    citation_intent_match: str  # How well this matches the detected intent
```

#### Implementation

The synthesis agent is the thinnest agent — mostly formatting and quality filtering:

```python
class SynthesisAgent:
    def synthesize(
        self,
        grounded_papers: list[GroundedPaper],
        query_analysis: QueryAnalysis,
        confidence_threshold: float = 0.4,
    ) -> list[Recommendation]:
        # Filter by confidence
        filtered = [p for p in grounded_papers if p.confidence >= confidence_threshold]

        # Sort by composite score (rerank_score * confidence)
        filtered.sort(
            key=lambda p: p.rerank_score * p.confidence, reverse=True
        )

        # Build final recommendations
        return [
            Recommendation(
                rank=i + 1,
                paper_id=p.paper_id,
                title=p.title,
                # ... fill remaining fields from metadata lookup
                justification=p.justification,
                supporting_snippet=p.supporting_snippet,
                confidence=p.confidence,
                citation_intent_match=self._assess_intent_match(p, query_analysis),
            )
            for i, p in enumerate(filtered)
        ]
```

---

## 6. Orchestration Layer

### 6.1 Why LangGraph

LangGraph (from LangChain) provides:

- **Explicit state graphs** — not implicit chains. You define nodes and edges directly.
- **Typed state** — the state schema is a Python dataclass/TypedDict, enforced at each step.
- **Deterministic transitions** — no probabilistic routing (unless you want it).
- **Built-in checkpointing** — can resume from any state (useful for debugging).
- **Retry and error hooks** — configurable retry logic per node.
- **Streaming support** — intermediate results can be streamed to the UI.

LangGraph is chosen over raw LangChain because it is more **explicit, debuggable, and aligned with the state machine model** described in this design. It avoids the "prompt soup" problem of chained LLM calls with no structure.

### 6.2 State Machine Definition

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    user_text: str
    query_analysis: QueryAnalysis | None
    retrieval_candidates: list[ScoredPaper]
    expanded_candidates: list[ScoredPaper]
    reranked_candidates: list[RankedPaper]
    grounded_candidates: list[GroundedPaper]
    final_recommendations: list[Recommendation]
    errors: list[str]
    metadata: dict

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes (each is a function that takes and returns AgentState)
workflow.add_node("query_analysis", run_query_agent)
workflow.add_node("primary_retrieval", run_retrieval_agent)
workflow.add_node("citation_expansion", run_expansion_agent)
workflow.add_node("reranking", run_reranking_agent)
workflow.add_node("grounding", run_grounding_agent)
workflow.add_node("synthesis", run_synthesis_agent)

# Define edges (linear pipeline)
workflow.set_entry_point("query_analysis")
workflow.add_edge("query_analysis", "primary_retrieval")
workflow.add_edge("primary_retrieval", "citation_expansion")
workflow.add_edge("citation_expansion", "reranking")
workflow.add_edge("reranking", "grounding")
workflow.add_edge("grounding", "synthesis")
workflow.add_edge("synthesis", END)

# Compile
app = workflow.compile()
```

### 6.3 Error Handling & Retry Strategy

Each node wrapper implements:

1. **Try/except** around the core logic
2. **Retry** with exponential backoff for LLM calls (max 2 retries)
3. **Graceful degradation:** If an agent fails, log the error to `state["errors"]` and pass through with the best available data. For example, if citation expansion fails, the pipeline continues with just the primary retrieval results.
4. **Timeout:** Each agent has a configurable timeout (default: 30s for LLM agents, 10s for retrieval agents)

### 6.4 Observability

Every agent logs:
- Input/output sizes
- Latency (wall clock time)
- Any errors or retries
- LLM token usage (for cost tracking)

This data is stored in `state["metadata"]` and displayed in the Streamlit UI's debug panel.

---

## 7. User Interface

### 7.1 Technology Choice: Streamlit

**Why Streamlit:**
- Fastest path from Python to interactive web UI
- Native support for session state, caching, and streaming
- No frontend/JavaScript skills required
- Widely used in ML demos and data apps
- Sufficient for a non-production prototype

### 7.2 UI Layout

```
┌──────────────────────────────────────────────────────────────────┐
│                    Citation Recommendation System                  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Paste your paragraph, claim, or draft section below:        │ │
│  │                                                               │ │
│  │  ┌──────────────────────────────────────────────────────────┐│ │
│  │  │                                                          ││ │
│  │  │  [Text area — multi-line input]                          ││ │
│  │  │                                                          ││ │
│  │  └──────────────────────────────────────────────────────────┘│ │
│  │                                                               │ │
│  │  [Find Citations]  ⚙️ Advanced options (year range, top-K)   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── Pipeline Status ──────────────────────────────────────────┐  │
│  │  ✓ Intent: METHOD | Keywords: attention, transformer        │  │
│  │  ✓ Retrieved 30 candidates (245ms)                          │  │
│  │  ✓ Expanded to 58 candidates (120ms)                        │  │
│  │  ✓ Reranked to top 10 (1.8s)                                │  │
│  │  ✓ Grounded top 5 (2.1s)                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌── Recommendations ─────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  1. "Attention Is All You Need" (2017)      Confidence: 95% │   │
│  │     Vaswani et al. | Cited by 90,000+                       │   │
│  │     ▶ Justification [expandable]                            │   │
│  │     ▶ Supporting Snippet [expandable]                       │   │
│  │                                                             │   │
│  │  2. "BERT: Pre-training of Deep..." (2018)  Confidence: 91% │   │
│  │     Devlin et al. | Cited by 75,000+                        │   │
│  │     ▶ Justification [expandable]                            │   │
│  │     ▶ Supporting Snippet [expandable]                       │   │
│  │                                                             │   │
│  │  ... (up to 5 recommendations)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌── Debug Panel (collapsed by default) ──────────────────────┐   │
│  │  Full PipelineState JSON | Token usage | Latency breakdown  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 7.3 Key Streamlit Implementation Details

- **`st.session_state`** stores the current `PipelineState` so results survive Streamlit reruns
- **`st.status`** containers show real-time pipeline progress (each agent updates a status block as it completes)
- **`st.expander`** for justification/snippet panels — keeps the UI clean
- **`st.cache_resource`** for the embedding model and Qdrant client — loaded once, reused across requests
- **`st.sidebar`** for advanced options (year range, number of results, debug toggle)

### 7.4 Caching Strategy

```python
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

@st.cache_resource
def load_qdrant():
    return QdrantClient(host="localhost", port=6333)

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("BAAI/bge-reranker-base")
```

These are loaded once when the Streamlit app starts and shared across all user requests.

---

## 8. Evaluation Strategy

This is the most important section for academic credibility. A recommendation system without rigorous evaluation is a demo, not a project.

### 8.1 Evaluation Paradigm: Citation Prediction

The evaluation task is:

> Given a paragraph from a known paper, can the system recover the actual citations that the original authors used?

This is a **retrospective citation prediction** task — a well-established evaluation paradigm in information retrieval and scientometrics.

### 8.2 Evaluation Dataset Construction

**Source:** Papers in the corpus that have known references (from OpenAlex).

**Construction procedure:**

1. Select ~200–500 papers from the corpus that have ≥5 references also present in the corpus
2. For each selected paper, use its abstract as the "input text"
3. The known references that exist in the corpus are the **ground truth citations**
4. Exclude the paper itself from the candidate pool during retrieval (to prevent trivial self-matching)

```python
@dataclass
class EvalSample:
    query_paper_id: str
    query_text: str                     # Abstract of the query paper
    ground_truth_ids: list[str]         # Known references in corpus
    ground_truth_count: int
```

**Why abstracts as input:**
- Abstracts naturally contain citable claims and context
- They are the most consistently available text across all papers
- Using abstracts (rather than hand-crafted queries) makes the evaluation realistic and reproducible

### 8.3 Metrics

| Metric | Definition | What It Measures |
|---|---|---|
| **Recall@K** | Fraction of ground truth citations found in top K results | Coverage — how many relevant papers did we find? |
| **MRR** (Mean Reciprocal Rank) | Average of 1/rank of the first relevant result | How quickly do we find *something* relevant? |
| **MAP** (Mean Average Precision) | Average precision at each relevant result, averaged | Overall ranking quality |

These are **standard IR metrics** — well-understood, reproducible, and directly comparable to published baselines.

### 8.4 Evaluation Loop Implementation

```python
def evaluate_pipeline(
    pipeline,
    eval_samples: list[EvalSample],
    k_values: list[int] = [5, 10, 20],
) -> dict:
    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    results["map"] = []

    for sample in tqdm(eval_samples):
        # Run the pipeline (excluding the query paper from results)
        recommendations = pipeline.run(
            sample.query_text,
            exclude_ids={sample.query_paper_id},
        )

        recommended_ids = [r.paper_id for r in recommendations]

        # Compute metrics
        for k in k_values:
            recall = compute_recall_at_k(
                recommended_ids[:k], sample.ground_truth_ids
            )
            results[f"recall@{k}"].append(recall)

        results["mrr"].append(
            compute_mrr(recommended_ids, sample.ground_truth_ids)
        )
        results["map"].append(
            compute_map(recommended_ids, sample.ground_truth_ids)
        )

    # Average across all samples
    return {k: sum(v) / len(v) for k, v in results.items()}
```

### 8.5 Ablation Studies

The modular architecture enables clean ablation experiments:

| Ablation | What to Disable | What It Measures |
|---|---|---|
| No citation expansion | Skip Agent 3 | Value of graph-based expansion |
| No reranking | Skip Agent 4, Stage A | Value of cross-encoder reranking |
| No intent analysis | Skip Agent 1, use raw text | Value of query reformulation |
| Different embeddings | Swap BGE for MiniLM or SPECTER2 | Embedding model impact |

Each ablation isolates one component's contribution — essential for the engineering narrative in the final report.

### 8.6 Baselines

| Baseline | Description |
|---|---|
| **BM25** | Classic sparse retrieval (using `rank_bm25` library) |
| **Dense retrieval only** | Agents 1+2 only (no expansion, no reranking) |
| **Full pipeline** | All 5 agents |

Showing improvement from BM25 → dense → expanded → reranked tells a compelling engineering story.

---

## 9. Project Structure

```
citation-recommender/
├── README.md                       # Project overview and setup instructions
├── SYSTEM_OVERVIEW.md              # This document
├── pyproject.toml                  # Dependencies and project metadata
├── .env.example                    # Template for environment variables
├── docker-compose.yml              # Qdrant service
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Settings (Pydantic BaseSettings)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py              # Paper, CitationEdge, EvalSample dataclasses
│   │   ├── openalex_client.py     # OpenAlex API wrapper
│   │   ├── corpus_builder.py      # Orchestrates corpus collection
│   │   └── db.py                  # SQLite operations
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embedder.py            # Embedding pipeline
│   │   └── qdrant_store.py        # Qdrant collection management
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── query_agent.py         # Agent 1: Query & Intent Analysis
│   │   ├── retrieval_agent.py     # Agent 2: Primary Retrieval
│   │   ├── expansion_agent.py     # Agent 3: Citation Expansion
│   │   ├── reranking_agent.py     # Agent 4: Reranking + Grounding
│   │   └── synthesis_agent.py     # Agent 5: Synthesis
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── state.py               # PipelineState, AgentState schemas
│   │   └── graph.py               # LangGraph workflow definition
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── dataset.py             # Evaluation dataset construction
│       ├── metrics.py             # Recall@K, MRR, MAP implementations
│       └── runner.py              # Evaluation loop orchestration
│
├── app/
│   └── streamlit_app.py           # Streamlit UI entry point
│
├── scripts/
│   ├── build_corpus.py            # Run corpus collection from OpenAlex
│   ├── index_corpus.py            # Embed and index into Qdrant
│   ├── build_eval_dataset.py      # Construct evaluation dataset
│   └── run_evaluation.py          # Execute evaluation suite
│
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── test_agents/
│   │   ├── test_query_agent.py
│   │   ├── test_retrieval_agent.py
│   │   ├── test_expansion_agent.py
│   │   ├── test_reranking_agent.py
│   │   └── test_synthesis_agent.py
│   ├── test_data/
│   │   ├── test_openalex_client.py
│   │   └── test_db.py
│   ├── test_indexing/
│   │   └── test_qdrant_store.py
│   └── test_evaluation/
│       └── test_metrics.py
│
├── data/                          # Local data directory (gitignored)
│   └── .gitkeep
│
└── notebooks/                     # Exploratory notebooks
    ├── 01_openalex_exploration.ipynb
    ├── 02_embedding_comparison.ipynb
    └── 03_evaluation_analysis.ipynb
```

---

## 10. Dependency Stack

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `python` | ≥3.11 | f-strings, typing, dataclasses, tomllib |
| `langgraph` | ≥0.2.0 | Agent orchestration state machine |
| `langchain-openai` | ≥0.2.0 | OpenAI LLM integration for LangGraph |
| `openai` | ≥1.50.0 | OpenAI API client (direct use + via instructor) |
| `instructor` | ≥1.5.0 | Structured LLM output extraction |
| `sentence-transformers` | ≥3.0 | Embedding model + cross-encoder |
| `qdrant-client` | ≥1.12.0 | Vector store client |
| `pyalex` | ≥0.14 | OpenAlex API wrapper |
| `streamlit` | ≥1.39.0 | Web UI framework |
| `pydantic` | ≥2.0 | Data validation and settings |
| `pydantic-settings` | ≥2.0 | Environment variable management |

### Data & Evaluation

| Package | Version | Purpose |
|---|---|---|
| `rank-bm25` | ≥0.2.2 | BM25 baseline retrieval |
| `pandas` | ≥2.0 | Data manipulation for evaluation |
| `tqdm` | ≥4.66 | Progress bars |

### Development & Testing

| Package | Version | Purpose |
|---|---|---|
| `pytest` | ≥8.0 | Test framework |
| `pytest-asyncio` | ≥0.23 | Async test support |
| `ruff` | ≥0.7.0 | Linting + formatting |
| `python-dotenv` | ≥1.0 | Environment variable loading |

### Infrastructure

| Component | Version | Purpose |
|---|---|---|
| `Docker` | ≥24.0 | Container runtime for Qdrant |
| `docker-compose` | ≥2.20 | Service orchestration |

---

## 11. Testing Strategy

### Unit Tests

Each agent has unit tests with **mocked dependencies:**

- **Agent 1:** Mock the OpenAI API, verify structured output parsing
- **Agent 2:** Mock the Qdrant client, verify query construction and result mapping
- **Agent 3:** Use an in-memory SQLite database with known citation edges, verify graph traversal
- **Agent 4:** Mock the cross-encoder and LLM, verify score ordering and grounding output
- **Agent 5:** Pure logic — verify filtering, sorting, and output formatting

### Integration Tests

- **End-to-end pipeline test:** Run the full LangGraph pipeline with a small test corpus (~100 papers) and verify that structured output is produced
- **Qdrant integration:** Verify indexing and retrieval against a test Qdrant collection (using Docker)

### Evaluation as Testing

The evaluation suite doubles as a comprehensive integration test. If Recall@10 drops below a threshold after a code change, something is broken.

### Testing Priority (Given Time Constraints)

1. **Evaluation metrics** — most critical (proves the system works)
2. **Agent unit tests** — catch regressions fast
3. **Integration tests** — if time permits

---

## 12. Risk Mitigation & Fallback Plans

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **OpenAI rate limits / cost overruns** | Medium | High | Budget tracking per run, `gpt-4o-mini` is very cheap (~$0.15/1M tokens). Fallback: batch LLM calls, reduce grounding to top-3 instead of top-5 |
| **OpenAlex API downtime** | Low | High (at data collection time) | Collect corpus early, cache all responses locally. Once the corpus is built, the API is not needed at runtime |
| **Qdrant issues** | Low | Medium | Qdrant is mature and stable in Docker. Fallback: switch to Chroma (in-process) or FAISS |
| **Embedding quality insufficient** | Medium | Medium | BGE-base is well-benchmarked. Ablation with SPECTER2 or MiniLM provides alternatives. This is a knob, not a blocker |
| **Cross-encoder too slow** | Low | Low | MiniLM cross-encoder is very fast. If needed, reduce candidate set size before reranking |
| **LangGraph complexity** | Low-Medium | Medium | The graph is purely linear for v1 — no branching, no conditionals. LangGraph handles this trivially. Fallback: replace with plain function composition |
| **Evaluation dataset too small** | Medium | Medium | Even 100 eval samples with 5+ ground truth citations each provides statistically meaningful metrics. The key is quality over quantity |
| **Scope creep** | High | High | The design is intentionally minimal. Every "nice-to-have" (full-text chunking, multi-hop graph traversal, user feedback loops) is explicitly deferred. The linear pipeline is the MVP |

---

## Appendix: Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| LLM provider | OpenAI (gpt-4o-mini) | Most reliable API, best structured output support, cheapest for quality tier |
| Embedding model | BAAI/bge-base-en-v1.5 | Top-tier MTEB scores, good tooling support, instruction-prefixed retrieval |
| Vector store | Qdrant (Docker) | Simple deployment, excellent filtering, mature Python client |
| Metadata store | SQLite | Zero-config, perfect for single-developer, fast enough for 25k records |
| Orchestration | LangGraph | Explicit state machine, typed state, deterministic, debuggable |
| Cross-encoder | BAAI/bge-reranker-base | Best quality-to-speed ratio in its class |
| Data source | OpenAlex (primary) + arXiv (supplement) | Free, comprehensive, citation graph included |
| UI | Streamlit | Fastest path from Python to interactive web app |
| Structured output | instructor + Pydantic | Type-safe LLM output extraction with automatic retries |
| Testing | pytest + evaluation suite | Evaluation metrics are the primary quality signal |

---

*This document was generated on 2026-02-10 from the initial technical design draft and elaborated with implementation-specific detail to serve as the basis for the development roadmap.*

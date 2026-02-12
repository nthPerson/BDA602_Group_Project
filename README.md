# Agentic Academic Paper Citation Recommendation System

A multi-agent Retrieval-Augmented Generation (RAG) system that recommends relevant citation candidates for AI/ML research papers. Given a paragraph, claim, or draft section from a paper, the system returns ranked citation recommendations with confidence scores, supporting snippets, and natural-language justifications.

Built as a group project for **BDA602: Machine Learning Engineering** at **San Diego State University**, Spring 2026.

## Team

- Robert Ashe
- Rohini Panikara
- Shivani Shankar
- Yuvaraj Murugan

## How It Works

The system models a researcher's citation workflow as a five-agent pipeline, orchestrated by a LangGraph state machine:

```
User text
  │
  ▼
Agent 1: Query & Intent Analysis ──── Classifies citation intent, extracts keywords,
  │                                   reformulates query for retrieval
  ▼
Agent 2: Dense Retrieval ──────────── Embeds query and searches Qdrant for candidate papers
  │                                   using cosine similarity
  ▼
Agent 3: Citation Expansion ───────── Traverses the citation graph (depth-1 BFS) to find
  │                                   papers connected to the initial candidates
  ▼
Agent 4: Reranking & Grounding ────── Cross-encoder reranking for precision, then LLM-generated
  │                                   justifications grounded in each paper's abstract
  ▼
Agent 5: Synthesis ────────────────── Confidence filtering, composite scoring, final output
  │
  ▼
Ranked Recommendations with justifications and supporting snippets
```

Each agent reads from and writes to a shared typed state object, making every intermediate result inspectable and every agent independently testable.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph |
| LLM | OpenAI gpt-4o-mini via `instructor` |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim, sentence-transformers) |
| Reranking | BAAI/bge-reranker-base (cross-encoder) |
| Vector Store | Qdrant (Docker) |
| Metadata Store | SQLite |
| Data Source | OpenAlex (CS.AI/LG/CL/CV, 2015-2025) |
| UI | Streamlit |
| Testing | pytest |
| Linting | ruff |

## Project Structure (under construction)

```
src/
├── config.py                    # Pydantic Settings (loads from .env)
├── data/
│   ├── models.py                # Paper, CitationEdge, ScoredPaper, and other data models
│   ├── db.py                    # SQLite CRUD for papers and citation edges
│   ├── openalex_client.py       # OpenAlex API wrapper with pagination
│   └── corpus_builder.py        # Orchestrates corpus collection
├── indexing/
│   ├── embedder.py              # BGE embedding pipeline
│   └── qdrant_store.py          # Qdrant collection management + search
├── agents/
│   ├── query_agent.py           # Agent 1: intent classification + keyword extraction
│   ├── retrieval_agent.py       # Agent 2: dense retrieval from Qdrant
│   ├── expansion_agent.py       # Agent 3: citation graph BFS
│   ├── reranking_agent.py       # Agent 4: cross-encoder rerank + LLM grounding
│   └── synthesis_agent.py       # Agent 5: confidence filter + composite scoring
├── orchestration/
│   ├── state.py                 # AgentState TypedDict
│   └── graph.py                 # LangGraph StateGraph definition
└── evaluation/
    ├── metrics.py               # Recall@K, MRR, MAP
    ├── dataset.py               # Eval dataset construction
    ├── baselines.py             # BM25 baseline
    └── runner.py                # Evaluation loop

app/
└── streamlit_app.py             # Streamlit UI

scripts/
├── build_corpus.py              # CLI: collect papers from OpenAlex
├── index_corpus.py              # CLI: embed + index into Qdrant
├── build_eval_dataset.py        # CLI: construct eval dataset
└── run_evaluation.py            # CLI: run evaluation suite

tests/                           # Unit and integration tests for all modules
```

## Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/nthPerson/BDA602_Group_Project.git
cd BDA602_Group_Project

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package with dev dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# Start Qdrant
docker compose up -d

# Verify the installation
pytest -m "not integration and not slow"
```

See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for the full setup guide.

### Building the Corpus and Index

```bash
# Collect papers from OpenAlex (~30-60 min, one-time operation)
python scripts/build_corpus.py

# Embed and index papers into Qdrant
python scripts/index_corpus.py
```

## Usage

### Streamlit UI (under construction)

```bash
streamlit run app/streamlit_app.py
```

Paste a paragraph or claim from an AI/ML paper, click **Find Citations**, and the system will return ranked recommendations with justifications.

### Running the Pipeline Programmatically

```python
from src.orchestration.graph import build_graph

graph = build_graph()
result = graph.invoke({"user_text": "Your paragraph here..."})

for rec in result["final_recommendations"]:
    print(f"{rec.rank}. {rec.title} (Confidence: {rec.confidence:.0%})")
    print(f"   {rec.justification}\n")
```

## Evaluation (under construction)

The system is evaluated using retrospective citation prediction: given a paper's abstract, can the system recover the actual citations the original authors used?

**Metrics:** Recall@K, Mean Reciprocal Rank (MRR), Mean Average Precision (MAP)

```bash
# Build the evaluation dataset
python scripts/build_eval_dataset.py

# Run evaluation (BM25 baseline is free; full pipeline costs ~$1-2 in API calls)
python scripts/run_evaluation.py --mode full
```

## Development

```bash
# Run fast unit tests
pytest -m "not integration and not slow" -v

# Run all tests (requires Qdrant running)
pytest -v

# Lint and format
ruff check src/ tests/ --fix
ruff format src/ tests/
```

## Documentation

| Document | Description |
|---|---|
| [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) | Authoritative technical design: architecture, data models, agent specs |
| [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) | 10-stage incremental build plan with acceptance criteria |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Step-by-step setup instructions |
| [docs/stages/](docs/stages/) | Per-stage implementation details and test instructions |

## License

This project was developed for academic purposes as part of BDA602 at San Diego State University.

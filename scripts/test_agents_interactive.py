"""Interactive test script for the full 5-agent pipeline.

This script demonstrates the complete pipeline:
1. Agent 1 analyzes user text (LLM-based intent + keyword extraction)
2. Agent 2 retrieves relevant papers from Qdrant
3. Agent 3 expands candidates via the citation graph
4. Agent 4 reranks candidates with cross-encoder + grounds top papers with LLM
5. Agent 5 synthesizes final ranked recommendations with confidence filtering

Run this script to test all agents with real API calls and see the results.

Requirements:
    - OPENAI_API_KEY set in .env
    - Qdrant running (docker compose up -d)
    - Corpus built and indexed (build_corpus.py + index_corpus.py)
"""

import sqlite3

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.agents.expansion_agent import ExpansionAgent, ExpansionAgentConfig
from src.agents.query_agent import QueryAgent, QueryAgentConfig
from src.agents.reranking_agent import RerankingAgent, RerankingAgentConfig
from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from src.agents.synthesis_agent import SynthesisAgent, SynthesisAgentConfig
from src.config import Settings
from src.indexing.embedder import Embedder, EmbedderConfig
from src.indexing.qdrant_store import QdrantConfig, QdrantStore

# Load environment variables from .env
load_dotenv()

# Load settings
settings = Settings()

# ==================== Initialize All Agents ====================

print("Initializing Agent 1 (Query Analysis)...")
openai_client = OpenAI()  # Reads OPENAI_API_KEY from environment
agent1 = QueryAgent(QueryAgentConfig(), openai_client)

print("Initializing Agent 2 (Retrieval)...")
embedder = Embedder(EmbedderConfig(show_progress=False))
qdrant_store = QdrantStore(QdrantConfig())
agent2 = RetrievalAgent(RetrievalAgentConfig(top_n=10), embedder, qdrant_store)

print("Initializing Agent 3 (Citation Expansion)...")
db = sqlite3.connect(settings.db_path)
agent3 = ExpansionAgent(ExpansionAgentConfig(), db)

print("Initializing Agent 4 (Reranking & Grounding)...")
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
agent4 = RerankingAgent(
    RerankingAgentConfig(rerank_top_k=10, ground_top_k=3),
    cross_encoder,
    openai_client,
)

print("Initializing Agent 5 (Synthesis)...")
agent5 = SynthesisAgent(SynthesisAgentConfig(), db)

print("\n" + "=" * 70)
print("Ready! Testing with sample queries...")
print("=" * 70)

# Test cases
test_cases = [
    (
        "Background Citation",
        """
        Recent advances in transformer architectures have shown that scaling
        model parameters improves few-shot learning capabilities. Large language
        models such as GPT-3 demonstrate emergent abilities on diverse NLP tasks.
        """,
    ),
    (
        "Method Citation",
        """
        We employ the BERT pre-training approach with masked language modeling
        to learn contextualized word representations from unlabeled text.
        """,
    ),
    (
        "Comparison Citation",
        """
        Unlike traditional RNNs which process sequences sequentially, our approach
        uses self-attention to capture long-range dependencies more efficiently.
        """,
    ),
]

for test_name, user_text in test_cases:
    print(f"\n{'=' * 70}")
    print(f"Test: {test_name}")
    print("=" * 70)
    print(f"Input text: {user_text.strip()[:100]}...")

    # Build pipeline state
    state: dict = {
        "user_text": user_text.strip(),
        "metadata": {},
        "errors": [],
    }

    # Agent 1: Query Analysis
    print(f"\n{'─' * 70}")
    print("Agent 1: Query Analysis")
    print("─" * 70)
    state.update(agent1.run(state))
    analysis = state["query_analysis"]
    print(f"Intent: {analysis.citation_intent.value.upper()}")
    print(f"Keywords: {', '.join(analysis.topic_keywords)}")
    print(f"Expanded query: {analysis.expanded_query}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Latency: {state['metadata'].get('agent1_latency_s', 'N/A')}s")

    # Agent 2: Primary Retrieval
    print(f"\n{'─' * 70}")
    print("Agent 2: Primary Retrieval")
    print("─" * 70)
    state.update(agent2.run(state))
    retrieval_count = len(state["retrieval_candidates"])
    print(f"Retrieved {retrieval_count} papers")
    print(f"Latency: {state['metadata'].get('agent2_latency_s', 'N/A')}s")
    for i, paper in enumerate(state["retrieval_candidates"][:5], 1):
        print(f"  {i}. [{paper.similarity_score:.3f}] {paper.title}")

    # Agent 3: Citation Expansion
    print(f"\n{'─' * 70}")
    print("Agent 3: Citation Expansion")
    print("─" * 70)
    state.update(agent3.run(state))
    expanded_count = len(state["expanded_candidates"])
    new_count = state["metadata"].get("agent3_new_papers", 0)
    print(f"Expanded {retrieval_count} -> {expanded_count} candidates (+{new_count} new)")
    print(f"Latency: {state['metadata'].get('agent3_latency_s', 'N/A')}s")

    # Agent 4: Reranking & Grounding
    print(f"\n{'─' * 70}")
    print("Agent 4: Reranking & Grounding")
    print("─" * 70)
    state.update(agent4.run(state))
    reranked = state["reranked_candidates"]
    grounded = state["grounded_candidates"]
    print(f"Reranked to top {len(reranked)} papers")
    print(f"Grounded {len(grounded)} papers with justifications")
    print(f"Rerank latency: {state['metadata'].get('agent4a_latency_s', 'N/A')}s")
    print(f"Grounding latency: {state['metadata'].get('agent4b_latency_s', 'N/A')}s")
    print(f"Total Agent 4 latency: {state['metadata'].get('agent4_total_latency_s', 'N/A')}s")

    # Show top reranked papers
    print(f"\nTop {min(5, len(reranked))} reranked papers:")
    for i, paper in enumerate(reranked[:5], 1):
        print(
            f"  {i}. [rerank={paper.rerank_score:.3f}, sim={paper.similarity_score:.3f}] "
            f"{paper.title}"
        )
        print(f"     Year: {paper.year} | Citations: {paper.citation_count:,}")

    # Show grounded papers with justifications
    if grounded:
        print("\nGrounded papers:")
        for i, paper in enumerate(grounded, 1):
            print(f"\n  {i}. {paper.title}")
            print(f"     Rerank score: {paper.rerank_score:.3f}")
            print(f"     Confidence: {paper.confidence:.2f}")
            print(f"     Snippet: {paper.supporting_snippet[:150]}...")
            print(f"     Justification: {paper.justification[:200]}...")

    # Agent 5: Synthesis
    print(f"\n{'─' * 70}")
    print("Agent 5: Synthesis")
    print("─" * 70)
    state.update(agent5.run(state))
    recommendations = state["final_recommendations"]
    print(f"Produced {len(recommendations)} final recommendations")
    print(f"Latency: {state['metadata'].get('agent5_latency_s', 'N/A')}s")

    # Show final recommendations
    if recommendations:
        print("\nFinal Recommendations:")
        for rec in recommendations:
            print(f"\n  {rec.rank}. {rec.title}")
            print(f"     Paper ID: {rec.paper_id}")
            if rec.authors:
                print(f"     Authors: {', '.join(rec.authors[:3])}")
                if len(rec.authors) > 3:
                    print(f"              ... and {len(rec.authors) - 3} more")
            print(f"     Year: {rec.year} | Citations: {rec.citation_count:,}")
            print(f"     Confidence: {rec.confidence:.2f}")
            print(f"     Intent match: {rec.citation_intent_match}")
            print(f"     Snippet: {rec.supporting_snippet[:150]}...")
            print(f"     Justification: {rec.justification[:200]}...")
    else:
        print("\n  No recommendations produced (all below confidence threshold?)")

    # Show any errors
    if state["errors"]:
        print(f"\nErrors: {state['errors']}")

db.close()

print("\n" + "=" * 70)
print("All tests complete!")
print("=" * 70)

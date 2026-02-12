"""Interactive test script for Agents 1 & 2.

This script demonstrates the full Agent 1 → Agent 2 pipeline:
1. Agent 1 analyzes user text (LLM-based intent + keyword extraction)
2. Agent 2 retrieves relevant papers from Qdrant

Run this script to test the agents with real API calls and see the results.
"""

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.query_agent import QueryAgent, QueryAgentConfig
from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from src.indexing.embedder import Embedder, EmbedderConfig
from src.indexing.qdrant_store import QdrantConfig, QdrantStore

# Load environment variables from .env
load_dotenv()

# Initialize Agent 1 (Query Analysis)
print("Initializing Agent 1 (Query Analysis)...")
openai_client = OpenAI()  # Reads OPENAI_API_KEY from environment
agent1 = QueryAgent(QueryAgentConfig(), openai_client)

# Initialize Agent 2 (Retrieval)
print("Initializing Agent 2 (Retrieval)...")
embedder = Embedder(EmbedderConfig(show_progress=False))
qdrant_store = QdrantStore(QdrantConfig())
agent2 = RetrievalAgent(RetrievalAgentConfig(top_n=10), embedder, qdrant_store)

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

    # Agent 1: Query Analysis
    print(f"\n{'─' * 70}")
    print("Agent 1: Query Analysis")
    print("─" * 70)
    analysis = agent1.analyze(user_text.strip())
    print(f"Intent: {analysis.citation_intent.value.upper()}")
    print(f"Keywords: {', '.join(analysis.topic_keywords)}")
    print(f"Expanded query: {analysis.expanded_query}")
    print(f"Confidence: {analysis.confidence:.2f}")

    # Agent 2: Primary Retrieval
    print(f"\n{'─' * 70}")
    print("Agent 2: Primary Retrieval")
    print("─" * 70)
    results = agent2.retrieve(analysis, top_n=5)
    print(f"Retrieved {len(results)} papers:\n")

    for i, paper in enumerate(results, 1):
        print(f"{i}. [{paper.similarity_score:.3f}] {paper.title}")
        print(f"   Year: {paper.year} | Citations: {paper.citation_count:,}")
        print(f"   Abstract: {paper.abstract[:150]}...")
        print()

print("\n" + "=" * 70)
print("All tests complete!")
print("=" * 70)

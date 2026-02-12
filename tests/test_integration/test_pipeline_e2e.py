"""End-to-end integration tests for the full LangGraph pipeline.

These tests require all services running:
- Qdrant (docker compose up -d)
- Corpus built and indexed
- OPENAI_API_KEY set in .env

Test inventory:
- test_e2e_produces_recommendations: Full pipeline → non-empty recommendations
- test_e2e_recommendation_structure: Output has correct fields and values
- test_e2e_performance: Full pipeline completes in <30 seconds
"""

import os
import sqlite3
import time

import pytest
from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.agents.expansion_agent import ExpansionAgent, ExpansionAgentConfig
from src.agents.query_agent import QueryAgent, QueryAgentConfig
from src.agents.reranking_agent import RerankingAgent, RerankingAgentConfig
from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from src.agents.synthesis_agent import SynthesisAgent, SynthesisAgentConfig
from src.config import Settings
from src.data.models import Recommendation
from src.indexing.embedder import Embedder, EmbedderConfig
from src.indexing.qdrant_store import QdrantConfig, QdrantStore
from src.orchestration.graph import build_graph

# All tests in this module require external services
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline():
    """Build the full pipeline with real dependencies.

    Skips if OPENAI_API_KEY is not set or services are unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping integration tests")

    settings = Settings()
    if not settings.db_path.exists():
        pytest.skip(f"Database not found at {settings.db_path} — build corpus first")

    openai_client = OpenAI()

    # Agent 1
    query_agent = QueryAgent(QueryAgentConfig(), openai_client)

    # Agent 2
    embedder = Embedder(EmbedderConfig(show_progress=False))
    try:
        qdrant_store = QdrantStore(QdrantConfig())
    except Exception:
        pytest.skip("Qdrant not available — start with docker compose up -d")
    retrieval_agent = RetrievalAgent(RetrievalAgentConfig(top_n=10), embedder, qdrant_store)

    # Agent 3
    db = sqlite3.connect(settings.db_path)
    expansion_agent = ExpansionAgent(ExpansionAgentConfig(), db)

    # Agent 4
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    reranking_agent = RerankingAgent(
        RerankingAgentConfig(rerank_top_k=10, ground_top_k=3),
        cross_encoder,
        openai_client,
    )

    # Agent 5
    synthesis_agent = SynthesisAgent(SynthesisAgentConfig(), db)

    graph = build_graph(
        query_agent,
        retrieval_agent,
        expansion_agent,
        reranking_agent,
        synthesis_agent,
    )

    yield graph

    db.close()


SAMPLE_INPUT = (
    "Recent work on transformer architectures has demonstrated that "
    "self-attention mechanisms can effectively capture long-range dependencies "
    "in sequence data, outperforming recurrent neural networks on machine "
    "translation and other sequence-to-sequence tasks."
)


class TestE2EProducesRecommendations:
    """Full pipeline run should produce non-empty recommendations."""

    def test_e2e_produces_recommendations(self, pipeline) -> None:
        """Pipeline produces at least one recommendation for a valid query."""
        result = pipeline.invoke({"user_text": SAMPLE_INPUT, "metadata": {}, "errors": []})

        recs = result.get("final_recommendations", [])
        assert len(recs) > 0, f"Expected recommendations, got errors: {result.get('errors')}"
        assert all(isinstance(r, Recommendation) for r in recs)


class TestE2ERecommendationStructure:
    """Output has correct fields and reasonable values."""

    def test_e2e_recommendation_structure(self, pipeline) -> None:
        """Each recommendation has all required fields with valid values."""
        result = pipeline.invoke({"user_text": SAMPLE_INPUT, "metadata": {}, "errors": []})

        recs = result.get("final_recommendations", [])
        assert len(recs) > 0

        for rec in recs:
            assert rec.rank > 0
            assert len(rec.paper_id) > 0
            assert len(rec.title) > 0
            assert isinstance(rec.authors, list)
            assert rec.year >= 2018
            assert rec.citation_count >= 0
            assert len(rec.justification) > 0
            assert len(rec.supporting_snippet) > 0
            assert 0.0 <= rec.confidence <= 1.0
            assert len(rec.citation_intent_match) > 0

        # Ranks should be sequential
        ranks = [r.rank for r in recs]
        assert ranks == list(range(1, len(recs) + 1))

        # Metadata should have all agent timings
        metadata = result.get("metadata", {})
        assert "agent1_latency_s" in metadata
        assert "agent5_latency_s" in metadata

        # No errors in a successful run
        assert result.get("errors", []) == []


class TestE2EPerformance:
    """Full pipeline should complete within a reasonable time."""

    def test_e2e_performance(self, pipeline) -> None:
        """Full pipeline completes in under 30 seconds."""
        start = time.time()
        result = pipeline.invoke({"user_text": SAMPLE_INPUT, "metadata": {}, "errors": []})
        elapsed = time.time() - start

        assert elapsed < 30.0, f"Pipeline took {elapsed:.1f}s (limit: 30s)"
        assert len(result.get("final_recommendations", [])) > 0

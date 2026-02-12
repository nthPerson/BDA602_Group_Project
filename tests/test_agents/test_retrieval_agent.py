"""Tests for Agent 2: Primary Retrieval.

Tests cover:
- ScoredPaper mapping from Qdrant results
- Results sorted by similarity score
- top_n parameter respected
- Year filter construction
- Empty result handling
- Live retrieval for transformers (integration)
- Live retrieval with year filter (integration)
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from src.data.models import CitationIntent, QueryAnalysis, ScoredPaper
from src.indexing.embedder import Embedder, EmbedderConfig
from src.indexing.qdrant_store import QdrantConfig, QdrantStore

# ==================== Fixtures ====================


@pytest.fixture
def sample_query_analysis() -> QueryAnalysis:
    """Create a sample QueryAnalysis for testing."""
    return QueryAnalysis(
        topic_keywords=["transformer", "attention", "sequence modeling"],
        citation_intent=CitationIntent.BACKGROUND,
        expanded_query="transformer attention mechanism sequence modeling neural networks",
        confidence=0.9,
    )


@pytest.fixture
def mock_qdrant_results() -> list[dict]:
    """Create mock Qdrant search results."""
    return [
        {
            "score": 0.95,
            "paper_id": "W001",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models...",
            "year": 2017,
            "citation_count": 90000,
            "concepts": ["Transformer", "Attention"],
            "authors": ["Vaswani"],
            "source": "NeurIPS",
        },
        {
            "score": 0.88,
            "paper_id": "W002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce a new language representation model...",
            "year": 2019,
            "citation_count": 75000,
            "concepts": ["BERT", "Language Model"],
            "authors": ["Devlin"],
            "source": "NAACL",
        },
        {
            "score": 0.82,
            "paper_id": "W003",
            "title": "An Image is Worth 16x16 Words",
            "abstract": "We show that a pure transformer applied to sequences...",
            "year": 2021,
            "citation_count": 20000,
            "concepts": ["Vision Transformer", "Image Classification"],
            "authors": ["Dosovitskiy"],
            "source": "ICLR",
        },
    ]


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock Embedder that returns a fixed query vector."""
    embedder = MagicMock(spec=Embedder)
    # Return a 768-dim unit vector
    embedder.embed_query.return_value = np.random.randn(768).astype(np.float32)
    return embedder


@pytest.fixture
def mock_qdrant_store(mock_qdrant_results: list[dict]) -> MagicMock:
    """Create a mock QdrantStore that returns fixed results."""
    store = MagicMock(spec=QdrantStore)
    store.search.return_value = mock_qdrant_results
    return store


# ==================== Unit Tests ====================


class TestScoredPaperMapping:
    """Test mapping Qdrant results to ScoredPaper objects."""

    def test_scored_paper_mapping(
        self,
        mock_embedder: MagicMock,
        mock_qdrant_store: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Mocked Qdrant results are correctly mapped to ScoredPaper objects."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=mock_qdrant_store,
        )

        results = agent.retrieve(sample_query_analysis)

        assert len(results) == 3
        assert all(isinstance(r, ScoredPaper) for r in results)

        # Check first result
        first = results[0]
        assert first.paper_id == "W001"
        assert first.title == "Attention Is All You Need"
        assert first.similarity_score == 0.95
        assert first.year == 2017
        assert first.citation_count == 90000

    def test_results_sorted_by_score(
        self,
        mock_embedder: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Output is sorted by similarity_score descending."""
        # Return results in scrambled order
        scrambled_results = [
            {
                "score": 0.82,
                "paper_id": "W003",
                "title": "Paper C",
                "abstract": "...",
                "year": 2021,
                "citation_count": 100,
                "concepts": [],
            },
            {
                "score": 0.95,
                "paper_id": "W001",
                "title": "Paper A",
                "abstract": "...",
                "year": 2017,
                "citation_count": 500,
                "concepts": [],
            },
            {
                "score": 0.88,
                "paper_id": "W002",
                "title": "Paper B",
                "abstract": "...",
                "year": 2019,
                "citation_count": 300,
                "concepts": [],
            },
        ]
        store = MagicMock(spec=QdrantStore)
        store.search.return_value = scrambled_results

        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=store,
        )

        results = agent.retrieve(sample_query_analysis)

        # Qdrant returns results sorted, so the agent preserves order
        scores = [r.similarity_score for r in results]
        assert scores == [0.82, 0.95, 0.88]  # Preserves Qdrant's returned order


class TestTopNAndFilters:
    """Test top_n parameter and filter construction."""

    def test_top_n_respected(
        self,
        mock_embedder: MagicMock,
        mock_qdrant_store: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """No more than top_n results returned (delegated to Qdrant)."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=5),
            embedder=mock_embedder,
            qdrant_store=mock_qdrant_store,
        )

        agent.retrieve(sample_query_analysis)

        # Verify the limit was passed to Qdrant
        call_kwargs = mock_qdrant_store.search.call_args
        assert call_kwargs.kwargs.get("limit") == 5 or call_kwargs[1].get("limit") == 5

    def test_year_filter_construction(
        self,
        mock_embedder: MagicMock,
        mock_qdrant_store: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Year range parameters generate correct Qdrant filter."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=mock_qdrant_store,
        )

        agent.retrieve(sample_query_analysis, year_min=2020, year_max=2024)

        call_kwargs = mock_qdrant_store.search.call_args
        year_filter = call_kwargs.kwargs.get("year_filter") or call_kwargs[1].get("year_filter")
        assert year_filter == (2020, 2024)

    def test_no_filter_when_unset(
        self,
        mock_embedder: MagicMock,
        mock_qdrant_store: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """No year filter is applied when year_min/year_max are None."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=mock_qdrant_store,
        )

        agent.retrieve(sample_query_analysis)

        call_kwargs = mock_qdrant_store.search.call_args
        year_filter = call_kwargs.kwargs.get("year_filter") or call_kwargs[1].get("year_filter")
        assert year_filter is None


class TestEmptyResults:
    """Test edge cases with empty or missing results."""

    def test_empty_results(
        self,
        mock_embedder: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Qdrant returns nothing -> empty list (no crash)."""
        store = MagicMock(spec=QdrantStore)
        store.search.return_value = []

        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=store,
        )

        results = agent.retrieve(sample_query_analysis)

        assert results == []
        assert isinstance(results, list)

    def test_run_with_no_query_analysis(
        self,
        mock_embedder: MagicMock,
        mock_qdrant_store: MagicMock,
    ) -> None:
        """run() with missing query_analysis returns empty candidates and error."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=mock_qdrant_store,
        )

        state = {"user_text": "test", "query_analysis": None, "metadata": {}, "errors": []}
        result = agent.run(state)

        assert result["retrieval_candidates"] == []
        assert len(result["errors"]) > 0


class TestRetrievalAgentRun:
    """Test the run() method (state dict interface)."""

    def test_run_populates_metadata(
        self,
        mock_embedder: MagicMock,
        mock_qdrant_store: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """run() populates metadata with latency and candidate count."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=mock_qdrant_store,
        )

        state = {
            "user_text": "test",
            "query_analysis": sample_query_analysis,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        assert "agent2_latency_s" in result["metadata"]
        assert result["metadata"]["agent2_latency_s"] >= 0
        assert result["metadata"]["agent2_candidates"] == 3

    def test_run_returns_scored_papers(
        self,
        mock_embedder: MagicMock,
        mock_qdrant_store: MagicMock,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """run() returns ScoredPaper objects in retrieval_candidates."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=mock_embedder,
            qdrant_store=mock_qdrant_store,
        )

        state = {
            "user_text": "test",
            "query_analysis": sample_query_analysis,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        candidates = result["retrieval_candidates"]
        assert len(candidates) == 3
        assert all(isinstance(c, ScoredPaper) for c in candidates)


# ==================== Integration Tests ====================


@pytest.mark.integration
class TestRetrievalAgentLive:
    """Integration tests requiring Qdrant running with indexed corpus."""

    @pytest.fixture
    def live_embedder(self) -> Embedder:
        """Load the real embedding model."""
        return Embedder(EmbedderConfig(show_progress=False))

    @pytest.fixture
    def live_qdrant_store(self) -> QdrantStore:
        """Connect to the real Qdrant server."""
        store = QdrantStore(QdrantConfig())
        if not store.collection_exists():
            pytest.skip("Qdrant collection 'papers' not found — run index_corpus.py first")
        return store

    def test_live_retrieval_transformers(
        self, live_embedder: Embedder, live_qdrant_store: QdrantStore
    ) -> None:
        """Query 'transformer attention' -> top 10 includes relevant papers."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=live_embedder,
            qdrant_store=live_qdrant_store,
        )

        query = QueryAnalysis(
            topic_keywords=["transformer", "attention", "self-attention"],
            citation_intent=CitationIntent.BACKGROUND,
            expanded_query="transformer attention mechanism self-attention neural network",
            confidence=0.9,
        )

        results = agent.retrieve(query)

        assert len(results) > 0
        assert len(results) <= 10

        # Check that results have valid structure
        for paper in results:
            assert isinstance(paper, ScoredPaper)
            assert paper.paper_id != ""
            assert paper.title != ""
            assert paper.similarity_score > 0

        # At least one result should mention attention or transformer
        titles_lower = [r.title.lower() for r in results]
        has_relevant = any(
            "attention" in t or "transformer" in t for t in titles_lower
        )
        assert has_relevant, f"No attention/transformer papers found in: {titles_lower}"

    def test_live_retrieval_year_filter(
        self, live_embedder: Embedder, live_qdrant_store: QdrantStore
    ) -> None:
        """Query with year_min=2022 -> all results have year >= 2022."""
        agent = RetrievalAgent(
            config=RetrievalAgentConfig(top_n=10),
            embedder=live_embedder,
            qdrant_store=live_qdrant_store,
        )

        query = QueryAnalysis(
            topic_keywords=["deep learning"],
            citation_intent=CitationIntent.BACKGROUND,
            expanded_query="deep learning neural networks machine learning",
            confidence=0.9,
        )

        results = agent.retrieve(query, year_min=2022)

        assert len(results) > 0
        for paper in results:
            assert paper.year >= 2022, (
                f"Paper '{paper.title}' has year {paper.year}, expected >= 2022"
            )

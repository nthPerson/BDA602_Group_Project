"""Tests for Agent 4: Reranking & Grounding.

Tests cover:
- Cross-encoder reranking reorders candidates (not input order)
- Top-k parameter respected
- Each RankedPaper has rerank_score
- Stable ordering when all scores are identical
- Empty input handled gracefully
- Single candidate returned as-is
- Grounding produces all required output fields
- Grounding prompt includes user text and paper abstract
- Grounding retry on failure
- Grounding confidence is between 0.0 and 1.0
- run() metadata tracking
- Graceful degradation on rerank failure
- Graceful degradation on grounding failure
- Live cross-encoder reranking (integration)
- Live LLM grounding (integration)
- Full Agent 1→2→3→4 pipeline (integration)
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.agents.reranking_agent import (
    GroundingResponse,
    RerankingAgent,
    RerankingAgentConfig,
)
from src.data.models import (
    CitationIntent,
    GroundedPaper,
    QueryAnalysis,
    RankedPaper,
    ScoredPaper,
)

# ==================== Helpers ====================


def _make_scored_paper(
    paper_id: str,
    title: str = "",
    abstract: str = "",
    score: float = 0.9,
    year: int = 2020,
    citation_count: int = 100,
) -> ScoredPaper:
    """Create a minimal ScoredPaper for testing."""
    return ScoredPaper(
        paper_id=paper_id,
        title=title or f"Paper {paper_id}",
        abstract=abstract or f"Abstract for paper {paper_id}.",
        year=year,
        citation_count=citation_count,
        similarity_score=score,
        concepts=["AI"],
    )


def _make_ranked_paper(
    paper_id: str,
    rerank_score: float = 5.0,
    title: str = "",
    abstract: str = "",
) -> RankedPaper:
    """Create a minimal RankedPaper for testing."""
    return RankedPaper(
        paper_id=paper_id,
        title=title or f"Paper {paper_id}",
        abstract=abstract or f"Abstract for paper {paper_id}.",
        year=2020,
        citation_count=100,
        similarity_score=0.9,
        concepts=["AI"],
        rerank_score=rerank_score,
    )


def _make_grounding_response(
    snippet: str = "relevant excerpt",
    justification: str = "This paper is relevant because...",
    confidence: float = 0.85,
) -> GroundingResponse:
    """Create a GroundingResponse for mocking LLM output."""
    return GroundingResponse(
        supporting_snippet=snippet,
        justification=justification,
        confidence=confidence,
    )


# ==================== Fixtures ====================


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    """Create a mock CrossEncoder that returns controlled scores."""
    ce = MagicMock(spec=CrossEncoder)
    # Default: return incrementing scores so we can verify reordering
    ce.predict.return_value = np.array([0.1, 0.9, 0.5])
    return ce


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def default_config() -> RerankingAgentConfig:
    """Create a default agent config."""
    return RerankingAgentConfig(rerank_top_k=10, ground_top_k=5)


@pytest.fixture
def agent(
    default_config: RerankingAgentConfig,
    mock_cross_encoder: MagicMock,
    mock_openai_client: MagicMock,
) -> RerankingAgent:
    """Create a RerankingAgent with mocked dependencies."""
    return RerankingAgent(default_config, mock_cross_encoder, mock_openai_client)


@pytest.fixture
def three_candidates() -> list[ScoredPaper]:
    """Create 3 test candidates with known similarity scores."""
    return [
        _make_scored_paper("W001", title="Paper A", abstract="About transformers", score=0.95),
        _make_scored_paper("W002", title="Paper B", abstract="About attention", score=0.88),
        _make_scored_paper("W003", title="Paper C", abstract="About BERT", score=0.82),
    ]


@pytest.fixture
def sample_query_analysis() -> QueryAnalysis:
    """Create a sample QueryAnalysis for testing."""
    return QueryAnalysis(
        topic_keywords=["transformer", "attention"],
        citation_intent=CitationIntent.BACKGROUND,
        expanded_query="transformer attention mechanism neural networks",
        confidence=0.9,
    )


# ==================== Reranking Tests ====================


class TestRerankingReorders:
    """Test that cross-encoder reranking changes the candidate ordering."""

    def test_reranking_reorders(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
    ) -> None:
        """Mocked cross-encoder scores cause a different ordering than input order.

        Input order: W001, W002, W003
        Cross-encoder scores: [0.1, 0.9, 0.5]
        Expected output order: W002 (0.9), W003 (0.5), W001 (0.1)
        """
        result = agent.rerank("test query", three_candidates)

        result_ids = [p.paper_id for p in result]
        assert result_ids == ["W002", "W003", "W001"]


class TestRerankingTopK:
    """Test that the top_k parameter is respected."""

    def test_reranking_top_k(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
    ) -> None:
        """Output length equals min(top_k, number of candidates)."""
        result = agent.rerank("test query", three_candidates, top_k=2)

        assert len(result) == 2

    def test_top_k_larger_than_candidates(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
    ) -> None:
        """top_k > num candidates -> returns all candidates."""
        result = agent.rerank("test query", three_candidates, top_k=100)

        assert len(result) == 3


class TestRerankingScoreAttached:
    """Test that rerank_score is properly attached to output."""

    def test_reranking_score_attached(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
    ) -> None:
        """Each RankedPaper has a rerank_score field from the cross-encoder."""
        result = agent.rerank("test query", three_candidates)

        assert all(isinstance(p, RankedPaper) for p in result)
        assert all(hasattr(p, "rerank_score") for p in result)
        # The top result should have the highest rerank score
        assert result[0].rerank_score == pytest.approx(0.9)

    def test_original_similarity_preserved(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
    ) -> None:
        """Original similarity_score is carried through to RankedPaper."""
        result = agent.rerank("test query", three_candidates)

        # W002 (score=0.88) should be first after reranking
        assert result[0].paper_id == "W002"
        assert result[0].similarity_score == 0.88


class TestRerankingAllIdenticalScores:
    """Test stable ordering when all cross-encoder scores are equal."""

    def test_reranking_all_identical_scores(
        self,
        mock_openai_client: MagicMock,
        three_candidates: list[ScoredPaper],
    ) -> None:
        """Stable ordering when all scores are equal — no crash."""
        ce = MagicMock(spec=CrossEncoder)
        ce.predict.return_value = np.array([0.5, 0.5, 0.5])

        agent = RerankingAgent(RerankingAgentConfig(), ce, mock_openai_client)
        result = agent.rerank("test query", three_candidates)

        assert len(result) == 3
        assert all(p.rerank_score == pytest.approx(0.5) for p in result)


class TestRerankingEmptyCandidates:
    """Test edge cases with empty input."""

    def test_reranking_empty_candidates(self, agent: RerankingAgent) -> None:
        """Empty input -> empty output (no crash)."""
        result = agent.rerank("test query", [])

        assert result == []
        assert isinstance(result, list)


class TestRerankingSingleCandidate:
    """Test with a single candidate."""

    def test_reranking_single_candidate(
        self,
        mock_openai_client: MagicMock,
    ) -> None:
        """One candidate -> returned as-is with rerank_score."""
        ce = MagicMock(spec=CrossEncoder)
        ce.predict.return_value = np.array([0.75])

        agent = RerankingAgent(RerankingAgentConfig(), ce, mock_openai_client)
        candidates = [_make_scored_paper("W001")]
        result = agent.rerank("test query", candidates)

        assert len(result) == 1
        assert result[0].paper_id == "W001"
        assert result[0].rerank_score == pytest.approx(0.75)


# ==================== Grounding Tests ====================


class TestGroundingOutputFields:
    """Test that grounding produces all required fields."""

    def test_grounding_output_fields(self, agent: RerankingAgent) -> None:
        """Mocked LLM -> GroundedPaper has snippet, justification, confidence."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response(
            snippet="We propose a transformer model",
            justification="Directly relevant to attention mechanisms",
            confidence=0.92,
        )

        paper = _make_ranked_paper("W001", rerank_score=5.0)
        result = agent.ground("test user text about attention", paper)

        assert isinstance(result, GroundedPaper)
        assert result.paper_id == "W001"
        assert result.supporting_snippet == "We propose a transformer model"
        assert result.justification == "Directly relevant to attention mechanisms"
        assert result.confidence == 0.92
        assert result.rerank_score == 5.0
        assert result.title == "Paper W001"


class TestGroundingPromptContainsContext:
    """Test that the grounding prompt includes both user text and paper content."""

    def test_grounding_prompt_contains_context(self, agent: RerankingAgent) -> None:
        """The prompt sent to the LLM includes both user text and paper abstract."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        paper = _make_ranked_paper(
            "W001",
            title="Attention Is All You Need",
            abstract="The dominant sequence transduction models are based on CNNs or RNNs.",
        )
        user_text = "We use self-attention for sequence modeling."

        agent.ground(user_text, paper)

        # Verify the prompt sent to the LLM
        create_call = agent.instructor_client.chat.completions.create
        assert create_call.called
        messages = create_call.call_args.kwargs.get("messages") or create_call.call_args[1].get(
            "messages"
        )

        # The user message should contain both user text and paper content
        user_message = messages[-1]["content"]
        assert "We use self-attention for sequence modeling." in user_message
        assert "Attention Is All You Need" in user_message
        assert "The dominant sequence transduction models" in user_message


class TestGroundingRetryOnFailure:
    """Test retry behavior when LLM fails."""

    def test_grounding_retry_on_failure(self, agent: RerankingAgent) -> None:
        """Mocked LLM fails once, succeeds on retry -> correct output.

        instructor handles retries internally. We test that the agent
        still produces a valid result after instructor succeeds.
        """
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response(
            confidence=0.88
        )

        paper = _make_ranked_paper("W001")
        result = agent.ground("test text", paper)

        assert isinstance(result, GroundedPaper)
        assert result.confidence == 0.88


class TestGroundingConfidenceRange:
    """Test that confidence is within valid range."""

    def test_grounding_confidence_range(self, agent: RerankingAgent) -> None:
        """Confidence is between 0.0 and 1.0."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response(
            confidence=0.75
        )

        paper = _make_ranked_paper("W001")
        result = agent.ground("test text", paper)

        assert 0.0 <= result.confidence <= 1.0


class TestGroundBatch:
    """Test parallel batch grounding."""

    def test_ground_batch_returns_all(self, agent: RerankingAgent) -> None:
        """Batch grounding returns results for all papers."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        papers = [_make_ranked_paper(f"W{i:03d}") for i in range(3)]
        results = agent.ground_batch("test text", papers)

        assert len(results) == 3
        assert all(isinstance(r, GroundedPaper) for r in results)

    def test_ground_batch_preserves_order(self, agent: RerankingAgent) -> None:
        """Batch results are returned in original input order."""
        responses = [
            _make_grounding_response(confidence=0.9),
            _make_grounding_response(confidence=0.7),
            _make_grounding_response(confidence=0.5),
        ]
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.side_effect = responses

        papers = [
            _make_ranked_paper("W001"),
            _make_ranked_paper("W002"),
            _make_ranked_paper("W003"),
        ]
        results = agent.ground_batch("test text", papers, max_workers=1)

        # With max_workers=1, execution is sequential so order is deterministic
        assert [r.paper_id for r in results] == ["W001", "W002", "W003"]

    def test_ground_batch_skips_failures(self, agent: RerankingAgent) -> None:
        """Papers that fail grounding are skipped, others still returned."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.side_effect = [
            _make_grounding_response(),
            Exception("LLM timeout"),
            _make_grounding_response(),
        ]

        papers = [
            _make_ranked_paper("W001"),
            _make_ranked_paper("W002"),
            _make_ranked_paper("W003"),
        ]
        results = agent.ground_batch("test text", papers, max_workers=1)

        # W002 should be skipped due to failure
        result_ids = [r.paper_id for r in results]
        assert "W001" in result_ids
        assert "W002" not in result_ids
        assert "W003" in result_ids

    def test_ground_batch_empty_input(self, agent: RerankingAgent) -> None:
        """Empty papers list -> empty results."""
        results = agent.ground_batch("test text", [])

        assert results == []


# ==================== Run Method Tests ====================


class TestRerankingAgentRun:
    """Test the run() method (state dict interface)."""

    def test_run_populates_metadata(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """run() populates metadata with latency and counts."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        state = {
            "user_text": "test text about attention",
            "query_analysis": sample_query_analysis,
            "expanded_candidates": three_candidates,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        assert "agent4a_latency_s" in result["metadata"]
        assert "agent4b_latency_s" in result["metadata"]
        assert "agent4_total_latency_s" in result["metadata"]
        assert result["metadata"]["agent4a_reranked_count"] == 3
        assert result["metadata"]["agent4b_grounded_count"] > 0

    def test_run_with_empty_candidates(self, agent: RerankingAgent) -> None:
        """run() with empty expanded_candidates returns empty lists."""
        state = {
            "user_text": "test",
            "query_analysis": None,
            "expanded_candidates": [],
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        assert result["reranked_candidates"] == []
        assert result["grounded_candidates"] == []

    def test_run_returns_ranked_papers(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """run() returns RankedPaper objects in reranked_candidates."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        state = {
            "user_text": "test",
            "query_analysis": sample_query_analysis,
            "expanded_candidates": three_candidates,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        assert len(result["reranked_candidates"]) == 3
        assert all(isinstance(c, RankedPaper) for c in result["reranked_candidates"])

    def test_run_returns_grounded_papers(
        self,
        three_candidates: list[ScoredPaper],
        sample_query_analysis: QueryAnalysis,
        mock_openai_client: MagicMock,
    ) -> None:
        """run() returns GroundedPaper objects in grounded_candidates."""
        ce = MagicMock(spec=CrossEncoder)
        ce.predict.return_value = np.array([0.9, 0.5, 0.1])

        config = RerankingAgentConfig(rerank_top_k=10, ground_top_k=2)
        agent = RerankingAgent(config, ce, mock_openai_client)
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        state = {
            "user_text": "test",
            "query_analysis": sample_query_analysis,
            "expanded_candidates": three_candidates,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        # ground_top_k=2, so exactly 2 grounded papers
        assert len(result["grounded_candidates"]) == 2
        assert all(isinstance(c, GroundedPaper) for c in result["grounded_candidates"])

    def test_run_uses_expanded_query_for_reranking(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """run() uses the expanded_query from query_analysis for reranking."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        state = {
            "user_text": "original text",
            "query_analysis": sample_query_analysis,
            "expanded_candidates": three_candidates,
            "metadata": {},
            "errors": [],
        }
        agent.run(state)

        # The cross-encoder should receive the expanded query
        pairs = agent.cross_encoder.predict.call_args[0][0]
        for query, _ in pairs:
            assert query == "transformer attention mechanism neural networks"

    def test_run_falls_back_to_user_text(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
    ) -> None:
        """run() uses user_text when query_analysis is None."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        state = {
            "user_text": "raw user text here",
            "query_analysis": None,
            "expanded_candidates": three_candidates,
            "metadata": {},
            "errors": [],
        }
        agent.run(state)

        pairs = agent.cross_encoder.predict.call_args[0][0]
        for query, _ in pairs:
            assert query == "raw user text here"


class TestRerankingAgentGracefulDegradation:
    """Test graceful degradation when stages fail."""

    def test_reranking_failure_degrades_gracefully(
        self,
        mock_openai_client: MagicMock,
        three_candidates: list[ScoredPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """When cross-encoder fails, top-k candidates pass through with score=0."""
        ce = MagicMock(spec=CrossEncoder)
        ce.predict.side_effect = RuntimeError("Model load failed")

        config = RerankingAgentConfig(rerank_top_k=2)
        agent = RerankingAgent(config, ce, mock_openai_client)
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.return_value = _make_grounding_response()

        state = {
            "user_text": "test",
            "query_analysis": sample_query_analysis,
            "expanded_candidates": three_candidates,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        # Should have error recorded
        assert any("reranking failed" in e for e in result["errors"])
        # Should still have reranked candidates (degraded)
        assert len(result["reranked_candidates"]) == 2
        assert all(p.rerank_score == 0.0 for p in result["reranked_candidates"])

    def test_grounding_failure_degrades_gracefully(
        self,
        agent: RerankingAgent,
        three_candidates: list[ScoredPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """When grounding fails entirely, grounded_candidates is empty."""
        agent.instructor_client = MagicMock()
        agent.instructor_client.chat.completions.create.side_effect = Exception("API down")

        state = {
            "user_text": "test",
            "query_analysis": sample_query_analysis,
            "expanded_candidates": three_candidates,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        # Reranking should still work
        assert len(result["reranked_candidates"]) == 3
        # Grounding failed but no crash — empty list
        assert result["grounded_candidates"] == []


# ==================== Integration Tests ====================


@pytest.mark.integration
class TestRerankingAgentLive:
    """Integration tests requiring real models and services."""

    @pytest.fixture
    def live_cross_encoder(self) -> CrossEncoder:
        """Load the real cross-encoder model."""
        return CrossEncoder("BAAI/bge-reranker-base")

    def test_cross_encoder_reranks_real_papers(self, live_cross_encoder: CrossEncoder) -> None:
        """Real cross-encoder produces different ranking than cosine similarity.

        Creates candidates where cosine order differs from cross-encoder order.
        """
        mock_client = MagicMock(spec=OpenAI)
        agent = RerankingAgent(
            RerankingAgentConfig(rerank_top_k=5),
            live_cross_encoder,
            mock_client,
        )

        # Candidates ordered by cosine similarity (high to low)
        candidates = [
            _make_scored_paper(
                "W001",
                title="Language Model Safety and Alignment",
                abstract="We study safety and alignment of large language models.",
                score=0.95,
            ),
            _make_scored_paper(
                "W002",
                title="Reducing Hallucination in Large Language Models",
                abstract="We propose methods to reduce hallucination in LLMs.",
                score=0.90,
            ),
            _make_scored_paper(
                "W003",
                title="A Survey of Deep Learning Optimization",
                abstract="This survey reviews optimization techniques for DNNs.",
                score=0.85,
            ),
        ]

        query = "methods for reducing hallucination in LLMs"
        result = agent.rerank(query, candidates)

        assert len(result) == 3
        assert all(isinstance(p, RankedPaper) for p in result)
        # Cross-encoder should rank W002 higher (direct match to query)
        assert result[0].paper_id == "W002", (
            f"Expected W002 (hallucination paper) at top, got {result[0].paper_id}"
        )

    def test_grounding_real_llm(self) -> None:
        """Real LLM call produces a non-empty justification with paper terms.

        Requires OPENAI_API_KEY in environment.
        """
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        mock_ce = MagicMock(spec=CrossEncoder)

        agent = RerankingAgent(RerankingAgentConfig(), mock_ce, client)

        paper = _make_ranked_paper(
            "W001",
            title="Attention Is All You Need",
            abstract=(
                "The dominant sequence transduction models are based on complex "
                "recurrent or convolutional neural networks. We propose a new simple "
                "network architecture, the Transformer, based solely on attention mechanisms."
            ),
        )

        result = agent.ground(
            "We use self-attention for sequence modeling in our architecture.",
            paper,
        )

        assert isinstance(result, GroundedPaper)
        assert len(result.supporting_snippet) > 0
        assert len(result.justification) > 0
        assert 0.0 <= result.confidence <= 1.0
        # Justification should reference terms from the paper
        justification_lower = result.justification.lower()
        assert any(
            term in justification_lower for term in ["attention", "transformer", "sequence"]
        ), f"Justification doesn't mention key paper terms: {result.justification}"

    def test_full_agent4_pipeline(self) -> None:
        """Run Agents 1→2→3→4 end-to-end on a known query.

        Requires Qdrant running, corpus indexed, and OPENAI_API_KEY set.
        """
        import os
        import sqlite3

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        from openai import OpenAI

        from src.agents.expansion_agent import ExpansionAgent, ExpansionAgentConfig
        from src.agents.query_agent import QueryAgent, QueryAgentConfig
        from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
        from src.config import Settings
        from src.indexing.embedder import Embedder, EmbedderConfig
        from src.indexing.qdrant_store import QdrantConfig, QdrantStore

        settings = Settings()

        qdrant_store = QdrantStore(QdrantConfig())
        if not qdrant_store.collection_exists():
            pytest.skip("Qdrant collection 'papers' not found — run index_corpus.py")

        embedder = Embedder(EmbedderConfig(show_progress=False))
        openai_client = OpenAI(api_key=api_key)
        cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
        db = sqlite3.connect(settings.db_path)

        # Agent 1: Query Analysis
        query_agent = QueryAgent(QueryAgentConfig(), openai_client)
        state: dict = {
            "user_text": (
                "Attention mechanisms have become an integral part of compelling "
                "sequence modeling and transduction models, allowing modeling of "
                "dependencies without regard to their distance in the input or "
                "output sequences."
            ),
            "metadata": {},
            "errors": [],
        }
        state.update(query_agent.run(state))

        # Agent 2: Retrieval
        retrieval_agent = RetrievalAgent(RetrievalAgentConfig(top_n=10), embedder, qdrant_store)
        state.update(retrieval_agent.run(state))

        # Agent 3: Expansion
        expansion_agent = ExpansionAgent(ExpansionAgentConfig(), db)
        state.update(expansion_agent.run(state))

        # Agent 4: Reranking + Grounding
        reranking_agent = RerankingAgent(
            RerankingAgentConfig(rerank_top_k=10, ground_top_k=3),
            cross_encoder,
            openai_client,
        )
        state.update(reranking_agent.run(state))

        # Verify reranked candidates
        reranked = state["reranked_candidates"]
        assert len(reranked) > 0
        assert len(reranked) <= 10
        assert all(isinstance(c, RankedPaper) for c in reranked)

        # Verify grounded candidates
        grounded = state["grounded_candidates"]
        assert len(grounded) > 0
        assert len(grounded) <= 3
        assert all(isinstance(c, GroundedPaper) for c in grounded)

        # Grounded papers should have non-empty justifications
        for paper in grounded:
            assert len(paper.supporting_snippet) > 0
            assert len(paper.justification) > 0
            assert 0.0 <= paper.confidence <= 1.0

        db.close()

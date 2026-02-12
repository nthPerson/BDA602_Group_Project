"""Tests for Agent 5: Synthesis.

Test inventory:
- Confidence filtering (below threshold excluded)
- Composite score sorting (rerank_score * confidence, descending)
- Rank numbering (sequential 1, 2, 3, ...)
- All required fields present on each Recommendation
- Empty input produces empty recommendations
- All papers below threshold produces empty output
- Authors list is preserved from database lookup
- run() populates metadata
- Intent match assessment
- Max recommendations limit
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.agents.synthesis_agent import SynthesisAgent, SynthesisAgentConfig
from src.data.models import (
    CitationIntent,
    GroundedPaper,
    QueryAnalysis,
    RankedPaper,
    Recommendation,
)

# ==================== Helpers ====================


def _make_grounded_paper(
    paper_id: str = "W001",
    title: str = "Test Paper",
    rerank_score: float = 3.0,
    snippet: str = "A relevant excerpt from the abstract.",
    justification: str = "This paper is relevant because...",
    confidence: float = 0.8,
) -> GroundedPaper:
    """Create a GroundedPaper for testing."""
    return GroundedPaper(
        paper_id=paper_id,
        title=title,
        rerank_score=rerank_score,
        supporting_snippet=snippet,
        justification=justification,
        confidence=confidence,
    )


def _make_ranked_paper(
    paper_id: str = "W001",
    title: str = "Test Paper",
    abstract: str = "An abstract about testing.",
    year: int = 2023,
    citation_count: int = 100,
    similarity_score: float = 0.85,
    rerank_score: float = 3.0,
) -> RankedPaper:
    """Create a RankedPaper for testing."""
    return RankedPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year,
        citation_count=citation_count,
        similarity_score=similarity_score,
        concepts=["machine learning"],
        rerank_score=rerank_score,
    )


def _make_query_analysis(
    intent: CitationIntent = CitationIntent.BACKGROUND,
) -> QueryAnalysis:
    """Create a QueryAnalysis for testing."""
    return QueryAnalysis(
        topic_keywords=["transformers", "attention"],
        citation_intent=intent,
        expanded_query="transformer attention mechanisms in NLP",
        confidence=0.9,
    )


# ==================== Fixtures ====================


@pytest.fixture
def config() -> SynthesisAgentConfig:
    """Default synthesis agent config."""
    return SynthesisAgentConfig(confidence_threshold=0.4, max_recommendations=10)


@pytest.fixture
def agent(config: SynthesisAgentConfig) -> SynthesisAgent:
    """Synthesis agent without database."""
    return SynthesisAgent(config)


@pytest.fixture
def sample_grounded() -> list[GroundedPaper]:
    """Three grounded papers with varying confidence and rerank scores."""
    return [
        _make_grounded_paper("W001", "Paper A", rerank_score=5.0, confidence=0.9),
        _make_grounded_paper("W002", "Paper B", rerank_score=3.0, confidence=0.7),
        _make_grounded_paper("W003", "Paper C", rerank_score=4.0, confidence=0.5),
    ]


@pytest.fixture
def sample_reranked() -> list[RankedPaper]:
    """Matching reranked papers with metadata."""
    return [
        _make_ranked_paper("W001", "Paper A", year=2023, citation_count=500, rerank_score=5.0),
        _make_ranked_paper("W002", "Paper B", year=2022, citation_count=200, rerank_score=3.0),
        _make_ranked_paper("W003", "Paper C", year=2021, citation_count=300, rerank_score=4.0),
    ]


@pytest.fixture
def sample_query_analysis() -> QueryAnalysis:
    """Sample query analysis."""
    return _make_query_analysis()


# ==================== Test Classes ====================


class TestConfidenceFilter:
    """Test that papers below confidence threshold are excluded."""

    def test_confidence_filter(
        self,
        agent: SynthesisAgent,
        sample_reranked: list[RankedPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Papers below threshold are excluded from recommendations."""
        grounded = [
            _make_grounded_paper("W001", confidence=0.9),
            _make_grounded_paper("W002", confidence=0.3),  # Below 0.4 threshold
            _make_grounded_paper("W003", confidence=0.5),
        ]

        result = agent.synthesize(grounded, sample_reranked, sample_query_analysis)

        paper_ids = [r.paper_id for r in result]
        assert "W001" in paper_ids
        assert "W003" in paper_ids
        assert "W002" not in paper_ids  # Filtered out


class TestCompositeScoreSorting:
    """Test output sorted by rerank_score * confidence (descending)."""

    def test_composite_score_sorting(
        self,
        agent: SynthesisAgent,
        sample_reranked: list[RankedPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Output is sorted by composite score (rerank_score * confidence)."""
        grounded = [
            _make_grounded_paper("W001", rerank_score=2.0, confidence=0.9),  # 1.8
            _make_grounded_paper("W002", rerank_score=5.0, confidence=0.5),  # 2.5
            _make_grounded_paper("W003", rerank_score=3.0, confidence=0.8),  # 2.4
        ]

        result = agent.synthesize(grounded, sample_reranked, sample_query_analysis)

        assert len(result) == 3
        assert result[0].paper_id == "W002"  # 2.5
        assert result[1].paper_id == "W003"  # 2.4
        assert result[2].paper_id == "W001"  # 1.8


class TestRankNumbering:
    """Test ranks are 1, 2, 3, ... (sequential, starting at 1)."""

    def test_rank_numbering(
        self,
        agent: SynthesisAgent,
        sample_grounded: list[GroundedPaper],
        sample_reranked: list[RankedPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Ranks are sequential starting at 1."""
        result = agent.synthesize(sample_grounded, sample_reranked, sample_query_analysis)

        ranks = [r.rank for r in result]
        assert ranks == list(range(1, len(result) + 1))


class TestAllFieldsPresent:
    """Test each Recommendation has every required field populated."""

    def test_all_fields_present(
        self,
        agent: SynthesisAgent,
        sample_grounded: list[GroundedPaper],
        sample_reranked: list[RankedPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Every Recommendation has all required fields."""
        result = agent.synthesize(sample_grounded, sample_reranked, sample_query_analysis)

        assert len(result) > 0
        for rec in result:
            assert isinstance(rec, Recommendation)
            assert rec.rank > 0
            assert len(rec.paper_id) > 0
            assert len(rec.title) > 0
            assert isinstance(rec.authors, list)
            assert rec.year > 0
            assert rec.citation_count >= 0
            assert len(rec.justification) > 0
            assert len(rec.supporting_snippet) > 0
            assert 0.0 <= rec.confidence <= 1.0
            assert len(rec.citation_intent_match) > 0


class TestEmptyInput:
    """Test no grounded papers produces empty recommendations."""

    def test_empty_input(
        self,
        agent: SynthesisAgent,
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """Empty grounded papers list returns empty recommendations."""
        result = agent.synthesize([], [], sample_query_analysis)
        assert result == []

    def test_run_with_empty_state(self, agent: SynthesisAgent) -> None:
        """run() with empty state returns empty recommendations."""
        state = {
            "user_text": "test",
            "grounded_candidates": [],
            "reranked_candidates": [],
            "query_analysis": None,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)
        assert result["final_recommendations"] == []


class TestAllBelowThreshold:
    """Test all papers below confidence produces empty output."""

    def test_all_below_threshold(
        self,
        agent: SynthesisAgent,
        sample_reranked: list[RankedPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """When all papers are below threshold, output is empty."""
        grounded = [
            _make_grounded_paper("W001", confidence=0.1),
            _make_grounded_paper("W002", confidence=0.2),
            _make_grounded_paper("W003", confidence=0.3),
        ]

        result = agent.synthesize(grounded, sample_reranked, sample_query_analysis)
        assert len(result) == 0


class TestFormatsMultipleAuthors:
    """Test that authors list is preserved from database lookup."""

    def test_formats_multiple_authors(self) -> None:
        """Authors from database are included in recommendations."""
        # Create a temporary database with paper records
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

        db = sqlite3.connect(str(db_path))
        db.execute(
            """
            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                year INTEGER,
                citation_count INTEGER,
                doi TEXT,
                arxiv_id TEXT,
                authors TEXT,
                concepts TEXT,
                source TEXT,
                "references" TEXT,
                cited_by_count INTEGER,
                chunk_texts TEXT
            )
            """
        )
        import json

        db.execute(
            """
            INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "W001",
                "Paper A",
                "Abstract",
                2023,
                500,
                None,
                None,
                json.dumps(["Alice Smith", "Bob Jones", "Charlie Brown"]),
                json.dumps(["ML"]),
                None,
                json.dumps([]),
                500,
                json.dumps(["Abstract"]),
            ),
        )
        db.commit()

        try:
            config = SynthesisAgentConfig(confidence_threshold=0.0)
            agent = SynthesisAgent(config, db)

            grounded = [_make_grounded_paper("W001", confidence=0.9)]
            reranked = [_make_ranked_paper("W001", year=2023)]
            query_analysis = _make_query_analysis()

            result = agent.synthesize(grounded, reranked, query_analysis)

            assert len(result) == 1
            assert result[0].authors == ["Alice Smith", "Bob Jones", "Charlie Brown"]
        finally:
            db.close()
            db_path.unlink()


class TestRunPopulatesMetadata:
    """Test that run() populates metadata with latency and counts."""

    def test_run_populates_metadata(
        self,
        agent: SynthesisAgent,
        sample_grounded: list[GroundedPaper],
        sample_reranked: list[RankedPaper],
        sample_query_analysis: QueryAnalysis,
    ) -> None:
        """run() populates metadata with latency and recommendation count."""
        state = {
            "user_text": "test text",
            "grounded_candidates": sample_grounded,
            "reranked_candidates": sample_reranked,
            "query_analysis": sample_query_analysis,
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        assert "agent5_latency_s" in result["metadata"]
        assert "agent5_recommendation_count" in result["metadata"]
        assert result["metadata"]["agent5_recommendation_count"] == len(
            result["final_recommendations"]
        )


class TestIntentMatchAssessment:
    """Test citation intent match assessment."""

    def test_strong_match(self, agent: SynthesisAgent) -> None:
        """High confidence produces 'Strong' match."""
        paper = _make_grounded_paper(confidence=0.85)
        qa = _make_query_analysis(CitationIntent.METHOD)

        match = agent._assess_intent_match(paper, qa)
        assert "Strong" in match
        assert "METHOD" in match

    def test_moderate_match(self, agent: SynthesisAgent) -> None:
        """Medium confidence produces 'Moderate' match."""
        paper = _make_grounded_paper(confidence=0.6)
        qa = _make_query_analysis(CitationIntent.COMPARISON)

        match = agent._assess_intent_match(paper, qa)
        assert "Moderate" in match
        assert "COMPARISON" in match

    def test_weak_match(self, agent: SynthesisAgent) -> None:
        """Low confidence produces 'Weak' match."""
        paper = _make_grounded_paper(confidence=0.45)
        qa = _make_query_analysis(CitationIntent.BACKGROUND)

        match = agent._assess_intent_match(paper, qa)
        assert "Weak" in match
        assert "BACKGROUND" in match

    def test_no_query_analysis(self, agent: SynthesisAgent) -> None:
        """Without query analysis, match is 'unknown'."""
        paper = _make_grounded_paper(confidence=0.9)

        match = agent._assess_intent_match(paper, None)
        assert match == "unknown"


class TestMaxRecommendations:
    """Test max_recommendations limit is respected."""

    def test_max_recommendations_limit(self) -> None:
        """Output is limited to max_recommendations."""
        config = SynthesisAgentConfig(confidence_threshold=0.0, max_recommendations=2)
        agent = SynthesisAgent(config)

        grounded = [
            _make_grounded_paper(f"W{i:03d}", confidence=0.9, rerank_score=float(i))
            for i in range(5)
        ]
        reranked = [_make_ranked_paper(f"W{i:03d}", rerank_score=float(i)) for i in range(5)]
        qa = _make_query_analysis()

        result = agent.synthesize(grounded, reranked, qa)
        assert len(result) == 2

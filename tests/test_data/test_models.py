"""Unit tests for data models."""

from src.data.models import (
    CitationEdge,
    CitationIntent,
    EvalSample,
    GroundedPaper,
    Paper,
    QueryAnalysis,
    RankedPaper,
    Recommendation,
    ScoredPaper,
)


def test_paper_creation() -> None:
    """Test constructing a Paper with all fields."""
    paper = Paper(
        paper_id="W123456",
        title="Attention Is All You Need",
        abstract="We propose a new architecture...",
        year=2017,
        citation_count=90000,
        doi="10.1234/arxiv.1706.03762",
        arxiv_id="1706.03762",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        concepts=["Transformer", "Attention Mechanism"],
        source="NeurIPS",
        references=["W111", "W222"],
        cited_by_count=90000,
        chunk_texts=["We propose a new architecture..."],
    )

    assert paper.paper_id == "W123456"
    assert paper.title == "Attention Is All You Need"
    assert paper.year == 2017
    assert len(paper.authors) == 2
    assert "Transformer" in paper.concepts


def test_paper_optional_fields() -> None:
    """Test that doi and arxiv_id can be None."""
    paper = Paper(
        paper_id="W789",
        title="Some Paper",
        abstract="Abstract text",
        year=2020,
        citation_count=10,
        doi=None,
        arxiv_id=None,
        authors=["Author One"],
        concepts=["ML"],
        source=None,
        references=[],
        cited_by_count=10,
        chunk_texts=["Abstract text"],
    )

    assert paper.doi is None
    assert paper.arxiv_id is None
    assert paper.source is None


def test_citation_edge_creation() -> None:
    """Test constructing a CitationEdge."""
    edge = CitationEdge(source_id="W123", target_id="W456")

    assert edge.source_id == "W123"
    assert edge.target_id == "W456"


def test_query_analysis_creation() -> None:
    """Test constructing a QueryAnalysis with valid intent."""
    analysis = QueryAnalysis(
        topic_keywords=["attention", "transformer"],
        citation_intent=CitationIntent.METHOD,
        expanded_query="neural attention mechanisms for sequence modeling",
        confidence=0.95,
    )

    assert len(analysis.topic_keywords) == 2
    assert analysis.citation_intent == CitationIntent.METHOD
    assert analysis.confidence == 0.95


def test_citation_intent_values() -> None:
    """Test that all 4 citation intent enum values exist."""
    assert CitationIntent.BACKGROUND == "background"
    assert CitationIntent.METHOD == "method"
    assert CitationIntent.COMPARISON == "comparison"
    assert CitationIntent.BENCHMARK == "benchmark"


def test_scored_paper_creation() -> None:
    """Test constructing a ScoredPaper."""
    scored = ScoredPaper(
        paper_id="W123",
        title="Test Paper",
        abstract="Test abstract",
        year=2020,
        citation_count=100,
        similarity_score=0.85,
        concepts=["ML", "NLP"],
    )

    assert scored.paper_id == "W123"
    assert scored.similarity_score == 0.85
    assert len(scored.concepts) == 2


def test_ranked_paper_creation() -> None:
    """Test constructing a RankedPaper."""
    ranked = RankedPaper(
        paper_id="W123",
        title="Test Paper",
        abstract="Test abstract",
        year=2020,
        citation_count=100,
        similarity_score=0.85,
        concepts=["ML"],
        rerank_score=0.92,
    )

    assert ranked.paper_id == "W123"
    assert ranked.similarity_score == 0.85
    assert ranked.rerank_score == 0.92


def test_grounded_paper_creation() -> None:
    """Test constructing a GroundedPaper."""
    grounded = GroundedPaper(
        paper_id="W123",
        title="Test Paper",
        rerank_score=0.92,
        supporting_snippet="We introduce a novel approach...",
        justification="This paper is relevant because...",
        confidence=0.88,
    )

    assert grounded.paper_id == "W123"
    assert grounded.confidence == 0.88
    assert "relevant" in grounded.justification


def test_recommendation_creation() -> None:
    """Test constructing a Recommendation with all fields."""
    rec = Recommendation(
        rank=1,
        paper_id="W123",
        title="Test Paper",
        authors=["Author One", "Author Two"],
        year=2020,
        citation_count=1000,
        justification="This paper is relevant because...",
        supporting_snippet="We propose...",
        confidence=0.90,
        citation_intent_match="Strong match for METHOD citations",
    )

    assert rec.rank == 1
    assert rec.paper_id == "W123"
    assert len(rec.authors) == 2
    assert rec.confidence == 0.90


def test_eval_sample_creation() -> None:
    """Test constructing an EvalSample."""
    sample = EvalSample(
        query_paper_id="W123",
        query_text="Abstract of the query paper...",
        ground_truth_ids=["W456", "W789", "W101"],
        ground_truth_count=3,
    )

    assert sample.query_paper_id == "W123"
    assert sample.ground_truth_count == 3
    assert len(sample.ground_truth_ids) == 3

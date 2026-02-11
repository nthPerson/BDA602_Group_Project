"""Unit tests for OpenAlex API client."""

import pytest

from src.data.models import CitationEdge, Paper
from src.data.openalex_client import OpenAlexClient, OpenAlexConfig


@pytest.fixture
def openalex_config() -> OpenAlexConfig:
    """Create a test OpenAlex configuration.

    Returns:
        Test configuration with default values.
    """
    return OpenAlexConfig(
        email="test@example.com",
        from_year=2018,
        to_year=2025,
        per_page=200,
        rate_limit_delay=0.0,  # No delay for tests
        cache_dir=None,  # No caching for tests
    )


@pytest.fixture
def sample_openalex_work() -> dict:
    """Create a sample OpenAlex work response.

    Returns:
        Dictionary mimicking OpenAlex API work response.
    """
    return {
        "id": "https://openalex.org/W2100837269",
        "title": "Attention Is All You Need",
        "abstract_inverted_index": {
            "We": [0, 10],
            "propose": [1],
            "a": [2],
            "new": [3],
            "architecture": [4],
            "based": [5],
            "solely": [6],
            "on": [7],
            "attention": [8],
            "mechanisms.": [9],
            "demonstrate": [11],
            "this": [12],
            "approach.": [13],
        },
        "publication_year": 2017,
        "cited_by_count": 90000,
        "doi": "https://doi.org/10.1234/arxiv.1706.03762",
        "ids": {
            "openalex": "https://openalex.org/W2100837269",
            "doi": "https://doi.org/10.1234/arxiv.1706.03762",
            "arxiv": "https://arxiv.org/abs/1706.03762",
        },
        "authorships": [
            {"author": {"display_name": "Ashish Vaswani"}},
            {"author": {"display_name": "Noam Shazeer"}},
            {"author": {"display_name": "Niki Parmar"}},
        ],
        "concepts": [
            {"display_name": "Transformer", "score": 0.95},
            {"display_name": "Attention mechanism", "score": 0.88},
            {"display_name": "Neural network", "score": 0.42},
            {"display_name": "Low confidence", "score": 0.15},  # Should be filtered out
        ],
        "primary_location": {
            "source": {
                "display_name": "NeurIPS",
            }
        },
        "referenced_works": [
            "https://openalex.org/W2964185660",
            "https://openalex.org/W2963091681",
            "W123456",  # Some IDs don't have URL prefix
        ],
    }


@pytest.fixture
def openalex_client(openalex_config: OpenAlexConfig) -> OpenAlexClient:
    """Create an OpenAlex client for testing.

    Args:
        openalex_config: Test configuration.

    Returns:
        Configured OpenAlexClient instance.
    """
    return OpenAlexClient(openalex_config)


def test_config_initialization(openalex_config: OpenAlexConfig) -> None:
    """Test that OpenAlexConfig initializes with correct defaults."""
    assert openalex_config.email == "test@example.com"
    assert openalex_config.from_year == 2018
    assert openalex_config.to_year == 2025
    assert openalex_config.per_page == 200
    assert openalex_config.rate_limit_delay == 0.0


def test_client_initialization(openalex_client: OpenAlexClient) -> None:
    """Test that OpenAlexClient initializes correctly."""
    assert openalex_client.config.email == "test@example.com"


def test_has_abstract_true(openalex_client: OpenAlexClient, sample_openalex_work: dict) -> None:
    """Test that _has_abstract returns True for works with abstracts."""
    assert openalex_client._has_abstract(sample_openalex_work) is True


def test_has_abstract_false(openalex_client: OpenAlexClient) -> None:
    """Test that _has_abstract returns False for works without abstracts."""
    work_no_abstract = {
        "id": "https://openalex.org/W123",
        "abstract_inverted_index": None,
    }
    assert openalex_client._has_abstract(work_no_abstract) is False

    work_empty_abstract = {
        "id": "https://openalex.org/W123",
        "abstract_inverted_index": {},
    }
    assert openalex_client._has_abstract(work_empty_abstract) is False


def test_reconstruct_abstract(openalex_client: OpenAlexClient) -> None:
    """Test abstract reconstruction from inverted index."""
    inverted_index = {
        "Hello": [0],
        "world": [1],
        "this": [2],
        "is": [3],
        "a": [4],
        "test": [5],
    }
    abstract = openalex_client._reconstruct_abstract(inverted_index)
    assert abstract == "Hello world this is a test"


def test_reconstruct_abstract_empty(openalex_client: OpenAlexClient) -> None:
    """Test abstract reconstruction with empty inverted index."""
    abstract = openalex_client._reconstruct_abstract({})
    assert abstract == ""


def test_parse_work_to_paper(
    openalex_client: OpenAlexClient,
    sample_openalex_work: dict,
) -> None:
    """Test parsing an OpenAlex work into a Paper object."""
    paper = openalex_client._parse_work_to_paper(sample_openalex_work)

    # Check basic fields
    assert paper.paper_id == "W2100837269"
    assert paper.title == "Attention Is All You Need"
    assert paper.year == 2017
    assert paper.citation_count == 90000
    assert paper.cited_by_count == 90000

    # Check DOI
    assert paper.doi == "https://doi.org/10.1234/arxiv.1706.03762"

    # Check arXiv ID (should be extracted from URL)
    assert paper.arxiv_id == "1706.03762"

    # Check authors
    assert len(paper.authors) == 3
    assert "Ashish Vaswani" in paper.authors
    assert "Noam Shazeer" in paper.authors

    # Check concepts (should only include high-confidence ones)
    assert len(paper.concepts) >= 2
    assert "Transformer" in paper.concepts
    assert "Attention mechanism" in paper.concepts
    # Low confidence concept should be filtered out
    assert "Low confidence" not in paper.concepts

    # Check source
    assert paper.source == "NeurIPS"

    # Check references (should have OpenAlex URL prefix removed)
    assert len(paper.references) == 3
    assert "W2964185660" in paper.references
    assert "W2963091681" in paper.references
    assert "W123456" in paper.references

    # Check abstract
    assert len(paper.abstract) > 0
    assert "We propose a new architecture" in paper.abstract

    # Check chunk_texts (should contain the abstract)
    assert len(paper.chunk_texts) == 1
    assert paper.chunk_texts[0] == paper.abstract


def test_parse_work_with_missing_fields(openalex_client: OpenAlexClient) -> None:
    """Test parsing a work with minimal/missing fields."""
    minimal_work = {
        "id": "https://openalex.org/W123",
        "title": "Minimal Paper",
        "abstract_inverted_index": {"Test": [0], "abstract": [1]},
        "publication_year": None,  # Missing year
        "cited_by_count": 0,
        "doi": None,  # No DOI
        "ids": {},  # No arXiv ID
        "authorships": [],  # No authors
        "concepts": [],  # No concepts
        "primary_location": None,  # No source
        "referenced_works": [],  # No references
    }

    paper = openalex_client._parse_work_to_paper(minimal_work)

    assert paper.paper_id == "W123"
    assert paper.title == "Minimal Paper"
    assert paper.year == 0  # Default for None
    assert paper.citation_count == 0
    assert paper.doi is None
    assert paper.arxiv_id is None
    assert len(paper.authors) == 0
    assert len(paper.concepts) == 0
    assert paper.source is None
    assert len(paper.references) == 0


def test_extract_citation_edges(openalex_client: OpenAlexClient) -> None:
    """Test citation edge extraction from a paper."""
    paper = Paper(
        paper_id="W123",
        title="Test Paper",
        abstract="Test abstract",
        year=2020,
        citation_count=100,
        doi=None,
        arxiv_id=None,
        authors=["Author"],
        concepts=["ML"],
        source=None,
        references=["W456", "W789", "W101"],
        cited_by_count=100,
        chunk_texts=["Test abstract"],
    )

    edges = openalex_client.extract_citation_edges(paper)

    assert len(edges) == 3
    assert all(isinstance(edge, CitationEdge) for edge in edges)
    assert all(edge.source_id == "W123" for edge in edges)

    target_ids = {edge.target_id for edge in edges}
    assert target_ids == {"W456", "W789", "W101"}


def test_extract_citation_edges_empty(openalex_client: OpenAlexClient) -> None:
    """Test citation edge extraction from a paper with no references."""
    paper = Paper(
        paper_id="W123",
        title="Test Paper",
        abstract="Test abstract",
        year=2020,
        citation_count=100,
        doi=None,
        arxiv_id=None,
        authors=["Author"],
        concepts=["ML"],
        source=None,
        references=[],  # No references
        cited_by_count=100,
        chunk_texts=["Test abstract"],
    )

    edges = openalex_client.extract_citation_edges(paper)
    assert len(edges) == 0

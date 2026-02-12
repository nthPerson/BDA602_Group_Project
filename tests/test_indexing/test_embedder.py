"""Unit tests for the embedding pipeline.

These tests verify that the embedder:
- Produces correct-dimensional vectors
- Normalizes embeddings properly
- Handles batching correctly
- Formats paper text correctly
"""

import numpy as np
import pytest

from src.data.models import Paper
from src.indexing import Embedder, EmbedderConfig


@pytest.fixture
def embedder():
    """Create an embedder instance for testing."""
    config = EmbedderConfig(show_progress=False, batch_size=2)
    return Embedder(config)


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return Paper(
        paper_id="W12345",
        title="Attention Is All You Need",
        abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        year=2017,
        citation_count=50000,
        doi="10.1234/test",
        arxiv_id="1706.03762",
        authors=["Vaswani", "Shazeer"],
        concepts=["Transformer", "Attention"],
        source="NeurIPS",
        references=[],
        cited_by_count=50000,
        chunk_texts=[],
    )


def test_embedding_dimensions(embedder):
    """Test that embeddings have the correct dimensions (768 for BGE-base)."""
    text = "This is a test sentence."
    embedding = embedder.embed_texts([text])

    assert embedding.shape == (1, 768), f"Expected shape (1, 768), got {embedding.shape}"


def test_embedding_normalization(embedder):
    """Test that embeddings are L2-normalized (norm ≈ 1.0)."""
    text = "This is a test sentence."
    embedding = embedder.embed_texts([text])

    # Calculate L2 norm
    norm = np.linalg.norm(embedding[0])

    assert abs(norm - 1.0) < 1e-5, f"Expected norm ≈ 1.0, got {norm}"


def test_batch_equals_single(embedder):
    """Test that batch encoding and single encoding produce the same vectors."""
    texts = ["First sentence.", "Second sentence."]

    # Batch encoding
    batch_embeddings = embedder.embed_texts(texts)

    # Single encoding
    single_embeddings = np.array(
        [embedder.embed_texts([text])[0] for text in texts]
    )

    # Should be very similar (within floating point tolerance)
    assert np.allclose(
        batch_embeddings, single_embeddings, atol=1e-5
    ), "Batch and single encodings differ"


def test_empty_text_handling(embedder):
    """Test that empty or whitespace-only input is handled gracefully."""
    # Empty string
    empty_embedding = embedder.embed_texts([""])
    assert empty_embedding.shape == (1, 768)

    # Whitespace only
    whitespace_embedding = embedder.embed_texts(["   "])
    assert whitespace_embedding.shape == (1, 768)


def test_different_texts_different_vectors(embedder):
    """Test that distinct texts produce distinct embeddings."""
    text1 = "Transformers are powerful neural networks."
    text2 = "Convolutional neural networks process images."

    embedding1 = embedder.embed_texts([text1])[0]
    embedding2 = embedder.embed_texts([text2])[0]

    # Cosine similarity should be less than 1.0 (not identical)
    cosine_sim = np.dot(embedding1, embedding2)
    assert cosine_sim < 0.99, f"Distinct texts should have different embeddings (sim={cosine_sim})"


def test_make_paper_text():
    """Test that paper text is formatted correctly."""
    paper = Paper(
        paper_id="W12345",
        title="Test Paper",
        abstract="This is the abstract.",
        year=2020,
        citation_count=100,
        doi=None,
        arxiv_id=None,
        authors=[],
        concepts=[],
        source=None,
        references=[],
        cited_by_count=100,
        chunk_texts=[],
    )

    text = Embedder.make_paper_text(paper)
    expected = "Test Paper\n\nThis is the abstract."

    assert text == expected, f"Expected '{expected}', got '{text}'"


def test_embed_papers(embedder, sample_paper):
    """Test embedding a list of papers."""
    papers = [sample_paper, sample_paper]  # Duplicate for testing
    embeddings = embedder.embed_papers(papers)

    assert embeddings.shape == (2, 768), f"Expected shape (2, 768), got {embeddings.shape}"

    # Both embeddings should be identical (same paper)
    assert np.allclose(embeddings[0], embeddings[1]), "Identical papers should have identical embeddings"


def test_embed_query(embedder):
    """Test that query embedding includes instruction prefix."""
    query = "transformer attention mechanism"
    query_embedding = embedder.embed_query(query)

    assert query_embedding.shape == (768,), f"Expected shape (768,), got {query_embedding.shape}"

    # Query embedding should be normalized
    norm = np.linalg.norm(query_embedding)
    assert abs(norm - 1.0) < 1e-5, f"Expected norm ≈ 1.0, got {norm}"


def test_get_embedding_dimension(embedder):
    """Test that get_embedding_dimension returns correct value."""
    dim = embedder.get_embedding_dimension()
    assert dim == 768, f"Expected dimension 768, got {dim}"

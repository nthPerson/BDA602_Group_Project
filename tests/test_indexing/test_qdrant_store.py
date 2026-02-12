"""Integration tests for Qdrant vector store.

These tests require Qdrant to be running in Docker.
Run: docker compose up -d

All tests are marked with @pytest.mark.integration.
"""

import numpy as np
import pytest

from src.data.models import Paper
from src.indexing import QdrantConfig, QdrantStore


@pytest.fixture
def qdrant_config():
    """Create a Qdrant config for testing with a test collection."""
    return QdrantConfig(collection_name="test_papers")


@pytest.fixture
def qdrant_store(qdrant_config):
    """Create a Qdrant store instance for testing."""
    store = QdrantStore(qdrant_config)

    # Clean up any existing test collection
    if store.collection_exists():
        store.delete_collection()

    yield store

    # Clean up after tests
    if store.collection_exists():
        store.delete_collection()


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        Paper(
            paper_id="W1",
            title="Attention Is All You Need",
            abstract="We propose the Transformer, based solely on attention mechanisms.",
            year=2017,
            citation_count=50000,
            doi="10.1234/test1",
            arxiv_id="1706.03762",
            authors=["Vaswani"],
            concepts=["Transformer", "Attention"],
            source="NeurIPS",
            references=[],
            cited_by_count=50000,
            chunk_texts=[],
        ),
        Paper(
            paper_id="W2",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="We introduce BERT, a language representation model.",
            year=2019,
            citation_count=40000,
            doi="10.1234/test2",
            arxiv_id="1810.04805",
            authors=["Devlin"],
            concepts=["BERT", "Language Model"],
            source="NAACL",
            references=[],
            cited_by_count=40000,
            chunk_texts=[],
        ),
        Paper(
            paper_id="W3",
            title="ImageNet Classification with Deep Convolutional Neural Networks",
            abstract="We train a large, deep convolutional neural network.",
            year=2012,
            citation_count=75000,
            doi="10.1234/test3",
            arxiv_id=None,
            authors=["Krizhevsky"],
            concepts=["CNN", "ImageNet"],
            source="NeurIPS",
            references=[],
            cited_by_count=75000,
            chunk_texts=[],
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings (random 768-dim vectors)."""
    # Create random embeddings and normalize them
    embeddings = np.random.randn(3, 768).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.mark.integration
def test_create_collection(qdrant_store):
    """Test that collection is created with correct vector config."""
    qdrant_store.create_collection()

    assert qdrant_store.collection_exists(), "Collection should exist after creation"

    # Verify collection config
    info = qdrant_store.client.get_collection(qdrant_store.config.collection_name)
    assert info.config.params.vectors.size == 768, "Vector size should be 768"


@pytest.mark.integration
def test_upsert_and_count(qdrant_store, sample_papers, sample_embeddings):
    """Test upserting vectors and counting points."""
    qdrant_store.create_collection()

    # Upsert 3 papers
    qdrant_store.upsert_papers(sample_papers, sample_embeddings, batch_size=2)

    # Verify point count
    count = qdrant_store.get_point_count()
    assert count == 3, f"Expected 3 points, got {count}"


@pytest.mark.integration
def test_search_returns_results(qdrant_store, sample_papers, sample_embeddings):
    """Test that searching with a known vector returns results."""
    qdrant_store.create_collection()
    qdrant_store.upsert_papers(sample_papers, sample_embeddings)

    # Search with the first paper's embedding
    query_vector = sample_embeddings[0]
    results = qdrant_store.search(query_vector, limit=3)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all("score" in r for r in results), "All results should have scores"
    assert all("paper_id" in r for r in results), "All results should have paper_id"
    assert all("title" in r for r in results), "All results should have title"


@pytest.mark.integration
def test_search_top_result_most_similar(qdrant_store, sample_papers, sample_embeddings):
    """Test that the most similar vector is ranked first."""
    qdrant_store.create_collection()
    qdrant_store.upsert_papers(sample_papers, sample_embeddings)

    # Search with the first paper's embedding
    query_vector = sample_embeddings[0]
    results = qdrant_store.search(query_vector, limit=3)

    # First result should be the query paper itself (highest similarity)
    assert results[0]["paper_id"] == "W1", "Top result should be the query paper"
    assert results[0]["score"] > 0.99, f"Self-similarity should be ~1.0, got {results[0]['score']}"


@pytest.mark.integration
def test_filtered_search_by_year(qdrant_store, sample_papers, sample_embeddings):
    """Test that year filter excludes non-matching papers."""
    qdrant_store.create_collection()
    qdrant_store.upsert_papers(sample_papers, sample_embeddings)

    # Search with year filter: only 2017-2019
    query_vector = sample_embeddings[0]
    results = qdrant_store.search(
        query_vector,
        limit=10,
        year_filter=(2017, 2019),
    )

    # Should only return papers from 2017 and 2019 (W1 and W2)
    assert len(results) == 2, f"Expected 2 results (2017-2019), got {len(results)}"
    assert all(2017 <= r["year"] <= 2019 for r in results), "All results should be within year range"


@pytest.mark.integration
def test_filtered_search_by_citation_count(qdrant_store, sample_papers, sample_embeddings):
    """Test that citation count filter excludes low-cited papers."""
    qdrant_store.create_collection()
    qdrant_store.upsert_papers(sample_papers, sample_embeddings)

    # Search with citation count filter: min 45000
    query_vector = sample_embeddings[0]
    results = qdrant_store.search(
        query_vector,
        limit=10,
        min_citation_count=45000,
    )

    # Should only return papers with >=45000 citations (W1 and W3)
    assert len(results) == 2, f"Expected 2 results (>=45k citations), got {len(results)}"
    assert all(r["citation_count"] >= 45000 for r in results), "All results should have >=45k citations"


@pytest.mark.integration
def test_delete_collection(qdrant_store):
    """Test that collection is cleanly deleted."""
    qdrant_store.create_collection()
    assert qdrant_store.collection_exists(), "Collection should exist"

    qdrant_store.delete_collection()
    assert not qdrant_store.collection_exists(), "Collection should not exist after deletion"


@pytest.mark.integration
def test_upsert_overwrites(qdrant_store, sample_papers, sample_embeddings):
    """Test that re-upserting a point updates (not duplicates) it."""
    qdrant_store.create_collection()

    # Upsert paper 1
    qdrant_store.upsert_papers([sample_papers[0]], sample_embeddings[0:1])
    count1 = qdrant_store.get_point_count()
    assert count1 == 1, "Should have 1 point after first upsert"

    # Upsert paper 1 again (with same paper_id)
    qdrant_store.upsert_papers([sample_papers[0]], sample_embeddings[0:1])
    count2 = qdrant_store.get_point_count()
    assert count2 == 1, "Should still have 1 point after re-upsert (not 2)"


@pytest.mark.integration
def test_collection_exists(qdrant_store):
    """Test collection_exists returns correct status."""
    assert not qdrant_store.collection_exists(), "Collection should not exist initially"

    qdrant_store.create_collection()
    assert qdrant_store.collection_exists(), "Collection should exist after creation"

    qdrant_store.delete_collection()
    assert not qdrant_store.collection_exists(), "Collection should not exist after deletion"

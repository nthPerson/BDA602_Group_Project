"""Qdrant vector store client for similarity search and vector management.

This module provides a wrapper around the Qdrant Python client for:
- Creating and managing collections
- Upserting vectors with metadata payloads
- Similarity search with optional metadata filters
- Collection management (delete, point counts, etc.)
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from src.data.models import Paper


@dataclass
class QdrantConfig:
    """Configuration for Qdrant client.

    Attributes:
        host: Qdrant server host.
        port: Qdrant server port.
        collection_name: Name of the collection to use.
        vector_size: Dimension of vectors (768 for BGE-base).
        distance: Distance metric (cosine for normalized vectors).
    """

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "papers"
    vector_size: int = 768
    distance: Distance = Distance.COSINE


class QdrantStore:
    """Qdrant vector store client for papers.

    This class wraps the Qdrant client and provides methods for:
    - Creating collections with appropriate vector configuration
    - Upserting paper embeddings with metadata payloads
    - Similarity search with optional filters
    - Collection management
    """

    def __init__(self, config: QdrantConfig) -> None:
        """Initialize Qdrant client and connect to server.

        Args:
            config: Configuration for Qdrant connection and collection.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Connecting to Qdrant at {config.host}:{config.port}")
        self.client = QdrantClient(host=config.host, port=config.port)

        # Verify connection
        try:
            self.client.get_collections()
            self.logger.info("Successfully connected to Qdrant")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, recreate: bool = False) -> None:
        """Create the papers collection with appropriate vector configuration.

        Creates a collection with:
        - Vector size: 768 (BGE-base embedding dimension)
        - Distance: Cosine (appropriate for normalized vectors)
        - Payload indexes on: year, citation_count, concepts

        Args:
            recreate: If True, delete existing collection first.
        """
        collection_name = self.config.collection_name

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists:
            if recreate:
                self.logger.info(
                    f"Collection '{collection_name}' exists. Deleting for recreation..."
                )
                self.client.delete_collection(collection_name)
            else:
                self.logger.info(
                    f"Collection '{collection_name}' already exists. Skipping creation."
                )
                return

        # Create collection
        self.logger.info(
            f"Creating collection '{collection_name}' "
            f"(vector_size={self.config.vector_size}, "
            f"distance={self.config.distance})"
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=self.config.distance,
            ),
        )

        # Create payload indexes for filtered search
        self.logger.info("Creating payload indexes...")
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="year",
            field_schema="integer",
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="citation_count",
            field_schema="integer",
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="concepts",
            field_schema="keyword",
        )

        self.logger.info(f"Collection '{collection_name}' created successfully.")

    def upsert_papers(
        self, papers: list[Paper], embeddings: np.ndarray, batch_size: int = 100
    ) -> None:
        """Upsert papers and their embeddings into Qdrant.

        Each paper is stored as a point with:
        - ID: paper_id (hashed to UUID)
        - Vector: 768-dim embedding
        - Payload: {paper_id, title, abstract, year, citation_count, concepts, ...}

        Args:
            papers: List of Paper objects.
            embeddings: numpy array of shape (len(papers), 768).
            batch_size: Number of points to upload per batch.
        """
        collection_name = self.config.collection_name

        if len(papers) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(papers)} papers but {len(embeddings)} embeddings"
            )

        self.logger.info(
            f"Upserting {len(papers)} papers into '{collection_name}' "
            f"(batch_size={batch_size})..."
        )

        # Convert to PointStruct list
        points = []
        for paper, embedding in zip(papers, embeddings):
            # Create payload with paper metadata
            payload = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
                "citation_count": paper.citation_count,
                "concepts": paper.concepts,
                "authors": paper.authors,
                "source": paper.source,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
            }

            # Use paper_id as the point ID (Qdrant will hash it to UUID)
            point = PointStruct(
                id=hash(paper.paper_id) & 0xFFFFFFFFFFFFFFFF,  # Convert to positive int
                vector=embedding.tolist(),
                payload=payload,
            )
            points.append(point)

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

            if (i // batch_size) % 10 == 0:
                self.logger.info(
                    f"Upserted {min(i + batch_size, len(points))}/{len(points)} points..."
                )

        self.logger.info(f"Successfully upserted {len(points)} points.")

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        year_filter: tuple[int, int] | None = None,
        min_citation_count: int | None = None,
        concept_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar papers using a query vector.

        Args:
            query_vector: Query embedding (768-dim numpy array).
            limit: Number of results to return.
            year_filter: Optional (min_year, max_year) tuple to filter by year.
            min_citation_count: Optional minimum citation count filter.
            concept_filter: Optional concept keyword filter (exact match).

        Returns:
            List of search results, each a dict with:
                - score: Similarity score (higher is better)
                - paper_id: Paper ID
                - title: Paper title
                - year: Publication year
                - citation_count: Citation count
                - abstract: Paper abstract (first 200 chars)
                - All other payload fields
        """
        collection_name = self.config.collection_name

        # Build filter
        filter_conditions = []

        if year_filter:
            min_year, max_year = year_filter
            filter_conditions.append(
                FieldCondition(
                    key="year",
                    range=Range(gte=min_year, lte=max_year),
                )
            )

        if min_citation_count is not None:
            filter_conditions.append(
                FieldCondition(
                    key="citation_count",
                    range=Range(gte=min_citation_count),
                )
            )

        if concept_filter:
            filter_conditions.append(
                FieldCondition(
                    key="concepts",
                    match=MatchValue(value=concept_filter),
                )
            )

        # Create Filter object if we have conditions
        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Perform search (using query_points for qdrant-client >=1.8)
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            query_filter=search_filter,
            limit=limit,
        ).points

        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                "score": result.score,
                **result.payload,  # Unpack all payload fields
            }
            formatted_results.append(formatted_result)

        return formatted_results

    def get_point_count(self) -> int:
        """Get the number of points in the collection.

        Returns:
            Number of points in the collection.
        """
        collection_name = self.config.collection_name
        info = self.client.get_collection(collection_name)
        return info.points_count

    def delete_collection(self) -> None:
        """Delete the papers collection.

        Warning: This permanently deletes all vectors and metadata.
        """
        collection_name = self.config.collection_name
        self.logger.warning(f"Deleting collection '{collection_name}'...")
        self.client.delete_collection(collection_name)
        self.logger.info(f"Collection '{collection_name}' deleted.")

    def collection_exists(self) -> bool:
        """Check if the papers collection exists.

        Returns:
            True if collection exists, False otherwise.
        """
        collection_name = self.config.collection_name
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

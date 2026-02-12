"""Embedding pipeline for converting papers into dense vector representations.

This module provides a wrapper around sentence-transformers for encoding
paper text (title + abstract) into 768-dimensional embeddings using the
BAAI/bge-base-en-v1.5 model.

Key features:
- Instruction-prefixed queries for improved retrieval quality
- Batch encoding with progress tracking
- GPU acceleration when available (automatic fallback to CPU)
- Normalized embeddings for cosine similarity search
"""

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.models import Paper


@dataclass
class EmbedderConfig:
    """Configuration for the embedding pipeline.

    Attributes:
        model_name: HuggingFace model ID for sentence-transformers.
        batch_size: Number of texts to encode per batch.
        show_progress: If True, display progress bar during encoding.
        normalize: If True, L2-normalize embeddings (required for cosine similarity).
        instruction_prefix: Optional instruction prefix for queries (BGE-specific).
    """

    model_name: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 64
    show_progress: bool = True
    normalize: bool = True
    instruction_prefix: str = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Embedding pipeline for papers using sentence-transformers.

    This class wraps the sentence-transformers library and provides methods for:
    - Loading the BGE embedding model
    - Encoding papers (title + abstract) into 768-dim vectors
    - Encoding queries with instruction prefixes
    - Batch processing with progress tracking
    """

    def __init__(self, config: EmbedderConfig) -> None:
        """Initialize the embedder and load the model.

        Args:
            config: Configuration for embedding behavior.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading embedding model: {config.model_name}")
        self.model = SentenceTransformer(config.model_name)

        # Log model info
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Model loaded. Embedding dimension: {embedding_dim}")

        # Verify expected dimensions
        if config.model_name == "BAAI/bge-base-en-v1.5" and embedding_dim != 768:
            self.logger.warning(
                f"Expected 768 dimensions for BGE-base, got {embedding_dim}"
            )

    def embed_papers(
        self, papers: list[Paper], batch_size: int | None = None
    ) -> np.ndarray:
        """Encode a list of papers into embeddings.

        Each paper is converted to text via make_paper_text() (title + abstract),
        then encoded into a 768-dimensional vector.

        Args:
            papers: List of Paper objects to embed.
            batch_size: Override config batch size. If None, uses config value.

        Returns:
            numpy array of shape (num_papers, 768) with L2-normalized embeddings.
        """
        batch_size = batch_size or self.config.batch_size

        # Convert papers to text
        texts = [self.make_paper_text(paper) for paper in papers]

        # Encode with progress bar
        self.logger.info(f"Encoding {len(texts)} papers (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=self.config.show_progress,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )

        self.logger.info(f"Encoded {len(embeddings)} papers.")
        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        """Encode a query string into an embedding.

        For BGE models, queries should be prefixed with an instruction to improve
        retrieval quality. This is automatically handled.

        Args:
            query_text: The query string (e.g., user's input paragraph or search query).

        Returns:
            numpy array of shape (768,) with L2-normalized embedding.
        """
        # Add instruction prefix for queries (BGE-specific optimization)
        prefixed_query = f"{self.config.instruction_prefix}{query_text}"

        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )

        return embedding

    def embed_texts(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Encode a list of arbitrary text strings into embeddings.

        This is a generic method for encoding any text, not just papers.
        Useful for testing or encoding other content types.

        Args:
            texts: List of text strings to encode.
            batch_size: Override config batch size. If None, uses config value.

        Returns:
            numpy array of shape (num_texts, 768) with L2-normalized embeddings.
        """
        batch_size = batch_size or self.config.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=self.config.show_progress,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )

        return embeddings

    @staticmethod
    def make_paper_text(paper: Paper) -> str:
        """Convert a Paper object into a single text string for embedding.

        Format: "Title\n\nAbstract"

        This simple concatenation preserves both the title (high-signal keywords)
        and the abstract (context and details).

        Args:
            paper: Paper object with title and abstract.

        Returns:
            Formatted text string.
        """
        return f"{paper.title}\n\n{paper.abstract}"

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the loaded model.

        Returns:
            Embedding dimension (768 for BGE-base).
        """
        return self.model.get_sentence_embedding_dimension()

"""BM25 baseline for evaluation comparison.

Implements a simple BM25 keyword-based retrieval system as a lower baseline.
If the neural pipeline can't beat BM25, something is wrong.

BM25 tokenizes paper abstracts, builds an index, and ranks by term-frequency
matching. It uses no embeddings, no LLMs, and no citation graph — purely
statistical keyword matching.
"""

import logging
import re
import sqlite3

from rank_bm25 import BM25Okapi

from src.data.db import get_all_papers
from src.data.models import Paper

logger = logging.getLogger(__name__)

# Common English stop words to exclude from BM25 tokenization
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it that this with by from as are was "
    "were be been being have has had do does did will would could should may might "
    "shall can not no nor so if then than too very also just about above after again "
    "all any because before between both each few how into more most much other "
    "out over own same some such there these they those through under until up "
    "we what when where which while who why our their its his her he him them me my "
    "your i you us".split()
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, split on non-alphanumeric, remove stop words.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens with stop words removed.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


class BM25Baseline:
    """BM25-based retrieval baseline.

    Builds an index over all paper abstracts in the corpus and provides
    a search method that returns ranked paper IDs by BM25 score.
    """

    def __init__(self, papers: list[Paper]) -> None:
        """Initialize the BM25 index from a list of papers.

        Args:
            papers: List of Paper instances to index. Papers without abstracts
                are skipped.
        """
        self.papers: list[Paper] = []
        corpus_tokens: list[list[str]] = []

        for paper in papers:
            if paper.abstract and paper.abstract.strip():
                self.papers.append(paper)
                corpus_tokens.append(_tokenize(paper.abstract))

        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"BM25 index built over {len(self.papers)} papers")

    @classmethod
    def from_db(cls, db: sqlite3.Connection) -> "BM25Baseline":
        """Build a BM25 index from all papers in the database.

        Args:
            db: SQLite database connection.

        Returns:
            BM25Baseline instance indexed over the corpus.
        """
        papers = get_all_papers(db)
        return cls(papers)

    def search(
        self,
        query: str,
        top_k: int = 20,
        exclude_ids: set[str] | None = None,
    ) -> list[str]:
        """Search for papers matching the query text.

        Args:
            query: Query text to search for.
            top_k: Number of top results to return.
            exclude_ids: Optional set of paper IDs to exclude from results
                (e.g., the query paper itself).

        Returns:
            Ordered list of paper IDs, ranked by BM25 score (descending).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with paper IDs and sort
        scored_papers = sorted(
            zip(self.papers, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[str] = []
        for paper, _score in scored_papers:
            if exclude_ids and paper.paper_id in exclude_ids:
                continue
            results.append(paper.paper_id)
            if len(results) >= top_k:
                break

        return results

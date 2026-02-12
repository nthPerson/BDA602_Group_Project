"""Agent 3: Citation Expansion.

This agent expands the candidate set by traversing the citation graph.
For each top-K seed paper from Agent 2, it looks up direct references
(papers it cites) and direct citations (papers that cite it), then
deduplicates, prioritizes, and limits the expansion pool.

No LLM is involved — this agent is purely deterministic, using only
SQLite queries on the citation_edges table. Given the same inputs and
citation graph, it always produces the same output.
"""

import logging
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

from src.data.db import (
    filter_to_corpus,
    get_cited_by_ids,
    get_papers_by_ids,
    get_reference_ids,
)
from src.data.models import ScoredPaper
from src.orchestration.state import PipelineState


@dataclass
class ExpansionAgentConfig:
    """Configuration for the Citation Expansion agent.

    Attributes:
        seed_top_k: Number of top retrieval candidates to use as seeds
            for citation graph expansion.
        max_expansion: Maximum number of additional papers to add beyond
            the original candidates.
    """

    seed_top_k: int = 10
    max_expansion: int = 40


class ExpansionAgent:
    """Agent 3: Citation Expansion via depth-1 BFS on the citation graph.

    Takes the top seed papers from Agent 2's retrieval results and
    discovers additional relevant papers by following citation links.
    Papers that appear in multiple seed papers' citation neighborhoods
    are prioritized.
    """

    def __init__(self, config: ExpansionAgentConfig, db: sqlite3.Connection) -> None:
        """Initialize the expansion agent.

        Args:
            config: Agent configuration.
            db: SQLite database connection for citation graph queries.
        """
        self.config = config
        self.db = db
        self.logger = logging.getLogger(__name__)

    def run(self, state: PipelineState) -> dict:
        """Execute the citation expansion step.

        Reads `retrieval_candidates` from state, produces `expanded_candidates`.

        Args:
            state: Current pipeline state with `retrieval_candidates` populated.

        Returns:
            Dict with `expanded_candidates` and updated `metadata`.
        """
        retrieval_candidates = state.get("retrieval_candidates", [])
        metadata = dict(state.get("metadata", {}))
        errors = list(state.get("errors", []))

        if not retrieval_candidates:
            self.logger.warning("No retrieval candidates — skipping expansion")
            return {
                "expanded_candidates": [],
                "metadata": metadata,
                "errors": errors,
            }

        start_time = time.time()

        try:
            expanded = self.expand(retrieval_candidates)
            new_count = len(expanded) - len(retrieval_candidates)
            self.logger.info(
                f"Expanded {len(retrieval_candidates)} candidates by "
                f"{new_count} to {len(expanded)} total"
            )
        except Exception as e:
            self.logger.error(f"Expansion failed: {e}")
            errors.append(f"Agent 3 expansion failed: {e}")
            # Graceful degradation: pass through originals unchanged
            expanded = list(retrieval_candidates)

        elapsed = time.time() - start_time
        metadata["agent3_latency_s"] = round(elapsed, 3)
        metadata["agent3_expanded_count"] = len(expanded)
        metadata["agent3_new_papers"] = len(expanded) - len(retrieval_candidates)

        return {
            "expanded_candidates": expanded,
            "metadata": metadata,
            "errors": errors,
        }

    def expand(
        self,
        candidates: list[ScoredPaper],
        seed_top_k: int | None = None,
        max_expansion: int | None = None,
    ) -> list[ScoredPaper]:
        """Expand the candidate set via depth-1 BFS on the citation graph.

        For each seed paper, looks up both forward references (papers it cites)
        and backward citations (papers that cite it). Deduplicates the expansion
        pool, prioritizes by multi-citation count then citation count, and limits
        to max_expansion additional papers.

        Args:
            candidates: Retrieval candidates from Agent 2.
            seed_top_k: Override config seed_top_k. If None, uses config value.
            max_expansion: Override config max_expansion. If None, uses config value.

        Returns:
            Combined list: original candidates + expanded papers (deduplicated).
        """
        seed_top_k = seed_top_k if seed_top_k is not None else self.config.seed_top_k
        max_expansion = max_expansion if max_expansion is not None else self.config.max_expansion

        # Use the top-k candidates as seeds for expansion
        seed_papers = candidates[:seed_top_k]
        seed_ids = {p.paper_id for p in candidates}

        # Track how many seed papers cite each expansion candidate
        expansion_counter: Counter[str] = Counter()

        for paper in seed_papers:
            # Forward: papers this seed cites
            ref_ids = get_reference_ids(self.db, paper.paper_id)
            for ref_id in ref_ids:
                if ref_id not in seed_ids:
                    expansion_counter[ref_id] += 1

            # Backward: papers that cite this seed
            cited_by_ids = get_cited_by_ids(self.db, paper.paper_id)
            for citer_id in cited_by_ids:
                if citer_id not in seed_ids:
                    expansion_counter[citer_id] += 1

        if not expansion_counter:
            return list(candidates)

        # Filter to only papers that exist in the corpus
        all_expansion_ids = list(expansion_counter.keys())
        in_corpus = filter_to_corpus(self.db, all_expansion_ids)
        expansion_counter = Counter(
            {pid: count for pid, count in expansion_counter.items() if pid in in_corpus}
        )

        if not expansion_counter:
            return list(candidates)

        # Fetch paper metadata to get citation_count for prioritization
        expansion_ids = list(expansion_counter.keys())
        expansion_papers = get_papers_by_ids(self.db, expansion_ids)

        # Build a lookup for citation counts
        citation_counts = {p.paper_id: p.citation_count for p in expansion_papers}

        # Prioritize: (1) multi-citation count desc, (2) citation count desc
        sorted_expansion_ids = sorted(
            expansion_counter.keys(),
            key=lambda pid: (expansion_counter[pid], citation_counts.get(pid, 0)),
            reverse=True,
        )

        # Limit expansion size
        selected_ids = sorted_expansion_ids[:max_expansion]

        # Build a lookup from paper_id -> Paper for selected papers
        paper_lookup = {p.paper_id: p for p in expansion_papers}

        # Convert expanded papers to ScoredPaper (similarity_score=0.0)
        expanded_scored = []
        for pid in selected_ids:
            paper = paper_lookup.get(pid)
            if paper is None:
                continue
            expanded_scored.append(
                ScoredPaper(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    year=paper.year,
                    citation_count=paper.citation_count,
                    similarity_score=0.0,  # No embedding score for expanded papers
                    concepts=paper.concepts,
                )
            )

        # Combine: originals first, then expanded
        return list(candidates) + expanded_scored

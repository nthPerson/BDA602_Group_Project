"""LangGraph pipeline definition for the citation recommendation system.

This module wires all five agents into a linear LangGraph StateGraph:

    query_analysis → primary_retrieval → citation_expansion →
    reranking_and_grounding → synthesis → END

Each node wraps its agent's run() method with error handling so the pipeline
degrades gracefully: if any agent raises an unhandled exception, the error
is logged to state["errors"] and subsequent agents receive whatever state
was available prior to the failure.

Usage:
    # With dependency injection (for testing / custom configuration):
    graph = build_graph(agent1, agent2, agent3, agent4, agent5)
    result = graph.invoke({"user_text": "...", "metadata": {}, "errors": []})

    # Convenience builder (creates all dependencies automatically):
    pipeline = build_pipeline()
    result = pipeline.invoke({"user_text": "...", "metadata": {}, "errors": []})
"""

import logging
import sqlite3

from langgraph.graph import END, StateGraph
from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.agents.expansion_agent import ExpansionAgent, ExpansionAgentConfig
from src.agents.query_agent import QueryAgent, QueryAgentConfig
from src.agents.reranking_agent import RerankingAgent, RerankingAgentConfig
from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from src.agents.synthesis_agent import SynthesisAgent, SynthesisAgentConfig
from src.config import Settings
from src.indexing.embedder import Embedder, EmbedderConfig
from src.indexing.qdrant_store import QdrantConfig, QdrantStore
from src.orchestration.state import PipelineState

logger = logging.getLogger(__name__)


# ==================== Node wrappers with error handling ====================


def _make_node(agent: object, node_name: str):
    """Create a LangGraph node function that wraps an agent with error handling.

    If the agent's run() raises an unhandled exception, the error is logged
    to state["errors"] and the pipeline continues with the existing state.

    Args:
        agent: An agent instance with a run(state) -> dict method.
        node_name: Human-readable name for error messages.

    Returns:
        A callable suitable for StateGraph.add_node().
    """

    def node_fn(state: PipelineState) -> dict:
        try:
            return agent.run(state)
        except Exception as e:
            logger.error(f"{node_name} failed with unhandled exception: {e}")
            errors = list(state.get("errors", []))
            errors.append(f"{node_name} failed: {e}")
            return {"errors": errors}

    return node_fn


# ==================== Graph construction ====================


def build_graph(
    query_agent: QueryAgent,
    retrieval_agent: RetrievalAgent,
    expansion_agent: ExpansionAgent,
    reranking_agent: RerankingAgent,
    synthesis_agent: SynthesisAgent,
) -> object:
    """Build and compile the LangGraph pipeline from pre-built agents.

    This is the primary entry point for constructing the pipeline.
    All agent dependencies are injected, making this fully testable.

    Args:
        query_agent: Agent 1 — intent classification + query expansion.
        retrieval_agent: Agent 2 — dense retrieval from Qdrant.
        expansion_agent: Agent 3 — citation graph expansion.
        reranking_agent: Agent 4 — cross-encoder reranking + LLM grounding.
        synthesis_agent: Agent 5 — confidence filtering + final formatting.

    Returns:
        Compiled LangGraph application (invoke-able).
    """
    workflow = StateGraph(PipelineState)

    # Add nodes — each wraps an agent with error handling
    workflow.add_node("query_analysis", _make_node(query_agent, "query_analysis"))
    workflow.add_node("primary_retrieval", _make_node(retrieval_agent, "primary_retrieval"))
    workflow.add_node("citation_expansion", _make_node(expansion_agent, "citation_expansion"))
    workflow.add_node(
        "reranking_and_grounding",
        _make_node(reranking_agent, "reranking_and_grounding"),
    )
    workflow.add_node("synthesis", _make_node(synthesis_agent, "synthesis"))

    # Define linear edges
    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "primary_retrieval")
    workflow.add_edge("primary_retrieval", "citation_expansion")
    workflow.add_edge("citation_expansion", "reranking_and_grounding")
    workflow.add_edge("reranking_and_grounding", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


def build_pipeline() -> object:
    """Convenience builder that creates all agents and returns a compiled pipeline.

    Creates all dependencies (OpenAI client, embedder, Qdrant, SQLite, cross-encoder)
    from the project settings and environment variables.

    Requires:
        - OPENAI_API_KEY set in environment
        - Qdrant running (docker compose up -d)
        - Corpus built and indexed (papers.db + Qdrant collection)

    Returns:
        Compiled LangGraph application ready for invoke().
    """
    settings = Settings()
    openai_client = OpenAI()

    # Agent 1: Query Analysis
    query_agent = QueryAgent(QueryAgentConfig(), openai_client)

    # Agent 2: Retrieval
    embedder = Embedder(EmbedderConfig(show_progress=False))
    qdrant_store = QdrantStore(QdrantConfig())
    retrieval_agent = RetrievalAgent(RetrievalAgentConfig(top_n=10), embedder, qdrant_store)

    # Agent 3: Citation Expansion
    db = sqlite3.connect(settings.db_path)
    expansion_agent = ExpansionAgent(ExpansionAgentConfig(), db)

    # Agent 4: Reranking & Grounding
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    reranking_agent = RerankingAgent(
        RerankingAgentConfig(rerank_top_k=10, ground_top_k=3),
        cross_encoder,
        openai_client,
    )

    # Agent 5: Synthesis
    synthesis_agent = SynthesisAgent(SynthesisAgentConfig(), db)

    return build_graph(
        query_agent,
        retrieval_agent,
        expansion_agent,
        reranking_agent,
        synthesis_agent,
    )

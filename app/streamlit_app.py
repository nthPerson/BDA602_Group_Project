"""Streamlit UI for the Citation Recommendation System.

This is the interactive web interface for the multi-agent RAG pipeline.
Users paste a paragraph of text and receive ranked citation recommendations
with justifications and supporting snippets.

Usage:
    streamlit run app/streamlit_app.py
"""

import json
import logging
import sqlite3
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.agents.expansion_agent import ExpansionAgent, ExpansionAgentConfig
from src.agents.query_agent import QueryAgent, QueryAgentConfig
from src.agents.reranking_agent import RerankingAgent, RerankingAgentConfig
from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from src.agents.synthesis_agent import SynthesisAgent, SynthesisAgentConfig
from src.config import Settings
from src.data.models import Recommendation
from src.indexing.embedder import Embedder, EmbedderConfig
from src.indexing.qdrant_store import QdrantConfig, QdrantStore
from src.orchestration.graph import build_graph

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="Citation Recommendation System",
    page_icon="\U0001f4da",
    layout="wide",
)


# ==================== Cached Resources ====================


@st.cache_resource
def load_settings() -> Settings:
    """Load application settings (cached across reruns)."""
    return Settings()


@st.cache_resource
def load_embedder() -> Embedder:
    """Load the BGE embedding model (cached, ~400 MB download on first use)."""
    return Embedder(EmbedderConfig(show_progress=False))


@st.cache_resource
def load_qdrant_store() -> QdrantStore:
    """Load the Qdrant vector store client (cached)."""
    return QdrantStore(QdrantConfig())


@st.cache_resource
def load_cross_encoder() -> CrossEncoder:
    """Load the BGE cross-encoder reranker (cached, ~400 MB download on first use)."""
    return CrossEncoder("BAAI/bge-reranker-base")


@st.cache_resource
def load_openai_client() -> OpenAI:
    """Load the OpenAI client (cached)."""
    return OpenAI()


@st.cache_resource
def load_db_connection() -> sqlite3.Connection:
    """Load the SQLite database connection (cached)."""
    settings = load_settings()
    return sqlite3.connect(settings.db_path, check_same_thread=False)


def build_app_pipeline(
    top_n: int = 10,
    rerank_top_k: int = 10,
    ground_top_k: int = 3,
    confidence_threshold: float = 0.4,
    max_recommendations: int = 10,
) -> object:
    """Build the LangGraph pipeline with configurable parameters.

    Args:
        top_n: Number of candidates for Agent 2 retrieval.
        rerank_top_k: Number of candidates to keep after reranking.
        ground_top_k: Number of candidates to ground with LLM.
        confidence_threshold: Minimum confidence for Agent 5 filtering.
        max_recommendations: Maximum final recommendations.

    Returns:
        Compiled LangGraph pipeline.
    """
    openai_client = load_openai_client()
    embedder = load_embedder()
    qdrant_store = load_qdrant_store()
    cross_encoder = load_cross_encoder()
    db = load_db_connection()

    query_agent = QueryAgent(QueryAgentConfig(), openai_client)
    retrieval_agent = RetrievalAgent(RetrievalAgentConfig(top_n=top_n), embedder, qdrant_store)
    expansion_agent = ExpansionAgent(ExpansionAgentConfig(), db)
    reranking_agent = RerankingAgent(
        RerankingAgentConfig(rerank_top_k=rerank_top_k, ground_top_k=ground_top_k),
        cross_encoder,
        openai_client,
    )
    synthesis_agent = SynthesisAgent(
        SynthesisAgentConfig(
            confidence_threshold=confidence_threshold,
            max_recommendations=max_recommendations,
        ),
        db,
    )

    return build_graph(
        query_agent, retrieval_agent, expansion_agent, reranking_agent, synthesis_agent
    )


# ==================== Sidebar ====================


def render_sidebar() -> dict:
    """Render the sidebar with advanced options and return settings dict."""
    with st.sidebar:
        st.header("Advanced Options")

        year_range = st.slider(
            "Publication year range",
            min_value=2015,
            max_value=2025,
            value=(2015, 2025),
            help="Filter results to papers published within this range.",
        )

        max_results = st.slider(
            "Maximum results",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of citation recommendations to return.",
        )

        st.divider()
        st.subheader("Pipeline Parameters")

        top_n = st.number_input(
            "Initial retrieval candidates",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of papers retrieved by dense search (Agent 2).",
        )

        ground_top_k = st.number_input(
            "Papers to ground with LLM",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of top-ranked papers to justify via LLM (Agent 4).",
        )

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum confidence score to include a recommendation.",
        )

        st.divider()
        show_debug = st.checkbox("Show debug panel", value=False)

    return {
        "year_min": year_range[0],
        "year_max": year_range[1],
        "max_results": max_results,
        "top_n": top_n,
        "ground_top_k": ground_top_k,
        "confidence_threshold": confidence_threshold,
        "show_debug": show_debug,
    }


# ==================== Recommendation Display ====================


def render_recommendation(rec: Recommendation, index: int) -> None:
    """Render a single recommendation card.

    Args:
        rec: The Recommendation dataclass to display.
        index: 0-based index for unique widget keys.
    """
    confidence_pct = int(rec.confidence * 100)

    # Color-code confidence
    if confidence_pct >= 80:
        conf_color = "green"
    elif confidence_pct >= 50:
        conf_color = "orange"
    else:
        conf_color = "red"

    authors_str = ", ".join(rec.authors[:3]) if rec.authors else "Unknown authors"
    if len(rec.authors) > 3:
        authors_str += " et al."

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"**{rec.rank}. {rec.title}** ({rec.year})")
        st.caption(f"{authors_str} | Citations: {rec.citation_count:,}")
    with col2:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span style='color:{conf_color};font-size:1.5em;font-weight:bold'>"
            f"{confidence_pct}%</span><br>"
            f"<small>confidence</small></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Justification", expanded=False):
        st.write(rec.justification)

    with st.expander("Supporting Snippet", expanded=False):
        st.info(rec.supporting_snippet)

    st.caption(f"*{rec.citation_intent_match}*")
    st.divider()


# ==================== Pipeline Execution ====================


def run_pipeline_with_status(user_text: str, options: dict) -> dict | None:
    """Run the full pipeline with real-time status updates.

    Args:
        user_text: The user's input text.
        options: Sidebar options dict.

    Returns:
        The final pipeline state dict, or None if an error occurred.
    """
    with st.status("Running citation recommendation pipeline...", expanded=True) as status:
        total_start = time.time()

        # Step 1: Build pipeline
        st.write("Loading pipeline components...")
        try:
            pipeline = build_app_pipeline(
                top_n=options["top_n"],
                rerank_top_k=10,
                ground_top_k=options["ground_top_k"],
                confidence_threshold=options["confidence_threshold"],
                max_recommendations=options["max_results"],
            )
        except Exception as e:
            st.error(f"Failed to build pipeline: {e}")
            status.update(label="Pipeline failed", state="error")
            return None

        # Step 2: Run pipeline
        st.write("Analyzing intent...")
        try:
            state = pipeline.invoke({"user_text": user_text, "metadata": {}, "errors": []})
        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")
            status.update(label="Pipeline failed", state="error")
            return None

        total_elapsed = time.time() - total_start

        # Display pipeline summary from metadata
        metadata = state.get("metadata", {})

        # Query analysis summary
        query_analysis = state.get("query_analysis")
        if query_analysis:
            intent = query_analysis.citation_intent.value.upper()
            keywords = ", ".join(query_analysis.topic_keywords[:5])
            agent1_time = metadata.get("agent1_latency_s", "?")
            st.write(f"Intent: **{intent}** | Keywords: {keywords} ({agent1_time}s)")

        # Retrieval summary
        retrieval = state.get("retrieval_candidates", [])
        agent2_time = metadata.get("agent2_latency_s", "?")
        st.write(f"Retrieved **{len(retrieval)}** candidates ({agent2_time}s)")

        # Expansion summary
        expanded = state.get("expanded_candidates", [])
        agent3_time = metadata.get("agent3_latency_s", "?")
        st.write(f"Expanded to **{len(expanded)}** candidates ({agent3_time}s)")

        # Reranking summary
        reranked = state.get("reranked_candidates", [])
        grounded = state.get("grounded_candidates", [])
        agent4_time = metadata.get("agent4_latency_s", "?")
        st.write(f"Reranked to **{len(reranked)}**, grounded **{len(grounded)}** ({agent4_time}s)")

        # Synthesis summary
        recommendations = state.get("final_recommendations", [])
        agent5_time = metadata.get("agent5_latency_s", "?")
        st.write(f"Synthesized **{len(recommendations)}** recommendations ({agent5_time}s)")

        status.update(
            label=f"Pipeline complete ({total_elapsed:.1f}s total)",
            state="complete",
        )

        # Store total time
        metadata["total_latency_s"] = round(total_elapsed, 3)

    return state


# ==================== Main App ====================


def main() -> None:
    """Main Streamlit application entry point."""
    st.title("\U0001f4da Citation Recommendation System")
    st.markdown(
        "Paste a paragraph from an AI/ML research paper and get relevant citation "
        "recommendations with justifications."
    )

    # Render sidebar and get options
    options = render_sidebar()

    # Input area
    user_text = st.text_area(
        "Paste your paragraph, claim, or draft section:",
        height=150,
        placeholder=(
            "e.g., Recent advances in transformer architectures have shown that "
            "scaling model parameters significantly improves few-shot learning "
            "capabilities across a wide range of NLP tasks..."
        ),
    )

    # Submit button
    if st.button("Find Citations", type="primary", use_container_width=True):
        # Validate input
        if not user_text or not user_text.strip():
            st.error("Please enter some text to find citations for.")
            return

        if len(user_text.strip()) < 20:
            st.warning(
                "Very short input may produce less accurate results. "
                "Consider pasting a full paragraph for best results."
            )

        # Run pipeline
        state = run_pipeline_with_status(user_text, options)

        if state is None:
            return

        # Store results in session state
        st.session_state["last_state"] = state
        st.session_state["last_options"] = options

    # Display results (either from this run or from session state)
    if "last_state" in st.session_state:
        state = st.session_state["last_state"]
        recommendations = state.get("final_recommendations", [])
        errors = state.get("errors", [])

        # Show errors if any
        if errors:
            with st.expander("Pipeline warnings", expanded=False):
                for err in errors:
                    st.warning(err)

        # Show recommendations
        if recommendations:
            st.subheader(f"Recommendations ({len(recommendations)})")
            for i, rec in enumerate(recommendations):
                render_recommendation(rec, i)
        else:
            st.info(
                "No recommendations met the confidence threshold. "
                "Try lowering the confidence threshold in the sidebar, "
                "or increasing the number of papers to ground."
            )

        # Debug panel
        if options.get("show_debug", False):
            with st.expander("Debug Panel", expanded=False):
                metadata = state.get("metadata", {})

                st.subheader("Timing")
                timing_data = {
                    k: v for k, v in metadata.items() if k.endswith("_s") or k.endswith("_count")
                }
                if timing_data:
                    st.json(timing_data)

                st.subheader("Query Analysis")
                query_analysis = state.get("query_analysis")
                if query_analysis:
                    st.json(
                        {
                            "topic_keywords": query_analysis.topic_keywords,
                            "citation_intent": query_analysis.citation_intent.value,
                            "expanded_query": query_analysis.expanded_query,
                            "confidence": query_analysis.confidence,
                        }
                    )

                st.subheader("Full Pipeline State")
                # Serialize state for display
                display_state = {}
                for key, value in state.items():
                    if isinstance(value, list) and value:
                        if hasattr(value[0], "__dataclass_fields__"):
                            display_state[key] = [
                                {k: getattr(v, k) for k in v.__dataclass_fields__} for v in value
                            ]
                        else:
                            display_state[key] = value
                    elif hasattr(value, "__dataclass_fields__"):
                        display_state[key] = {
                            k: getattr(value, k) for k in value.__dataclass_fields__
                        }
                    else:
                        display_state[key] = value

                st.json(json.loads(json.dumps(display_state, default=str)))


if __name__ == "__main__":
    main()

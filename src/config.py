"""Configuration management using Pydantic Settings.

All application settings are defined in the Settings class and loaded from
environment variables (via .env file or system environment).
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Configuration is loaded from .env file and/or system environment variables.
    Missing required fields will raise validation errors.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==================== OpenAI Configuration ====================
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.0

    # ==================== Qdrant Configuration ====================
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "papers"

    # ==================== Database Configuration ====================
    db_path: str = "data/papers.db"
    data_dir: Path = Path("data")

    # ==================== Embedding Model Configuration ====================
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    # ==================== Reranker Configuration ====================
    reranker_model_name: str = "BAAI/bge-reranker-base"

    # ==================== Retrieval Configuration ====================
    top_k_retrieval: int = 50
    top_k_rerank: int = 10
    top_k_final: int = 5

    # ==================== Citation Expansion Configuration ====================
    citation_expansion_depth: int = 1
    max_expansion_candidates: int = 100

    # ==================== OpenAlex Configuration ====================
    openalex_email: str = "your-email@example.com"
    openalex_rate_limit_delay: float = 0.1

    # ==================== Logging ====================
    log_level: str = "INFO"

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Ensure OpenAI API key is provided."""
        if not v or v == "":
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Set it in your .env file or environment variables."
            )
        return v

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """Ensure database directory exists."""
        db_path = Path(v)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return v


def get_settings() -> Settings:
    """Load and return the application settings.

    Returns:
        Settings instance populated from environment variables.

    Raises:
        ValidationError: If required settings are missing or invalid.
    """
    return Settings()

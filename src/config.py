"""Configuration management using Pydantic Settings.

This module will be expanded in Stage 1 to include all application settings.
For now, it provides a basic stub that loads from .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Stage 0 stub — will be expanded in Stage 1 with all configuration options.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Placeholder settings for Stage 0
    openai_api_key: str = ""
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Load and return the application settings.

    Returns:
        Settings instance populated from environment variables.
    """
    return Settings()

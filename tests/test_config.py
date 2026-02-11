"""Unit tests for configuration loading."""

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


def test_config_loads_defaults(mock_env_vars: dict[str, str]) -> None:
    """Test that config initializes with default values when env vars provided."""
    settings = Settings(openai_api_key="test-key-12345")

    # Check defaults are applied
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.top_k_retrieval == 50
    assert settings.embedding_model_name == "BAAI/bge-base-en-v1.5"


def test_config_loads_from_env(mock_env_vars: dict[str, str]) -> None:
    """Test that config reads from environment variables."""
    settings = get_settings()

    assert settings.openai_api_key == "test-key-12345"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.log_level == "DEBUG"


def test_config_openai_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing OPENAI_API_KEY raises a clear error."""
    # Remove OPENAI_API_KEY from environment and set to empty
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    error_message = str(exc_info.value)
    assert "OPENAI_API_KEY is required" in error_message


def test_config_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that custom environment variables override defaults."""
    monkeypatch.setenv("OPENAI_API_KEY", "custom-key")
    monkeypatch.setenv("TOP_K_RETRIEVAL", "100")
    monkeypatch.setenv("QDRANT_PORT", "7333")

    settings = get_settings()

    assert settings.openai_api_key == "custom-key"
    assert settings.top_k_retrieval == 100
    assert settings.qdrant_port == 7333


def test_config_db_path_creates_directory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that db_path validator creates the parent directory."""
    db_path = tmp_path / "test_data" / "test.db"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DB_PATH", str(db_path))

    settings = get_settings()

    assert settings.db_path == str(db_path)
    assert db_path.parent.exists()

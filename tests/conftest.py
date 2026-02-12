"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file at test startup (before any tests run)
# This ensures OPENAI_API_KEY and other vars are available from .env
load_dotenv()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory.

    Returns:
        Path to tests/test_data/ directory.
    """
    return Path(__file__).parent / "test_data"


@pytest.fixture
def tmp_db_path() -> Generator[Path, None, None]:
    """Create a temporary SQLite database file for testing.

    Yields:
        Path to a temporary database file that will be cleaned up after the test.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up mock environment variables for testing.

    Args:
        monkeypatch: pytest's monkeypatch fixture for modifying environment.

    Returns:
        Dictionary of environment variables that were set.
    """
    env_vars = {
        "OPENAI_API_KEY": "test-key-12345",
        "OPENAI_MODEL": "gpt-4o-mini",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "DB_PATH": "test_papers.db",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment to avoid test pollution.

    This fixture runs automatically for every test.
    """
    # Ensure we're not accidentally using real API keys in tests
    if "OPENAI_API_KEY" in os.environ and not os.environ.get("PYTEST_ALLOW_REAL_API"):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

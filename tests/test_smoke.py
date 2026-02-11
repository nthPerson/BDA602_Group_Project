"""Smoke tests to verify basic project setup."""


def test_src_is_importable() -> None:
    """Verify that the src package can be imported."""
    import src

    assert src is not None
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_subpackages_are_importable() -> None:
    """Verify that all src subpackages can be imported."""
    import src.agents
    import src.data
    import src.evaluation
    import src.indexing
    import src.orchestration

    assert src.data is not None
    assert src.indexing is not None
    assert src.agents is not None
    assert src.orchestration is not None
    assert src.evaluation is not None


def test_config_loads(mock_env_vars: dict[str, str]) -> None:
    """Verify that the config module can be imported and settings load.

    Args:
        mock_env_vars: Fixture providing mock environment variables.
    """
    from src.config import Settings, get_settings

    # Test that Settings class exists
    assert Settings is not None

    # Test that get_settings() works
    settings = get_settings()
    assert settings is not None
    assert settings.openai_api_key == "test-key-12345"
    assert settings.log_level == "DEBUG"


def test_config_loads_with_defaults() -> None:
    """Verify that config loads with default values when env vars are missing."""
    from src.config import Settings

    # Create settings without environment variables
    settings = Settings(openai_api_key="", log_level="INFO")
    assert settings.openai_api_key == ""
    assert settings.log_level == "INFO"

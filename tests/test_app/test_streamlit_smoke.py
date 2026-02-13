"""Smoke tests for the Streamlit application.

These tests verify that the app module imports correctly and that core
functions are callable — without launching the full Streamlit server or
requiring external services.
"""

import importlib


def test_app_imports():
    """Verify that the streamlit_app module imports without errors."""
    mod = importlib.import_module("app.streamlit_app")
    assert mod is not None


def test_main_function_exists():
    """Verify that the main() entry point is defined and callable."""
    from app.streamlit_app import main

    assert callable(main)


def test_render_sidebar_exists():
    """Verify that render_sidebar() is defined and callable."""
    from app.streamlit_app import render_sidebar

    assert callable(render_sidebar)


def test_render_recommendation_exists():
    """Verify that render_recommendation() is defined and callable."""
    from app.streamlit_app import render_recommendation

    assert callable(render_recommendation)


def test_build_app_pipeline_exists():
    """Verify that build_app_pipeline() is defined and callable."""
    from app.streamlit_app import build_app_pipeline

    assert callable(build_app_pipeline)


def test_run_pipeline_with_status_exists():
    """Verify that run_pipeline_with_status() is defined and callable."""
    from app.streamlit_app import run_pipeline_with_status

    assert callable(run_pipeline_with_status)

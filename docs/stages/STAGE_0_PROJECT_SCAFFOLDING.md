# Stage 0 — Project Scaffolding

> **Status:** ✅ Complete
> **Depends on:** Nothing — this is the first stage
> **Estimated effort:** 1–2 hours

---

## What This Stage Builds

This stage creates the project's skeleton — the directory structure, configuration files, dependency management, and development tooling. After this stage, every team member can clone the repo, install dependencies, and run tests.

**No application logic is written in this stage.** It is pure infrastructure.

### Components Created

| File / Directory | Purpose |
|---|---|
| `pyproject.toml` | Declares all Python dependencies, project metadata, and tool configs (ruff, pytest) |
| `docker-compose.yml` | Defines the Qdrant vector store container |
| `.env.example` | Template for environment variables (API keys, config) |
| `.gitignore` | Excludes data files, virtual environments, IDE configs from Git |
| `src/__init__.py` | Makes `src` an importable Python package |
| `src/config.py` | Stub for Pydantic settings (expanded in Stage 1) |
| `src/data/__init__.py` | Empty init for data subpackage |
| `src/indexing/__init__.py` | Empty init for indexing subpackage |
| `src/agents/__init__.py` | Empty init for agents subpackage |
| `src/orchestration/__init__.py` | Empty init for orchestration subpackage |
| `src/evaluation/__init__.py` | Empty init for evaluation subpackage |
| `app/` | Directory for Streamlit app (empty until Stage 9) |
| `scripts/` | Directory for CLI scripts (empty until Stage 2) |
| `tests/conftest.py` | Shared pytest fixtures and configuration |
| `tests/test_smoke.py` | A single test that verifies the project is importable |
| `data/.gitkeep` | Keeps the data directory in Git (contents are gitignored) |
| `notebooks/` | Directory for Jupyter notebooks |
| `docs/SETUP_GUIDE.md` | Team setup instructions (already created) |

---

## Acceptance Criteria

- [ ] Fresh clone → `pip install -e ".[dev]"` succeeds with no errors
- [ ] `docker compose up -d` starts Qdrant without errors
- [ ] `docker compose ps` shows the Qdrant container running
- [ ] `pytest tests/` runs and passes (smoke test)
- [ ] `ruff check src/` reports no violations
- [ ] `python -c "import src; print('OK')"` prints "OK"

---

## How to Test This Stage

### Automated Tests

```bash
# Run the smoke test
pytest tests/test_smoke.py -v
```

**Expected output:**
```
tests/test_smoke.py::test_src_is_importable PASSED
tests/test_smoke.py::test_config_loads PASSED
```

### Manual Verification

```bash
# Verify install
pip install -e ".[dev]" && echo "PASS: install" || echo "FAIL: install"

# Verify Docker
docker compose up -d && docker compose ps

# Verify linting
ruff check src/ && echo "PASS: lint" || echo "FAIL: lint"
```

---

## Key Files to Understand

### `pyproject.toml`

This file defines:
- The project's name, version, and Python requirement
- All runtime dependencies (openai, langgraph, sentence-transformers, etc.)
- Development-only dependencies under `[project.optional-dependencies] dev = [...]`
- Tool configurations for `ruff` and `pytest`

When you run `pip install -e ".[dev]"`, pip reads this file to know what to install.

### `docker-compose.yml`

This file tells Docker to run a Qdrant container. Qdrant is the vector database where we store paper embeddings for similarity search. You start it with `docker compose up -d` and stop it with `docker compose down`.

### `.env` / `.env.example`

`.env.example` is a template checked into Git. `.env` is your local copy with real values (API keys). **Never commit `.env` to Git** — it contains secrets.

### `tests/conftest.py`

This file contains shared pytest configuration:
- Custom markers (`integration`, `slow`)
- Shared fixtures (like a test database or test configuration)

---

## Notes for Teammates

- **If this is your first time working on the project**, follow the [Setup Guide](../SETUP_GUIDE.md) end-to-end.
- **You don't need an OpenAI API key** for this stage. The smoke test doesn't call any external APIs.
- **You DO need Docker** to start Qdrant, but you can skip it for now and just verify the Python install.

---

*Completed by: Claude Code on 2026-02-10*

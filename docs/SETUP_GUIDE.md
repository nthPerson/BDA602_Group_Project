# Project Setup Guide

> **Audience:** This guide is for all team members, including those with minimal programming experience. Follow every step in order. If you get stuck, check the [Troubleshooting](#troubleshooting) section at the bottom.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Install Project Dependencies](#4-install-project-dependencies)
5. [Environment Variables](#5-environment-variables)
6. [Start Docker Services (Qdrant)](#6-start-docker-services-qdrant)
7. [Verify Everything Works](#7-verify-everything-works)
8. [Day-to-Day Development Commands](#8-day-to-day-development-commands)
9. [Project Structure Quick Tour](#9-project-structure-quick-tour)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

You need the following software installed on your machine **before** starting:

### Required

| Software | Minimum Version | How to Check | Install Link |
|---|---|---|---|
| **Python** | 3.11+ | `python3 --version` | [python.org](https://www.python.org/downloads/) |
| **Git** | 2.30+ | `git --version` | [git-scm.com](https://git-scm.com/downloads) |
| **Docker Desktop** | 24.0+ | `docker --version` | [docker.com](https://docs.docker.com/get-docker/) |
| **Docker Compose** | 2.20+ | `docker compose version` | Included with Docker Desktop |

### Recommended (but optional)

| Software | Purpose |
|---|---|
| **VS Code** | Recommended code editor |
| **VS Code Python extension** | Python language support, debugging |
| **A terminal** | Any terminal works — VS Code's built-in terminal, Terminal.app, iTerm2, Windows Terminal, etc. |

### Operating System Notes

- **macOS / Linux:** All commands below work as-is.
- **Windows:** Use **Git Bash**, **WSL2**, or **PowerShell**. If using PowerShell, replace `source .venv/bin/activate` with `.venv\Scripts\Activate.ps1`.

---

## 2. Clone the Repository

Open your terminal and run:

```bash
# Clone the repository
git clone https://github.com/nthPerson/BDA602_Group_Project.git

# Enter the project directory
cd BDA602_Group_Project
```

If you've already cloned it before and want to get the latest changes:

```bash
cd BDA602_Group_Project
git pull origin main
```

---

## 3. Python Environment Setup

We use a **virtual environment** to keep this project's packages isolated from your system Python. This prevents conflicts with other projects.

```bash
# Create a virtual environment (one-time setup)
python3 -m venv .venv

# Activate it (you must do this every time you open a new terminal)
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
# .venv\Scripts\activate.bat  # Windows Command Prompt
```

**How do I know it's activated?** Your terminal prompt will show `(.venv)` at the beginning:
```
(.venv) $ 
```

> **Important:** Always make sure the virtual environment is activated before running any Python command. If you see errors like `ModuleNotFoundError`, the first thing to check is whether `.venv` is activated.

---

## 4. Install Project Dependencies

With the virtual environment activated:

```bash
# Install the project and all dependencies (including dev tools)
pip install -e ".[dev]"
```

**What does this do?**
- `-e` = "editable" install. Any changes you make to the source code take effect immediately without reinstalling.
- `.[dev]` = install the project itself, plus development dependencies (pytest, ruff, etc.).

This may take a few minutes the first time (it downloads ML models and libraries).

### Verify the install worked:

```bash
python -c "import src; print('Import successful')"
```

---

## 5. Environment Variables

The project needs certain configuration values (like API keys) that should NOT be committed to Git. We use a `.env` file for this.

```bash
# Copy the template
cp .env.example .env

# Open .env in your editor and fill in the required values
```

**Required variables:**

| Variable | Where to Get It | Example |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `sk-proj-abc123...` |

**Optional variables (have sensible defaults):**

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `DB_PATH` | `data/papers.db` | Path to SQLite database file |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence transformer model name |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for agents |

> **Note on the OpenAI API key:** You need an OpenAI account with API access. The project uses `gpt-4o-mini`, which is very inexpensive (~$0.15 per million tokens). A full evaluation run costs less than $1. If you don't have a key yet, ask the project lead — most tests can run without it (using mocks).

---

## 6. Start Docker Services (Qdrant)

The project uses **Qdrant** as a vector store, running in a Docker container.

```bash
# Start Qdrant in the background
docker compose up -d
```

**Verify it's running:**

```bash
# Check container status
docker compose ps

# You should see something like:
# NAME      IMAGE                    STATUS
# qdrant    qdrant/qdrant:v1.12.5    Up

# Or test the API directly:
curl http://localhost:6333/healthz
# Should return: {"title":"qdrant - vectorass engine","version":"..."}
```

**To stop Qdrant:**
```bash
docker compose down
```

**To stop AND delete all stored data:**
```bash
docker compose down -v
```

> **Note:** You only need Qdrant running when working on Stages 3+ (anything involving vector search). Stages 0–1 work without it.

---

## 7. Verify Everything Works

Run the full check:

```bash
# 1. Check Python environment
python3 --version              # Should be 3.11+

# 2. Check virtual env is active
which python                   # Should point to .venv/bin/python

# 3. Check the project is installed
python -c "import src; print('OK')"

# 4. Check Docker is running Qdrant
docker compose ps              # Should show qdrant container running

# 5. Run the test suite
pytest tests/ -v               # Should pass all available tests

# 6. Check code formatting
ruff check src/                # Should show no issues
```

If all 6 steps pass, **you're ready to develop!**

---

## 8. Day-to-Day Development Commands

Here are the commands you'll use most often:

### Starting a work session

```bash
cd BDA602_Group_Project
source .venv/bin/activate      # Activate virtual environment
docker compose up -d           # Start Qdrant (if needed)
```

### Running tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_data/ -v          # Data layer tests
pytest tests/test_agents/ -v        # Agent tests
pytest tests/test_evaluation/ -v    # Evaluation tests

# Run only fast unit tests (no external services needed)
pytest tests/ -m "not integration" -v

# Run a single test file
pytest tests/test_data/test_db.py -v

# Run a single test function
pytest tests/test_data/test_db.py::test_insert_and_query_paper -v
```

### Code formatting

```bash
# Check for style issues
ruff check src/ tests/

# Auto-fix style issues
ruff check src/ tests/ --fix

# Format code
ruff format src/ tests/
```

### Git workflow

```bash
# Check what branch you're on
git branch

# Create a new branch for your work
git checkout -b feature/my-feature-name

# Stage and commit your changes
git add -A
git commit -m "Brief description of what you changed"

# Push your branch
git push origin feature/my-feature-name

# Then create a Pull Request on GitHub
```

---

## 9. Project Structure Quick Tour

```
BDA602_Group_Project/
│
├── SYSTEM_OVERVIEW.md        ← Technical design document (read this first)
├── DEVELOPMENT_ROADMAP.md    ← Development plan with all stages
├── docs/
│   ├── SETUP_GUIDE.md        ← You are here
│   └── stages/               ← One document per development stage
│       ├── STAGE_0_...md
│       ├── STAGE_1_...md
│       └── ...
│
├── src/                      ← All source code lives here
│   ├── data/                 ← Data models, database, OpenAlex client
│   ├── indexing/             ← Embedding and Qdrant operations
│   ├── agents/               ← The 5 AI agents
│   ├── orchestration/        ← LangGraph pipeline wiring
│   └── evaluation/           ← Metrics and evaluation framework
│
├── app/                      ← Streamlit web UI
├── scripts/                  ← Runnable scripts (build corpus, index, evaluate)
├── tests/                    ← All test files (mirrors src/ structure)
├── data/                     ← Local data files (gitignored)
├── notebooks/                ← Jupyter notebooks for exploration
│
├── pyproject.toml            ← Project dependencies and metadata
├── docker-compose.yml        ← Qdrant container configuration
├── .env.example              ← Template for environment variables
└── .gitignore                ← Files excluded from Git
```

### Where do I look?

| I want to understand... | Look at... |
|---|---|
| The overall system design | `SYSTEM_OVERVIEW.md` |
| What was built in each stage | `docs/stages/STAGE_N_*.md` |
| How a specific agent works | `src/agents/<agent_name>.py` |
| How the pipeline is wired | `src/orchestration/graph.py` |
| How evaluation works | `src/evaluation/` |
| Test examples | `tests/` (mirrors `src/` structure) |

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Cause:** The virtual environment isn't activated, or the project isn't installed.

**Fix:**
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

### "docker: command not found"

**Cause:** Docker isn't installed or isn't in your PATH.

**Fix:** Install Docker Desktop from [docker.com](https://docs.docker.com/get-docker/) and restart your terminal.

### "Connection refused" when connecting to Qdrant

**Cause:** The Qdrant container isn't running.

**Fix:**
```bash
docker compose up -d
docker compose ps    # Verify it says "Up"
```

### "openai.AuthenticationError" or "Invalid API key"

**Cause:** The OpenAI API key isn't set or is invalid.

**Fix:**
1. Make sure `.env` exists and contains `OPENAI_API_KEY=sk-...`
2. Make sure the key is valid (check [platform.openai.com](https://platform.openai.com))
3. Make sure you haven't accidentally committed `.env` to Git

### Tests fail with "connection refused" or "timeout"

**Cause:** Some tests require Qdrant to be running (integration tests).

**Fix:** Either start Qdrant (`docker compose up -d`) or run only unit tests:
```bash
pytest tests/ -m "not integration" -v
```

### "Permission denied" errors on Linux/macOS

**Fix:**
```bash
chmod +x scripts/*.py
```

### Python version too old

**Cause:** System Python is <3.11.

**Fix:** Install Python 3.11+ from [python.org](https://www.python.org/downloads/) or use `pyenv`:
```bash
pyenv install 3.11.8
pyenv local 3.11.8
```

### I pulled new changes and things broke

**Fix:** Reinstall dependencies (new packages might have been added):
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

---

*Last updated: 2026-02-10. If you encounter an issue not covered here, let the team know and we'll add it.*

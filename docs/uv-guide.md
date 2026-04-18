# uv — The Modern Python Package Manager

## The Before / Now Story

### Before uv (the dark ages)

Setting up a Python project used to require juggling 5+ separate tools:

```bash
# The old painful workflow
pyenv install 3.12.3          # Step 1: install Python (pyenv)
pyenv local 3.12.3            # Step 2: pin version
python -m venv .venv           # Step 3: create virtualenv (venv/virtualenv)
source .venv/bin/activate      # Step 4: activate it
pip install -r requirements.txt # Step 5: install deps (pip)
pip freeze > requirements.txt  # Step 6: lock deps (pip-tools/pip freeze)
pipx install ruff              # Step 7: install CLI tools (pipx)
```

Problems with this approach:
- **Slow**: pip resolves dependencies serially in Python. Large installs take minutes.
- **Fragmented**: pyenv, venv, pip, pip-tools, pipx — five tools, five mental models.
- **No lockfile**: `pip freeze` is lossy. It captures what's installed, not what was requested.
- **No cross-platform guarantee**: requirements.txt on Mac may fail on Linux.
- **Reproducibility**: "works on my machine" is the default, not the exception.

### Now (uv era — 2024+)

uv replaces ALL of the above with a single binary written in Rust:

```bash
# The new workflow — one tool does everything
uv init smartdesk              # Creates project + pyproject.toml + .venv
cd smartdesk
uv add fastapi pydantic        # Adds deps, resolves, installs, updates lockfile
uv add pytest ruff --dev       # Dev dependencies in a separate group
uv run python main.py          # Runs inside the virtualenv automatically
uv run pytest                  # No manual activation needed
```

Built by Astral (the Ruff team). Written in Rust. 10–100x faster than pip.
One binary. No Python dependency to bootstrap. Cross-platform lockfile.

### Who uses uv?

Any serious Python shop in 2026. Vercel made uv the default for Python builds.
Companies migrating from Poetry/pip/Conda to uv for speed and simplicity.
AI/ML teams especially benefit — installing PyTorch + deps goes from 3 min to 15 sec.


---

## Part 1: Installation

### macOS / Linux (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This installs a standalone binary — no Python needed.

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Via pip (if you must)

```bash
pip install uv
```

### Via Homebrew (macOS)

```bash
brew install uv
```

### Verify installation

```bash
uv --version
# uv 0.11.x (your-platform)
```

### Self-update

```bash
uv self update
```


---

## Part 2: Python Version Management (replaces pyenv)

uv manages Python installations directly. No more pyenv.

### Install Python versions

```bash
uv python install 3.12        # Install latest 3.12.x
uv python install 3.11 3.12   # Install multiple versions
uv python install 3.12.3      # Install exact patch version
```

### List installed versions

```bash
uv python list                # Show all available + installed
uv python list --only-installed  # Only what's on your machine
```

### Pin version for a project

```bash
cd my-project
uv python pin 3.12             # Creates .python-version file
```

This file goes into git. Any collaborator running `uv sync` gets the same Python.

### Where are Pythons stored?

```bash
# uv stores managed Pythons in its own directory:
# Linux/Mac: ~/.local/share/uv/python/
# Windows:   %APPDATA%\uv\python\
```

No conflicts with system Python. No shims. Clean.


---

## Part 3: Project Setup (replaces poetry init / pip + venv)

### Create a new project

```bash
uv init my-project             # Library-style project
cd my-project

# What you get:
# my-project/
# ├── .python-version           # Pinned Python version
# ├── .venv/                    # Virtual environment (auto-created)
# ├── pyproject.toml            # Project metadata + dependencies
# ├── README.md
# └── src/
#     └── my_project/
#         └── __init__.py
```

### Create with specific Python version

```bash
uv init my-project --python 3.12
```

### Create an application (not a library)

```bash
uv init my-app --app           # No src/ layout, has hello.py
```

### The pyproject.toml uv generates

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

This is standard PEP 621. Works with pip, Poetry, everything. Not vendor-locked.


---

## Part 4: Dependency Management (replaces pip install + pip freeze)

### Add dependencies

```bash
uv add requests                # Latest version
uv add "requests>=2.31"        # With version constraint
uv add fastapi uvicorn         # Multiple at once
uv add "numpy>=1.26,<2.0"     # Range constraint
```

What happens under the hood:
1. Updates `pyproject.toml` [project.dependencies]
2. Resolves the full dependency tree
3. Updates `uv.lock` (cross-platform lockfile)
4. Installs into `.venv/`

All in under 1 second for most packages.

### Add dev dependencies

```bash
uv add pytest ruff mypy --dev
# Goes into [dependency-groups] dev = [...]
```

### Add to custom groups

```bash
uv add sphinx --group docs
uv add locust --group loadtest
```

Your pyproject.toml:
```toml
[project]
dependencies = [
    "fastapi>=0.110",
    "pydantic>=2.6",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.3",
    "mypy>=1.9",
]
docs = [
    "sphinx>=7.0",
]
```

### Remove dependencies

```bash
uv remove requests
uv remove pytest --dev
```

### Sync environment (install everything from lockfile)

```bash
uv sync                        # Install all deps from uv.lock
uv sync --frozen               # Don't update lockfile, just install
uv sync --no-dev               # Production: skip dev dependencies
```

### Upgrade dependencies

```bash
uv lock --upgrade              # Upgrade ALL to latest compatible
uv lock --upgrade-package requests  # Upgrade just one
```

### View dependency tree

```bash
uv tree                        # Full dependency tree
uv tree --depth 1              # Just direct deps
```


---

## Part 5: Running Code (replaces `source .venv/bin/activate`)

The killer feature: **you never need to activate the virtualenv manually**.

### Run scripts

```bash
uv run python main.py          # Runs with project's .venv Python
uv run pytest                  # Runs pytest from dev deps
uv run ruff check src/         # Runs any installed tool
```

`uv run` auto-creates `.venv` if missing, syncs deps if needed, then runs.

### Run with extra packages (not in project)

```bash
uv run --with httpx script.py  # Temporarily add httpx for this run
```

### Interactive Python shell

```bash
uv run python                  # Opens Python REPL with project deps
uv run ipython                 # If ipython is installed
```

### Run a module

```bash
uv run python -m pytest tests/ -v
uv run python -m smartdesk.scripts.demo_phase1
```


---

## Part 6: The Lockfile (uv.lock)

### What is uv.lock?

A cross-platform, deterministic lockfile. Unlike `requirements.txt`:
- Records exact versions for ALL platforms (Linux, Mac, Windows)
- Records hashes for integrity verification
- Tracks the full dependency graph, not just leaf packages
- Human-readable TOML format

### Rules

```
DO commit uv.lock to git        → reproducible builds for everyone
DO NOT edit uv.lock manually     → it's auto-generated
DO run `uv lock` after editing pyproject.toml manually
DO run `uv sync` on a fresh clone to install from lockfile
```

### Generate requirements.txt (for legacy systems / Docker)

```bash
uv export --format requirements-txt > requirements.txt
uv export --format requirements-txt --no-dev > requirements-prod.txt
```


---

## Part 7: CLI Tools (replaces pipx)

### Run tools without installing

```bash
uvx ruff check .               # Runs ruff in ephemeral env
uvx black --check .            # Runs black without installing
uvx cowsay "hello uv"          # Any PyPI tool
```

`uvx` is an alias for `uv tool run`.

### Install tools globally

```bash
uv tool install ruff
uv tool install pre-commit
uv tool install httpie

# Now available everywhere:
ruff check .
pre-commit install
http GET https://api.example.com
```

### List installed tools

```bash
uv tool list
```

### Upgrade tools

```bash
uv tool upgrade ruff
uv tool upgrade --all
```


---

## Part 8: Inline Script Dependencies (PEP 723)

For one-off scripts that need specific packages:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests>=2.31",
#     "rich>=13.0",
# ]
# ///

import requests
from rich import print

response = requests.get("https://api.github.com")
print(response.json())
```

Run it:
```bash
uv run script.py
# uv reads the inline metadata, creates a temp env, installs deps, runs.
# No project setup needed. Share the file — anyone with uv can run it.
```

Add deps to a script:
```bash
uv add --script script.py pandas
# Adds pandas to the inline metadata automatically
```


---

## Part 9: Workspaces (Monorepo Support)

For projects with multiple packages (like microservices):

```
my-org/
├── pyproject.toml              # Root workspace config
├── uv.lock                    # Shared lockfile
├── packages/
│   ├── core/
│   │   └── pyproject.toml     # Shared library
│   ├── api-service/
│   │   └── pyproject.toml     # FastAPI service
│   └── ml-pipeline/
│       └── pyproject.toml     # ML training pipeline
```

Root pyproject.toml:
```toml
[tool.uv.workspace]
members = ["packages/*"]
```

Now all packages share one lockfile and can depend on each other:
```bash
cd packages/api-service
uv add core --workspace        # Depend on sibling package
```


---

## Part 10: Docker Integration (Enterprise Essential)

### Production Dockerfile with uv

```dockerfile
# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

# Install uv (standalone binary, no Python needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (Docker layer caching!)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps in production)
RUN uv sync --frozen --no-dev --no-editable

# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Use the virtualenv Python
ENV PATH="/app/.venv/bin:$PATH"

# Run
CMD ["python", "-m", "uvicorn", "smartdesk.api.main:app", "--host", "0.0.0.0"]
```

Key points:
- Multi-stage build keeps image small (no uv binary in final image)
- `--frozen` means "use lockfile exactly, don't resolve"
- pyproject.toml + uv.lock copied first → Docker caches this layer
- Code changes don't re-install dependencies


### docker-compose.yml

```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```


---

## Part 11: CI/CD with GitHub Actions

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true           # Cache .uv for speed

      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --frozen

      - name: Lint
        run: uv run ruff check src/ tests/

      - name: Type check
        run: uv run mypy src/

      - name: Test
        run: uv run pytest tests/ -v --cov=src/ --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

Key: `astral-sh/setup-uv@v5` is the official GitHub Action. `enable-cache: true` caches the uv store across runs.


---

## Part 12: Advanced uv Features

### Override dependency versions

```toml
# pyproject.toml
[tool.uv]
override-dependencies = [
    "numpy==1.26.4",             # Force specific version
]
```

### Platform-specific dependencies

```toml
[project]
dependencies = [
    "torch==2.2.0; sys_platform == 'linux'",
    "torch==2.2.0; sys_platform == 'darwin'",
]
```

### Resolution strategy

```toml
[tool.uv]
resolution = "lowest-direct"    # Use lowest compatible versions
# Options: "highest" (default), "lowest", "lowest-direct"
```

### Constraint files

```bash
uv add --constraint constraints.txt
```

### Build and publish

```bash
uv build                       # Creates wheel + sdist in dist/
uv publish                     # Publishes to PyPI
uv publish --token $PYPI_TOKEN # With auth token
```

### Cache management

```bash
uv cache dir                   # Show cache location
uv cache clean                 # Clear all cached packages
uv cache prune                 # Remove unused entries
```


---

## Part 13: Migration Cheat Sheet

### From pip

| pip command | uv equivalent |
|---|---|
| `pip install package` | `uv add package` |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `pip freeze > requirements.txt` | `uv export --format requirements-txt` |
| `pip uninstall package` | `uv remove package` |
| `pip list` | `uv pip list` |
| `pip show package` | `uv pip show package` |
| `python -m venv .venv` | `uv venv` (or auto-created) |
| `pip install -e .` | `uv sync` (editable by default) |

### From Poetry

| Poetry command | uv equivalent |
|---|---|
| `poetry init` | `uv init` |
| `poetry add package` | `uv add package` |
| `poetry add --group dev pytest` | `uv add pytest --dev` |
| `poetry install` | `uv sync` |
| `poetry lock` | `uv lock` |
| `poetry run python script.py` | `uv run python script.py` |
| `poetry shell` | `uv run python` (no shell needed) |
| `poetry build` | `uv build` |
| `poetry publish` | `uv publish` |

### From Conda

| Conda command | uv equivalent |
|---|---|
| `conda create -n myenv python=3.12` | `uv init --python 3.12` |
| `conda activate myenv` | Not needed (uv run handles it) |
| `conda install numpy` | `uv add numpy` |
| `conda list` | `uv pip list` |
| `conda env export` | `uv export` |

Note: uv cannot manage non-Python packages (C libraries, CUDA).
For those, keep using conda alongside uv, or use system packages.


---

## Part 14: Troubleshooting

### Common issues

**"No Python found"**
```bash
uv python install 3.12         # Let uv install it
```

**"Resolution failed"**
```bash
uv lock --verbose               # See what's conflicting
uv add package --resolution lowest  # Try lowest compatible
```

**"Cached package is corrupted"**
```bash
uv cache clean
uv sync --reinstall
```

**VSCode not finding the virtualenv**
Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

**PyCharm not finding the virtualenv**
Settings → Project → Python Interpreter → Add Local → Existing → Select `.venv/bin/python`


---

## Quick Reference Card

```bash
# Project lifecycle
uv init project-name           # Create project
uv add package                 # Add dependency
uv add package --dev           # Add dev dependency
uv remove package              # Remove dependency
uv sync                        # Install from lockfile
uv lock                        # Update lockfile
uv run command                 # Run in project env

# Python management
uv python install 3.12         # Install Python
uv python pin 3.12             # Pin for project
uv python list                 # List versions

# Tools
uvx tool-name                  # Run tool once
uv tool install tool-name      # Install globally

# Maintenance
uv lock --upgrade              # Upgrade all deps
uv tree                        # Show dep tree
uv cache clean                 # Clear cache
uv self update                 # Update uv itself
```

# ADR-001: Use uv for Python package management

## Status

Accepted

## Date

2026-04-17

## Context

SmartDesk needs a dependency management tool that handles:
- Virtual environment creation and management
- Dependency resolution and lockfile generation
- Python version management
- Fast CI/CD builds in Docker
- Developer experience (one tool, not five)

Options considered: pip + venv, Poetry, Conda, uv.

## Decision

We chose uv (by Astral, the Ruff team) as our sole package management tool.

## Rationale

- **Speed**: 10–100x faster than pip. Docker builds go from minutes to seconds.
- **Single binary**: Replaces pyenv + venv + pip + pip-tools + pipx.
- **Cross-platform lockfile**: uv.lock works on Linux, Mac, Windows — critical for team reproducibility.
- **Standards-compliant**: Uses PEP 621 pyproject.toml. Not vendor-locked. Compatible with pip if we ever need to switch.
- **Docker-friendly**: Standalone binary, no Python needed to bootstrap. Multi-stage builds are trivial.
- **Industry momentum**: Adopted by Vercel, growing rapidly in the Python ecosystem.

## Trade-offs

- uv is newer (v0.x) — less battle-tested than pip or Poetry in very large orgs.
- Cannot manage non-Python dependencies (C libraries, CUDA). For those, we'd layer conda or system packages.
- Team members need to learn one new tool (but it's simpler than what it replaces).

## Consequences

- All dependency commands go through uv (no pip install anywhere).
- Makefile wraps uv commands for discoverability.
- CI uses astral-sh/setup-uv GitHub Action.
- Docker uses uv sync --frozen for deterministic builds.
- .python-version file pins Python version for uv and editors.

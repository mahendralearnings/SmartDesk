# ============================================================================
# SmartDesk Production Dockerfile — powered by uv
#
# Build:  docker build -f docker/Dockerfile -t smartdesk:latest .
# Run:    docker run -p 8000:8000 --env-file .env smartdesk:latest
# ============================================================================

# ── Stage 1: Builder ───────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv — standalone binary, no Python dependency
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set uv environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy ONLY dependency files first (Docker layer caching)
# If pyproject.toml or uv.lock hasn't changed, this layer is cached
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev, no editable)
RUN uv sync --frozen --no-dev --no-editable

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home app

WORKDIR /app

# Copy the virtualenv from builder (contains all installed packages)
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy application source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/ ./configs/

# Put the virtualenv's Python first in PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "smartdesk.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

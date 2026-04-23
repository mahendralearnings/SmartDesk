# src/smartdesk/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

from smartdesk.api.routers import chat as chat_router
from smartdesk.api.services.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lifespan — startup & shutdown (modern FastAPI pattern)
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("SmartDesk API starting up...")
    app.state.ollama_client = OllamaClient()
    logger.info("OllamaClient initialised")
    yield
    # SHUTDOWN
    await app.state.ollama_client.close()
    logger.info("OllamaClient closed — goodbye")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartDesk API",
    description="Enterprise Document Intelligence — Chat Endpoint",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS (allow all origins for local dev — lock this down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Latency logging middleware ───
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    ms = (time.monotonic() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.1f}ms)")
    return response


# ─── Wire the dependency override ───
# Makes `Depends(get_ollama_client)` return our shared instance
from smartdesk.api.routers.chat import get_ollama_client

def _get_client() -> OllamaClient:
    return app.state.ollama_client

app.dependency_overrides[get_ollama_client] = _get_client

# ─── Include router ───
app.include_router(chat_router.router)
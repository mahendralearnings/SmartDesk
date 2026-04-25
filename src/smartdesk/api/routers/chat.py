# src/smartdesk/api/routers/chat.py
from fastapi import APIRouter, HTTPException, Depends
from smartdesk.api.models.chat_models import ChatRequest, ChatResponse, HealthResponse
from smartdesk.api.services.chat_service import ChatService
from smartdesk.api.services.ollama_client import OllamaClient
import logging

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Dependency injection — FastAPI creates these per-request
# ─────────────────────────────────────────────────────────────

def get_ollama_client() -> OllamaClient:
    """Returns the shared client (set by app startup)."""
    # We'll override this in main.py using app.state
    pass  # replaced below

def get_chat_service(client: OllamaClient = Depends(get_ollama_client)) -> ChatService:
    return ChatService(client)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    """
    Main chat endpoint.
    - Validates request (Pydantic, automatic 422 on bad input)
    - Calls ChatService
    - Returns ChatResponse JSON
    """
    logger.info(f"Chat request: '{request.message[:60]}...' | history={len(request.history)} turns")
    try:
        return await service.chat(request)
    except RuntimeError as e:
        # OllamaClient raises RuntimeError for connection/timeout
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in chat endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


# @router.get("/health", response_model=HealthResponse)
# async def health_check(client: OllamaClient = Depends(get_ollama_client)):
#     """
#     Liveness + readiness probe.
#     Returns 200 if FastAPI is up, reports whether Ollama is reachable.
#     """
#     healthy = await client.is_healthy()
#     return HealthResponse(
#         status="ok",
#         ollama_reachable=healthy,
#         model=client.model,
#     )
@router.get("/health", response_model=HealthResponse)
async def health_check():
    import anthropic
    import os

    # Check Claude API
    claude_ok = bool(os.getenv("ANTHROPIC_API_KEY"))

    return HealthResponse(
        status="ok",
        ollama_reachable=claude_ok,
        model="claude-sonnet-4-5",
    )
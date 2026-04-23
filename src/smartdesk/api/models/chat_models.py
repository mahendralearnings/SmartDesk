# src/smartdesk/api/models/chat_models.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ConversationTurn(BaseModel):
    """One message in history — role + content."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """What the client sends us."""
    message: str = Field(..., min_length=1, max_length=4000,
                         description="User's current message")
    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=20,  # cap at 20 turns so we don't blow the context window
        description="Previous turns for multi-turn chat"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional system prompt override"
    )
    stream: bool = Field(default=False, description="Enable streaming response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """What we send back."""
    reply: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
# src/smartdesk/api/services/chat_service.py
from smartdesk.api.models.chat_models import ChatRequest, ChatResponse, ConversationTurn
from smartdesk.api.services.ollama_client import OllamaClient

DEFAULT_SYSTEM_PROMPT = """You are SmartDesk, an enterprise document intelligence assistant.
You help users analyse contracts, invoices, and reports.
Be concise, factual, and cite specific details when possible."""


class ChatService:
    """
    Owns the business logic:
      - builds the messages list from history + current message
      - applies the system prompt
      - calls OllamaClient
      - returns a typed ChatResponse
    """

    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client

    def _build_messages(self, request: ChatRequest) -> list[dict]:
        """
        Construct the messages list in Ollama/OpenAI format:
        [ {role: system, content: ...},
          {role: user, content: ...},
          {role: assistant, content: ...},
          ...
          {role: user, content: <current message>} ]
        """
        messages = []

        # 1. System prompt (custom or default)
        system = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system})

        # 2. History — already validated by Pydantic (max 20 turns)
        for turn in request.history:
            messages.append({"role": turn.role, "content": turn.content})

        # 3. Current user message
        messages.append({"role": "user", "content": request.message})

        return messages

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = self._build_messages(request)

        result = await self.ollama.chat(
            messages=messages,
            temperature=request.temperature,
        )

        return ChatResponse(
            reply=result["reply"],
            model=result["model"],
            tokens_used=result.get("tokens_used"),
            latency_ms=result["latency_ms"],
        )
# src/smartdesk/api/services/ollama_client.py
import httpx
import time
import logging

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "tinyllama"


class OllamaClient:
    """
    Async client for local Ollama.
    Uses httpx — production-grade async HTTP, not requests.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.model = model
        # async client with timeout + connection limits
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> dict:
        """
        Send messages list to Ollama /api/chat.
        Returns full response dict.
        """
        start = time.monotonic()

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()  # 4xx/5xx → exception
            data = response.json()
            latency_ms = (time.monotonic() - start) * 1000
            logger.info(f"Ollama responded in {latency_ms:.1f}ms")
            return {
                "reply": data["message"]["content"],
                "model": data.get("model", self.model),
                "tokens_used": data.get("eval_count"),
                "latency_ms": round(latency_ms, 2),
            }

        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot reach Ollama. Is it running? → `ollama serve`"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama timed out after {self.client.timeout.read}s. "
                "Try a shorter message or increase timeout."
            )

    async def is_healthy(self) -> bool:
        """Quick ping to check if Ollama is alive."""
        try:
            r = await self.client.get("/")
            return r.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Release connection pool — call on app shutdown."""
        await self.client.aclose()
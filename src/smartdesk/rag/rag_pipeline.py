# src/smartdesk/rag/rag_pipeline.py
"""
Orchestrates everything:
  Ingestion: doc → chunks → embeddings → ChromaDB
  Query: question → retrieve → prompt → LLM → answer
"""
from smartdesk.rag.chunker import DocumentChunker
from smartdesk.rag.vector_store import SmartDeskVectorStore
from smartdesk.rag.retriever import RAGRetriever
import httpx
import json
import logging

logger = logging.getLogger(__name__)

# RAG_SYSTEM_PROMPT = """You are SmartDesk, an enterprise document assistant.
# Answer questions using ONLY the provided document excerpts below.
# If the answer is not in the excerpts, respond with: NOT FOUND IN DOCUMENT.
# Never guess. Never use outside knowledge. Cite the source number."""

RAG_SYSTEM_PROMPT = """You are a document extraction assistant.
Answer using ONLY information from the document excerpt provided.
Do NOT use your own knowledge. Do NOT add context not in the excerpt.
If the answer is not in the excerpt say: NOT FOUND IN DOCUMENT."""

class RAGPipeline:
    def __init__(self, persist_dir: str = "data/chroma"):
        self.chunker = DocumentChunker(chunk_size=300, overlap=50)
        self.store = SmartDeskVectorStore(persist_dir=persist_dir)
        self.retriever = RAGRetriever(self.store, use_reranker=True)

    # ── INGESTION ─────────────────────────────────────────────

    def ingest(self, doc_id: str, text: str) -> int:
        """Chunk and index one document. Returns chunk count."""
        chunks = self.chunker.chunk(text, doc_id)
        self.store.add_chunks(chunks)
        logger.info(f"Ingested '{doc_id}' → {len(chunks)} chunks")
        return len(chunks)

    def ingest_many(self, docs: dict[str, str]) -> dict[str, int]:
        """Ingest multiple documents. docs = {doc_id: text}"""
        return {doc_id: self.ingest(doc_id, text) for doc_id, text in docs.items()}

    # ── RETRIEVAL + GENERATION ────────────────────────────────

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Full RAG query:
          1. Retrieve relevant chunks
          2. Build grounded prompt
          3. Call Ollama
          4. Return answer + sources
        """
        # 1. Retrieve
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "NOT FOUND IN DOCUMENT",
                "sources": [],
                "chunks_used": 0,
            }

        # 2. Build prompt
        context = self.retriever.format_context(chunks)
        
        user_message = f"""Excerpt: {context}
               Answer this question using only the excerpt above: {question}
        Answer:"""


        # 3. Call Ollama
        answer = self._call_ollama(user_message)

        return {
            "answer": answer,
            "sources": [c["doc_id"] for c in chunks],
            "chunks_used": len(chunks),
            "top_chunk_score": chunks[0].get("rerank_score", chunks[0]["score"]),
        }

    # def _call_ollama(self, user_message: str) -> str:
    #     payload = {
    #         "model": "tinyllama",
    #         "messages": [
    #             {"role": "system", "content": RAG_SYSTEM_PROMPT},
    #             {"role": "user", "content": user_message},
    #         ],
    #         "stream": False,
    #         "options": {"temperature": 0.0},  # low temp for factual extraction
    #     }
    #     try:
    #         r = httpx.post(
    #             "http://localhost:11434/api/chat",
    #             json=payload,
    #             timeout=60.0,
    #         )
    #         r.raise_for_status()
    #         return r.json()["message"]["content"]
    #     except Exception as e:
    #         raise RuntimeError(f"Ollama call failed: {e}")


    def _call_ollama(self, user_message: str) -> str:
        import anthropic
        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-haiku-4-5",   # cheapest Claude — still 10x better than tinyllama
            max_tokens=512,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()
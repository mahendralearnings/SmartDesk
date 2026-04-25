# src/smartdesk/rag/retriever.py
"""
Two-stage retrieval:
  Stage 1 — Vector search: fast, approximate (top-5)
  Stage 2 — Reranker: slow, precise (top-3)

Why two stages?
  Vector search alone sometimes returns topically related chunks
  that don't actually answer the question. The cross-encoder
  reranker reads query + chunk together and scores relevance
  directly — much more accurate but too slow to run on 10,000 chunks.
  Solution: use vector search to shortlist, reranker to re-order.
"""
from sentence_transformers import CrossEncoder
from smartdesk.rag.vector_store import SmartDeskVectorStore
import logging

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RAGRetriever:
    def __init__(self, vector_store: SmartDeskVectorStore, use_reranker: bool = True):
        self.store = vector_store
        self.use_reranker = use_reranker
        if use_reranker:
            logger.info("Loading cross-encoder reranker...")
            self.reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Full two-stage retrieval.
        Returns top_k chunks ranked by relevance to query.
        """
        # Stage 1: vector search — get 5 candidates
        candidates = self.store.search(query, top_k=5)

        if not candidates:
            return []

        if not self.use_reranker:
            return candidates[:top_k]

        # Stage 2: rerank — cross-encoder scores each (query, chunk) pair
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        # Attach reranker scores and sort
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Retrieved {len(reranked[:top_k])} chunks for: '{query[:50]}'"
        )
        return reranked[:top_k]

    def format_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a context block for the LLM prompt.
        Each chunk is clearly delimited — helps LLM attribute answers.
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i} — {chunk['doc_id']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)
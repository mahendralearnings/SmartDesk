# src/smartdesk/rag/embedder.py
"""
Thin wrapper around Phase 5 DocumentEncoder.
Key rule: SAME model for ingestion AND retrieval.
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class RAGEmbedder:
    """
    Wraps sentence-transformers for RAG.
    Uses mean pooling + L2 normalisation (Phase 5 decisions).
    """
    MODEL_NAME = "all-MiniLM-L6-v2"
    # Why this model?
    #   384-dim vectors (small = fast FAISS/Chroma search)
    #   Trained specifically for semantic similarity
    #   6x faster than BERT-base at inference

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.dimension = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.
        Returns: float32 array of shape (N, 384), L2-normalised.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,          # GPU processes 32 at once
            normalize_embeddings=True,  # L2 norm → cosine = dot product
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding — convenience wrapper."""
        return self.embed([query])[0]
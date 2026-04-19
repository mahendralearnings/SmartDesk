"""Semantic search engine for SmartDesk.

This replaces Phase 2's keyword search (TF-IDF) with meaning-based search.
A user searching "physician experience" now finds documents containing "doctor skills".

Architecture:
  1. Encode all documents into vectors using a pre-trained embedding model
  2. Store vectors (in production: vector DB like Chroma/FAISS. Here: numpy array)
  3. At query time: encode the query → find closest document vectors → return results

WHY sentence-transformers and not raw Word2Vec?
  Word2Vec gives you one vector PER WORD. To get a document vector,
  you'd average all word vectors — which loses word order (failure #3).
  Sentence-transformers gives you one vector PER SENTENCE/DOCUMENT
  using a Transformer model (BERT-based). It understands context.
  Think of it as: Word2Vec for words, sentence-transformers for documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smartdesk.observability.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Domain model
# ============================================================================
@dataclass
class SearchResult:
    """One search result with score and metadata.

    WHY a dataclass and not a dict?
    Types, autocomplete, and you can't accidentally misspell a key.
    """

    document: str
    score: float  # cosine similarity, 0.0 to 1.0
    index: int    # position in the original corpus
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# The search engine
# ============================================================================
class SemanticSearchEngine:
    """Embedding-based document search.

    This is the core of Phase 3. Everything in RAG (Phase 8) builds on this:
      - Phase 8 adds chunking (splitting long docs into pieces)
      - Phase 8 adds a real vector DB (Chroma/Qdrant instead of numpy)
      - Phase 8 adds reranking (a second model that re-scores top results)

    But the FOUNDATION — encode, store, search — is right here.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with a sentence-transformer model.

        WHY all-MiniLM-L6-v2?
          - 384-dimensional embeddings (small, fast)
          - Trained on 1B+ sentence pairs
          - Best speed/quality trade-off for prototyping
          - 80MB model size vs 1.3GB for large models

        In production you'd benchmark alternatives:
          - all-mpnet-base-v2 (768-dim, better quality, slower)
          - BGE-large-en (1024-dim, state-of-the-art, GPU recommended)
          - E5-large-v2 (1024-dim, great for retrieval)
        """
        # Lazy import — don't load the model until actually needed.
        # This is the Lazy Loading pattern. In production, model loading
        # happens at startup, not on first request (cold start problem).
        from sentence_transformers import SentenceTransformer

        logger.info("loading_embedding_model", model=model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # These get populated when you call index()
        self._embeddings: np.ndarray | None = None
        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Encode and store all documents.

        This is the "write" path. In production this runs as a batch job
        (Celery/Arq) because encoding 100K docs takes minutes.

        WHY batch encode and not one-by-one?
          GPU parallelism. Encoding 1000 docs at once is 10x faster
          than encoding them one at a time because the GPU can process
          multiple vectors simultaneously.
        """
        self._documents = documents
        self._metadata = metadata or [{} for _ in documents]

        logger.info("indexing_documents", count=len(documents))

        # encode() returns a numpy array of shape (n_docs, embedding_dim)
        # normalize_embeddings=True → unit vectors → cosine sim = dot product
        # WHY normalize? If all vectors have length 1, then:
        #   cosine_similarity(a, b) = dot_product(a, b)
        # Dot product is MUCH faster than computing cosine from scratch.
        # This is the standard optimization in all vector databases.
        self._embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=len(documents) > 100,
            batch_size=64,  # WHY 64? Balances GPU memory vs throughput
        )

        logger.info(
            "indexing_complete",
            count=len(documents),
            embedding_dim=self._embeddings.shape[1],
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Find the most similar documents to a query.

        This is your scratch code's find_neighbors() — but using
        matrix multiplication instead of a Python loop.

        WHY matrix multiply instead of a loop?
          Loop: for each doc, compute cosine → O(n × d) in Python (slow)
          Matrix: query_vec @ doc_matrix.T → O(n × d) in C/BLAS (1000x faster)

          For 1M documents, the loop takes 30 seconds.
          The matrix multiply takes 0.03 seconds.
          This is why numpy exists.
        """
        if self._embeddings is None:
            raise RuntimeError("No documents indexed. Call index() first.")

        # Encode the query into a vector
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        # Compute cosine similarity against ALL documents in one operation
        # Shape: (1, dim) @ (dim, n_docs) → (1, n_docs)
        # This is your dot_product() from scratch — but for ALL docs at once
        scores = (query_vec @ self._embeddings.T)[0]

        # Get the top_k highest-scoring indices
        # WHY argpartition instead of argsort?
        #   argsort sorts ALL scores → O(n log n)
        #   argpartition finds top K without full sort → O(n)
        #   For 1M docs, this saves ~100ms per query.
        #   In Phase 8 (RAG), we'll use FAISS/HNSW for O(log n) search.
        if len(scores) <= top_k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = [
            SearchResult(
                document=self._documents[i],
                score=float(scores[i]),
                index=int(i),
                metadata=self._metadata[i],
            )
            for i in top_indices
        ]

        logger.info(
            "search_complete",
            query=query[:50],
            top_score=round(results[0].score, 3) if results else 0,
            results=len(results),
        )

        return results

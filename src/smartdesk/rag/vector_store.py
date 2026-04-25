# src/smartdesk/rag/vector_store.py
import chromadb
from chromadb.config import Settings
from smartdesk.rag.chunker import Chunk
from smartdesk.rag.embedder import RAGEmbedder
import logging

logger = logging.getLogger(__name__)


class SmartDeskVectorStore:
    """
    ChromaDB wrapper.
    Stores chunk text + embeddings + metadata on disk.
    Persists between runs — ingest once, query forever.
    """
    COLLECTION_NAME = "smartdesk_docs"

    def __init__(self, persist_dir: str = "data/chroma"):
        # Persistent client — survives process restarts
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        self.embedder = RAGEmbedder()
        logger.info(f"Vector store at {persist_dir} — "
                    f"{self.collection.count()} chunks indexed")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks. Skips already-indexed chunks."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        self.collection.add(
            ids=[f"{c.doc_id}_{c.chunk_index}" for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "start_word": c.start_word,
            } for c in chunks],
        )
        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top-k most semantically similar chunks.
        Returns list of dicts with text, doc_id, score.
        """
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": text,
                "doc_id": meta["doc_id"],
                "chunk_index": meta["chunk_index"],
                "score": 1 - dist,  # ChromaDB returns distance; convert to similarity
            })

        return chunks

    def count(self) -> int:
        return self.collection.count()
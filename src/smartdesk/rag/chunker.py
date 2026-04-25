# src/smartdesk/rag/chunker.py
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    start_word: int
    end_word: int


class DocumentChunker:
    """
    Splits documents into overlapping word-based chunks.

    Why word-based not character-based?
      Characters are meaningless units. Words respect sentence
      boundaries much better. 300 words ≈ 1-2 paragraphs —
      enough context for the LLM, small enough to be precise.

    Why overlap?
      A payment term sentence at position 299 would be split
      across two chunks without overlap. Overlap guarantees
      every sentence appears complete in at least one chunk.
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap  # how far we advance each step
        i = 0
        chunk_index = 0

        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_text = " ".join(words[i:end])
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=chunk_index,
                start_word=i,
                end_word=end,
            ))
            if end == len(words):
                break
            i += step
            chunk_index += 1

        return chunks

    def chunk_many(self, docs: dict[str, str]) -> list[Chunk]:
        """Chunk multiple documents. docs = {doc_id: text}"""
        all_chunks = []
        for doc_id, text in docs.items():
            all_chunks.extend(self.chunk(text, doc_id))
        return all_chunks
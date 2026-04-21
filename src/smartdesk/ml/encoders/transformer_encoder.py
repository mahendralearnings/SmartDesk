"""
Production Transformer encoder for SmartDesk.

WHY this file exists: our scratch Phase 5 code taught us the mechanism.
This file is what actually runs in production — HuggingFace does the
heavy lifting, but every line below has a reason.

Design decisions:
  - Two output modes: token-level (for NER) and sentence-level (for search)
  - Automatic GPU/CPU detection
  - Batch processing with padding + attention masks
  - Model is loaded once, reused across calls (same lru_cache pattern as NER)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BatchEncoding

logger = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class EncoderConfig:
    """Runtime config for DocumentEncoder.

    model_name: any HuggingFace model ID.
      - "bert-base-uncased"          → general purpose, 768-dim
      - "sentence-transformers/all-MiniLM-L6-v2" → fast, 384-dim (we used this in Phase 3)
      - "sentence-transformers/all-mpnet-base-v2" → slower, better quality

    WHY we default to MiniLM: SmartDesk is an enterprise tool.
    Latency matters. MiniLM is 5× faster than bert-base with
    ~95% of the quality on most retrieval benchmarks.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512          # BERT's hard limit — see Phase 5 quadratic cost
    batch_size: int = 32
    pooling: Literal["cls", "mean", "max"] = "mean"
    device: str | None = None      # None = auto-detect


# ── Pooling strategies ───────────────────────────────────────────────────────

def _cls_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the [CLS] token embedding only.

    WHY CLS: BERT prepends a special [CLS] token. During pre-training,
    the model is trained to pack sentence-level meaning into this token
    for the Next Sentence Prediction task. For classification tasks,
    CLS is often the best single-vector representation.

    Weakness: [CLS] is only as good as fine-tuning made it. On models
    NOT fine-tuned for sentence similarity, CLS performs poorly.
    """
    return token_embeddings[:, 0, :]   # first token = [CLS]


def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average all token embeddings, ignoring padding tokens.

    WHY mean pooling beats CLS for semantic search:
    Sentence-transformers paper (Reimers 2019) showed mean pooling
    outperforms CLS on STS benchmarks by 2-3 points when the model
    is fine-tuned for sentence similarity.

    The attention_mask multiplication is critical — padding tokens
    are zeros in the mask, so they contribute nothing to the average.
    Without this, padding tokens corrupt the sentence vector.
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)   # avoid division by zero
    return sum_embeddings / sum_mask


def _max_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the max value across tokens for each dimension.

    WHY max pooling: captures the most "activated" signal per feature.
    Useful for classification where you want the strongest evidence,
    not the average evidence. Less common than mean for retrieval.
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings = token_embeddings.clone()
    token_embeddings[mask == 0] = -1e9   # mask padding before max
    return torch.max(token_embeddings, dim=1).values


_POOLING_FNS = {
    "cls":  _cls_pooling,
    "mean": _mean_pooling,
    "max":  _max_pooling,
}


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class EncodingResult:
    """Output of a single document encoding.

    Two tensors, two use cases:
      token_embeddings  → per-token vectors → NER, token classification
      sentence_embedding → one vector → semantic search, clustering, RAG
    """
    text: str
    token_embeddings: torch.Tensor    # (num_tokens, hidden_size)
    sentence_embedding: torch.Tensor  # (hidden_size,) — after pooling + normalisation
    tokens: list[str]                 # the actual wordpieces, useful for debugging
    hidden_size: int = field(init=False)

    def __post_init__(self):
        self.hidden_size = self.sentence_embedding.shape[-1]

    @property
    def embedding_numpy(self):
        """Convenience: sentence embedding as numpy array for sklearn/faiss."""
        return self.sentence_embedding.cpu().numpy()


# ── Main class ────────────────────────────────────────────────────────────────

class DocumentEncoder:
    """Encode documents into contextual token and sentence embeddings.

    Usage:
        encoder = create_encoder()
        result  = encoder.encode("Invoice INV-0042 from Acme Corp is overdue.")

        # sentence-level: semantic search, RAG retrieval
        vec = result.embedding_numpy   # (384,) for MiniLM

        # token-level: NER, token classification (Phase 6)
        tok_vecs = result.token_embeddings   # (num_tokens, 384)
    """

    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, config: EncoderConfig):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._pool_fn = _POOLING_FNS[config.pooling]

        logger.info(
            "DocumentEncoder ready | model=%s | pooling=%s | device=%s | hidden=%d",
            config.model_name,
            config.pooling,
            self._device,
            self._model.config.hidden_size,
        )

    def encode(self, text: str) -> EncodingResult:
        """Encode a single document."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[EncodingResult]:
        """Encode multiple documents efficiently.

        WHY batch > loop: the GPU processes a whole batch in one
        kernel launch. Single-doc loop = N kernel launches.
        At batch_size=32, throughput is ~10× higher on GPU.
        """
        results = []

        for i in range(0, len(texts), self._config.batch_size):
            batch_texts = texts[i : i + self._config.batch_size]
            batch_results = self._encode_batch_internal(batch_texts)
            results.extend(batch_results)

        return results

    def _encode_batch_internal(self, texts: list[str]) -> list[EncodingResult]:
        """Core encoding logic for one mini-batch."""
        # Tokenise — padding + truncation handles variable-length docs
        encoded: BatchEncoding = self._tokenizer(
            texts,
            padding=True,           # pad shorter sequences to batch max length
            truncation=True,        # cut anything beyond max_length
            max_length=self._config.max_length,
            return_tensors="pt",    # return PyTorch tensors, not lists
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        # Forward pass — no_grad because we're inferring, not training
        # WHY no_grad: skips building the computation graph → 2× faster,
        # uses half the memory. Always use this at inference time.
        with torch.no_grad():
            outputs = self._model(**encoded)

        # token_embeddings shape: (batch_size, seq_len, hidden_size)
        token_embeddings = outputs.last_hidden_state

        # Pool to sentence vectors: (batch_size, hidden_size)
        sentence_embeddings = self._pool_fn(
            token_embeddings, encoded["attention_mask"]
        )

        # L2 normalise — makes cosine similarity = dot product
        # WHY normalise: cosine_sim(a, b) = dot(a,b) / (|a| * |b|)
        # If both vectors are unit length, |a|=|b|=1, so cosine = dot.
        # Dot product is much faster than full cosine at search time
        # (especially with FAISS). Normalise once here, dot everywhere.
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        results = []
        for j, text in enumerate(texts):
            token_ids = encoded["input_ids"][j]
            tokens = self._tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())

            # Remove padding tokens from token_embeddings
            real_len = encoded["attention_mask"][j].sum().item()
            real_token_embeddings = token_embeddings[j, :int(real_len), :]

            results.append(EncodingResult(
                text=text,
                token_embeddings=real_token_embeddings.cpu(),
                sentence_embedding=sentence_embeddings[j].cpu(),
                tokens=[t for t in tokens[:int(real_len)] if t != "[PAD]"],
            ))

        return results

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two documents. 1.0 = identical, 0.0 = unrelated."""
        results = self.encode_batch([text_a, text_b])
        # After L2 normalisation, dot product == cosine similarity
        score = torch.dot(results[0].sentence_embedding, results[1].sentence_embedding)
        return round(float(score), 4)


# ── Factory ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _load_model_and_tokenizer(model_name: str) -> tuple[AutoModel, AutoTokenizer]:
    """Load model + tokenizer once, cache forever in this process.

    WHY cache both together: they must be paired — a tokenizer from
    model A with weights from model B produces garbage. Caching the
    tuple prevents any possibility of mismatch.
    """
    logger.info("Loading transformer model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()   # WHY eval(): disables dropout. At inference,
                   # you want deterministic outputs, not stochastic.
    return model, tokenizer


def create_encoder(config: EncoderConfig | None = None) -> DocumentEncoder:
    """Factory — the only entry point callers should use."""
    if config is None:
        config = EncoderConfig()

    model, tokenizer = _load_model_and_tokenizer(config.model_name)
    return DocumentEncoder(model=model, tokenizer=tokenizer, config=config)
"""Text vectorization: BoW, TF-IDF, n-grams.

Strategy pattern again: swap vectorizers without touching the classifier.
In Phase 6, we'll add a BERTVectorizer that drops in the same way.

Why this design matters:
  - Config-driven: yaml says "tfidf" → factory returns TfidfVectorizer
  - Serializable: save/load fitted vectorizer for production inference
  - Testable: each vectorizer tested independently
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from smartdesk.observability.logging import get_logger

logger = get_logger(__name__)


class VectorizerType(str, Enum):
    BOW = "bow"
    TFIDF = "tfidf"
    NGRAM_TFIDF = "ngram_tfidf"


def create_vectorizer(
    method: VectorizerType = VectorizerType.TFIDF,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 1),
    **kwargs: Any,
) -> CountVectorizer | TfidfVectorizer:
    """Factory function — returns the right vectorizer based on config.

    This is the Factory pattern. In Phase 8 we'll use the same pattern
    to create embedding models and retrievers.

    Args:
        method: Which vectorization strategy to use.
        max_features: Cap vocabulary size (prevents memory explosion).
        ngram_range: (1,1) for unigrams, (1,2) for uni+bigrams, etc.
    """
    common = {
        "max_features": max_features,
        "strip_accents": "unicode",
        "token_pattern": r"(?u)\b\w\w+\b",  # at least 2 chars
        **kwargs,
    }

    if method == VectorizerType.BOW:
        logger.info("vectorizer_created", method="bow", max_features=max_features)
        return CountVectorizer(ngram_range=ngram_range, **common)

    elif method == VectorizerType.TFIDF:
        logger.info("vectorizer_created", method="tfidf", max_features=max_features)
        return TfidfVectorizer(
            ngram_range=ngram_range,
            sublinear_tf=True,  # log(1 + tf) — dampens high-freq words
            **common,
        )

    elif method == VectorizerType.NGRAM_TFIDF:
        logger.info(
            "vectorizer_created",
            method="ngram_tfidf",
            ngram_range=(1, 2),
            max_features=max_features,
        )
        return TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            **common,
        )

    raise ValueError(f"Unknown vectorizer: {method}")

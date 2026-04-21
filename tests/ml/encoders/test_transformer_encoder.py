"""5 tests for DocumentEncoder — covering the core contract."""
import pytest
import torch

from smartdesk.ml.encoders.transformer_encoder import (
    EncoderConfig,
    create_encoder,
)

@pytest.fixture(scope="module")
def encoder():
    """Load once for the whole test module — model load is ~2s."""
    return create_encoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))


def test_sentence_embedding_shape(encoder):
    result = encoder.encode("Invoice INV-0042 from Acme Corp is overdue.")
    assert result.sentence_embedding.shape == (384,), "MiniLM output dim is 384"


def test_sentence_embedding_is_unit_normalised(encoder):
    result = encoder.encode("Test document.")
    norm = torch.norm(result.sentence_embedding).item()
    assert abs(norm - 1.0) < 1e-5, "Embedding must be L2-normalised"


def test_token_embeddings_shape(encoder):
    text = "Acme Corp owes payment."
    result = encoder.encode(text)
    # token count > 0 and < max_length
    assert result.token_embeddings.shape[0] > 0
    assert result.token_embeddings.shape[1] == 384


def test_similar_docs_score_higher_than_unrelated(encoder):
    score_related = encoder.similarity(
        "Invoice INV-0042 is overdue.",
        "Payment for invoice INV-0042 has not been received.",
    )
    score_unrelated = encoder.similarity(
        "Invoice INV-0042 is overdue.",
        "The quarterly earnings report exceeded expectations.",
    )
    assert score_related > score_unrelated, (
        f"Related docs ({score_related:.3f}) should outscore "
        f"unrelated ({score_unrelated:.3f})"
    )


def test_batch_matches_single(encoder):
    texts = [
        "Acme Corp has an outstanding invoice.",
        "Payment terms are net 30 days.",
    ]
    batch   = encoder.encode_batch(texts)
    singles = [encoder.encode(t) for t in texts]

    for b, s in zip(batch, singles):
        diff = torch.max(torch.abs(b.sentence_embedding - s.sentence_embedding)).item()
        assert diff < 1e-4, "Batch and single encoding must agree"
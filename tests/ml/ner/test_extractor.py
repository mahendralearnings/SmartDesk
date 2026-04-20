"""Tests for EntityExtractor — 6 tests covering the core contract."""
import pytest

from smartdesk.ml.ner.entities import DocumentEntity, EntityLabel, ExtractionResult
from smartdesk.ml.ner.extractor import NERConfig, create_extractor


@pytest.fixture(scope="module")
def extractor():
    """Module-scoped: model loads once, all tests reuse it.
    WHY module scope: loading spaCy per-test would add ~1s per test.
    """
    return create_extractor(NERConfig(model_name="en_core_web_sm"))


# ── 1. Happy path: spaCy entities ─────────────────────────────────────────

def test_extracts_person_and_org(extractor):
    result = extractor.extract("John Smith signed a contract with Acme Corp.")
    persons = result.by_label(EntityLabel.PERSON)
    orgs = result.by_label(EntityLabel.ORG)
    assert any("John" in e.text for e in persons), "Expected a PERSON entity"
    assert any("Acme" in e.text for e in orgs), "Expected an ORG entity"


def test_extracts_date(extractor):
    result = extractor.extract("The invoice is due on March 15, 2024.")
    dates = result.by_label(EntityLabel.DATE)
    assert len(dates) >= 1
    assert dates[0].source == "spacy"


# ── 2. Custom INVOICE_ID component ────────────────────────────────────────

def test_extracts_invoice_id(extractor):
    result = extractor.extract("Please process invoice INV-00123 immediately.")
    invoices = result.by_label(EntityLabel.INVOICE_ID)
    assert len(invoices) == 1
    assert invoices[0].text == "INV-00123"
    assert invoices[0].source == "regex"


def test_invoice_id_does_not_overlap_person(extractor):
    """Invoice ID in a sentence that also has a person — both should appear."""
    result = extractor.extract(
        "Sarah Connor approved INV-99999 from Cyberdyne Systems."
    )
    labels = {e.label for e in result.entities}
    assert EntityLabel.INVOICE_ID in labels
    assert EntityLabel.PERSON in labels


# ── 3. Edge cases ──────────────────────────────────────────────────────────

def test_empty_string_returns_empty_result(extractor):
    result = extractor.extract("")
    assert result.entity_count == 0
    assert result.entities == []


def test_batch_extract_matches_single(extractor):
    """Batch results must equal calling extract() individually."""
    texts = [
        "Alice works at OpenAI.",
        "Invoice INV-00042 from Bob LLC.",
    ]
    batch = extractor.extract_batch(texts)
    singles = [extractor.extract(t) for t in texts]

    for b, s in zip(batch, singles):
        b_labels = {(e.text, e.label) for e in b.entities}
        s_labels = {(e.text, e.label) for e in s.entities}
        assert b_labels == s_labels
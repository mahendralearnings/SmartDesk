"""Production NER using spaCy + custom INVOICE_ID component.

Patterns used:
- Factory (create_extractor) — caller doesn't construct directly
- Strategy (NERConfig.model_name) — swap sm/md/trf without changing caller
- Decorator (@Language.component) — extend spaCy's pipeline non-invasively
"""
from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import Iterable

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from smartdesk.ml.ner.entities import DocumentEntity, EntityLabel, ExtractionResult
from smartdesk.core.config import settings  # your existing Pydantic Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class NERConfig:
    """Runtime config for EntityExtractor.

    WHY a plain dataclass-style class (not Pydantic): this is a *constructor*
    argument, not a settings file. Keep it lightweight.
    """
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        invoice_pattern: str = r"\bINV-\d{4,8}\b",
        labels_to_extract: Iterable[EntityLabel] | None = None,
        confidence_threshold: float = 0.0,
    ):
        self.model_name = model_name
        self.invoice_pattern = re.compile(invoice_pattern)
        # default: extract all supported labels
        self.labels_to_extract = set(
            labels_to_extract or list(EntityLabel)
        )
        self.confidence_threshold = confidence_threshold


# ---------------------------------------------------------------------------
# Custom spaCy pipeline component — the enterprise secret sauce
# ---------------------------------------------------------------------------

def _make_invoice_component(pattern: re.Pattern) -> callable:
    """Factory that closes over the compiled regex.

    WHY a closure: spaCy components must be stateless callables, but we want
    the regex compiled once (not on every Doc). The closure captures it.
    """
    @Language.component("invoice_id_detector")
    def invoice_id_detector(doc: Doc) -> Doc:
        """Add INVOICE_ID spans to doc.ents without overwriting spaCy's spans.

        The tricky bit: spaCy ents are immutable tuples. We must rebuild
        the list, check for overlaps, then re-assign.
        """
        new_ents = list(doc.ents)   # existing spaCy entities

        for match in pattern.finditer(doc.text):
            # Convert char offsets → token offsets (spaCy works in tokens)
            span = doc.char_span(
                match.start(), match.end(),
                label="INVOICE_ID",
                alignment_mode="expand",  # "expand" absorbs partial token matches
            )
            if span is None:
                continue  # match doesn't align with token boundaries → skip

            # Check for overlap with existing entities — first-wins policy
            # WHY: spaCy crashes if you assign overlapping spans
            overlap = any(
                span.start < e.end and span.end > e.start
                for e in new_ents
            )
            if not overlap:
                new_ents.append(span)

        doc.ents = tuple(new_ents)
        return doc

    return invoice_id_detector


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extract named entities from documents.

    Usage:
        extractor = create_extractor()
        result = extractor.extract("Invoice INV-00123 from Acme Corp due 2024-03-15")
        persons = result.by_label(EntityLabel.PERSON)
    """

    # WHY class-level mapping: spaCy labels → our EntityLabel enum
    # This insulates our code from spaCy internals; if spaCy renames "PER"
    # we fix it in one place.
    _LABEL_MAP: dict[str, EntityLabel] = {
        "PERSON": EntityLabel.PERSON,
        "PER":    EntityLabel.PERSON,   # some spaCy models use PER
        "ORG":    EntityLabel.ORG,
        "DATE":   EntityLabel.DATE,
        "GPE":    EntityLabel.GPE,
        "MONEY":  EntityLabel.MONEY,
        "INVOICE_ID": EntityLabel.INVOICE_ID,
    }

    def __init__(self, nlp: Language, config: NERConfig) -> None:
        self._nlp = nlp
        self._config = config

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities from a single document."""
        if not text or not text.strip():
            return ExtractionResult(
                text=text,
                entities=[],
                model_name=self._config.model_name,
            )

        doc = self._nlp(text)
        entities = self._doc_to_entities(doc)

        return ExtractionResult(
            text=text,
            entities=entities,
            model_name=self._config.model_name,
        )

    def extract_batch(self, texts: list[str], batch_size: int = 64) -> list[ExtractionResult]:
        """Batch extraction — 3-5× faster than calling extract() in a loop.

        WHY nlp.pipe: spaCy processes documents in parallel mini-batches
        internally. Single-doc calls have Python overhead per call.
        At 64-doc batches the overhead is ~1% of total time.
        """
        results = []
        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            results.append(ExtractionResult(
                text=doc.text,
                entities=self._doc_to_entities(doc),
                model_name=self._config.model_name,
            ))
        return results

    def _doc_to_entities(self, doc: Doc) -> list[DocumentEntity]:
        """Convert spaCy doc spans → our DocumentEntity objects."""
        entities: list[DocumentEntity] = []

        for ent in doc.ents:
            our_label = self._LABEL_MAP.get(ent.label_)
            if our_label is None:
                continue  # spaCy found something we don't care about (e.g. CARDINAL)
            if our_label not in self._config.labels_to_extract:
                continue

            source = "regex" if ent.label_ == "INVOICE_ID" else "spacy"

            try:
                entity = DocumentEntity(
                    text=ent.text,
                    label=our_label,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=None,    # en_core_web_sm doesn't expose scores
                    source=source,
                )
                entities.append(entity)
            except Exception:
                # Validation failed (e.g. zero-length span) — log and skip
                logger.debug("Skipped invalid entity span: %r", ent.text)

        return entities


# ---------------------------------------------------------------------------
# Factory — the public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_nlp(model_name: str) -> Language:
    """Load and cache spaCy model.

    WHY lru_cache: loading en_core_web_sm takes ~150ms. In a web server
    you'd load it once at startup, but in tests and scripts this prevents
    repeated cold loads on every create_extractor() call.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found. "
            f"Run: python -m spacy download {model_name}"
        ) from exc
    return nlp


def create_extractor(config: NERConfig | None = None) -> EntityExtractor:
    """Factory function — the only entry point callers should use.

    WHY a factory: hides the two-step (load model, register component).
    Callers don't need to know about spaCy internals.
    """
    if config is None:
        config = NERConfig()

    nlp = _load_nlp(config.model_name)

    # Register the custom component if it isn't already in the pipeline
    # WHY the guard: calling add_pipe twice on the same nlp raises an error
    if "invoice_id_detector" not in nlp.pipe_names:
        _make_invoice_component(config.invoice_pattern)   # registers via decorator
        nlp.add_pipe("invoice_id_detector", last=True)

    logger.info(
        "EntityExtractor ready | model=%s | pipeline=%s",
        config.model_name,
        nlp.pipe_names,
    )
    return EntityExtractor(nlp=nlp, config=config)
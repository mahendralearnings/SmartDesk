"""Value objects for NER results.

WHY Pydantic here: immutable, serializable, auto-validated.
A DocumentEntity is a *value object* — two entities with the same
fields are equal, regardless of which extraction run produced them.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class EntityLabel(str, Enum):
    """Supported entity types in SmartDesk.

    WHY an Enum: prevents typos like "PERSN" silently passing through.
    String mixin lets us serialize directly to JSON without .value.
    """
    PERSON = "PERSON"
    ORG = "ORG"
    DATE = "DATE"
    GPE = "GPE"           # geopolitical entity (city, country)
    MONEY = "MONEY"
    INVOICE_ID = "INVOICE_ID"   # our custom business entity


class DocumentEntity(BaseModel):
    """A single extracted entity — immutable value object.

    WHY frozen=True: entities are facts about a document, not mutable state.
    Freezing makes them hashable → you can put them in sets for deduplication.
    """
    text: str = Field(..., min_length=1, description="Surface form as it appeared")
    label: EntityLabel
    start_char: int = Field(..., ge=0, description="Char offset in original text")
    end_char: int = Field(..., gt=0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: str = Field("spacy", description="'spacy' or 'regex' — helps debug")

    model_config = {"frozen": True}   # makes instances hashable

    @model_validator(mode="after")
    def end_after_start(self) -> "DocumentEntity":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be > start_char")
        return self

    @property
    def span_length(self) -> int:
        return self.end_char - self.start_char


class ExtractionResult(BaseModel):
    """Full extraction result for one document.

    WHY a wrapper class: gives us a clean place to add metadata later
    (latency, model version, document_id) without changing the entity schema.
    """
    text: str
    entities: list[DocumentEntity] = Field(default_factory=list)
    model_name: str = "unknown"

    def by_label(self, label: EntityLabel) -> list[DocumentEntity]:
        """Filter entities by type — the most common access pattern."""
        return [e for e in self.entities if e.label == label]

    @property
    def entity_count(self) -> int:
        return len(self.entities)
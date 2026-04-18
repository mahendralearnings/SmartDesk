"""Text preprocessing pipeline.

Design: Strategy pattern. Every step is a PreprocessingStep with a uniform interface.
The pipeline runs steps in order. Swap, add, or remove steps without touching others.

Why this matters: in Phase 2 we'll reuse the same pipeline for classical ML,
in Phase 6 we'll reuse it before BERT tokenization, in Phase 8 we'll reuse it
before chunking for RAG. One pipeline, many consumers.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import emoji
import ftfy
from bs4 import BeautifulSoup

from smartdesk.config import PreprocessingConfig
from smartdesk.observability.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Domain model
# ============================================================================
@dataclass
class ProcessedText:
    """Result of running text through the pipeline.

    We keep the original so downstream consumers can audit.
    Metadata tracks what each step did — critical for debugging in production.
    """

    original: str
    cleaned: str
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


# ============================================================================
# Strategy pattern: every step implements this Protocol
# ============================================================================
class PreprocessingStep(Protocol):
    """Interface every preprocessing step must satisfy.

    Using Protocol (structural typing) not ABC — any class with an `apply` method works.
    This is more Pythonic and testable than inheritance.
    """

    name: str

    def apply(self, text: str) -> str: ...


# ============================================================================
# Concrete steps
# ============================================================================
class FixEncoding:
    """Step 1: Repair mojibake (broken encoding). "cafÃ©" -> "café"."""

    name = "fix_encoding"

    def apply(self, text: str) -> str:
        return ftfy.fix_text(text)


class StripHtml:
    """Step 2: Remove HTML tags, keep inner text."""

    name = "strip_html"

    def apply(self, text: str) -> str:
        if "<" not in text:  # fast path — most inputs are plain text
            return text
        return BeautifulSoup(text, "lxml").get_text(separator=" ")


class RemoveUrls:
    """Step 3: Strip URLs. Pre-compiled regex for speed."""

    name = "remove_urls"
    _pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    def apply(self, text: str) -> str:
        return self._pattern.sub(" ", text)


class RemoveEmails:
    """Step 4: Strip email addresses."""

    name = "remove_emails"
    _pattern = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

    def apply(self, text: str) -> str:
        return self._pattern.sub(" ", text)


class RemoveEmojis:
    """Step 5: Strip emoji characters."""

    name = "remove_emojis"

    def apply(self, text: str) -> str:
        return emoji.replace_emoji(text, replace=" ")


class NormalizeWhitespace:
    """Step 6: Collapse multiple spaces/tabs/newlines into single space."""

    name = "normalize_whitespace"
    _pattern = re.compile(r"\s+")

    def apply(self, text: str) -> str:
        return self._pattern.sub(" ", text).strip()


class Lowercase:
    """Step 7: Lowercase. Task-dependent — sometimes you want to preserve case."""

    name = "lowercase"

    def apply(self, text: str) -> str:
        return text.lower()


# ============================================================================
# The pipeline itself
# ============================================================================
class PreprocessingPipeline:
    """Composes a sequence of preprocessing steps.

    Enterprise touches:
      - Config-driven: steps enabled/disabled via PreprocessingConfig.
      - Observable: every step logs what it changed (length delta).
      - Immutable after construction: steps list is frozen once set.
      - Testable: each step is independently unit-testable.
    """

    def __init__(self, steps: list[PreprocessingStep]) -> None:
        self._steps: tuple[PreprocessingStep, ...] = tuple(steps)
        logger.info("pipeline_initialized", steps=[s.name for s in self._steps])

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> PreprocessingPipeline:
        """Factory method — builds pipeline based on config toggles.

        This is the Factory pattern. In Phase 8 we'll use this same approach
        to build swappable retriever pipelines.
        """
        steps: list[PreprocessingStep] = [FixEncoding(), StripHtml()]

        if config.remove_urls:
            steps.append(RemoveUrls())
        if config.remove_emails:
            steps.append(RemoveEmails())
        if config.remove_emojis:
            steps.append(RemoveEmojis())

        steps.append(NormalizeWhitespace())

        if config.lowercase:
            steps.append(Lowercase())

        return cls(steps)

    def process(self, text: str) -> ProcessedText:
        """Run text through all steps. Returns ProcessedText with audit trail."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        original = text
        current = text
        step_log: dict[str, int] = {}

        for step in self._steps:
            before_len = len(current)
            current = step.apply(current)
            step_log[step.name] = before_len - len(current)

        return ProcessedText(
            original=original,
            cleaned=current,
            metadata={"step_deltas": step_log},
        )

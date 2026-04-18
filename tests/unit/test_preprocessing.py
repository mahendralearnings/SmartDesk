"""Tests for Phase 1 preprocessing components."""
import pytest

from smartdesk.config import PreprocessingConfig
from smartdesk.ml.preprocessing.pipeline import (
    FixEncoding,
    Lowercase,
    NormalizeWhitespace,
    PreprocessingPipeline,
    RemoveEmails,
    RemoveEmojis,
    RemoveUrls,
    StripHtml,
)
from smartdesk.ml.preprocessing.trie import Trie


# ============================================================================
# Individual step tests — each step tested in isolation
# ============================================================================
class TestIndividualSteps:
    def test_fix_encoding_repairs_mojibake(self) -> None:
        assert FixEncoding().apply("cafÃ©") == "café"

    def test_strip_html_removes_tags(self) -> None:
        result = StripHtml().apply("<p>Hello <b>World</b></p>")
        assert "Hello" in result and "World" in result
        assert "<" not in result

    def test_strip_html_fast_path_no_tags(self) -> None:
        # When no HTML present, should return unchanged (perf optimization)
        plain = "Just plain text"
        assert StripHtml().apply(plain) == plain

    def test_remove_urls_strips_http_and_www(self) -> None:
        step = RemoveUrls()
        assert "http" not in step.apply("visit https://acme.com now")
        assert "www" not in step.apply("go to www.acme.com")

    def test_remove_emails(self) -> None:
        result = RemoveEmails().apply("contact alice@acme.com please")
        assert "@" not in result

    def test_remove_emojis(self) -> None:
        result = RemoveEmojis().apply("Hello 👋 world 🌍")
        assert "👋" not in result and "🌍" not in result

    def test_normalize_whitespace_collapses(self) -> None:
        assert NormalizeWhitespace().apply("hello    world\n\ttest") == "hello world test"

    def test_lowercase(self) -> None:
        assert Lowercase().apply("HELLO World") == "hello world"


# ============================================================================
# Pipeline integration tests
# ============================================================================
class TestPipelineIntegration:
    def test_full_pipeline_on_messy_document(self) -> None:
        """The invoice example from our phase 1 story."""
        messy = (
            "Hi!! I received the invoice #INV-2024-00891 on 12/03/2024. "
            "The amount $1,250.00 USD is INCORRECT!!! "
            "Please check https://portal.acme.com/invoice/891... "
            "Email: accounts@acme.com. Regards, Priya 👋"
        )
        config = PreprocessingConfig(remove_emojis=True)
        pipeline = PreprocessingPipeline.from_config(config)

        result = pipeline.process(messy)

        assert result.original == messy  # original preserved
        assert "https://" not in result.cleaned
        assert "accounts@acme.com" not in result.cleaned
        assert "👋" not in result.cleaned
        assert result.cleaned == result.cleaned.lower()
        assert "step_deltas" in result.metadata

    def test_pipeline_raises_on_non_string(self) -> None:
        pipeline = PreprocessingPipeline.from_config(PreprocessingConfig())
        with pytest.raises(TypeError):
            pipeline.process(12345)  # type: ignore[arg-type]

    def test_config_toggles_steps(self) -> None:
        """Strategy pattern: turning off a step actually skips it."""
        config = PreprocessingConfig(remove_urls=False)
        pipeline = PreprocessingPipeline.from_config(config)
        result = pipeline.process("go to https://acme.com")
        assert "https://acme.com" in result.cleaned


# ============================================================================
# Trie tests
# ============================================================================
class TestTrie:
    def test_insert_and_contains(self) -> None:
        trie = Trie.from_words(["the", "this", "that", "is", "it"])
        assert trie.contains("the")
        assert trie.contains("this")
        assert not trie.contains("th")  # prefix, not a word
        assert not trie.contains("xyz")

    def test_starts_with(self) -> None:
        trie = Trie.from_words(["the", "this", "that"])
        assert trie.starts_with("th")
        assert trie.starts_with("the")
        assert not trie.starts_with("xy")

    def test_size_tracking(self) -> None:
        trie = Trie()
        trie.insert("hello")
        trie.insert("world")
        trie.insert("hello")  # duplicate, should not increment
        assert len(trie) == 2

    def test_payload_storage(self) -> None:
        """Production use: store POS tag or category at word end."""
        trie = Trie()
        trie.insert("bank", payload="NOUN")
        node = trie._traverse("bank")
        assert node is not None
        assert node.payload == "NOUN"

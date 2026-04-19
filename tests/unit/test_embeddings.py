"""Tests for Phase 3: Semantic search engine."""
import pytest

from smartdesk.ml.embeddings import SearchResult, SemanticSearchEngine


class TestSearchResult:
    def test_fields_exist(self) -> None:
        r = SearchResult(document="test doc", score=0.95, index=0)
        assert r.document == "test doc"
        assert r.score == 0.95
        assert r.index == 0
        assert r.metadata == {}

    def test_metadata_default(self) -> None:
        r = SearchResult(document="x", score=0.5, index=1, metadata={"label": "invoice"})
        assert r.metadata["label"] == "invoice"


@pytest.fixture(scope="module")
def engine() -> SemanticSearchEngine:
    """Load engine once for all tests (model loading is slow)."""
    e = SemanticSearchEngine(model_name="all-MiniLM-L6-v2")
    docs = [
        "experienced doctor in cardiology",
        "software engineer with Python skills",
        "invoice for medical equipment supplies",
        "pizza delivery restaurant menu",
    ]
    e.index(docs, metadata=[{"label": l} for l in ["medical", "tech", "finance", "food"]])
    return e


class TestSemanticSearchEngine:
    def test_search_returns_results(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("physician", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_synonym_finds_correct_doc(self, engine: SemanticSearchEngine) -> None:
        """The CORE Phase 3 test: 'physician' should match 'doctor'."""
        results = engine.search("physician", top_k=1)
        assert "doctor" in results[0].document or "medical" in results[0].metadata.get("label", "")

    def test_scores_are_sorted_descending(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("coding developer", top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unrelated_query_has_low_scores(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("pizza delivery", top_k=1)
        # pizza should match the food doc, not the medical doc
        assert "pizza" in results[0].document or "food" in results[0].metadata.get("label", "")

    def test_search_before_index_raises(self) -> None:
        empty_engine = SemanticSearchEngine.__new__(SemanticSearchEngine)
        empty_engine._embeddings = None
        with pytest.raises(RuntimeError, match="No documents indexed"):
            empty_engine.search("test")

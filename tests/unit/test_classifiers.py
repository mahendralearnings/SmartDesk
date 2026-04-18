"""Tests for Phase 2: Vectorizers and Classifiers."""
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from smartdesk.ml.classifiers import VectorizerType, create_vectorizer
from smartdesk.ml.classifiers.document_classifier import (
    ClassifierType,
    DocumentClassifier,
    EvaluationResult,
    create_classifier,
)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def sample_dataset():
    """Small labeled dataset for testing."""
    docs = [
        "invoice number 1234 payment due",
        "billing statement amount owed",
        "payment terms net thirty days",
        "please remit invoice amount",
        "disappointed with service terrible",
        "want refund immediately unacceptable",
        "worst experience ever frustrated",
        "filing complaint about quality",
        "experienced engineer python years",
        "data scientist machine learning",
        "full stack developer react node",
        "senior developer cloud infrastructure",
    ]
    labels = ["invoice"] * 4 + ["complaint"] * 4 + ["resume"] * 4
    return docs, labels


# ============================================================================
# Vectorizer tests
# ============================================================================
class TestVectorizers:
    def test_bow_creates_count_vectorizer(self):
        vec = create_vectorizer(method=VectorizerType.BOW, max_features=100)
        assert hasattr(vec, "fit_transform")

    def test_tfidf_creates_tfidf_vectorizer(self):
        vec = create_vectorizer(method=VectorizerType.TFIDF)
        assert isinstance(vec, TfidfVectorizer)

    def test_ngram_tfidf_uses_bigrams(self):
        vec = create_vectorizer(method=VectorizerType.NGRAM_TFIDF)
        assert vec.ngram_range == (1, 2)

    def test_vectorizer_produces_sparse_matrix(self, sample_dataset):
        docs, _ = sample_dataset
        vec = create_vectorizer(method=VectorizerType.TFIDF, max_features=50)
        X = vec.fit_transform(docs)
        assert X.shape[0] == len(docs)
        assert X.shape[1] <= 50


# ============================================================================
# Classifier tests
# ============================================================================
class TestClassifiers:
    def test_factory_creates_all_types(self):
        for clf_type in ClassifierType:
            clf = create_classifier(clf_type)
            assert hasattr(clf, "fit")
            assert hasattr(clf, "predict")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_classifier("random_forest")  # type: ignore

    def test_train_and_evaluate_returns_result(self, sample_dataset):
        docs, labels = sample_dataset
        vec = create_vectorizer(method=VectorizerType.TFIDF, max_features=50)
        X = vec.fit_transform(docs)
        y = np.array(labels)

        # Need at least 2 per class in test set for stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.34, random_state=42, stratify=y
        )

        clf = DocumentClassifier(classifier_type=ClassifierType.NAIVE_BAYES)
        result = clf.train_and_evaluate(
            X_train, y_train, X_test, y_test, cv_folds=2
        )

        assert isinstance(result, EvaluationResult)
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1_macro <= 1
        assert result.classifier_name == "naive_bayes"

    def test_predict_before_train_raises(self):
        clf = DocumentClassifier(classifier_type=ClassifierType.NAIVE_BAYES)
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(np.array([[1, 2, 3]]))


# ============================================================================
# Evaluation result tests
# ============================================================================
class TestEvaluationResult:
    def test_to_dict_has_all_keys(self):
        result = EvaluationResult(
            classifier_name="test",
            accuracy=0.95,
            precision_macro=0.94,
            recall_macro=0.93,
            f1_macro=0.935,
            cv_f1_mean=0.93,
            cv_f1_std=0.02,
        )
        d = result.to_dict()
        assert "classifier" in d
        assert "accuracy" in d
        assert "f1" in d
        assert "cv_f1" in d

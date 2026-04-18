"""Document classifier with multiple algorithm support.

This module implements the core ML pipeline for SmartDesk:
  1. Vectorize text (delegates to vectorizer factory)
  2. Train multiple classifiers
  3. Evaluate with proper metrics (not just accuracy!)
  4. Select the best model

Design patterns at work:
  - Strategy: swap classifiers and vectorizers independently
  - Factory: create_classifier() builds the right model from config
  - Template Method: train_and_evaluate() follows a fixed sequence with variable internals

Enterprise touches:
  - Cross-validation (not just train/test split)
  - Classification report with per-class metrics
  - Confusion matrix for error analysis
  - Model serialization for production deployment
  - Structured logging of all training runs
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from smartdesk.observability.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Enums and data classes
# ============================================================================
class ClassifierType(str, Enum):
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"


@dataclass
class EvaluationResult:
    """Full evaluation metrics for a single model.

    Why dataclass, not dict? Types, autocomplete, immutability.
    This is what we log, compare, and ship to Grafana.
    """

    classifier_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cv_f1_mean: float       # cross-validated F1 — the metric that matters
    cv_f1_std: float
    confusion_mat: list[list[int]] = field(default_factory=list)
    class_report: str = ""
    roc_auc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision_macro, 4),
            "recall": round(self.recall_macro, 4),
            "f1": round(self.f1_macro, 4),
            "cv_f1": f"{self.cv_f1_mean:.4f} ± {self.cv_f1_std:.4f}",
        }


# ============================================================================
# Factory
# ============================================================================
def create_classifier(clf_type: ClassifierType) -> ClassifierMixin:
    """Factory: returns a sklearn classifier instance.

    Interview Q: "Why not just instantiate directly?"
    Answer: Decouples config from construction. Tomorrow we add XGBoost —
    one line in this factory, zero changes in the training pipeline.
    """
    if clf_type == ClassifierType.NAIVE_BAYES:
        return MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing

    elif clf_type == ClassifierType.LOGISTIC_REGRESSION:
        return LogisticRegression(
            C=1.0,            # inverse regularization strength
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )

    elif clf_type == ClassifierType.SVM:
        return LinearSVC(
            C=1.0,
            max_iter=2000,
            dual="auto",
            random_state=42,
        )

    raise ValueError(f"Unknown classifier: {clf_type}")


# ============================================================================
# Training and evaluation
# ============================================================================
class DocumentClassifier:
    """End-to-end document classification system.

    Usage:
        clf = DocumentClassifier(classifier_type=ClassifierType.LOGISTIC_REGRESSION)
        result = clf.train_and_evaluate(X_train, y_train, X_test, y_test, labels)
        clf.save("models/logreg_v1.joblib")
    """

    def __init__(self, classifier_type: ClassifierType) -> None:
        self.classifier_type = classifier_type
        self.model: ClassifierMixin = create_classifier(classifier_type)
        self._is_fitted = False

    def train_and_evaluate(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        label_names: list[str] | None = None,
        cv_folds: int = 5,
    ) -> EvaluationResult:
        """Train, cross-validate, and evaluate on held-out test set.

        This is the Template Method pattern: fixed steps, variable internals.
        Every classifier follows the same evaluate flow.
        """
        name = self.classifier_type.value
        logger.info("training_started", classifier=name, train_size=X_train.shape[0])

        # 1. Cross-validation on training data (guards against overfitting)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv, scoring="f1_macro"
        )
        logger.info(
            "cv_complete",
            classifier=name,
            cv_f1_mean=round(float(cv_scores.mean()), 4),
            cv_f1_std=round(float(cv_scores.std()), 4),
        )

        # 2. Train on full training set
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        # 3. Predict on test set
        y_pred = self.model.predict(X_test)

        # 4. Compute all metrics
        result = EvaluationResult(
            classifier_name=name,
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision_macro=float(precision_score(y_test, y_pred, average="macro")),
            recall_macro=float(recall_score(y_test, y_pred, average="macro")),
            f1_macro=float(f1_score(y_test, y_pred, average="macro")),
            cv_f1_mean=float(cv_scores.mean()),
            cv_f1_std=float(cv_scores.std()),
            confusion_mat=confusion_matrix(y_test, y_pred).tolist(),
            class_report=classification_report(
                y_test, y_pred, target_names=label_names or None
            ),
        )

        logger.info("evaluation_complete", **result.to_dict())
        return result

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for new documents."""
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train_and_evaluate() first.")
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        """Serialize model to disk. Production deployment uses this."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path, classifier_type: ClassifierType) -> DocumentClassifier:
        """Load a saved model."""
        instance = cls(classifier_type)
        instance.model = joblib.load(path)
        instance._is_fitted = True
        return instance

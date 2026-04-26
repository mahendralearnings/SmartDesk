"""Phase 2 Demo: Document Classification Pipeline.

Generates synthetic documents, preprocesses them (Phase 1 pipeline),
vectorizes with TF-IDF, trains 3 classifiers, and compares with full metrics.

Usage: PYTHONPATH=src python scripts/demo_phase2.py
"""
import random

import numpy as np
from sklearn.model_selection import train_test_split

from smartdesk.config import PreprocessingConfig
from smartdesk.ml.classifiers import VectorizerType, create_vectorizer
from smartdesk.ml.classifiers.document_classifier import (
    ClassifierType,
    DocumentClassifier,
    EvaluationResult,
)
from smartdesk.ml.preprocessing.pipeline import PreprocessingPipeline
from smartdesk.observability.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ============================================================================
# Synthetic data generation
# ============================================================================
TEMPLATES: dict[str, list[str]] = {
    "invoice": [
        "Invoice number {id} for amount ${amount} due on {date}. Payment terms net 30.",
        "Please find attached invoice {id}. Total amount due: ${amount} USD.",
        "Billing statement for services rendered. Invoice {id}, amount ${amount}.",
        "This invoice {id} covers professional consulting fees totaling ${amount}.",
        "Account receivable notice: invoice {id} pending payment of ${amount}.",
        "Monthly billing summary. Reference: {id}. Outstanding balance ${amount}.",
        "Payment request for invoice {id}. Amount payable: ${amount} before {date}.",
        "Tax invoice {id} issued for ${amount} inclusive of applicable taxes.",
    ],
    "complaint": [
        "I am extremely disappointed with the service. Order {id} was never delivered.",
        "This is unacceptable! I have been waiting {days} days for a response.",
        "Your product broke after {days} days. I demand a full refund immediately.",
        "Terrible customer service experience. Nobody answered my {days} emails.",
        "I want to escalate this complaint about order {id}. Very frustrated.",
        "The quality of item {id} is far below what was advertised. Misleading.",
        "I am filing a formal complaint. {days} days without resolution is outrageous.",
        "Worst experience ever. Order {id} arrived damaged and nobody is helping.",
    ],
    "resume": [
        "Experienced software engineer with {years} years in Python and machine learning.",
        "Senior data scientist skilled in NLP, deep learning, and statistical modeling.",
        "Full stack developer with expertise in React, Node.js, and cloud infrastructure.",
        "Project manager with {years} years leading cross-functional agile teams.",
        "Machine learning engineer experienced in PyTorch, transformers, and MLOps.",
        "Data analyst with strong SQL skills and {years} years of business intelligence.",
        "DevOps engineer specializing in Kubernetes, CI/CD pipelines, and monitoring.",
        "Technical lead with {years} years building scalable distributed systems.",
    ],
    "contract": [
        "This agreement is entered into between Party A and Party B effective {date}.",
        "The parties hereby agree to the following terms and conditions for {years} years.",
        "Confidentiality clause: all proprietary information shall remain confidential.",
        "Termination: either party may terminate with {days} days written notice.",
        "Indemnification: Party A shall indemnify Party B against all claims.",
        "Force majeure: neither party liable for failure due to unforeseen circumstances.",
        "Governing law: this contract shall be governed by the laws of the jurisdiction.",
        "Amendment: no modification shall be valid unless agreed in writing by both parties.",
    ],
}


def generate_dataset(samples_per_class: int = 200, seed: int = 42) -> tuple[list[str], list[str]]:
    """Generate synthetic documents for classification.

    Returns (documents, labels). In production, this would be
    a database query or file reader.
    """
    random.seed(seed)
    docs: list[str] = []
    labels: list[str] = []

    for category, templates in TEMPLATES.items():
        for _ in range(samples_per_class):
            template = random.choice(templates)
            doc = template.format(
                id=f"INV-{random.randint(1000, 9999)}",
                amount=f"{random.randint(100, 50000):,.2f}",
                date=f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                days=random.randint(1, 90),
                years=random.randint(1, 20),
            )
            # Add noise (like real data)
            if random.random() < 0.3:
                doc = doc.upper() if random.random() < 0.5 else doc + "  \n\t"
            docs.append(doc)
            labels.append(category)

    return docs, labels


# ============================================================================
# Main pipeline
# ============================================================================
def main() -> None:
    print("=" * 70)
    print("PHASE 2: SmartDesk Document Classification")
    print("=" * 70)

    # 1. Generate data
    docs, labels = generate_dataset(samples_per_class=250)
    print(f"\nDataset: {len(docs)} documents, {len(set(labels))} classes")
    print(f"Classes: {sorted(set(labels))}")

    # 2. Preprocess (reuse Phase 1 pipeline!)
    config = PreprocessingConfig(remove_emojis=True, lowercase=True)
    pipeline = PreprocessingPipeline.from_config(config)
    cleaned = [pipeline.process(doc).cleaned for doc in docs]
    print(f"Preprocessing complete. Sample: '{cleaned[0][:80]}...'")

    # 3. Vectorize with TF-IDF (+ bigrams)
    vectorizer = create_vectorizer(
        method=VectorizerType.NGRAM_TFIDF,
        max_features=5000,
    )

    # 4. Train/test split (stratified!)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"\nTrain: {X_train.shape[0]} docs, Test: {X_test.shape[0]} docs")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")

    # 5. Train all 3 classifiers and compare
    label_names = sorted(set(labels))
    results: list[EvaluationResult] = []

    for clf_type in ClassifierType:
        print(f"\n{'─' * 50}")
        print(f"Training: {clf_type.value}")
        print(f"{'─' * 50}")

        clf = DocumentClassifier(classifier_type=clf_type)
        result = clf.train_and_evaluate(
            X_train, y_train, X_test, y_test, label_names=label_names
        )
        results.append(result)

        print(f"\n{result.class_report}")
        print(f"Confusion Matrix:\n{np.array(result.confusion_mat)}")

    # 6. Compare all models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV F1':>16}")
    print("─" * 85)
    for r in results:
        print(
            f"{r.classifier_name:<25} "
            f"{r.accuracy:>10.4f} "
            f"{r.precision_macro:>10.4f} "
            f"{r.recall_macro:>10.4f} "
            f"{r.f1_macro:>10.4f} "
            f"{r.cv_f1_mean:>8.4f}±{r.cv_f1_std:.4f}"
        )

    # 7. Pick best model
    best = max(results, key=lambda r: r.cv_f1_mean)
    print(f"\nBest model: {best.classifier_name} (CV F1: {best.cv_f1_mean:.4f})")

    # 8. Show top features for Logistic Regression (interpretability!)
    logreg_clf = DocumentClassifier(classifier_type=ClassifierType.LOGISTIC_REGRESSION)
    logreg_clf.train_and_evaluate(X_train, y_train, X_test, y_test, label_names=label_names)
    feature_names = vectorizer.get_feature_names_out()

    print("\n" + "=" * 70)
    print("TOP FEATURES PER CLASS (Logistic Regression weights)")
    print("=" * 70)
    coef = logreg_clf.model.coef_
    for i, class_name in enumerate(label_names):
        top_indices = np.argsort(coef[i])[-8:][::-1]
        top_features = [(feature_names[j], round(coef[i][j], 3)) for j in top_indices]
        print(f"\n  {class_name}:")
        for feat, weight in top_features:
            print(f"    {feat:30s} weight={weight:>7.3f}")


if __name__ == "__main__":
    main()

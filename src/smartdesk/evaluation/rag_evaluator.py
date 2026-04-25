# src/smartdesk/evaluation/rag_evaluator.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset


THRESHOLDS = {
    "faithfulness":      0.90,
    "answer_relevancy":  0.85,
    "context_recall":    0.85,
    "context_precision": 0.80,
}


class RAGEvaluator:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, questions: list[dict]) -> dict:

        print(f"Running RAGAS on {len(questions)} questions...")
        rows = []

        for q in questions:
            # Get answer from your Phase 8 pipeline
            result  = self.pipeline.query(q["question"])
            chunks  = self.pipeline.retriever.retrieve(q["question"])

            rows.append({
                "question":     q["question"],
                "answer":       result["answer"],
                "contexts":     [c["text"] for c in chunks],
                "ground_truth": q["ground_truth"],
            })

        # Run RAGAS scoring
        dataset = Dataset.from_list(rows)
        scores  = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
        )

        # Print results clearly
        print("\n" + "=" * 45)
        print("RAGAS RESULTS")
        print("=" * 45)

        results  = {}
        all_pass = True

        # Convert to pandas first — works with all RAGAS versions
        scores_df = scores.to_pandas()

        for metric in ["faithfulness", "answer_relevancy",
                        "context_recall", "context_precision"]:
            score     = round(float(scores_df[metric].mean()), 3)
            threshold = THRESHOLDS[metric]
            passed    = score >= threshold
            status    = "PASS" if passed else "FAIL"

            if not passed:
                all_pass = False

            results[metric] = score
            print(f"  {metric:<22} {score:.3f}   [{status}]  "
                f"(need > {threshold})")
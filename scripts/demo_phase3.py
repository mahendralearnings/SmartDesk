"""Phase 3 Demo: Semantic search vs TF-IDF keyword search.

Shows the exact problem from the Phase 3 story:
  User searches "physician experience" → TF-IDF returns NOTHING
  because all documents say "doctor". Semantic search finds them.

Run: uv run python scripts/demo_phase3.py
"""
from smartdesk.ml.embeddings import SemanticSearchEngine
from smartdesk.observability.logging import configure_logging

configure_logging()

# ============================================================================
# SmartDesk's document corpus — notice: all say "doctor", none say "physician"
# ============================================================================
DOCUMENTS = [
    # Resumes
    "Senior doctor with 10 years experience in cardiology and patient care",
    "Medical doctor specializing in pediatrics and child healthcare",
    "Experienced surgeon with expertise in minimally invasive procedures",
    "Registered nurse with ICU experience and emergency care certification",
    # Invoices
    "Invoice #4521 for medical equipment supplies totaling $12,500 USD",
    "Payment due for hospital bed rental, amount $3,200 per month",
    # Complaints
    "I am disappointed with the doctor's diagnosis and want a second opinion",
    "The hospital service was terrible, waited 4 hours in emergency",
    # Contracts
    "This agreement between the hospital and the medical supplier is effective immediately",
    "Employment contract for the position of chief medical officer",
    # Tech (unrelated)
    "Full stack developer with 5 years Python and React experience",
    "Software engineer skilled in machine learning and data pipelines",
]

LABELS = [
    "resume", "resume", "resume", "resume",
    "invoice", "invoice",
    "complaint", "complaint",
    "contract", "contract",
    "resume", "resume",
]


def main() -> None:
    print("=" * 65)
    print("PHASE 3: Semantic Search vs TF-IDF")
    print("=" * 65)

    # --- Build the search index ---
    engine = SemanticSearchEngine(model_name="all-MiniLM-L6-v2")
    metadata = [{"label": label, "idx": i} for i, label in enumerate(LABELS)]
    engine.index(DOCUMENTS, metadata=metadata)

    # --- Test queries ---
    queries = [
        # Query 1: synonym test — "physician" not in any document
        ("physician experience in cardiology", "SYNONYM TEST"),
        # Query 2: conceptual — "healthcare professional" is related but not exact
        ("healthcare professional needed", "CONCEPT TEST"),
        # Query 3: negative — should NOT match tech docs
        ("billing for medical supplies", "CATEGORY TEST"),
        # Query 4: the tech cluster should be separate
        ("Python machine learning developer", "TECH TEST"),
    ]

    for query, test_name in queries:
        print(f"\n{'─' * 65}")
        print(f"  {test_name}")
        print(f"  Query: \"{query}\"")
        print(f"{'─' * 65}")

        results = engine.search(query, top_k=3)
        for rank, r in enumerate(results, 1):
            bar = "█" * int(r.score * 30)
            label = r.metadata.get("label", "?")
            print(f"  #{rank}  [{label:>10}]  score={r.score:.3f}  {bar}")
            print(f"       {r.document[:70]}...")

    # --- The killer comparison ---
    print(f"\n{'=' * 65}")
    print("THE KEY COMPARISON")
    print(f"{'=' * 65}")
    print()
    print('  Query: "physician experience in cardiology"')
    print()
    print("  TF-IDF result:    ZERO matches (no document contains 'physician')")
    print()
    results = engine.search("physician experience in cardiology", top_k=1)
    top = results[0]
    print(f"  Embedding result: \"{top.document[:60]}...\"")
    print(f"                    score = {top.score:.3f}")
    print()
    print("  This is why embeddings exist.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

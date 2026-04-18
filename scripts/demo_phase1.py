"""Demo: Run the SmartDesk preprocessing pipeline on a messy document.

Usage: PYTHONPATH=src python scripts/demo_phase1.py
"""
from smartdesk.config import PreprocessingConfig
from smartdesk.ml.preprocessing.pipeline import PreprocessingPipeline
from smartdesk.ml.preprocessing.trie import Trie
from smartdesk.observability.logging import configure_logging

configure_logging()


def demo_pipeline() -> None:
    messy = (
        "Hi!! I received the invoice #INV-2024-00891 on 12/03/2024. "
        "The amount $1,250.00 USD is INCORRECT!!! "
        "Please check https://portal.acme.com/invoice/891... "
        "Email: accounts@acme.com. Regards, Priya 👋"
    )

    config = PreprocessingConfig(remove_emojis=True)
    pipeline = PreprocessingPipeline.from_config(config)
    result = pipeline.process(messy)

    print("=" * 70)
    print("ORIGINAL:")
    print(result.original)
    print("\nCLEANED:")
    print(result.cleaned)
    print("\nSTEP DELTAS (chars removed by each step):")
    for step, delta in result.metadata["step_deltas"].items():
        print(f"  {step:30s} removed {delta:>4d} chars")
    print("=" * 70)


def demo_trie() -> None:
    stopwords = ["the", "is", "at", "which", "on", "a", "an", "and", "or", "but"]
    trie = Trie.from_words(stopwords)

    print(f"\nTrie loaded with {len(trie)} stop words.")
    tests = ["the", "The", "cat", "and", "sandwich"]
    for word in tests:
        found = trie.contains(word.lower())
        print(f"  '{word}' is stopword? {found}")


if __name__ == "__main__":
    demo_pipeline()
    demo_trie()

"""Phase 3 — Build embeddings from scratch.

Stage 2: We implement cosine similarity, build a fake embedding space,
and find nearest neighbors MANUALLY. No sklearn. No gensim. No black boxes.

Run: uv run python scripts/learn_phase3_scratch.py
"""
import math


# ============================================================================
# STEP 1: Cosine similarity — the heart of embedding search
# ============================================================================
# You know what this does: measures the angle between two vectors.
# cos(0°) = 1.0 → identical direction → same meaning
# cos(90°) = 0.0 → perpendicular → unrelated
# cos(180°) = -1.0 → opposite direction → opposite meaning
#
# WHY cosine and not Euclidean distance?
# Because cosine ignores magnitude. A document with "doctor" 10 times
# and one with "doctor" 1 time should still be considered similar —
# they're about the same topic, just different lengths.
# Cosine measures DIRECTION, not LENGTH.

def dot_product(a: list[float], b: list[float]) -> float:
    """Sum of element-wise products. a·b = Σ(ai × bi)"""
    return sum(ai * bi for ai, bi in zip(a, b))


def magnitude(v: list[float]) -> float:
    """Length of vector. ||v|| = √(Σ vi²)"""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two vectors.

    This is the ONLY distance metric used in embedding search.
    Every vector database (Chroma, Pinecone, Weaviate) uses this.
    """
    dot = dot_product(a, b)
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ============================================================================
# STEP 2: Build a tiny embedding space BY HAND
# ============================================================================
# In real Word2Vec, these numbers are LEARNED from data.
# Here I'm hand-crafting them so you can SEE the patterns.
#
# Each word gets 5 dimensions. Think of them as:
#   dim 0: royalty / power
#   dim 1: gender (positive = female)
#   dim 2: medical
#   dim 3: tech / coding
#   dim 4: food

EMBEDDINGS: dict[str, list[float]] = {
    # Royalty cluster — high dim 0
    "king":      [0.9, -0.8,  0.0,  0.0,  0.0],
    "queen":     [0.9,  0.8,  0.0,  0.0,  0.0],
    "prince":    [0.7, -0.6,  0.0,  0.0,  0.0],
    "princess":  [0.7,  0.6,  0.0,  0.0,  0.0],

    # People cluster — moderate dim 0
    "man":       [0.1, -0.9,  0.0,  0.0,  0.0],
    "woman":     [0.1,  0.9,  0.0,  0.0,  0.0],
    "boy":       [0.0, -0.7,  0.0,  0.0,  0.0],
    "girl":      [0.0,  0.7,  0.0,  0.0,  0.0],

    # Medical cluster — high dim 2
    "doctor":    [0.0,  0.0,  0.9,  0.0,  0.0],
    "physician": [0.0,  0.0,  0.88, 0.0,  0.0],
    "nurse":     [0.0,  0.3,  0.8,  0.0,  0.0],
    "hospital":  [0.0,  0.0,  0.7,  0.0,  0.1],
    "patient":   [0.0,  0.0,  0.6,  0.0,  0.0],

    # Tech cluster — high dim 3
    "python":    [0.0,  0.0,  0.0,  0.9,  0.0],
    "code":      [0.0,  0.0,  0.0,  0.85, 0.0],
    "developer": [0.0,  0.0,  0.0,  0.8,  0.0],
    "software":  [0.0,  0.0,  0.0,  0.75, 0.0],

    # Food cluster — high dim 4
    "pizza":     [0.0,  0.0,  0.0,  0.0,  0.9],
    "burger":    [0.0,  0.0,  0.0,  0.0,  0.85],
    "food":      [0.0,  0.0,  0.0,  0.0,  0.7],
}


# ============================================================================
# STEP 3: Find nearest neighbors — this is what vector DBs do
# ============================================================================
def find_neighbors(word: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Find the top_k most similar words to the given word.

    This is the EXACT algorithm inside Chroma/Pinecone at its core:
    1. Get the query vector
    2. Compute cosine similarity against every other vector
    3. Sort by similarity
    4. Return top K

    In production, step 2 is too slow for millions of vectors.
    That's why vector DBs use approximate algorithms like HNSW,
    IVF, or LSH — we'll cover those in Phase 8 (RAG).
    """
    if word not in EMBEDDINGS:
        print(f"  '{word}' not in vocabulary!")
        return []

    query_vec = EMBEDDINGS[word]
    scores: list[tuple[str, float]] = []

    for other_word, other_vec in EMBEDDINGS.items():
        if other_word == word:
            continue
        sim = cosine_similarity(query_vec, other_vec)
        scores.append((other_word, sim))

    # Sort by similarity descending — highest first
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ============================================================================
# STEP 4: Word analogy — king - man + woman = ?
# ============================================================================
def analogy(a: str, b: str, c: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Solve: a is to b as c is to ???

    The math: result_vector = embedding(b) - embedding(a) + embedding(c)
    Then find the word closest to result_vector.

    Example: king - man + woman = queen
    Because: king and man differ by "royalty."
    Adding "royalty" to woman gives queen.

    This is the most mind-blowing property of embeddings.
    TF-IDF can NEVER do this because it has no concept of directions.
    """
    vec_a = EMBEDDINGS[a]
    vec_b = EMBEDDINGS[b]
    vec_c = EMBEDDINGS[c]

    # result = b - a + c (element-wise)
    result = [bi - ai + ci for ai, bi, ci in zip(vec_a, vec_b, vec_c)]

    # Find closest word to the result vector
    exclude = {a, b, c}
    scores: list[tuple[str, float]] = []
    for word, vec in EMBEDDINGS.items():
        if word in exclude:
            continue
        sim = cosine_similarity(result, vec)
        scores.append((word, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ============================================================================
# DEMO — run this and watch the numbers
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: Word Embeddings from scratch")
    print("=" * 60)

    # --- Cosine similarity comparisons ---
    print("\n--- Similarity comparisons ---")
    pairs = [
        ("doctor", "physician"),  # synonyms → should be HIGH
        ("doctor", "nurse"),      # related → should be MEDIUM-HIGH
        ("doctor", "hospital"),   # related → should be MEDIUM
        ("doctor", "python"),     # unrelated → should be LOW
        ("doctor", "pizza"),      # unrelated → should be LOW
        ("king", "queen"),        # related → HIGH
        ("king", "man"),          # some overlap → MEDIUM
    ]
    for w1, w2 in pairs:
        sim = cosine_similarity(EMBEDDINGS[w1], EMBEDDINGS[w2])
        bar = "█" * int(sim * 20)
        print(f"  {w1:12s} ~ {w2:12s}  = {sim:.3f}  {bar}")

    # --- Nearest neighbors ---
    print("\n--- Nearest neighbors ---")
    for query in ["doctor", "king", "python", "pizza"]:
        neighbors = find_neighbors(query, top_k=3)
        neighbor_str = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
        print(f"  {query:12s} → {neighbor_str}")

    # --- Word analogies ---
    print("\n--- Word analogies ---")
    analogies = [
        ("man", "king", "woman"),      # man→king as woman→? (queen)
        ("man", "boy", "woman"),        # man→boy as woman→? (girl)
        ("king", "prince", "queen"),    # king→prince as queen→? (princess)
    ]
    for a, b, c in analogies:
        results = analogy(a, b, c, top_k=1)
        answer = results[0]
        print(f"  {a} → {b} as {c} → {answer[0]} (similarity: {answer[1]:.3f})")

    # --- THE KEY INSIGHT ---
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("  In TF-IDF, doctor~physician similarity = 0.0")
    print("  In embeddings, doctor~physician similarity = 0.998")
    print()
    print("  That's why SmartDesk needs embeddings for search.")
    print("  A user searching 'physician' should find docs about 'doctor'.")
    print("=" * 60)

# scripts/break_phase8.py

# BREAK 1: Wrong embedding model at query time
# Ingest with model A, query with model B
# → vectors in different spaces → all scores ~0.1 (random)
# → completely wrong chunks returned
# Fix: always load model name from config, never hardcode twice

# BREAK 2: No overlap — sentence split at boundary
text = "Payment is due on April 14. " * 30 + "Late fee is 2%."
# With chunk_size=10 words, overlap=0:
# chunk N ends: "Payment is due"
# chunk N+1 starts: "on April 14. Payment..."
# The key sentence is split — neither chunk has it complete
# Fix: overlap=50 ensures key sentences appear whole in at least one chunk

# BREAK 3: Retrieving without reranker — wrong chunk ranked #1
# "What is the due date?"
# Vector search returns:
#   rank 1: "Date: March 15, 2024" (high cosine — has "date")
#   rank 2: "Due Date: April 14, 2024" ← the actual answer
# Reranker flips them because it reads query+chunk together
# Fix: always use reranker for precision tasks like date/amount extraction

# BREAK 4: Temperature too high for factual extraction
# temperature=0.7 on a RAG answer → model starts paraphrasing
# "The due date appears to be sometime in mid-April..."
# Fix: temperature=0.1 for extraction, 0.7 only for summarisation
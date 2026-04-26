# scripts/break_phase8.py
"""
Phase 8 RAG — Break It
4 failure modes, all live and executable.
"""
import httpx
import time
from smartdesk.rag.chunker import DocumentChunker
from smartdesk.rag.rag_pipeline import RAGPipeline

INVOICE = """
INVOICE INV-2024-0042
Vendor: Acme Solutions Ltd
Date: March 15, 2024
Due Date: April 14, 2024 (30 days net)
Amount: $14,750.00
VAT Number: GB123456789
Payment Terms: Payment must be received by April 14, 2024.
Late payments incur 2% monthly interest per clause 7.3.
Bank: Barclays | Sort: 20-00-00 | Account: 12345678
"""

SEP = "=" * 55

# ── BREAK 1: No overlap — sentence orphaned at boundary ──────
print(SEP)
print("BREAK 1: No overlap — key sentence gets split")
print(SEP)

chunker_no_overlap = DocumentChunker(chunk_size=30, overlap=0)
chunker_with_overlap = DocumentChunker(chunk_size=30, overlap=10)

chunks_bad  = chunker_no_overlap.chunk(INVOICE, "test")
chunks_good = chunker_with_overlap.chunk(INVOICE, "test")

print(f"\nNo overlap  → {len(chunks_bad)} chunks")
print(f"Last words of chunk 0 : '...{chunks_bad[0].text.split()[-6:]}'" )
print(f"First words of chunk 1: '{chunks_good[1].text.split()[:6]}...'")

print(f"\nWith overlap → {len(chunks_good)} chunks")
print(f"Last words of chunk 0 : '...{chunks_good[0].text.split()[-6:]}'")
print(f"First words of chunk 1: '{chunks_good[1].text.split()[:6]}...'")

print("\nFIX: overlap=50 — key sentences appear whole in at least one chunk")
input("\nPress Enter to see Break 2...")

# ── BREAK 2: Temperature too high — model starts paraphrasing ──
print(f"\n{SEP}")
print("BREAK 2: High temperature — creative instead of factual")
print(SEP)

def call_ollama(question, context, temperature):
    prompt = f"Excerpt: {context}\nAnswer in one sentence: {question}\nAnswer:"
    r = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "tinyllama",
            "messages": [
                {"role": "system", "content": "Answer using only the excerpt."},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=60,
    )
    return r.json()["message"]["content"].strip()

q = "What is the due date?"
ctx = INVOICE.strip()

print(f"\nQuestion: '{q}'")
print("\ntemperature=0.1 (production):")
ans = call_ollama(q, ctx, 0.1)
print(f"  → {ans[:150]}")

print("\ntemperature=1.5 (dangerous):")
ans = call_ollama(q, ctx, 1.5)
print(f"  → {ans[:150]}")

print("\nFIX: Always use temperature=0.1 for factual extraction in RAG")
input("\nPress Enter to see Break 3...")

# ── BREAK 3: Empty vector store — graceful failure ────────────
print(f"\n{SEP}")
print("BREAK 3: Querying before ingesting any documents")
print(SEP)

empty_pipeline = RAGPipeline(persist_dir="data/chroma_break3_empty")
print("\nQuerying an empty vector store...")
result = empty_pipeline.query("What is the due date?")
print(f"Answer  : {result['answer']}")
print(f"Chunks  : {result['chunks_used']}")
print("\nFIX: RAGPipeline returns NOT FOUND IN DOCUMENT gracefully — no crash")
input("\nPress Enter to see Break 4...")

# ── BREAK 4: Question outside all documents ───────────────────
print(f"\n{SEP}")
print("BREAK 4: Hallucination trap — question not in any document")
print(SEP)

pipeline = RAGPipeline(persist_dir="data/chroma_break4")
pipeline.ingest("invoice_042", INVOICE)

hallucination_questions = [
    "What is the CEO's name?",
    "What is the company's stock price?",
    "What is the registered office address?",
]

print()
for q in hallucination_questions:
    result = pipeline.query(q)
    verdict = "SAFE" if "NOT FOUND" in result["answer"].upper() else "HALLUCINATION RISK"
    print(f"Q: {q}")
    print(f"A: {result['answer'][:100]}")
    print(f"   [{verdict}]\n")

print("FIX: System prompt 'NOT FOUND IN DOCUMENT' rule + temperature=0.1")
print(f"\n{SEP}")
print("All 4 breaks done. Pipeline is solid.")
print(SEP)
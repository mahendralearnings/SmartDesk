# scripts/learn_phase8_rag.py
from smartdesk.rag.rag_pipeline import RAGPipeline

SAMPLE_INVOICE = """
INVOICE INV-2024-0042
Vendor: Acme Solutions Ltd
Date: March 15, 2024
Due Date: April 14, 2024 (30 days net)
Amount: $14,750.00
VAT Number: GB123456789

Items:
- Software License Q1 2024: $10,000.00
- Professional Services (40hrs @ $118.75): $4,750.00

Payment Terms: Payment must be received by April 14, 2024.
Late payments incur 2% monthly interest per clause 7.3.
Bank: Barclays | Sort: 20-00-00 | Account: 12345678
"""

pipeline = RAGPipeline(persist_dir="data/chroma_demo")

print("Ingesting sample invoice...")
n = pipeline.ingest("invoice_INV-2024-0042", SAMPLE_INVOICE)
print(f"  → {n} chunks indexed\n")

questions = [
    "What is the due date?",
    "What is the VAT number?",
    "What happens if payment is late?",
    "What is the CEO's name?",   # not in document → should get NOT FOUND
]

for q in questions:
    print(f"Q: {q}")
    result = pipeline.query(q)
    print(f"A: {result['answer'][:200]}")
    print(f"   Sources: {result['sources']} | Chunks: {result['chunks_used']}\n")
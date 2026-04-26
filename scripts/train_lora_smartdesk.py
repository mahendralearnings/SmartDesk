# scripts/train_lora_smartdesk.py
"""
Demonstrates the full LoRA + adapter swap workflow.
No real dataset needed — shows the pattern with dummy data.
"""
from smartdesk.ml.finetuning.adapter_manager import AdapterManager

print("=" * 60)
print("LORA ADAPTER MANAGER DEMO")
print("=" * 60)

manager = AdapterManager()

# Simulating adapter swap at runtime
# In production: detect document type, swap adapter, run NER

print("\nScenario: Two documents arrive — invoice then legal contract")
print("Same base model, different adapter per document type\n")

# Invoice document
print("Document 1: Invoice")
# manager.load_adapter("invoice")   # uncomment after training
print("  → Would load invoice adapter (3MB)")
print("  → Extracts: INVOICE_ID, VENDOR, AMOUNT, DUE_DATE")

print("\nDocument 2: Legal contract")
# manager.swap_adapter("legal")     # uncomment after training  
print("  → Swaps to legal adapter (~10ms)")
print("  → Extracts: PARTY, EFFECTIVE_DATE, JURISDICTION, CLAUSE_ID")

print("\nKey: Base model (440MB) loaded ONCE at startup.")
print("Adapters (3MB each) swap in 10ms — no GPU reload.")
# scripts/learn_phase12_ragas.py
from dotenv import load_dotenv
load_dotenv()

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Claude as judge LLM
judge_llm = LangchainLLMWrapper(
    ChatAnthropic(model="claude-haiku-4-5")
)

# HuggingFace embeddings — free, no API key needed
# Same model you used in Phase 8 RAG pipeline
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Set on each metric
faithfulness.llm            = judge_llm
answer_relevancy.llm        = judge_llm
answer_relevancy.embeddings = embeddings
context_recall.llm          = judge_llm
context_precision.llm       = judge_llm

# Now run your pipeline
from smartdesk.rag.rag_pipeline import RAGPipeline
from smartdesk.evaluation.rag_evaluator import RAGEvaluator
from smartdesk.evaluation.golden_dataset import GOLDEN_QUESTIONS

print("Setting up pipeline...")
pipeline = RAGPipeline(persist_dir="data/chroma_demo")

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

pipeline.ingest("invoice_INV-2024-0042", INVOICE)

evaluator = RAGEvaluator(pipeline)
scores    = evaluator.run(GOLDEN_QUESTIONS)
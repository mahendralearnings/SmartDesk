# SmartDesk — Enterprise Document Intelligence Platform

An end-to-end AI/ML project built layer by layer, from classical NLP to Agentic AI. Each phase adds a new capability to a single product — exactly how the industry evolved.

## What is this?

SmartDesk is a document intelligence platform that reads, classifies, searches, and answers questions about enterprise documents (invoices, complaints, resumes, contracts). It's built as a learning project that covers the **full modern AI stack**:

| Phase | Era | What SmartDesk gains | Status |
|-------|-----|----------------------|--------|
| 0 | System design | Architecture, project skeleton, CI/CD | Done |
| 1 | Text basics | Document ingestion and cleaning | Done |
| 2 | Classical ML | Auto-classify documents (NB, LogReg, SVM) | Done |
| 3 | Word embeddings | Semantic search v1 (Word2Vec, FAISS) | In progress |
| 4 | Sequence models | Entity extraction (RNN, LSTM) | Planned |
| 5 | Transformers | Upgraded summarizer, attention mechanism | Planned |
| 6 | BERT era | Better classifier + extractive QA | Planned |
| 7 | GPT / LLMs | Chat interface, prompting | Planned |
| 7.5 | Fine-tuning | LoRA, QLoRA on domain data | Planned |
| 8 | RAG | Chat over your documents | Planned |
| 9 | Agents | SmartDesk sends emails, updates tickets | Planned |
| 10 | Multi-agent | Researcher + Writer + Reviewer collaborate | Planned |
| 11 | MCP | Connect to Gmail, Jira, Slack | Planned |
| 12 | Enterprise | Docker, K8s, eval harness, observability | Planned |

## Why this project?

Most tutorials teach concepts in isolation. This project connects them:

- Phase 1's preprocessing feeds into Phase 2's classifiers
- Phase 2's TF-IDF limitations motivate Phase 3's embeddings
- Phase 3's static embeddings motivate Phase 5's Transformers
- Each phase keeps working code from previous phases

One repo. One product. The full AI/ML journey.

## Tech stack

- **Language**: Python 3.12+
- **Package manager**: [uv](https://docs.astral.sh/uv/) (replaces pip/poetry)
- **API**: FastAPI + Pydantic v2
- **ML**: scikit-learn, PyTorch, Hugging Face Transformers
- **Logging**: structlog (JSON in prod, pretty in dev)
- **Testing**: pytest with coverage
- **Linting**: ruff (replaces black + flake8 + isort)
- **CI/CD**: GitHub Actions
- **Containerization**: Docker with multi-stage builds

## Quick start

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/YOUR_USERNAME/smartdesk.git
cd smartdesk
make setup    # installs Python 3.12 + all dependencies

# Run tests
make test

# Run demos
make demo-phase1   # text preprocessing pipeline
make demo-phase2   # document classification
```

## Project structure

```
smartdesk/
├── src/smartdesk/
│   ├── api/                 # FastAPI routes
│   ├── core/                # Domain models, business logic
│   ├── ml/
│   │   ├── preprocessing/   # Phase 1: tokenization, cleaning, Trie
│   │   ├── classifiers/     # Phase 2: NB, LogReg, SVM, TF-IDF
│   │   ├── embeddings/      # Phase 3: Word2Vec, semantic search
│   │   └── ...              # Future phases
│   ├── infrastructure/      # DB, vector store, LLM adapters
│   ├── observability/       # Structured logging, tracing
│   ├── security/            # Auth, guardrails, PII redaction
│   └── evaluation/          # Eval harness, metrics
├── tests/                   # Unit + integration + eval tests
├── scripts/                 # Demo scripts per phase
├── configs/                 # YAML configs (dev/staging/prod)
├── docker/                  # Dockerfile + docker-compose
├── docs/                    # Architecture docs, uv guide
└── notebooks/               # Exploration only
```

## Design patterns used

- **Strategy pattern**: Swap preprocessing steps, vectorizers, classifiers independently
- **Factory pattern**: `create_vectorizer()`, `create_classifier()` — add new models without changing pipeline
- **Template method**: `train_and_evaluate()` — same evaluation flow for any model
- **Repository pattern**: Abstract data sources (coming in Phase 8)
- **Ports & Adapters**: Swap LLM providers without touching business logic (coming in Phase 7)

## Cross-cutting concerns

These aren't afterthoughts — they're applied from Phase 1:

- **Evaluation**: Precision, recall, F1, confusion matrix, cross-validation
- **Performance**: Pre-compiled regex, sparse matrices, Trie for O(L) lookup
- **Guardrails**: Type validation, config-driven toggles
- **Security**: PII redaction (emails, URLs), secrets in .env
- **Observability**: Structured JSON logging from day 1
- **Testing**: pytest for every module, 24+ tests and growing

## License

MIT

#!/usr/bin/env bash
# ============================================================================
# SmartDesk — Git Repository Setup Script
#
# DO NOT run this script directly. Read each section and run commands
# one at a time so you understand what each step does.
#
# Prerequisites:
#   - git installed
#   - GitHub account
#   - GitHub CLI (gh) installed: https://cli.github.com/
#     OR you can create the repo manually on github.com
# ============================================================================

echo "=== SmartDesk Git Setup Guide ==="
echo "Run these commands one at a time. Don't execute this script blindly."
echo ""

# ============================================================================
# STEP 1: Extract the project (you already downloaded the tarball)
# ============================================================================
# tar -xzf smartdesk_uv_migration.tar.gz
# cd smartdesk

# ============================================================================
# STEP 2: Install uv (if you haven't already)
# ============================================================================
# curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Verify:
# uv --version

# ============================================================================
# STEP 3: Initialize git and set up the repo
# ============================================================================
# git init
# git branch -M main

# ============================================================================
# STEP 4: Create commits that tell a STORY
#
# Why separate commits? When an interviewer looks at your repo, they see
# your commit history. Clean commits show you think in layers, not chaos.
# Each commit message follows Conventional Commits (an industry standard).
# ============================================================================

# --- Commit 1: Project foundation ---
# This shows you think about infrastructure BEFORE code.
#
# git add pyproject.toml Makefile .gitignore .python-version .env.example
# git add .pre-commit-config.yaml
# git add .github/
# git add docker/
# git add docs/uv-guide.md docs/adr/
# git add README.md
# git add configs/
#
# git commit -m "chore: initialize project with uv, Docker, CI/CD
#
# - Enterprise src/ layout following Python packaging standards
# - uv for dependency management (replaces pip/poetry/pyenv)
# - Multi-stage Dockerfile with non-root user
# - GitHub Actions CI: lint, typecheck, test, docker build
# - Makefile for developer experience
# - ADR-001: rationale for choosing uv
# - Pre-commit hooks with ruff"

# --- Commit 2: Observability layer ---
# Logging before features. This is what senior engineers do.
#
# git add src/smartdesk/__init__.py
# git add src/smartdesk/config.py
# git add src/smartdesk/observability/
#
# git commit -m "feat: add config management and structured logging
#
# - Pydantic Settings for typed, validated configuration
# - structlog for structured JSON logging (prod) / pretty console (dev)
# - Config singleton with lru_cache
# - Environment-specific overrides via .env files"

# --- Commit 3: Phase 1 — Preprocessing ---
#
# git add src/smartdesk/ml/__init__.py
# git add src/smartdesk/ml/preprocessing/
# git add tests/__init__.py tests/unit/__init__.py
# git add tests/unit/test_preprocessing.py
# git add scripts/demo_phase1.py
#
# git commit -m "feat(phase1): text preprocessing pipeline with Trie
#
# - Strategy pattern: composable, toggleable preprocessing steps
# - Steps: encoding fix, HTML strip, URL/email/emoji removal,
#   whitespace normalization, case folding
# - Trie data structure for O(word_length) stop-word lookup
# - Config-driven step toggling via Pydantic
# - 15 unit tests covering individual steps, integration, and Trie
#
# Algorithms: regex FSM, hashmap (Counter), Trie (prefix tree)
# Patterns: Strategy, Factory Method, Protocol-based interfaces"

# --- Commit 4: Phase 2 — Classical ML ---
#
# git add src/smartdesk/ml/classifiers/
# git add tests/unit/test_classifiers.py
# git add scripts/demo_phase2.py
#
# git commit -m "feat(phase2): document classification — NB, LogReg, SVM
#
# - TF-IDF vectorization with n-gram support (uni + bigrams)
# - Three classifiers: Naive Bayes, Logistic Regression, LinearSVC
# - Factory pattern for classifier and vectorizer creation
# - Full evaluation suite: accuracy, precision, recall, F1 (macro),
#   cross-validated F1, confusion matrix, classification report
# - Model serialization with joblib for production deployment
# - 9 additional tests (24 total, all passing)
#
# Algorithms: Bayes theorem, gradient descent, Laplace smoothing,
#             sigmoid/softmax, convex optimization (SVM)
# Patterns: Strategy, Factory, Template Method"

# ============================================================================
# STEP 5: Create the GitHub repo and push
# ============================================================================

# Option A: Using GitHub CLI (recommended)
# gh repo create smartdesk --public --description "Enterprise Document Intelligence — from regex to Agentic AI" --source=. --push

# Option B: Manual
# 1. Go to github.com → New Repository → name: smartdesk → Public → Create
# 2. Run:
# git remote add origin https://github.com/YOUR_USERNAME/smartdesk.git
# git push -u origin main

# ============================================================================
# STEP 6: Set up the development environment
# ============================================================================
# make setup
#
# This single command:
#   1. Installs Python 3.12 via uv
#   2. Pins the version (.python-version)
#   3. Creates .venv and installs all dependencies
#   4. Sets up pre-commit hooks
#
# Verify everything works:
# make test

# ============================================================================
# STEP 7: Set up branch protection (optional but professional)
# ============================================================================
# On GitHub: Settings → Branches → Add rule
#   Branch name pattern: main
#   Check: "Require status checks to pass before merging"
#   Check: "Require pull request reviews before merging"
#
# This means you'll work on feature branches:
# git checkout -b phase3/word-embeddings
# ... make changes ...
# git add . && git commit -m "feat(phase3): word embeddings and semantic search"
# git push -u origin phase3/word-embeddings
# ... create PR on GitHub ... review ... merge ...

# ============================================================================
# STEP 8: Verify your repo looks professional
# ============================================================================
# Check these things on github.com:
#   [ ] README renders nicely with the table and code blocks
#   [ ] Commit history shows 4 clean commits with descriptive messages
#   [ ] .github/workflows/ci.yml is visible (Actions tab should show)
#   [ ] No .venv, __pycache__, or .env files committed
#   [ ] Folder structure is clean (src/smartdesk/..., tests/...)

echo ""
echo "=== Done! Your SmartDesk repo is ready. ==="
echo "Share the URL: https://github.com/YOUR_USERNAME/smartdesk"

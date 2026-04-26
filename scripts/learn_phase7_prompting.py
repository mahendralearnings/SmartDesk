"""
Phase 7 — Prompting patterns for SmartDesk.

Learning path:
  1. Zero-shot   — baseline, no examples
  2. Few-shot    — examples improve accuracy dramatically
  3. CoT         — step-by-step reasoning reduces hallucination
  4. System prompt — persona + constraints + format
  5. Output parsing — structured JSON from LLM

WHY this file exists:
  Prompting is not guessing. Each pattern solves a specific
  failure mode. This file shows exactly which pattern to reach
  for and why.
"""

# ============================================================
# SETUP — works with OpenAI OR local Ollama (free)
# ============================================================

"""
Two options — pick one:

Option A (OpenAI API — paid):
  pip install openai
  Set OPENAI_API_KEY in your .env

Option B (Ollama — free, runs locally):
  Download: ollama.ai
  Run: ollama pull mistral
  Run: ollama serve
  No API key needed.

The prompting patterns are identical for both.
We abstract behind a single call_llm() function.
"""

import json
import os
from typing import Any

# ── LLM abstraction ───────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,   # 0.0 = deterministic, reproducible
    model: str = "tinyllama",     # swap to "gpt-4o" for OpenAI
) -> str:
    """Single function that works with Ollama (local) or OpenAI.

    WHY temperature=0.0 by default:
    For structured tasks (extraction, classification, parsing)
    you want the same answer every time. Temperature 0 picks
    the highest-probability token always — no randomness.

    Use temperature > 0 only for creative tasks (email drafting,
    summarisation where variety is acceptable).
    """
    try:
        # Try Ollama first (free, local)
        import requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    except Exception:
        # Fall back to OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


# ── Sample invoice ─────────────────────────────────────────────────────────────

SAMPLE_INVOICE = """
Invoice Number: INV-0042
Vendor: Acme Corp
Date Issued: 2024-01-15
Due Date: 2024-02-15
Amount Due: $4,500.00
Status: Unpaid

Line Items:
  - Software License Q1 2024: $3,000.00
  - Support Services Jan 2024: $1,500.00

Approved by: John Smith
"""


# ============================================================
# PATTERN 1 — Zero-shot
# ============================================================

print("=" * 55)
print("PATTERN 1: Zero-shot")
print("=" * 55)

"""
Zero-shot = give the task, no examples.

WHY use it: fastest to write, good for simple tasks the
model has seen billions of times (summarisation, translation).

WHY it fails: for domain-specific tasks (SmartDesk entity
types, custom output formats) the model guesses the format.
"""

zero_shot_prompt = f"""
Extract the following fields from this invoice:
- Invoice number
- Vendor name
- Amount due
- Due date
- Status

Invoice:
{SAMPLE_INVOICE}
"""

print("\nPrompt sent:")
print(zero_shot_prompt)
print("\nModel response:")
# response = call_llm(zero_shot_prompt)
# print(response)
print("[Run with Ollama or OpenAI to see live output]")
print("""
Expected output (approximate):
  Invoice number: INV-0042
  Vendor name: Acme Corp
  Amount due: $4,500.00
  Due date: 2024-02-15
  Status: Unpaid

PROBLEM: Format is inconsistent across runs. Sometimes
"Invoice number:", sometimes "Number:", sometimes JSON.
Hard to parse programmatically. → Fix with Few-shot.
""")


# ============================================================
# PATTERN 2 — Few-shot
# ============================================================

print("=" * 55)
print("PATTERN 2: Few-shot")
print("=" * 55)

"""
Few-shot = give 2-3 examples of input → output BEFORE
the actual task.

WHY it works: the model pattern-matches on your examples.
You're not explaining the format — you're demonstrating it.
3 examples is usually enough. More than 5 adds noise.

WHY it beats zero-shot for SmartDesk:
Our output format (pipe-separated, specific field names)
is not something the model has seen. Examples teach it.
"""

few_shot_prompt = """
Extract invoice fields in this exact format:
INVOICE_ID | VENDOR | AMOUNT | DUE_DATE | STATUS

Example 1:
Invoice: Invoice INV-1001 from TechCorp. Amount $2,000. Due 2024-03-01. Paid.
Output: INV-1001 | TechCorp | $2,000.00 | 2024-03-01 | Paid

Example 2:
Invoice: Invoice INV-0055 from GlobalTech Ltd. Amount $8,750. Due 2024-01-30. Unpaid.
Output: INV-0055 | GlobalTech Ltd | $8,750.00 | 2024-01-30 | Unpaid

Example 3:
Invoice: Invoice INV-0099 from BuildCo. Amount $550. Due 2024-04-15. Overdue.
Output: INV-0099 | BuildCo | $550.00 | 2024-04-15 | Overdue

Now extract from this invoice:
Invoice: Invoice INV-0042 from Acme Corp. Amount $4,500. Due 2024-02-15. Unpaid.
Output:"""

print("\nKey insight:")
print("""
  Zero-shot: "Extract these fields" → inconsistent format
  Few-shot:  3 examples shown      → model copies the format exactly

  Output will be: INV-0042 | Acme Corp | $4,500.00 | 2024-02-15 | Unpaid

  WHY this matters in production:
  Consistent format → reliable parsing → downstream code works.
  Inconsistent format → parsing fails → silent errors in production.
""")


# ============================================================
# PATTERN 3 — Chain-of-Thought (CoT)
# ============================================================

print("=" * 55)
print("PATTERN 3: Chain-of-Thought")
print("=" * 55)

"""
Chain-of-Thought = ask the model to think step by step
BEFORE giving the final answer.

WHY it reduces hallucination:
Without CoT, the model jumps to an answer immediately.
For complex reasoning (is this invoice overdue? by how many
days?) it skips steps and guesses.

With CoT, each reasoning step can be checked. Errors
surface in the chain before affecting the final answer.

Magic phrase: "Think step by step." or "Let's reason through this."
"""

today = "2024-03-01"

cot_prompt = f"""
Today's date is {today}.

Is invoice INV-0042 overdue? If so, by how many days?
Think step by step before giving your final answer.

Invoice details:
{SAMPLE_INVOICE}
"""

print("\nWithout CoT — model might say:")
print("  'Yes, it is overdue.'  ← no reasoning shown, hard to trust")

print("\nWith CoT — model reasons:")
print("""
  Step 1: Due date is 2024-02-15
  Step 2: Today is 2024-03-01
  Step 3: Days overdue = March 1 - Feb 15
           = 14 days in Feb remaining after 15th + 1 day in March
           = 13 + 1 = 14 days overdue
  Final answer: Yes, INV-0042 is overdue by 14 days.

WHY this is better:
  The reasoning is visible. If step 2 is wrong (wrong date)
  you can catch it. The final answer is traceable.
  This matters in finance — you can't just trust a number.
""")


# ============================================================
# PATTERN 4 — System prompt
# ============================================================

print("=" * 55)
print("PATTERN 4: System prompt")
print("=" * 55)

"""
System prompt = a hidden instruction that sets the model's
persona, constraints, and output rules BEFORE the conversation.

Three things every SmartDesk system prompt must include:
  1. ROLE    — who the model is
  2. RULES   — what it must/must not do
  3. FORMAT  — exact output structure

WHY format in system prompt beats format in every user prompt:
You write it once. Every conversation inherits it.
Changing the format means changing one place, not 1000 prompts.
"""

SMARTDESK_SYSTEM_PROMPT = """
You are SmartDesk Assistant, an enterprise document intelligence
system for Acme Corporation's accounts payable team.

ROLE:
  You extract, analyse, and answer questions about invoices,
  contracts, and financial documents.

RULES:
  1. Only answer questions about documents provided in the prompt.
     Never invent invoice numbers, amounts, or dates.
  2. If information is not in the document, say exactly:
     "This information is not available in the provided document."
  3. Never make payment decisions or recommendations.
     Only report facts from documents.
  4. All amounts must include currency symbol and two decimal places.
  5. All dates must be in YYYY-MM-DD format.

OUTPUT FORMAT:
  For extraction tasks: respond with valid JSON only.
  For questions: respond in 1-3 sentences maximum.
  Never include preamble like "Sure!" or "Great question!".
"""

user_message = f"""
Extract all entities from this invoice and return as JSON.

{SAMPLE_INVOICE}
"""

print("\nSystem prompt sets the rules once.")
print("User messages stay clean and simple.")
print("""
Expected JSON output:
{
  "invoice_id": "INV-0042",
  "vendor": "Acme Corp",
  "date_issued": "2024-01-15",
  "due_date": "2024-02-15",
  "amount_due": "$4,500.00",
  "status": "Unpaid",
  "line_items": [
    {"description": "Software License Q1 2024", "amount": "$3,000.00"},
    {"description": "Support Services Jan 2024", "amount": "$1,500.00"}
  ],
  "approved_by": "John Smith"
}

WHY JSON output matters:
  Your downstream Python code does json.loads(response)
  and gets a real dictionary. No string parsing. No regex.
  Type-safe. Reliable. Production-ready.
""")


# ============================================================
# PATTERN 5 — Output parsing + validation
# ============================================================

print("=" * 55)
print("PATTERN 5: Output parsing + validation")
print("=" * 55)

"""
LLMs sometimes return:
  - JSON wrapped in markdown: ```json { ... } ```
  - Trailing text after JSON: { ... } "Hope this helps!"
  - Invalid JSON: missing comma, trailing comma

Production code must handle all three.
"""

def parse_llm_json(raw_response: str) -> dict[str, Any]:
    """Safely extract JSON from LLM response.

    WHY not just json.loads(response):
    LLMs frequently wrap JSON in markdown code blocks.
    json.loads('```json {...} ```') raises JSONDecodeError.
    This function strips common wrappers before parsing.
    """
    text = raw_response.strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        # Remove opening ``` or ```json
        text = text.split("\n", 1)[-1]
        # Remove closing ```
        if text.endswith("```"):
            text = text[:-3].strip()

    # Find JSON object boundaries
    # WHY: sometimes model adds text after the closing brace
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {raw_response[:100]}")

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\nRaw: {json_str[:200]}")


def validate_invoice_json(data: dict) -> dict:
    """Validate required fields are present and correctly typed.

    WHY validate: LLMs occasionally miss a field or change
    field names ("invoice_number" vs "invoice_id"). Catching
    this early prevents silent errors downstream.
    """
    required_fields = [
        "invoice_id", "vendor", "due_date", "amount_due", "status"
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"LLM response missing fields: {missing}")

    # Validate amount format
    amount = data["amount_due"]
    if not amount.startswith("$"):
        raise ValueError(f"Amount must start with $, got: {amount}")

    return data


# Demo with a realistic LLM response (with markdown wrapper)
mock_llm_response = '''```json
{
  "invoice_id": "INV-0042",
  "vendor": "Acme Corp",
  "date_issued": "2024-01-15",
  "due_date": "2024-02-15",
  "amount_due": "$4,500.00",
  "status": "Unpaid",
  "approved_by": "John Smith"
}
```'''

parsed  = parse_llm_json(mock_llm_response)
valid   = validate_invoice_json(parsed)

print("\nRaw LLM response:")
print(mock_llm_response)
print("\nAfter parse_llm_json():")
print(json.dumps(parsed, indent=2))
print("\nValidation passed ✓")
print(f"\nNow usable in Python: invoice_id = {valid['invoice_id']}")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 55)
print("Summary — when to use each pattern")
print("=" * 55)
print("""
  Zero-shot    → simple tasks, standard formats, fast prototyping
  Few-shot     → custom formats, domain-specific output, consistency
  CoT          → multi-step reasoning, date calculations, logic
  System prompt→ production apps, persona, safety rules, output format
  Output parse → any structured output going into downstream code

SmartDesk uses ALL FIVE — layered:
  System prompt sets rules once
  Few-shot examples in user turn for format
  CoT for overdue calculations
  Output parsing for every extraction response
""")
# ```

# ---

# ### Run it

# ```powershell
# # Option A — Ollama (free, recommended)
# # 1. Download ollama.ai
# # 2. Run in terminal: ollama pull mistral
# # 3. Run: ollama serve
# uv run python scripts/learn_phase7_prompting.py

# # Option B — OpenAI
# # Add to .env: OPENAI_API_KEY=sk-...
# uv add openai --group ml
# uv run python scripts/learn_phase7_prompting.py
# ```

# ---

# ### Interview map — every pattern in one table

# | Pattern | Solves | Fails when |
# |---|---|---|
# | Zero-shot | Simple tasks, fast | Custom formats, domain-specific |
# | Few-shot | Consistent output format | Examples are bad quality |
# | Chain-of-Thought | Multi-step reasoning, calculations | Task is simple — adds noise |
# | System prompt | Production rules, persona, safety | Ignored on weak models |
# | Output parsing | Structured data into code | LLM changes schema unpredictably |

# ---

# ### తెలుగులో — 5 patterns summary

# **Zero-shot** — Example లేకుండా task ఇవ్వడం. Simple tasks కి OK. Custom formats కి fail.

# **Few-shot** — 2-3 examples చూపించడం. Model ఆ format copy చేస్తుంది. Consistent output కోసం best.

# **Chain-of-Thought** — "Step by step think" అని చెప్పడం. Complex reasoning లో hallucination తగ్గుతుంది.

# **System prompt** — Rules, persona, format ఒకసారి set చేయడం. Production apps లో mandatory.

# **Output parsing** — LLM response నుండి clean JSON extract చేయడం. Downstream code కి safe గా pass చేయడానికి.

# ---

# Stage 2 done. Stage 3 is the most important one before any interview — **hallucination, prompt injection, and token limits** live.

# Ready?
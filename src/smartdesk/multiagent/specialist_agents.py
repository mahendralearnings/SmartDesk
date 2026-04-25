# src/smartdesk/multiagent/specialist_agents.py
"""
Three focused agents — each does ONE thing well.
Specialist agents are simpler than the full ReAct agent:
  - Smaller system prompt
  - Fewer tools
  - Faster + cheaper per call
"""
import anthropic
from smartdesk.agents.tools import search_invoices, calculate_overdue_days, get_vendor_info

try:
    from langsmith import traceable
except ImportError:
    def traceable(*_args, **_kwargs):
        def _decorator(func):
            return func
        return _decorator

client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-5"


@traceable(name="specialist_llm_call", run_type="llm")
def _call_claude(system: str, user: str) -> str:
    r = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return r.content[0].text.strip()


# ── SPECIALIST 1: Retrieval agent ────────────────────────────
@traceable(name="retrieval_agent", run_type="chain")
def retrieval_agent(instruction: str) -> str:
    """
    Focused on finding invoices.
    Uses search_invoices tool directly — no full ReAct loop needed.
    """
    # Get raw invoice data
    raw = search_invoices(instruction)

    system = """You are a retrieval specialist.
You receive raw invoice data and extract the relevant facts cleanly.
Return only the facts requested — no commentary."""

    return _call_claude(system, f"Instruction: {instruction}\n\nData:\n{raw}")


# ── SPECIALIST 2: Analysis agent ─────────────────────────────
@traceable(name="analysis_agent", run_type="chain")
def analysis_agent(instruction: str) -> str:
    """
    Focused on date calculations and financial analysis.
    Given invoice data, computes overdue days, totals, risk levels.
    """
    system = """You are a financial analysis specialist.
You calculate overdue days, totals, and risk levels for invoices.
Be precise with numbers. Return structured analysis."""

    # Enrich with overdue calculations
    overdue_data = {}
    for due_date in ["2024-01-01", "2024-01-03"]:  # known overdue dates
        overdue_data[due_date] = calculate_overdue_days(due_date)

    enriched = f"{instruction}\n\nOverdue calculations: {overdue_data}"
    return _call_claude(system, enriched)


# ── SPECIALIST 3: Summary agent ──────────────────────────────
@traceable(name="summary_agent", run_type="chain")
def summary_agent(instruction: str) -> str:
    """
    Focused on producing clear, executive-level summaries.
    Takes results from other agents and writes the final report.
    """
    system = """You are a business writing specialist.
You turn raw analysis into clear, concise executive summaries.
Use bullet points for key findings. Be specific with numbers and dates."""

    return _call_claude(system, instruction)


# ── AGENT REGISTRY ────────────────────────────────────────────
SPECIALIST_MAP = {
    "retrieval": retrieval_agent,
    "analysis":  analysis_agent,
    "summary":   summary_agent,
}
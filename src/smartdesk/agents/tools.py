# src/smartdesk/agents/tools.py
"""
Tools are just Python functions.
The agent picks which one to call based on the description.
Description quality directly determines agent accuracy.
"""
from dataclasses import dataclass
from typing import Callable
from datetime import datetime, date
import json


@dataclass
class Tool:
    name: str
    description: str   # LLM reads this to decide when to use the tool
    func: Callable


# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────

def search_invoices(query: str) -> str:
    """
    Search SmartDesk invoice database.
    In production this calls RAGPipeline.query().
    Here we use a mock for clarity.
    """
    MOCK_INVOICES = [
        {"id": "INV-001", "vendor": "Acme Ltd",    "amount": 5000,  "due": "2024-01-01", "status": "overdue"},
        {"id": "INV-002", "vendor": "Beta Corp",   "amount": 12000, "due": "2024-04-30", "status": "pending"},
        {"id": "INV-003", "vendor": "Gamma Inc",   "amount": 3500,  "due": "2024-01-03", "status": "overdue"},
        {"id": "INV-004", "vendor": "Delta GmbH",  "amount": 8000,  "due": "2024-05-10", "status": "pending"},
    ]

    query_lower = query.lower()
    if "overdue" in query_lower:
        results = [i for i in MOCK_INVOICES if i["status"] == "overdue"]
    elif "pending" in query_lower:
        results = [i for i in MOCK_INVOICES if i["status"] == "pending"]
    else:
        results = MOCK_INVOICES

    return json.dumps(results, indent=2)


def calculate_overdue_days(due_date: str) -> str:
    """
    Calculate how many days overdue an invoice is.
    due_date format: YYYY-MM-DD
    """
    try:
        due = datetime.strptime(due_date, "%Y-%m-%d").date()
        today = date(2024, 3, 15)   # fixed for demo reproducibility
        delta = (today - due).days
        if delta > 0:
            return f"{delta} days overdue"
        elif delta == 0:
            return "Due today"
        else:
            return f"Due in {abs(delta)} days"
    except ValueError:
        return f"Invalid date format: {due_date}. Use YYYY-MM-DD."


def get_vendor_info(vendor_name: str) -> str:
    """Look up contact details for a vendor."""
    VENDORS = {
        "acme ltd":   {"email": "accounts@acme.com",  "contact": "John Smith",  "payment_terms": "30 days"},
        "gamma inc":  {"email": "finance@gamma.com",  "contact": "Sara Lee",    "payment_terms": "30 days"},
        "beta corp":  {"email": "billing@beta.com",   "contact": "Mike Jones",  "payment_terms": "60 days"},
        "delta gmbh": {"email": "rechnungen@delta.de","contact": "Hans Müller", "payment_terms": "45 days"},
    }
    info = VENDORS.get(vendor_name.lower())
    if not info:
        return f"Vendor '{vendor_name}' not found in database."
    return json.dumps(info, indent=2)


def calculate_total(invoice_ids: str) -> str:
    """
    Sum the amounts for given invoice IDs.
    invoice_ids: comma-separated string e.g. 'INV-001,INV-003'
    """
    AMOUNTS = {"INV-001": 5000, "INV-002": 12000, "INV-003": 3500, "INV-004": 8000}
    ids = [i.strip() for i in invoice_ids.split(",")]
    total = sum(AMOUNTS.get(i, 0) for i in ids)
    found = [i for i in ids if i in AMOUNTS]
    return f"Total for {found}: ${total:,}"


def final_answer(answer: str) -> str:
    """
    Call this when you have a complete answer for the user.
    This stops the agent loop.
    """
    return answer   # the loop checks for this tool name to exit


# ── TOOL REGISTRY ─────────────────────────────────────────────

TOOLS = [
    Tool(
        name="search_invoices",
        description="Search invoices by status (overdue/pending) or keyword. "
                    "Returns a list of matching invoices with amounts and due dates.",
        func=search_invoices,
    ),
    Tool(
        name="calculate_overdue_days",
        description="Calculate how many days overdue an invoice is. "
                    "Input: due date in YYYY-MM-DD format.",
        func=calculate_overdue_days,
    ),
    Tool(
        name="get_vendor_info",
        description="Get contact email and details for a vendor by name.",
        func=get_vendor_info,
    ),
    Tool(
        name="calculate_total",
        description="Sum invoice amounts for given invoice IDs. "
                    "Input: comma-separated IDs e.g. INV-001,INV-003",
        func=calculate_total,
    ),
    Tool(
        name="final_answer",
        description="Use this when you have a complete answer. "
                    "This ends the task. Input: your final answer as a string.",
        func=final_answer,
    ),
]

TOOL_MAP = {t.name: t for t in TOOLS}
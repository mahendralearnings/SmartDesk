# src/smartdesk/mcp/server.py
"""
SmartDesk MCP Server.
Exposes SmartDesk tools via the MCP protocol.
Any MCP-compatible agent can now use these tools
without knowing anything about our codebase.
"""
import json
import asyncio
from datetime import date, datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("smartdesk-mcp")

MOCK_INVOICES = [
    {"id": "INV-001", "vendor": "Acme Ltd",   "amount": 5000,  "due": "2024-01-01", "status": "overdue"},
    {"id": "INV-002", "vendor": "Beta Corp",   "amount": 12000, "due": "2024-04-30", "status": "pending"},
    {"id": "INV-003", "vendor": "Gamma Inc",   "amount": 3500,  "due": "2024-01-03", "status": "overdue"},
    {"id": "INV-004", "vendor": "Delta GmbH",  "amount": 8000,  "due": "2024-05-10", "status": "pending"},
]

VENDORS = {
    "acme ltd":   {"email": "accounts@acme.com",   "contact": "John Smith"},
    "gamma inc":  {"email": "finance@gamma.com",   "contact": "Sara Lee"},
    "beta corp":  {"email": "billing@beta.com",    "contact": "Mike Jones"},
    "delta gmbh": {"email": "rechnungen@delta.de", "contact": "Hans Müller"},
}


# ── PHASE 1: DISCOVERY ────────────────────────────────────────
# The agent calls tools/list to learn what this server can do.
# Descriptions are the most important part — the LLM reads
# these to decide which tool to call.

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_invoices",
            description=(
                "Search SmartDesk invoices by status or keyword. "
                "Use 'overdue' to find unpaid past-due invoices, "
                "'pending' for upcoming ones, or any vendor name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term: 'overdue', 'pending', or vendor name"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="calculate_overdue_days",
            description=(
                "Calculate how many days overdue an invoice is. "
                "Returns the number of days past the due date."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "due_date": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format"
                    }
                },
                "required": ["due_date"]
            }
        ),
        types.Tool(
            name="get_vendor_info",
            description=(
                "Get contact details for a vendor by name. "
                "Returns email address and contact person name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "vendor_name": {
                        "type": "string",
                        "description": "Vendor name, e.g. 'Acme Ltd'"
                    }
                },
                "required": ["vendor_name"]
            }
        ),
        types.Tool(
            name="calculate_total",
            description=(
                "Sum invoice amounts for a list of invoice IDs. "
                "Returns the total outstanding amount."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "invoice_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of invoice IDs, e.g. ['INV-001', 'INV-003']"
                    }
                },
                "required": ["invoice_ids"]
            }
        ),
    ]


# ── PHASE 2: INVOCATION ───────────────────────────────────────
# Agent calls tools/call with name + arguments.
# We route to the right function and return structured result.

@app.call_tool()
async def call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent]:

    if name == "search_invoices":
        result = _search_invoices(arguments["query"])

    elif name == "calculate_overdue_days":
        result = _calculate_overdue_days(arguments["due_date"])

    elif name == "get_vendor_info":
        result = _get_vendor_info(arguments["vendor_name"])

    elif name == "calculate_total":
        result = _calculate_total(arguments["invoice_ids"])

    else:
        result = f"Unknown tool: {name}"

    # MCP always returns content as a list of typed blocks
    # TextContent is the simplest — just a string result
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────

def _search_invoices(query: str) -> list[dict]:
    q = query.lower()
    if "overdue" in q:
        return [i for i in MOCK_INVOICES if i["status"] == "overdue"]
    elif "pending" in q:
        return [i for i in MOCK_INVOICES if i["status"] == "pending"]
    else:
        return [i for i in MOCK_INVOICES if q in i["vendor"].lower()]

def _calculate_overdue_days(due_date: str) -> dict:
    try:
        due = datetime.strptime(due_date, "%Y-%m-%d").date()
        today = date(2024, 3, 15)
        delta = (today - due).days
        return {
            "due_date": due_date,
            "days_overdue": delta if delta > 0 else 0,
            "status": "overdue" if delta > 0 else "not overdue"
        }
    except ValueError:
        return {"error": f"Invalid date: {due_date}. Use YYYY-MM-DD."}

def _get_vendor_info(vendor_name: str) -> dict:
    info = VENDORS.get(vendor_name.lower())
    if not info:
        return {"error": f"Vendor '{vendor_name}' not found."}
    return {"vendor": vendor_name, **info}

def _calculate_total(invoice_ids: list[str]) -> dict:
    amounts = {i["id"]: i["amount"] for i in MOCK_INVOICES}
    total = sum(amounts.get(id, 0) for id in invoice_ids)
    found = [id for id in invoice_ids if id in amounts]
    return {"invoice_ids": found, "total_amount": total}


# ── ENTRY POINT ───────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
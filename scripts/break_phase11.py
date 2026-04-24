# scripts/break_phase11.py
import asyncio
from dotenv import load_dotenv
load_dotenv()

SEP = "=" * 55

# BREAK 1: Wrong argument type — MCP validates schema
print(SEP)
print("BREAK 1: Wrong input type — MCP schema catches it")
print(SEP)
print("""
Tool schema says: invoice_ids must be an array of strings
If Claude sends:  {"invoice_ids": "INV-001"}  (string not array)
MCP validation:   rejects before tool even runs
Fix: inputSchema is the contract — define it precisely
""")

# BREAK 2: Tool name hallucination
print(SEP)
print("BREAK 2: Claude invents a tool that doesn't exist")
print(SEP)
print("""
Without MCP: Claude might output "Action: send_email" from Phase 9
             — our registry returns unknown tool error
With MCP:    Claude only sees tools in list_tools() response
             — it cannot call a tool that wasn't listed
Fix: MCP discovery prevents hallucinated tool names entirely
""")

# BREAK 3: Server not running
print(SEP)
print("BREAK 3: MCP server process fails to start")
print(SEP)

async def test_connection_failure():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    bad_params = StdioServerParameters(
        command="python",
        args=["-m", "smartdesk.mcp.nonexistent_server"],
    )
    try:
        async with stdio_client(bad_params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
    except Exception as e:
        print(f"Connection failed: {type(e).__name__}")
        print("Fix: wrap in try/except, return graceful error to user")

asyncio.run(test_connection_failure())

# BREAK 4: Missing required argument
print(f"\n{SEP}")
print("BREAK 4: Required argument missing from tool call")
print(SEP)
print("""
search_invoices requires: {"query": "..."}
If Claude omits it:       MCP raises validation error
Fix: mark all required fields in inputSchema "required" array
     Claude will always include them — it reads the schema
""")
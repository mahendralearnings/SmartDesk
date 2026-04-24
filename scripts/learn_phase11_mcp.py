# scripts/learn_phase11_mcp.py
from dotenv import load_dotenv
load_dotenv()

import asyncio
from smartdesk.mcp.client import run_mcp_agent

tasks = [
    "Find all overdue invoices and calculate the total outstanding amount.",
    "Which overdue invoice has been waiting the longest? How many days overdue is it?",
    "Get the contact email for the vendor with the largest overdue invoice.",
]

async def main():
    for task in tasks:
        await run_mcp_agent(task, verbose=True)
        input("\nPress Enter for next task...")

asyncio.run(main())
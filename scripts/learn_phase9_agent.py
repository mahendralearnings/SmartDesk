# scripts/learn_phase9_agent.py
import logging
logging.basicConfig(level=logging.WARNING)  # suppress httpx noise

from dotenv import load_dotenv
load_dotenv()  # ← loads .env before anything else

from smartdesk.agents.react_agent import ReActAgent

import os
if not os.getenv("LANGCHAIN_API_KEY"):
    raise RuntimeError("LANGCHAIN_API_KEY not set — tracing will be silent")

agent = ReActAgent(model="claude-sonnet-4-5")

tasks = [
    "Find all overdue invoices and tell me the total amount outstanding.",
    "Which overdue invoice has been waiting the longest? How many days?",
    "Get the contact email for Acme Ltd.",
]

for task in tasks:
    result = agent.run(task, verbose=True)
    input("\nPress Enter for next task...") 
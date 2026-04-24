# scripts/learn_phase9_agent.py
import logging
logging.basicConfig(level=logging.WARNING)  # suppress httpx noise

from dotenv import load_dotenv
load_dotenv()  # ← loads .env before anything else

from smartdesk.agents.react_agent import ReActAgent

agent = ReActAgent(model="claude-sonnet-4-5")

tasks = [
    "Find all overdue invoices and tell me the total amount outstanding.",
    "Which overdue invoice has been waiting the longest? How many days?",
    "Get the contact email for Acme Ltd.",
]

for task in tasks:
    result = agent.run(task, verbose=True)
    input("\nPress Enter for next task...")
# scripts/learn_phase10_multiagent.py
from dotenv import load_dotenv
load_dotenv()

from smartdesk.multiagent.orchestrator import Orchestrator

orchestrator = Orchestrator(verbose=True)

tasks = [
    "Audit all overdue invoices — find them, calculate how overdue each is, and write an executive summary.",
    "What is the total financial exposure from overdue invoices and which vendor owes the most?",
]

for task in tasks:
    result = orchestrator.run(task)
    input("\nPress Enter for next task...")
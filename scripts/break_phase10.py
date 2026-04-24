# scripts/break_phase10.py
from dotenv import load_dotenv
load_dotenv()

SEP = "=" * 55

# BREAK 1: Invalid agent name in plan → graceful fallback
print(SEP)
print("BREAK 1: Orchestrator plans with unknown agent")
print(SEP)
from smartdesk.multiagent.state import SubTask
from smartdesk.multiagent.specialist_agents import SPECIALIST_MAP

fake_subtask = SubTask(
    id="1", agent="email_sender",  # doesn't exist
    instruction="Send emails to all vendors",
    result=None, status="pending"
)
agent_fn = SPECIALIST_MAP.get(fake_subtask["agent"])
if not agent_fn:
    print(f"Agent '{fake_subtask['agent']}' not in registry.")
    print(f"Available: {list(SPECIALIST_MAP.keys())}")
    print("FIX: Orchestrator marks subtask as failed, continues with others")

input("\nPress Enter for Break 2...")

# BREAK 2: One specialist fails — others still complete
print(f"\n{SEP}")
print("BREAK 2: One agent fails — does orchestrator recover?")
print(SEP)

from smartdesk.multiagent.orchestrator import Orchestrator

class BrokenOrchestrator(Orchestrator):
    def _plan(self, task):
        # Inject a failing subtask
        from smartdesk.multiagent.state import SubTask
        return [
            SubTask(id="1", agent="retrieval", instruction=task,
                    result=None, status="pending"),
            SubTask(id="2", agent="broken_agent", instruction="This will fail",
                    result=None, status="pending"),
            SubTask(id="3", agent="summary", instruction=f"Summarise: {task}",
                    result=None, status="pending"),
        ]

orch = BrokenOrchestrator(verbose=True)
result = orch.run("Find overdue invoices")
print(f"\nResult despite broken agent: {result[:200]}")
print("FIX: Failed subtasks are skipped — synthesiser uses available results")

input("\nPress Enter for Break 3...")

# BREAK 3: No MAX_STEPS → orchestrator could loop forever
print(f"\n{SEP}")
print("BREAK 3: Orchestrator without step limit")
print(SEP)
print("If synthesiser is unsatisfied and re-plans indefinitely:")
print("  → Infinite LLM calls → infinite cost")
print("FIX: state['step'] counter + MAX_ORCHESTRATOR_STEPS = 3")
print("     If step > max: force final_answer from whatever is available")
# src/smartdesk/multiagent/state.py
"""
Shared state that flows between orchestrator and agents.
TypedDict gives type safety — catches bugs at definition time.
"""
from typing import TypedDict, Optional


class SubTask(TypedDict):
    id: str
    agent: str           # which specialist handles this
    instruction: str     # what to do
    result: Optional[str]  # filled in after execution
    status: str          # pending | running | done | failed


class OrchestratorState(TypedDict):
    task: str                    # original user task
    plan: list[SubTask]          # decomposed subtasks
    final_answer: Optional[str]  # synthesised result
    step: int                    # how many orchestrator loops ran
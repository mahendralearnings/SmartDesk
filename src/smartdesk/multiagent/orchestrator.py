# src/smartdesk/multiagent/orchestrator.py
"""
The orchestrator does three things:
  1. PLAN   — decompose the task into subtasks for specialists
  2. EXECUTE — run subtasks (sequentially here, async in production)
  3. SYNTHESISE — combine results into a final answer

Why not async here?
  asyncio adds complexity. We learn the pattern first,
  then Phase 12 Enterprise adds true parallelism.
"""
import json
import anthropic
from smartdesk.multiagent.state import OrchestratorState, SubTask
from smartdesk.multiagent.specialist_agents import SPECIALIST_MAP

try:
    from langsmith import traceable
except ImportError:
    def traceable(*_args, **_kwargs):
        def _decorator(func):
            return func
        return _decorator

client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-5"

PLANNER_SYSTEM = """You are an orchestrator for SmartDesk, an invoice intelligence system.
You decompose complex tasks into subtasks for specialist agents.

Available specialists:
- retrieval: finds and fetches invoice data from the database
- analysis: calculates overdue days, totals, risk levels
- summary: writes executive reports from analysis results

Respond ONLY with valid JSON — no markdown, no explanation:
{
  "subtasks": [
    {"id": "1", "agent": "retrieval", "instruction": "..."},
    {"id": "2", "agent": "analysis",  "instruction": "..."},
    {"id": "3", "agent": "summary",   "instruction": "..."}
  ]
}

Rules:
- Always end with a summary subtask
- Keep instructions specific and actionable
- 2-4 subtasks maximum"""

SYNTHESISER_SYSTEM = """You are the final synthesiser for SmartDesk.
Given results from specialist agents, produce a clean final answer.
Be concise. Lead with the most important finding."""


class Orchestrator:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @traceable(name="orchestrator_run", run_type="chain")
    def run(self, task: str) -> str:
        state: OrchestratorState = {
            "task": task,
            "plan": [],
            "final_answer": None,
            "step": 0,
        }

        self._log(f"\n{'='*55}")
        self._log(f"ORCHESTRATOR — TASK: {task}")
        self._log(f"{'='*55}")

        # ── PHASE 1: PLAN ──
        state["plan"] = self._plan(task)
        self._log(f"\nPLAN: {len(state['plan'])} subtasks")
        for st in state["plan"]:
            self._log(f"  [{st['id']}] {st['agent']} → {st['instruction'][:60]}")

        # ── PHASE 2: EXECUTE ──
        self._log("\nEXECUTING subtasks...")
        for subtask in state["plan"]:
            subtask["status"] = "running"
            self._log(f"\n  Running [{subtask['id']}] {subtask['agent']} agent...")

            agent_fn = SPECIALIST_MAP.get(subtask["agent"])
            if not agent_fn:
                subtask["result"] = f"Unknown agent: {subtask['agent']}"
                subtask["status"] = "failed"
                continue

            try:
                subtask["result"] = agent_fn(subtask["instruction"])
                subtask["status"] = "done"
                self._log(f"  Result: {subtask['result'][:120]}...")
            except Exception as e:
                subtask["result"] = f"Error: {e}"
                subtask["status"] = "failed"

        # ── PHASE 3: SYNTHESISE ──
        state["final_answer"] = self._synthesise(state)
        self._log(f"\n{'='*55}")
        self._log(f"FINAL ANSWER:\n{state['final_answer']}")
        self._log(f"{'='*55}")

        return state["final_answer"]

    @traceable(name="orchestrator_plan", run_type="chain")
    def _plan(self, task: str) -> list[SubTask]:
        """Ask Claude to decompose the task into subtasks."""
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": f"Task: {task}"}],
        )
        raw = response.content[0].text.strip()

        try:
            data = json.loads(raw)
            return [
                SubTask(
                    id=st["id"],
                    agent=st["agent"],
                    instruction=st["instruction"],
                    result=None,
                    status="pending",
                )
                for st in data["subtasks"]
            ]
        except json.JSONDecodeError:
            # Fallback plan if Claude doesn't return valid JSON
            return [
                SubTask(id="1", agent="retrieval", instruction=task,
                        result=None, status="pending"),
                SubTask(id="2", agent="summary",   instruction=f"Summarise: {task}",
                        result=None, status="pending"),
            ]

    @traceable(name="orchestrator_synthesise", run_type="chain")
    def _synthesise(self, state: OrchestratorState) -> str:
        """Combine all agent results into a final answer."""
        results_text = "\n\n".join(
            f"[{st['agent']} agent — subtask {st['id']}]:\n{st['result']}"
            for st in state["plan"]
            if st["status"] == "done"
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYNTHESISER_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Original task: {state['task']}\n\nAgent results:\n{results_text}"
            }],
        )
        return response.content[0].text.strip()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
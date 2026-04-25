# src/smartdesk/agents/react_agent.py
import httpx
import re
import logging
import os
from smartdesk.agents.tools import TOOLS, TOOL_MAP
from smartdesk.agents.prompt import build_system_prompt, build_user_prompt

try:
    from langsmith import traceable
except ImportError:
    def traceable(*_args, **_kwargs):
        def _decorator(func):
            return func
        return _decorator

logger = logging.getLogger(__name__)
MAX_STEPS = 8


class ReActAgent:
    """
    ReAct agent — supports tinyllama, Claude, or OpenAI.
    Switch model by passing model= parameter.
    """

    CLAUDE_MODELS = {"claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"}
    OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo"}

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.model = model
        self.system_prompt = build_system_prompt(TOOLS)
        self._setup_client()

    def _setup_client(self):
        if self.model in self.CLAUDE_MODELS:
            import anthropic
            self.client = anthropic.Anthropic()
            self.backend = "claude"
            print(f"Using Claude backend: {self.model}")

        elif self.model in self.OPENAI_MODELS:
            from openai import OpenAI
            self.client = OpenAI()
            self.backend = "openai"
            print(f"Using OpenAI backend: {self.model}")

        else:
            self.client = None
            self.backend = "ollama"
            print(f"Using Ollama backend: {self.model}")

    @traceable(name="react_agent_llm_call", run_type="llm")
    def _call_llm(self, user_prompt: str) -> str:
        if self.backend == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()

        elif self.backend == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        else:  # ollama / tinyllama
            r = httpx.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()

    @traceable(name="react_agent_run", run_type="chain")
    def run(self, task: str, verbose: bool = True) -> str:
        history = []
        step = 0

        if verbose:
            print(f"\n{'='*55}")
            print(f"TASK: {task}")
            print(f"MODEL: {self.model}")
            print(f"{'='*55}")

        while step < MAX_STEPS:
            step += 1
            user_prompt = build_user_prompt(task, history)
            llm_output = self._call_llm(user_prompt)

            if verbose:
                print(f"\n[Step {step}]")
                print(llm_output)

            action, action_input = self._parse_action(llm_output)

            if not action:
                history.append({
                    "content": f"{llm_output}\nObservation: "
                               f"Could not parse action. Use exact format."
                })
                continue

            if action == "final_answer":
                if verbose:
                    print(f"\n{'='*55}")
                    print(f"FINAL ANSWER: {action_input}")
                    print(f"Steps: {step}")
                    print(f"{'='*55}")
                return action_input

            tool = TOOL_MAP.get(action)
            if not tool:
                observation = (f"Error: '{action}' not found. "
                               f"Available: {list(TOOL_MAP.keys())}")
            else:
                try:
                    observation = tool.func(action_input)
                except Exception as e:
                    observation = f"Tool error: {e}"

            if verbose:
                print(f"Observation: {observation}")

            history.append({
                "content": f"{llm_output}\nObservation: {observation}"
            })

        return "Max steps reached without a final answer."

    def _parse_action(self, llm_output: str) -> tuple[str, str]:
        action_match = re.search(r"Action:\s*(.+)", llm_output)
        input_match  = re.search(r"Action Input:\s*(.+)", llm_output)
        if not action_match:
            return None, None
        action = action_match.group(1).strip().lower().replace(" ", "_")
        action_input = input_match.group(1).strip() if input_match else ""
        return action, action_input
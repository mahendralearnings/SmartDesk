# src/smartdesk/agents/prompt.py

def build_system_prompt(tools: list) -> str:
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in tools
    )

    return f"""You are SmartDesk Agent, an enterprise invoice assistant.
You solve tasks step by step using available tools.

Available tools:
{tool_descriptions}

STRICT FORMAT — every response must follow this exactly:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [the input to pass to the tool, as a plain string]

When you have enough information to answer:
Thought: I now have all the information needed.
Action: final_answer
Action Input: [your complete answer]

Rules:
- Never skip the Thought step.
- Only use tools listed above. Never invent tool names.
- One action per response — wait for the observation before continuing.
- Action Input must be a plain string — no JSON, no brackets.
"""


def build_user_prompt(task: str, history: list[dict]) -> str:
    """
    Builds the full prompt including conversation history.
    history = [{"role": "thought/observe", "content": "..."}]
    """
    lines = [f"Task: {task}\n"]
    for entry in history:
        lines.append(entry["content"])
    return "\n".join(lines)
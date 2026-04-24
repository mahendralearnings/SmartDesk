# src/smartdesk/mcp/client.py
"""
MCP Client — connects Claude to SmartDesk MCP server.
Uses Anthropic's native tool_use feature (function calling)
instead of our manual ReAct parser from Phase 9.

The difference from Phase 9:
  Phase 9: We parsed free text with regex to extract tool calls
  Phase 11: Claude outputs structured tool_use blocks — zero parsing
"""
import asyncio
import json
import subprocess
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are SmartDesk, an enterprise invoice intelligence assistant.
Use the available tools to answer questions accurately.
Always use tools to fetch data — never guess or invent invoice details."""


async def run_mcp_agent(task: str, verbose: bool = True) -> str:
    """
    Connect to SmartDesk MCP server and run a task.
    Claude uses native tool_use — no regex, no prompt parsing.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "smartdesk.mcp.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # ── 1. DISCOVERY: ask server what tools exist ──
            await session.initialize()
            tools_result = await session.list_tools()

            # Convert MCP tool schemas → Anthropic tool format
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_result.tools
            ]

            if verbose:
                print(f"\n{'='*55}")
                print(f"TASK: {task}")
                print(f"Tools available: {[t['name'] for t in anthropic_tools]}")
                print(f"{'='*55}")

            # ── 2. AGENTIC LOOP ───────────────────────────
            messages = [{"role": "user", "content": task}]

            while True:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=anthropic_tools,
                    messages=messages,
                )

                if verbose:
                    print(f"\nStop reason: {response.stop_reason}")

                # ── 3. STRUCTURED TOOL USE ────────────────
                # Claude returns tool_use blocks — no regex needed
                if response.stop_reason == "tool_use":
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            tool_name  = block.name
                            tool_input = block.input

                            if verbose:
                                print(f"\nTool call: {tool_name}({tool_input})")

                            # ── 4. INVOCATION via MCP ─────
                            result = await session.call_tool(tool_name, tool_input)
                            result_text = result.content[0].text if result.content else ""

                            if verbose:
                                print(f"Result: {result_text[:120]}")

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            })

                    # Append tool results to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                else:
                    # Claude finished — extract final text
                    final = next(
                        (b.text for b in response.content if hasattr(b, "text")),
                        "No response"
                    )
                    if verbose:
                        print(f"\n{'='*55}")
                        print(f"FINAL ANSWER:\n{final}")
                        print(f"{'='*55}")
                    return final
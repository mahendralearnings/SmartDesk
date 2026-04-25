# # scripts/break_langsmith.py
# from dotenv import load_dotenv
# load_dotenv() ##first always not afetr imports

# import os
# os.environ["LANGCHAIN_TRACING_V2"] = "false"  # disable after load

# from smartdesk.agents.react_agent import ReActAgent

# agent = ReActAgent(model="claude-sonnet-4-5")
# result = agent.run("Find all overdue invoices", verbose=True)

# print(f"\nResult: {result}")
# print("Check LangSmith — no trace should appear")


# scripts/break_langsmith.py — add this
import httpx

print("Testing LangSmith connectivity...")
try:
    r = httpx.get("https://api.smith.langchain.com/health", timeout=5)
    print(f"LangSmith reachable: {r.status_code}")
except httpx.ConnectError:
    print("LangSmith unreachable — traces will be lost")
    print("Agent still works — observability is gone")
    print("Fix: implement local fallback logging")
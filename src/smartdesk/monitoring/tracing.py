# src/smartdesk/monitoring/tracing.py
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class LocalTraceBuffer:
    """
    Fallback when LangSmith is unreachable.
    Writes traces to local JSONL file.
    """
    def __init__(self, path: str = "logs/traces.jsonl"):
        import os
        os.makedirs("logs", exist_ok=True)
        self.path = path

    def log(self, name: str, inputs: dict, output: str, latency_ms: float):
        trace = {
            "timestamp": datetime.utcnow().isoformat(),
            "name":      name,
            "inputs":    inputs,
            "output":    output[:500],   # truncate long outputs
            "latency_ms": latency_ms,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(trace) + "\n")
        logger.info(f"Trace buffered locally: {name}")

# Use it as fallback
buffer = LocalTraceBuffer()
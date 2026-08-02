#!/usr/bin/env python3
"""Direct test script that calls calculate_trade_setup.calculate_trade_setups() without MCP stdio transport.

Imports calculate_trade_setup module and calls calculate_trade_setups() directly, then prints the result as JSON.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/Users/haiiro/NoSync/momentum-mcp")
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root (same as mcp_server modules do).
# override=True ensures .env values win over any stale shell env vars
# (e.g. an old TWELVEDATA_API_KEY exported in the current shell session).
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Configure environment
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
os.environ.setdefault("MT5_MCP_URL", os.getenv("MT5_MCP_URL", "http://10.0.1.105:8080/sse"))

# Configure logging so INFO messages from mcp_server modules are visible.
# Set MCP_LOG_LEVEL env var to override (e.g. MCP_LOG_LEVEL=DEBUG).
logging.basicConfig(
    level=os.getenv("MCP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet down noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)

from mcp_server.calculate_trade_setup import calculate_trade_setups

TICKER = "BITUSD"
PERIOD = "5d"
INTERVAL = "1h"


async def main() -> dict:
    """Call calculate_trade_setups directly and return the result."""
    try:
        result = await asyncio.wait_for(
            calculate_trade_setups(TICKER, PERIOD, INTERVAL),
            timeout=30.0,
        )
        return result
    except asyncio.TimeoutError:
        return {"error": "calculate_trade_setups timed out (30s)"}
    except Exception as exc:
        return {"error": f"Exception: {exc}"}


if __name__ == "__main__":
    result = asyncio.run(main())

    # Convert any non-JSON-serializable values (e.g. numpy scalars, pandas Timestamps) to JSON-friendly types
    def _normalize(obj):
        try:
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()
        except ImportError:
            pass
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_normalize(v) for v in obj]
        return obj

    result = _normalize(result)

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if "error" not in result else 1)
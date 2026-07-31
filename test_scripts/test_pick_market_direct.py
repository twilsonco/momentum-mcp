#!/usr/bin/env python3
"""Direct test script that calls market_picker.pick_market() without MCP stdio transport.

Imports market_picker module and calls pick_market() directly, then prints the result as JSON.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/Users/haiiro/NoSync/momentum-mcp")
sys.path.insert(0, str(PROJECT_ROOT))

# Configure environment
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
os.environ.setdefault("MT5_MCP_URL", os.getenv("MT5_MCP_URL", "http://10.0.1.105:8080/sse"))

from mcp_server.market_picker import pick_market

MAX_POSITIONS = 30


async def main() -> dict:
    """Call pick_market directly and return the result."""
    try:
        result = await asyncio.wait_for(
            pick_market(max_positions=MAX_POSITIONS),
            timeout=30.0,
        )
        return result
    except asyncio.TimeoutError:
        return {"error": "pick_market timed out (30s)"}
    except Exception as exc:
        return {"error": f"Exception: {exc}"}


if __name__ == "__main__":
    result = asyncio.run(main())
    
    # Convert datetime objects to ISO format strings for JSON serialization
    if "time_utc" in result:
        result["time_utc"] = result["time_utc"].isoformat()
    if "time_local" in result:
        result["time_local"] = result["time_local"].isoformat()
    
    print(json.dumps(result, indent=2))
    sys.exit(0 if "error" not in result else 1)

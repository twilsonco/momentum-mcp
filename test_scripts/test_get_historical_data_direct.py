#!/usr/bin/env python3
"""Direct test script that calls data.get_historical_data() without MCP stdio transport.

Imports the mcp_server.data module and calls get_historical_data() directly, then prints
the result as JSON. Because it runs in-process (no stdio/SSE wrapper), all logging from
mcp_server modules streams straight to stdout so you can see which data source was used,
how many bars were returned, whether the date range covered the requested period, etc.

This is primarily for validating that get_historical_data can pull a *full* set of candles
from MetaTrader 5 MCP (the first source in the chain) — including symbols where the MT5
terminal itself must fetch history from the broker first — without falling back to any other
source.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

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

# Configure logging so INFO messages from mcp_server modules are visible in stdout.
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

from mcp_server.data import get_historical_data

# ── Configuration ─────────────────────────────────────────────────────────────
# The symbol/period/interval to test. Use a large period + coarse interval so we can
# verify MT5 MCP returns the FULL history (not just what's already cached locally in
# the terminal). For forex/metals, 1h bars over 6mo-1y is a good stress test.
TICKER = "EURUSD"
PERIOD = "1mo"
INTERVAL = "1h"

# If True, accept partial data (skip the date-range check that would otherwise trigger
# fallback to the next source). Set False to force the full-period completeness check —
# useful for confirming MT5 returns enough history without needing a fallback.
FALLBACK_FOR_INCOMPLETE_DATA = True


async def main() -> dict[str, Any]:
    """Fetch historical data and return it (plus metadata) as a JSON-friendly dict."""
    try:
        records = await asyncio.wait_for(
            get_historical_data(
                TICKER,
                period=PERIOD,
                interval=INTERVAL,
                fallback_for_incomplete_data=FALLBACK_FOR_INCOMPLETE_DATA,
            ),
            timeout=120.0,
        )
    except Exception as exc:
        logging.error(f"get_historical_data failed for {TICKER}: {exc}")
        return {
            "ticker": TICKER,
            "period": PERIOD,
            "interval": INTERVAL,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not records:
        return {
            "ticker": TICKER,
            "period": PERIOD,
            "interval": INTERVAL,
            "error": "get_historical_data returned no records",
        }

    # Summarize the result
    first = records[0]
    last = records[-1]
    sources: dict[str, int] = {}
    for r in records:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    summary = {
        "ticker": TICKER,
        "period": PERIOD,
        "interval": INTERVAL,
        "num_bars": len(records),
        "first_date": first["date"],
        "last_date": last["date"],
        "sources_used": sources,
        "fallback_for_incomplete_data": FALLBACK_FOR_INCOMPLETE_DATA,
    }
    logging.info("Summary: %s", summary)

    return {"summary": summary, "records": records}


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

    normalized = _normalize(result)

    # Exit non-zero on error so it's easy to spot failures in CI / scripts.
    has_error = "error" in result
    print(json.dumps(normalized, indent=2, default=str))
    sys.exit(1 if has_error else 0)

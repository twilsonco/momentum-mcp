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
from mcp_server.charts import generate_chart as _generate_chart
from mcp_server.data import get_historical_data

TICKER = "EURUSD"
# 6 months of H1 data gives the VW-KDE method enough history to find
# structural volume nodes on both sides of price. Shorter windows (e.g. 1mo)
# often leave all density below a rising market, so no resistance node exists.
PERIOD = "1mo"
INTERVAL = "1h"

# SL/TP determination strategies to compare.
STRATEGIES = ["swings", "vw_kde", "dbscan"]


async def run_strategy(strategy: str, records: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Call calculate_trade_setups for a single strategy and return the result."""
    try:
        return await asyncio.wait_for(
            calculate_trade_setups(TICKER, PERIOD, INTERVAL, input_records=records, strategy=strategy),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        return {"error": f"calculate_trade_setups ({strategy}) timed out (30s)"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def main() -> dict[str, Any]:
    """Run trade setups for every strategy and key the results by strategy name."""
    results: dict[str, dict] = {}
    records = None
    try:
        records = await get_historical_data(TICKER, period=PERIOD, interval=INTERVAL)
        if not records or len(records) < 2:
            records = None
    except Exception as e:
        logging.error(f"Failed to get historical data for {TICKER}: {e}")
    for strategy in STRATEGIES:
        logging.info(f"Running strategy: {strategy}")
        results[strategy] = await run_strategy(strategy, records)
        # Now make charts for resulting trade setups, if any. We pass the same records to avoid re-fetching.
        for setup_type in ["long_buy_setup", "short_sell_setup"]:
            if "Abort" not in results[strategy].get(setup_type, {}).get("status", ""):
                setup = results[strategy][setup_type]
                if setup.get("stop_loss_price") is not None and setup.get("take_profit_price") is not None:
                    try:
                        chart_data = await _generate_chart(
                            TICKER,
                            interval=INTERVAL,
                            period=PERIOD,
                            stop_loss_price=setup["stop_loss_price"],
                            take_profit_price=setup["take_profit_price"],
                            input_records=records,
                            file_name_suffix=f"{strategy}_{setup_type}",
                        )
                        results[strategy][setup_type]["chart_path"] = chart_data["path"]
                    except Exception as e:
                        logging.error(f"Failed to generate chart for {TICKER} ({strategy}, {setup_type}): {e}")
    return {"ticker": TICKER, "period": PERIOD, "interval": INTERVAL, "results": results}


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

    # Determine exit code from the typed results before normalization.
    has_error = any(
        "error" in setup
        for strategy_result in result["results"].values()
        if isinstance(strategy_result, dict)
        for setup in (strategy_result.get("long_buy_setup", {}), strategy_result.get("short_sell_setup", {}))
    )

    normalized = _normalize(result)

    print(json.dumps(normalized, indent=2, default=str))
    sys.exit(1 if has_error else 0)
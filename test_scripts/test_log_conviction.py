#!/usr/bin/env python3
"""Test the conviction journal's ``log_conviction()`` tool.

Verifies that logging a directional call works end-to-end:
  * price is fetched (preferring MetaTrader MCP get_symbol_price, falling back to yfinance)
  * an entry is written to data/conviction_journal.json
  * the returned confirmation dict matches expectations

Usage:
    python test_scripts/test_log_conviction.py [TICKER] [DIRECTION] [CONFIDENCE]
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
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
os.environ.setdefault(
    "MT5_MCP_URL",
    os.getenv("MT5_MCP_URL", "http://10.0.1.105:8080/sse"),
)

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

from mcp_server.conviction import _fetch_price, log_conviction
from mcp_server.utils.mt5_mcp_server import is_mt5_configured

# Journal path (same as conviction.py computes it) — used to inspect the write.
_JOURNAL_PATH = PROJECT_ROOT / "data" / "conviction_journal.json"


async def main() -> int:
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "EURUSD"
    direction = (sys.argv[2] if len(sys.argv) > 2 else "bullish").lower()
    confidence = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    print("=" * 70)
    print(f"Testing log_conviction(ticker={ticker}, direction={direction}, "
          f"confidence={confidence})")
    print("=" * 70)

    # --- Step 1: verify the price source path ---------------------------------
    mt5_on = is_mt5_configured()
    print(f"\n[1] MT5 MCP configured? {mt5_on} (MT5_MCP_URL="
          f"{os.getenv('MT5_MCP_URL', '')!r})")

    price = await _fetch_price(ticker)
    if price is None:
        print("❌ Could not fetch a price for the ticker. Aborting.")
        return 1
    print(f"✅ Price fetched via _fetch_price: {price:.4f} "
          f"(source={'MT5 get_symbol_price' if mt5_on else 'yfinance fallback'})")

    # --- Step 2: log a conviction ----------------------------------------------
    before = []
    if _JOURNAL_PATH.exists():
        with open(_JOURNAL_PATH) as f:
            before = json.load(f)
    n_before = len(before)

    print("\n[2] Calling log_conviction()...")
    result = await log_conviction(
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        reasoning="Automated test of the conviction journal.",
        signals="test",
    )
    print(json.dumps(result, indent=2))

    if "error" in result:
        print("\n❌ log_conviction returned an error.")
        return 1
    assert result.get("status") == "logged", f"Unexpected status: {result}"
    assert abs(result["entry_price"] - price) < 0.01, (
        f"Entry price mismatch: journal={result['entry_price']} fetched={price}"
    )

    # --- Step 3: verify the entry was persisted --------------------------------
    with open(_JOURNAL_PATH) as f:
        after = json.load(f)
    n_after = len(after)

    if n_after != n_before + 1:
        print("❌ Journal did not grow by exactly one entry "
              f"({n_before} -> {n_after}).")
        return 1

    newest = after[-1]
    assert newest["id"] == result["id"], "Newest journal id != returned id"
    assert newest["ticker"] == ticker
    assert newest["direction"] == direction
    assert newest["confidence"] == confidence
    assert newest.get("resolved") is False
    print(f"\n✅ Journal grew {n_before} -> {n_after}; newest entry:")
    print(json.dumps(newest, indent=2))

    # --- Step 4: invalid input handling ----------------------------------------
    print("\n[3] Testing invalid direction rejection...")
    bad = await log_conviction(
        ticker=ticker,
        direction="sideways",
        confidence=confidence,
        reasoning="should be rejected",
    )
    if "error" in bad:
        print(f"✅ Correctly rejected: {bad['error']}")
    else:
        print("❌ Invalid direction was NOT rejected.")
        return 1

    # --- Step 5: cleanup the test entry -----------------------------------------
    with open(_JOURNAL_PATH) as f:
        entries = json.load(f)
    if entries and entries[-1]["id"] == newest["id"]:
        entries.pop()
        _JOURNAL_PATH.write_text(json.dumps(entries, indent=2, default=str))
        print("\n🧹 Removed the test entry from the journal.")

    print("\n✅ log_conviction() passed all checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

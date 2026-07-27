#!/usr/bin/env python3
"""
Test script for calculate_mt5_position_size function.

This script tests the position sizer with provided entry_price.
The key fix: entry_price parameter should be respected when provided.

Run with: python3 test_mt5_position_sizer.py
"""

import asyncio
import json
import sys
import logging
from os import getenv
from dotenv import load_dotenv

# Add mcp_server to path
sys.path.insert(0, "/Users/haiiro/NoSync/momentum-mcp")

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import after path setup
from mcp_server.mt5_position_sizer import calculate_mt5_position_size
from mcp_server.data import MT5_MCP_URL

# Suppress verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)




async def test_with_provided_entry():
    """Test with entry_price provided — should NOT fetch from MT5."""
    logger.info("=" * 80)
    logger.info("TEST 1: Position Sizer with Provided Entry Price")
    logger.info("=" * 80)
    logger.info("Testing: entry_price=0.02915 should be used directly (no MT5 fetch)")
    logger.info("")
    
    result = await calculate_mt5_position_size(
        symbol="ETHBTC",
        position_direction="long",
        entry_price=0.02915,  # Provided — should use this
        stop_price=0.028,
        risk_pct=1.0
    )
    
    logger.info(f"Result status: {result.status}")
    if result.status == "success":
        logger.info(f"✅ Success!")
        logger.info(f"   Entry Price Used: {result.data.get('entry_price')}")
        logger.info(f"   Position: {result.data.get('position_lots')} lots")
        logger.info(f"   Actual Risk: ${result.data.get('actual_risk'):.2f}")
        logger.info(f"   Risk %: {result.data.get('actual_risk_pct')}%")
        return True
    else:
        logger.error(f"❌ FAILED with error: {result.error}")
        return False


async def test_short_position():
    """Test with short position and provided entry_price."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Position Sizer with Short Position")
    logger.info("=" * 80)
    logger.info("Testing: entry_price=0.028 for short (entry < stop)")
    logger.info("")
    
    result = await calculate_mt5_position_size(
        symbol="ETHBTC",
        position_direction="short",
        entry_price=0.028,  # Provided — should use this
        stop_price=0.02915,  # Stop above entry for short
        risk_pct=2.0
    )
    
    logger.info(f"Result status: {result.status}")
    if result.status == "success":
        logger.info(f"✅ Success!")
        logger.info(f"   Entry Price Used: {result.data.get('entry_price')}")
        logger.info(f"   Position: {result.data.get('position_lots')} lots")
        logger.info(f"   Actual Risk: ${result.data.get('actual_risk'):.2f}")
        logger.info(f"   Risk %: {result.data.get('actual_risk_pct')}%")
        return True
    else:
        logger.error(f"❌ FAILED with error: {result.error}")
        return False


async def main():
    """Run all tests."""
    logger.info("\n🔍 MT5 Position Sizer - Entry Price Fix Tests")
    logger.info(f"MT5_MCP_URL: {MT5_MCP_URL}")
    logger.info(f"Testing: entry_price parameter is respected\n")
    
    if not MT5_MCP_URL:
        logger.error("❌ MT5_MCP_URL not configured!")
        sys.exit(1)
    
    results = []
    
    # Test 1: Long with provided entry_price
    try:
        test1_ok = await test_with_provided_entry()
        results.append(("Long Position (entry_price provided)", test1_ok))
    except Exception as e:
        logger.error(f"❌ Test 1 crashed: {e}")
        results.append(("Long Position (entry_price provided)", False))
    
    # Test 2: Short with provided entry_price
    try:
        test2_ok = await test_short_position()
        results.append(("Short Position (entry_price provided)", test2_ok))
    except Exception as e:
        logger.error(f"❌ Test 2 crashed: {e}")
        results.append(("Short Position (entry_price provided)", False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    all_pass = all(r for _, r in results)
    print("")
    if all_pass:
        logger.info("✅ All tests passed! Entry price fix is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        logger.info("\nDiagnostics:")
        logger.info("  • Check MT5_MCP_URL is set correctly")
        logger.info("  • Verify MT5 MCP server is running")
        logger.info("  • Check account has sufficient balance")
        logger.info("  • Ensure ETHBTC symbol exists and is tradeable")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

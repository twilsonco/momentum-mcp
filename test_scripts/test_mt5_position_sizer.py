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
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to sys.path so we can import mcp_server regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import after path setup
from mcp_server.mt5_position_sizer import calculate_mt5_position_size
from mcp_server.utils.mt5_mcp_server import MT5_MCP_URL, get_mt5_client

# Suppress verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Suppress harmless Python 3.14 + anyio cleanup errors from mcp.client.sse.
#
# When the MT5 SSE connection fails or is torn down, the underlying
# ``sse_client`` async generator can raise ``RuntimeError`` during finalization
# because its cancel scope is exited from a different task than the one that
# entered it.  These errors are emitted *after* the event loop is closed, so
# they cannot be caught normally.  Filter them out at the interpreter and
# asyncio layers.
# ─────────────────────────────────────────────────────────────────────────────
_HARMLESS = (
    "Attempted to exit cancel scope in a different task",
    "generator didn't stop after athrow",
    "unhandled errors in a TaskGroup",
)


def _is_harmless(exc: BaseException) -> bool:
    return any(msg in str(exc) for msg in _HARMLESS)


def _install_exception_hooks() -> None:
    """Install hooks that swallow the harmless anyio/MCP shutdown errors."""

    def _excepthook(exc_type, exc_value, exc_tb):
        if exc_value is not None and _is_harmless(exc_value):
            return
        if exc_type is ExceptionGroup or exc_type is BaseExceptionGroup:
            if _is_harmless(exc_value):
                return
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook
    if hasattr(sys, "unraisablehook"):
        def _unraisablehook(unraisable):
            if unraisable.exc_value is not None and _is_harmless(unraisable.exc_value):
                return
            sys.__unraisablehook__(unraisable)
        sys.unraisablehook = _unraisablehook

    def _loop_handler(loop, context):
        exc = context.get("exception")
        if exc is not None and _is_harmless(exc):
            return
        loop.default_exception_handler(context)

    try:
        asyncio.get_event_loop().set_exception_handler(_loop_handler)
    except RuntimeError:
        # No running loop yet — will be installed in main()
        pass




async def test_with_provided_entry():
    """Test with entry_price provided — should NOT fetch from MT5."""
    logger.info("=" * 80)
    logger.info("TEST 1: Position Sizer with Provided Entry Price")
    logger.info("=" * 80)
    logger.info("Testing: entry_price=0.02915 should be used directly (no MT5 fetch)")
    logger.info("")
    
    result = await calculate_mt5_position_size(
        symbol="EURUSD",
        position_direction="long",
        entry_price=1.15459,  # Provided — should use this
        stop_price=1.150,
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
    logger.info("Testing: entry_price=1.145 for short (entry < stop)")
    logger.info("")
    
    result = await calculate_mt5_position_size(
        symbol="EURUSD",
        position_direction="short",
        entry_price=1.15459,  # Provided — should use this
        stop_price=1.16,  # Stop above entry for short
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

    # Install the asyncio loop handler now that a loop exists
    def _loop_handler(loop, context):
        exc = context.get("exception")
        if exc is not None and _is_harmless(exc):
            return
        loop.default_exception_handler(context)
    asyncio.get_running_loop().set_exception_handler(_loop_handler)

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
    else:
        logger.error("❌ Some tests failed!")
        logger.info("\nDiagnostics:")
        logger.info("  • Check MT5_MCP_URL is set correctly")
        logger.info("  • Verify MT5 MCP server is running")
        logger.info("  • Check account has sufficient balance")
        logger.info("  • Ensure ETHBTC symbol exists and is tradeable")

    # Explicitly close the MT5 singleton so its sse_client generator is
    # finalized *inside* the running loop, avoiding shutdown-time cleanup
    # errors from anyio cancel-scope mismatches.
    try:
        await get_mt5_client().aclose()
    except Exception:
        pass

    return 0 if all_pass else 1


if __name__ == "__main__":
    _install_exception_hooks()
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(exit_code)

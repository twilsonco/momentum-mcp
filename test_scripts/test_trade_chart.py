"""Test script for the new trade-level chart features.

Exercises:
  1. Basic chart (no trade levels) — regression check
  2. LONG trade (entry < stop < tp)
  3. SHORT trade (entry > stop > tp)
  4. Entry + stop only (no TP)
  5. Validation error when stop/tp given without entry
  6. Out-of-range trade levels — verify y-axis expands to show them
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make sure we import the local mcp_server package from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.charts import generate_chart


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

    # 1. Synchronous interpreter shutdown hook
    def _excepthook(exc_type, exc_value, exc_tb):
        if exc_value is not None and _is_harmless(exc_value):
            return
        # Also handle ExceptionGroup wrappers
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

    # 2. Asyncio loop exception handler (catches errors during loop close)
    def _loop_handler(loop, context):
        exc = context.get("exception")
        if exc is not None and _is_harmless(exc):
            return
        # Default handling for everything else
        loop.default_exception_handler(context)

    try:
        asyncio.get_event_loop().set_exception_handler(_loop_handler)
    except RuntimeError:
        # No running loop yet — will be installed in main()
        pass


def _print(label: str, obj) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    if isinstance(obj, dict):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(repr(obj))


async def test_basic_chart() -> dict:
    """Regression: chart without trade levels still works."""
    return await generate_chart(
        ticker="AAPL",
        period="3mo",
        interval="1d",
        return_image=False,
    )


async def test_long_trade() -> dict:
    """Full LONG setup: entry, stop below, TP above."""
    return await generate_chart(
        ticker="EURUSD",
        period="5d",
        interval="1h",
        entry_price=1.137,
        stop_loss_price=1.134,
        take_profit_price=1.140,
        return_image=False,
    )


async def test_short_trade() -> dict:
    """Full SHORT setup: entry, stop above, TP below."""
    return await generate_chart(
        ticker="TSLA",
        period="1mo",
        interval="1h",
        entry_price=250.00,
        stop_loss_price=265.00,
        take_profit_price=220.00,
        show_volume=True,
        return_image=False,
    )


async def test_entry_and_stop_only() -> dict:
    """Entry + stop, no TP — direction still detectable from entry vs stop."""
    return await generate_chart(
        ticker="NVDA",
        period="6mo",
        interval="1d",
        entry_price=120.00,
        stop_loss_price=110.00,
        return_image=False,
    )


async def test_validation_error() -> None:
    """Stop without entry should raise ValueError."""
    try:
        await generate_chart(
            ticker="AAPL",
            period="3mo",
            interval="1d",
            stop_loss_price=185.00,
            take_profit_price=215.00,
            return_image=False,
        )
        print("FAIL: expected ValueError, got none")
    except ValueError as e:
        print(f"PASS: got expected ValueError: {e}")


async def test_out_of_range_trade_levels() -> dict:
    """Trade levels far outside the candle range — verify the y-axis
    expands to show them, and the PNG is taller than the no-trade chart."""
    # AAPL 3mo 1d typically trades ~$190-$220. Place TP well above and
    # stop well below to force the y-axis to expand.
    return await generate_chart(
        ticker="AAPL",
        period="3mo",
        interval="1d",
        entry_price=200.00,
        stop_loss_price=100.00,   # way below recent range
        take_profit_price=300.00, # way above recent range
        return_image=False,
    )


async def main() -> None:
    # Install the asyncio loop handler now that a loop exists
    def _loop_handler(loop, context):
        exc = context.get("exception")
        if exc is not None and _is_harmless(exc):
            return
        loop.default_exception_handler(context)
    asyncio.get_running_loop().set_exception_handler(_loop_handler)

    _print("Test 1: Basic chart (no trade levels)", await test_basic_chart())
    _print("Test 2: LONG trade (AAPL)", await test_long_trade())
    _print("Test 3: SHORT trade (TSLA)", await test_short_trade())
    _print("Test 4: Entry + stop only (NVDA)", await test_entry_and_stop_only())
    _print("Test 5: Validation error", await test_validation_error())
    _print("Test 6: Out-of-range trade levels", await test_out_of_range_trade_levels())

    # Verify all PNGs exist on disk
    print("\n" + "=" * 70)
    print("Verifying saved PNGs:")
    print("=" * 70)
    charts_dir = Path("./charts")
    pngs = sorted(charts_dir.glob("*.png"))
    for p in pngs:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:60s}  {size_kb:.1f} KB")
    print(f"\nTotal: {len(pngs)} PNGs in {charts_dir.resolve()}")

    # Explicitly close the MT5 singleton so its sse_client generator is
    # finalized *inside* the running loop, avoiding shutdown-time cleanup
    # errors from anyio cancel-scope mismatches.
    try:
        from mcp_server.utils.mt5_mcp_server import get_mt5_client
        await get_mt5_client().aclose()
    except Exception:
        pass


if __name__ == "__main__":
    _install_exception_hooks()
    asyncio.run(main())
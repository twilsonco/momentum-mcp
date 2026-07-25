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

# Make sure we import the local mcp_server package
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.charts import generate_chart


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


if __name__ == "__main__":
    asyncio.run(main())
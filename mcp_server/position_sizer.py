"""
Position sizing module — Fixed Fractional, ATR-based, and Kelly Criterion.

Answers: "How many shares should I buy?"
Stock-focused position sizing with multiple methods for equities and options.

For Forex/Metals/Futures/Indices, see mt5_position_sizer.py which uses MetaTrader MCP.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from mcp_server.schema import SignalResult
from mcp_server.data import get_live_price
from mcp_server.technicals import analyze_technicals

logger = logging.getLogger(__name__)


async def calculate_position_size(
    ticker: str,
    account_size: float,
    risk_pct: float = 1.0,
    entry_price: float | None = None,
    stop_price: float | None = None,
    max_position_pct: float = 10.0,
    method: str = "fixed_fractional",
    win_rate: float | None = 0.5,
    avg_win: float | None = 2.0,
    avg_loss: float | None = 1.0,
) -> SignalResult:
    """Calculate risk-based position size for stock trading.
    
    Answers: "How many shares should I buy?" given account size and risk tolerance.
    
    Supports three position sizing methods:
    1. Fixed Fractional: Risk a fixed % of account per trade
    2. ATR-based: Scale position based on volatility (ATR)
    3. Kelly Criterion: Optimal bet size based on win rate and profit ratio
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT").
        account_size: Total account size in dollars (required).
        risk_pct: Percentage of account to risk per trade (default 1%).
        entry_price: Entry price. If None, fetches live price (5s timeout).
        stop_price: Stop loss price. If None, calculated from ATR (10s timeout, fallback 5%).
        max_position_pct: Maximum position size as % of account (default 10%).
        method: Position sizing method - "fixed_fractional" (default), "atr", or "kelly".
        win_rate: Win rate for Kelly method (default 0.5, i.e., 50%).
        avg_win: Average win size for Kelly (default 2.0, i.e., 2x loss).
        avg_loss: Average loss size for Kelly (default 1.0, i.e., base unit).
        
    Returns:
        SignalResult with position size in shares, detailed metrics, and method comparison.
    """
    try:
        ticker = ticker.strip().upper()

        # 1. Fetch live price if entry_price is None
        if entry_price is None:
            try:
                entry_price = await asyncio.wait_for(get_live_price(ticker), timeout=5.0)
            except asyncio.TimeoutError:
                return SignalResult.error_msg(f"Timeout fetching live price for {ticker} (exceeded 5s)")
            except Exception as e:
                return SignalResult.error_msg(f"Could not fetch live price for {ticker}: {e}")

        # 2. Get ATR(14) if needed (for stop_price or atr method), with timeout
        atr_14 = None
        if stop_price is None or method == "atr":
            try:
                tech_res = await asyncio.wait_for(analyze_technicals(ticker), timeout=10.0)
                if tech_res.status == "success":
                    atr_14 = tech_res.data.get("atr_14")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching technicals for {ticker}, using fallback stop")
            except Exception as e:
                logger.warning(f"Could not fetch technicals for {ticker}: {e}, using fallback stop")

        if stop_price is None:
            if atr_14:
                stop_price = entry_price - (atr_14 * 2)
            else:
                # Fallback stop if ATR fails: 5% below entry
                stop_price = entry_price * 0.95
                logger.warning(f"ATR unavailable for {ticker}, using 5% fallback stop.")

        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return SignalResult.error_msg(f"Invalid setup for {ticker}: Entry ({entry_price}) <= Stop ({stop_price})")

        risk_amount = account_size * (risk_pct / 100)

        # 3. Compute position sizing methods
        # Fixed Fractional
        ff_shares = int(risk_amount / risk_per_share)
        ff_value = ff_shares * entry_price

        # ATR-based
        if atr_14:
            atr_shares = int(risk_amount / (atr_14 * 2))
            atr_value = atr_shares * entry_price
        else:
            atr_shares = ff_shares
            atr_value = ff_value

        # Kelly
        kelly_fraction = None
        kelly_shares = 0
        kelly_value = 0
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            b = avg_win / avg_loss if avg_loss != 0 else 1.0
            p = win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b if b != 0 else 0
            if kelly_fraction > 0:
                kelly_value = account_size * kelly_fraction
                kelly_shares = int(kelly_value / entry_price)
            else:
                kelly_fraction = 0

        # Select primary method
        if method == "atr":
            shares = atr_shares
            position_value = atr_value
        elif method == "kelly":
            shares = kelly_shares
            position_value = kelly_value
        else:
            shares = ff_shares
            position_value = ff_value

        # 4. Apply position size constraint
        max_position_value = account_size * (max_position_pct / 100)
        max_shares = int(max_position_value / entry_price)

        constraints_applied = False
        if shares > max_shares:
            shares = max_shares
            position_value = shares * entry_price
            constraints_applied = True

        # Recalculate actual risk after constraints
        actual_risk_amount = shares * risk_per_share

        # Summary
        summary = (
            f"Position size for {ticker} @ ${entry_price:.2f}:\n"
            f"- Recommended: {shares} shares (~${position_value:,.2f})\n"
            f"- Method: {method.replace('_', ' ').title()}\n"
            f"- Actual Risk: ${actual_risk_amount:,.2f} ({actual_risk_amount/account_size*100:.2f}% of account)\n"
            f"- Stop Loss: ${stop_price:.2f} ({abs(entry_price-stop_price)/entry_price*100:.1f}% risk per share)"
        )
        if constraints_applied:
            summary += f"\n- ⚠️ Capped by {max_position_pct}% max position constraint."
            if actual_risk_amount < risk_amount:
                summary += f"\n  (Requested {risk_pct}% risk = ${risk_amount:,.2f}, got {actual_risk_amount/account_size*100:.2f}% = ${actual_risk_amount:,.2f})"

        data = {
            "ticker": ticker,
            "entry_price": round(entry_price, 2),
            "stop_price": round(stop_price, 2),
            "risk_per_share": round(risk_per_share, 2),
            "method": method,
            "account_size": account_size,
            "risk_pct": risk_pct,
            "risk_amount": round(risk_amount, 2),
            "actual_risk_amount": round(actual_risk_amount, 2),
            "actual_risk_pct": round(actual_risk_amount / account_size * 100, 2),
            "recommended_shares": shares,
            "position_value": round(position_value, 2),
            "position_pct_of_account": round((position_value / account_size) * 100, 2),
            "constraints": {
                "max_position_shares": max_shares,
                "max_position_binding": constraints_applied,
            },
            "methods_compared": {
                "fixed_fractional": {"shares": ff_shares, "value": round(ff_value, 2)},
                "atr_based": {"shares": atr_shares, "value": round(atr_value, 2), "atr_14": round(atr_14, 4) if atr_14 else None},
                "kelly": {"shares": kelly_shares, "value": round(kelly_value, 2), "kelly_fraction": round(kelly_fraction, 4)} if kelly_fraction is not None else None,
            },
            "summary": summary,
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error(f"Position sizing failed for {ticker}: {e}")
        return SignalResult.error_msg(str(e))

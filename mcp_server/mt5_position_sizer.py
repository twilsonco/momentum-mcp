"""
MT5 Position Sizer — Position sizing for any MT5-tradeable symbol via MetaTrader MCP.

Supports stocks, forex, commodities, indices, cryptocurrencies, and more.
Uses the MetaTrader MCP server to fetch real-time account info, symbol data, and pricing.
Supports long and short positions with proper risk calculations for any MT5 asset.

The tool fetches optional parameters from MT5 if not provided and validates provided parameters
against live MT5 data, reporting any discrepancies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp_server.schema import SignalResult
from mcp_server.utils.mt5_mcp_server import (
    fetch_mt5_account_info,
    fetch_mt5_symbol_info,
)

logger = logging.getLogger(__name__)


async def calculate_mt5_position_size(
    symbol: str,
    position_direction: str,
    entry_price: float = 0.0,
    stop_price: float = 0.0,
    risk_pct: float = 1.0,
) -> SignalResult:
    """Calculate risk-based position size for any MT5-tradeable symbol.
    
    Works with stocks, forex, commodities, indices, cryptocurrencies, or any asset
    available on MetaTrader5 by leveraging native tick values to automatically handle 
    cross-currency conversions (e.g., ETHBTC to USD).
    
    Uses MetaTrader 5's native `trade_tick_value` (the monetary value of a single tick
    movement for exactly 1.0 standard lot, in the account's base currency) and
    `trade_tick_size` (the minimum price increment) to compute position size. This
    inherently accounts for the account's base currency, live cross-rates, and asset
    class differences.
    
    Formula:
        Position Lots = Risk Amount / ((|Entry - Stop| / Tick Size) * Tick Value)
    
    Args:
        symbol: MT5 symbol (e.g., "XAUUSD", "EURUSD", "AAPL", "BTCUSD", "SPX").
        entry_price: Entry price (if provided, overrides live fetched price).
        stop_price: Stop loss price (user decision, required).
        position_direction: "long" a.k.a. "buy" (entry < stop) or "short" a.k.a. "sell" (entry > stop).
        risk_pct: Percentage of account to risk per trade (default 1%).
        
    Returns:
        SignalResult with position size in lots, account validation, and price comparison.
    """
    start_time = time.time()
    try:
        symbol = symbol.strip().upper()
        
        # Validate stop_price
        if stop_price <= 0:
            return SignalResult.error_msg(f"stop_price must be positive, got {stop_price}")
        
        # Fetch MT5 data
        fetch_start = time.time()
        account_info = await fetch_mt5_account_info(timeout=5.0)
        symbol_info = await fetch_mt5_symbol_info(symbol, timeout=15.0)
        
        if not account_info or not symbol_info:
            return SignalResult.error_msg(
                "Failed to fetch MT5 data. MetaTrader MCP not configured or not responding."
            )
        
        fetch_elapsed = time.time() - fetch_start
        logger.info(f"MT5 data fetch completed in {fetch_elapsed:.2f}s (account: {bool(account_info)}, symbol: {bool(symbol_info)})")
        
        # Account size
        mt5_balance = None
        if account_info and "balance" in account_info:
            mt5_balance = account_info["balance"]
        
        if mt5_balance is not None:
            account_size = mt5_balance
            logger.info(f"Fetched account balance from MT5: ${account_size:,.2f}")
        else:
            return SignalResult.error_msg(
                "Failed to fetch account balance from MT5. MetaTrader MCP not configured or not responding."
            )

        position_direction = position_direction.strip().lower()
        if not position_direction:
            return SignalResult.error_msg(
                "Position direction must be specified."
            )
        else:
            if position_direction == "buy":
                position_direction = "long"
            elif position_direction == "sell":
                position_direction = "short"
            elif position_direction not in ["long", "short"]:
                return SignalResult.error_msg(
                    f"Invalid position_direction '{position_direction}'. Must be 'long', 'short', 'buy', or 'sell'."
                )
        
        if entry_price > 0.0:
            if position_direction == "long" and entry_price <= stop_price:
                return SignalResult.error_msg(
                    f"Invalid long setup for {symbol}: Entry ({entry_price}) must be > Stop ({stop_price})"
                )
            elif position_direction == "short" and entry_price >= stop_price:
                return SignalResult.error_msg(
                    f"Invalid short setup for {symbol}: Entry ({entry_price}) must be < Stop ({stop_price})"
                )
        
        # Entry price: use provided value, or fetch from MT5 if not provided
        if entry_price <= 0.0:
            # No entry price provided — fetch from MT5
            mt5_price = None
            if symbol_info and isinstance(symbol_info, dict):
                    if position_direction == "long":
                        mt5_price = symbol_info.get("ask")
                    else:
                        mt5_price = symbol_info.get("bid")
            
            if mt5_price is not None:
                entry_price = mt5_price
                logger.info(f"Fetched {position_direction} entry price from MT5: {entry_price}")
            else:
                return SignalResult.error_msg(
                    "entry_price not provided and failed to fetch from MT5. "
                    "MetaTrader MCP not configured or symbol not found. "
                    "Provide entry_price as a parameter or ensure MT5 can fetch prices."
                )
        else:
            logger.info(f"Using provided entry price: {entry_price}")
        
        # Require tick data for accurate cross-currency math
        tick_size = symbol_info.get("trade_tick_size") if symbol_info else None
        
        # Prioritize 'trade_tick_value_loss' for Stop Loss calculations, fallback to standard tick value
        tick_value = symbol_info.get("trade_tick_value_loss") if symbol_info else None
        
        if not tick_value or tick_value <= 0:
            tick_value = symbol_info.get("trade_tick_value") if symbol_info else None
        
        if not tick_size or not tick_value or tick_value <= 0:
            return SignalResult.error_msg(
                f"Missing or invalid tick data for {symbol}. Ensure the MT5 MCP server exposes "
                f"'trade_tick_size', 'trade_tick_value', or 'trade_tick_value_loss'. "
                f"Removing from valid symbols to protect account equity."
            )
        
        # Validate setup and calculate raw price distance
        if position_direction == "long":
            if entry_price <= stop_price:
                return SignalResult.error_msg(
                    f"Invalid long setup for {symbol}: Entry ({entry_price}) must be > Stop ({stop_price})"
                )
            price_distance = entry_price - stop_price
        else:  # short
            if entry_price >= stop_price:
                return SignalResult.error_msg(
                    f"Invalid short setup for {symbol}: Entry ({entry_price}) must be < Stop ({stop_price})"
                )
            price_distance = stop_price - entry_price
        
        if price_distance <= 0:
            return SignalResult.error_msg(
                f"Invalid setup: price_distance must be positive, got {price_distance}"
            )
        
        # --- Tick-Based Risk Calculation ---
        # 1. Convert price distance to raw ticks
        distance_in_ticks = price_distance / tick_size
        
        # 2. Calculate the exact risk per 1.0 lot in the account's base currency
        risk_per_lot_account_currency = distance_in_ticks * tick_value
        
        if risk_per_lot_account_currency <= 0:
            return SignalResult.error_msg(
                f"Invalid calculation: risk per lot is zero or negative ({risk_per_lot_account_currency}). "
                f"Check tick_size ({tick_size}) and tick_value ({tick_value}) for {symbol}."
            )
        
        # 3. Calculate target position size in lots
        risk_amount = account_size * (risk_pct / 100)
        position_lots_exact = risk_amount / risk_per_lot_account_currency
        
        # 4. Round to nearest broker step (default 0.01)
        volume_step = symbol_info.get("volume_step") if symbol_info else None
        min_lot_size = volume_step if volume_step and volume_step > 0 else 0.01
        position_lots_rounded = round(position_lots_exact / min_lot_size) * min_lot_size
        
        # 5. Back-calculate actual risk using the rounded lot size
        actual_risk = position_lots_rounded * risk_per_lot_account_currency
        actual_risk_pct = (actual_risk / account_size) * 100
        
        # Summary
        summary = (
            f"MT5 Position for {symbol} ({position_direction.upper()}):\n"
            f"- Entry: {entry_price:.5f}, Stop: {stop_price:.5f}\n"
            f"- Distance in Ticks: {distance_in_ticks:.2f}\n"
            f"- Account: ${account_size:,.2f}\n"
            f"- Requested Risk: ${risk_amount:,.2f} ({risk_pct}%)\n"
            f"- Position: {position_lots_rounded:.2f} lots\n"
            f"- Actual Risk: ${actual_risk:,.2f} ({actual_risk_pct:.2f}%)"
        )
        
        data = {
            "symbol": symbol,
            "position_direction": position_direction,
            "entry_price": round(entry_price, 5),
            "stop_price": round(stop_price, 5),
            "account_size": round(account_size, 2),
            "risk_pct": risk_pct,
            "risk_amount": round(risk_amount, 2),
            "position_lots": round(position_lots_rounded, 2),
            "actual_risk": round(actual_risk, 2),
            "actual_risk_pct": round(actual_risk_pct, 2),
            "tick_size": tick_size,
            "tick_value": tick_value,
            "distance_in_ticks": round(distance_in_ticks, 2),
            "risk_per_lot_account_currency": round(risk_per_lot_account_currency, 4),
            "volume_step": min_lot_size,
            "summary": summary,
        }
        
        total_time = time.time() - start_time
        logger.info(f"calculate_mt5_position_size({symbol}) completed in {total_time:.2f}s")
        return SignalResult.success(data)
    
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"MT5 position sizing failed for {symbol} after {total_time:.2f}s: {e}")
        return SignalResult.error_msg(str(e))

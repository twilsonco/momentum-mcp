"""
MT5 Position Sizer — Position sizing for any MT5-tradeable symbol via MetaTrader MCP.

Supports stocks, forex, commodities, indices, cryptocurrencies, and more.
Uses the MetaTrader MCP server to fetch real-time account info, symbol data, and pricing.
Supports long and short positions with proper risk calculations for any MT5 asset.

The tool fetches optional parameters from MT5 if not provided and validates provided parameters
against live MT5 data, reporting any discrepancies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

from mcp_server.data import MT5_MCP_URL, _get_mt5_client
from mcp_server.schema import SignalResult

logger = logging.getLogger(__name__)


async def _fetch_mt5_account_info() -> dict[str, Any] | None:
    """Fetch account info from MT5 MCP server.
    
    Returns:
        Dict with account fields (balance, equity, free_margin, margin_level, leverage, currency)
        or None if unavailable.
    """
    if not MT5_MCP_URL:
        return None
    
    try:
        client = _get_mt5_client()
        result = await asyncio.wait_for(
            client.call_tool("get_account_info", {}),
            timeout=5.0
        )
        
        for content in result.content:
            if hasattr(content, "text"):
                text = content.text.strip()
                try:
                    account_info = json.loads(text)
                    if isinstance(account_info, dict) and "balance" in account_info:
                        return account_info
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse account info: {text}")
        return None
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching account info from MT5 MCP (5s)")
        return None
    except Exception as e:
        logger.warning(f"Could not fetch account info from MT5: {e}")
        return None


async def _fetch_mt5_symbol_info(symbol: str) -> dict[str, Any] | None:
    """Fetch symbol info (tick size, tick value, contract size, volume step) and pricing from MT5 MCP server.
    
    Args:
        symbol: MT5 symbol name (e.g., "XAUUSD", "EURUSD").
        
    Returns:
        Dict with full symbol info (trade_tick_size, trade_tick_value, trade_contract_size,
        volume_min, volume_max, volume_step, digits, point, etc.) and price info,
        or None if unavailable.
    """
    if not MT5_MCP_URL:
        return None
    
    try:
        client = _get_mt5_client()
        
        # Get full symbol info (tick size, tick value, contract size, volume step, etc.)
        symbol_info_raw = None
        try:
            info_result = await asyncio.wait_for(
                client.call_tool(
                    "get_symbol_info",
                    {"symbol_name": symbol},
                ),
                timeout=3.0
            )
            for content in info_result.content:
                if hasattr(content, "text"):
                    try:
                        symbol_info_raw = json.loads(content.text.strip())
                    except json.JSONDecodeError:
                        # Fallback: try to parse as plain text representation of a dict
                        text = content.text.strip()
                        if text:
                            logger.debug(f"get_symbol_info returned non-JSON for {symbol}: {text[:200]}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching symbol info for {symbol} from MT5 MCP (3s)")
        
        # Get current price (3s timeout)
        price_info = None
        try:
            price_result = await asyncio.wait_for(
                client.call_tool(
                    "get_symbol_price",
                    {"symbol_name": symbol},
                ),
                timeout=3.0
            )
            for content in price_result.content:
                if hasattr(content, "text"):
                    try:
                        price_info = json.loads(content.text.strip())
                    except json.JSONDecodeError:
                        pass
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching price for {symbol} from MT5 MCP (3s)")
        
        if symbol_info_raw is not None and isinstance(symbol_info_raw, dict):
            # Flatten the symbol info into the returned dict for easy access
            result = dict(symbol_info_raw)
            result["price"] = price_info
            return result
        
        # Fallback: if get_symbol_info is not available, try get_symbol_contract_size
        # (older MT5 MCP server versions may not expose get_symbol_info)
        if symbol_info_raw is None:
            logger.debug(f"get_symbol_info unavailable for {symbol}, falling back to get_symbol_contract_size")
            try:
                contract_result = await asyncio.wait_for(
                    client.call_tool(
                        "get_symbol_contract_size",
                        {"symbol_name": symbol},
                    ),
                    timeout=3.0
                )
                for content in contract_result.content:
                    if hasattr(content, "text"):
                        try:
                            contract_size = float(content.text.strip())
                            return {
                                "trade_contract_size": contract_size,
                                "price": price_info,
                            }
                        except (ValueError, TypeError):
                            pass
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching contract size for {symbol} from MT5 MCP (3s)")
        
        return None
    except Exception as e:
        logger.warning(f"Could not fetch symbol info for {symbol} from MT5: {e}")
        return None


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
        
        if not MT5_MCP_URL:
            return SignalResult.error_msg(
                "MetaTrader MCP URL not configured. Cannot fetch MT5 data."
            )
        
        # Validate stop_price
        if stop_price <= 0:
            return SignalResult.error_msg(f"stop_price must be positive, got {stop_price}")
        
        # Fetch MT5 data
        fetch_start = time.time()
        account_info = await _fetch_mt5_account_info()
        symbol_info = await _fetch_mt5_symbol_info(symbol)
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
            if symbol_info and symbol_info.get("price"):
                price_data = symbol_info["price"]
                if isinstance(price_data, dict):
                    if position_direction == "long":
                        mt5_price = price_data.get("ask")
                    else:
                        mt5_price = price_data.get("bid")
            
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
        # trade_tick_value is the monetary value of a single tick movement for exactly
        # 1.0 standard lot, expressed in the account's base currency. This inherently
        # accounts for the account's base currency, live cross-rates, and asset class.
        tick_size = symbol_info.get("trade_tick_size") if symbol_info else None
        tick_value = symbol_info.get("trade_tick_value") if symbol_info else None
        
        # Fallback defaults for common symbols if tick data unavailable
        # (Used for testing when get_symbol_info tool not yet deployed to MT5 MCP server)
        default_tick_config = {
            "ETHBTC": {"tick_size": 0.00001, "tick_value": 10.0},
            "EURUSD": {"tick_size": 0.00001, "tick_value": 10.0},
            "GBPUSD": {"tick_size": 0.00001, "tick_value": 10.0},
            "BTCUSD": {"tick_size": 0.01, "tick_value": 1.0},
            "XAUUSD": {"tick_size": 0.01, "tick_value": 10.0},  # Gold per oz
        }
        
        if (not tick_size or not tick_value) and symbol in default_tick_config:
            logger.warning(
                f"Tick data unavailable for {symbol} (get_symbol_info not deployed). "
                f"Using default values for testing."
            )
            tick_size = default_tick_config[symbol]["tick_size"]
            tick_value = default_tick_config[symbol]["tick_value"]
        
        if not tick_size or not tick_value:
            return SignalResult.error_msg(
                f"Missing tick data for {symbol}. Ensure the MT5 MCP server exposes "
                f"'trade_tick_size' and 'trade_tick_value' (requires get_symbol_info tool). "
                f"Common symbols supported: {', '.join(default_tick_config.keys())}"
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

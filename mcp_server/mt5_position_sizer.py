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
    """Fetch symbol contract size and pricing from MT5 MCP server.
    
    Args:
        symbol: MT5 symbol name (e.g., "XAUUSD", "EURUSD").
        
    Returns:
        Dict with contract_size and price info, or None if unavailable.
    """
    if not MT5_MCP_URL:
        return None
    
    try:
        client = _get_mt5_client()
        
        # Get contract size (3s timeout)
        contract_size = None
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
                    except (ValueError, TypeError):
                        pass
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching contract size for {symbol} from MT5 MCP (3s)")
        
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
        
        if contract_size is not None:
            return {
                "contract_size": contract_size,
                "price": price_info,
            }
        return None
    except Exception as e:
        logger.warning(f"Could not fetch symbol info for {symbol} from MT5: {e}")
        return None


async def calculate_mt5_position_size(
    symbol: str,
    stop_price: float,
    position_direction: str = "long",
    account_size: float | None = None,
    entry_price: float | None = None,
    risk_pct: float = 1.0,
    method: str = "fixed_fractional",
) -> SignalResult:
    """Calculate risk-based position size for any MT5-tradeable symbol.
    
    Works with stocks, forex, commodities, indices, cryptocurrencies, or any asset
    available on MetaTrader5. Fetches real-time account balance, symbol contract size, 
    and current price from MetaTrader MCP server (if configured). Validates provided 
    parameters against live MT5 data and reports any discrepancies.
    
    Args:
        symbol: MT5 symbol (e.g., "XAUUSD", "EURUSD", "AAPL", "BTCUSD", "SPX").
        stop_price: Stop loss price (user decision, required).
        position_direction: "long" a.k.a. "buy" (entry < stop) or "short" a.k.a. "sell" (entry > stop). Default "long".
        account_size: Account balance in account currency. If None, fetches from MT5.
        entry_price: Entry price. If None, uses current bid/ask from MT5.
        risk_pct: Percentage of account to risk per trade (default 1%).
        method: Position sizing method - "fixed_fractional" (default) or "kelly".
        
    Returns:
        SignalResult with position size in lots, account validation, and price comparison.
    """
    try:
        symbol = symbol.strip().upper()
        position_direction = position_direction.strip().lower()
        if position_direction == "buy":
            position_direction = "long"
        elif position_direction == "sell":
            position_direction = "short"
        
        if position_direction not in ("long", "short"):
            return SignalResult.error_msg(
                f"Invalid position_direction '{position_direction}'. Must be 'long', 'short', 'buy', or 'sell'."
            )
        
        # Validate stop_price
        if stop_price <= 0:
            return SignalResult.error_msg(f"stop_price must be positive, got {stop_price}")
        
        # Fetch MT5 data
        account_info = await _fetch_mt5_account_info()
        symbol_info = await _fetch_mt5_symbol_info(symbol)
        
        # Build validation report
        validation_report = {
            "mt5_configured": bool(MT5_MCP_URL),
            "account_data_available": bool(account_info),
            "symbol_data_available": bool(symbol_info),
            "discrepancies": [],
        }
        
        # Account size
        mt5_balance = None
        if account_info and "balance" in account_info:
            mt5_balance = account_info["balance"]
            validation_report["mt5_balance"] = mt5_balance
        
        if account_size is None:
            if mt5_balance is not None:
                account_size = mt5_balance
                logger.info(f"Fetched account balance from MT5: ${account_size:,.2f}")
            else:
                return SignalResult.error_msg(
                    "account_size parameter required. MetaTrader MCP not configured or not responding."
                )
        else:
            if mt5_balance is not None and abs(account_size - mt5_balance) > 0.01:
                validation_report["discrepancies"].append({
                    "field": "account_size",
                    "provided": account_size,
                    "mt5_value": mt5_balance,
                    "note": "MT5 value differs from provided parameter",
                })
        
        # Entry price
        mt5_price = None
        if symbol_info and symbol_info.get("price"):
            price_data = symbol_info["price"]
            if isinstance(price_data, dict):
                if position_direction == "long":
                    mt5_price = price_data.get("ask")
                else:
                    mt5_price = price_data.get("bid")
        
        if entry_price is None:
            if mt5_price is not None:
                entry_price = mt5_price
                logger.info(f"Fetched {position_direction} entry price from MT5: {entry_price}")
            else:
                return SignalResult.error_msg(
                    "entry_price parameter required. MetaTrader MCP not configured or symbol not found."
                )
        else:
            if mt5_price is not None and abs(entry_price - mt5_price) > 0.01:
                validation_report["discrepancies"].append({
                    "field": "entry_price",
                    "provided": entry_price,
                    "mt5_value": mt5_price,
                    "note": "MT5 current price differs from provided entry price",
                })
        
        # Contract size
        contract_size = None
        if symbol_info and symbol_info.get("contract_size"):
            contract_size = symbol_info["contract_size"]
            validation_report["contract_size"] = contract_size
        
        # Validate setup
        if position_direction == "long":
            if entry_price <= stop_price:
                return SignalResult.error_msg(
                    f"Invalid long setup for {symbol}: Entry ({entry_price}) must be > Stop ({stop_price})"
                )
            risk_per_unit = entry_price - stop_price
        else:  # short
            if entry_price >= stop_price:
                return SignalResult.error_msg(
                    f"Invalid short setup for {symbol}: Entry ({entry_price}) must be < Stop ({stop_price})"
                )
            risk_per_unit = stop_price - entry_price
        
        if risk_per_unit <= 0:
            return SignalResult.error_msg(
                f"Invalid setup: risk_per_unit must be positive, got {risk_per_unit}"
            )
        
        # Calculate position size
        risk_amount = account_size * (risk_pct / 100)
        
        # Position size in units
        position_units = risk_amount / risk_per_unit
        
        # Convert to lots if contract size is available
        position_lots = position_units / contract_size if contract_size else position_units
        
        # Round to nearest minimum lot size (0.01 for MT5) for accuracy
        # Rounding to nearest gives better risk target accuracy than floor()
        min_lot_size = 0.01
        position_lots_rounded = round(position_lots / min_lot_size) * min_lot_size
        position_units_final = position_lots_rounded * contract_size if contract_size else position_lots_rounded
        
        # Actual risk after rounding
        actual_risk = position_units_final * risk_per_unit
        actual_risk_pct = (actual_risk / account_size) * 100
        
        # Summary
        summary = (
            f"MT5 Position for {symbol} ({position_direction.upper()}):\n"
            f"- Entry: {entry_price:.5f}, Stop: {stop_price:.5f}\n"
            f"- Risk per unit: {risk_per_unit:.5f}\n"
            f"- Account: ${account_size:,.2f}\n"
            f"- Requested Risk: ${risk_amount:,.2f} ({risk_pct}%)\n"
            f"- Position: {position_lots_rounded:.2f} lots ({position_units_final:.2f} units)\n"
            f"- Actual Risk: ${actual_risk:,.2f} ({actual_risk_pct:.2f}%)"
        )
        
        data = {
            "symbol": symbol,
            "position_direction": position_direction,
            "entry_price": round(entry_price, 5),
            "stop_price": round(stop_price, 5),
            "risk_per_unit": round(risk_per_unit, 5),
            "account_size": round(account_size, 2),
            "risk_pct": risk_pct,
            "risk_amount": round(risk_amount, 2),
            "position_lots": round(position_lots_rounded, 2),
            "position_units": round(position_units_final, 2),
            "actual_risk": round(actual_risk, 2),
            "actual_risk_pct": round(actual_risk_pct, 2),
            "contract_size": round(contract_size, 4) if contract_size else None,
            "validation": validation_report,
            "summary": summary,
        }
        
        return SignalResult.success(data)
    
    except Exception as e:
        logger.error(f"MT5 position sizing failed for {symbol}: {e}")
        return SignalResult.error_msg(str(e))

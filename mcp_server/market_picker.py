"""Smart market picker that excludes open MetaTrader positions and validates symbols.

Instead of taking an explicit list of excluded symbols, this picker:
1. Fetches all open MetaTrader positions
2. Excludes symbols that have open positions
3. Checks max_positions limit (default 10)
4. Validates each randomly picked symbol with get_symbol_info
5. Returns a tradeable symbol, interval, and historical timeframe

Exposes as a single MCP tool: `pick_market`
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# MT5 MCP server URL (from environment)
MT5_MCP_URL: str = os.getenv("MT5_MCP_URL", "").strip()

# Symbol catalog (same as market_picker.py)
SYMBOLS = {
    "Crypto": [
        "BCHBTC", "BCHUSD", "BITUSD", "BTCUSD", "DASHBTC", "DASHUSD", "EDOBIT",
        "EDOUSD", "EOSBIT", "EOSUSD", "ETCUSD", "ETHBTC", "ETHUSD", "ETPBIT",
        "ETPUSD", "IOSTBIT", "IOTABIT", "IOTAUSD", "LTCBTC", "LTCUSD", "NEOBTC",
        "NEOUSD", "OMGBIT", "OMGUSD", "QTUMBIT", "SANBIT", "SANUSD", "TRXUSD",
        "USDTUSD", "XMRBTC", "XMRUSD", "XRPBIT", "XRPUSD", "ZECBTC", "ZECUSD",
    ],
    "FX_majors": ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    "FX_minors": [
        "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "CADCHF", "CADJPY", "CHFJPY",
        "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "GBPAUD",
        "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY",
    ],
    "Metals": ["XAGUSD", "XAUEUR", "XAUUSD", "XPTUSD"],
    "Indices": ["AUS200", "ESP35", "EUSTX50", "FRA40", "GER30", "JPN225", "NAS100", "SPX500", "UK100", "US30"],
    "Futures": ["DOLLAR"],
    "Energies": ["UKOil", "USOil"],
}

INTERVALS = ["M15", "M30", "H1", "H4", "D1"]


def _get_historical_timeframe(interval: str) -> str:
    """Map interval to recommended historical data timeframe."""
    if interval in ["M1", "M5"]:
        return "1 day"
    elif interval in ["M15", "M30"]:
        return "5 days"
    elif interval == "H1":
        return "1 month"
    elif interval == "H4":
        return "3 months"
    elif interval == "D1":
        return "6 months"
    elif interval == "D5":
        return "2 years"
    elif interval == "W1":
        return "5 years"
    elif interval == "MN":
        return "max"
    return "1 month"


def _is_market_open(asset_class: str, now_utc: datetime) -> bool:
    """Check if a market category is open at the given UTC time."""
    if asset_class == "Crypto":
        return True
    
    weekday = now_utc.weekday()  # Monday is 0, Sunday is 6
    hour = now_utc.hour
    
    # Global Weekend Closure: Friday 21:00 UTC -> Sunday 21:00 UTC
    if (weekday == 4 and hour >= 21) or (weekday == 5) or (weekday == 6 and hour < 21):
        return False
    
    # Daily rollover break for non-Forex (Metals, Indices, Energies)
    # Block hours 21 and 22 UTC to safely cover daylight savings time shifts
    if asset_class in ["Metals", "Indices", "Futures", "Energies"]:
        if hour == 21 or hour == 22:
            return False
    
    return True


async def _get_open_position_symbols() -> set[str]:
    """Fetch all currently open MetaTrader positions and return their symbols.
    
    Returns:
        Set of symbols with open positions (uppercase). Empty set if MT5 MCP unavailable
        or if an error occurs.
    """
    if not MT5_MCP_URL:
        logger.debug("MT5_MCP_URL not configured, no positions to exclude")
        return set()
    
    try:
        from mcp_server.mt5_position_sizer import _get_mt5_client
        
        client = _get_mt5_client()
        result = await asyncio.wait_for(
            client.call_tool("get_all_positions", {}),
            timeout=5.0,
        )
        
        # Parse the CSV response — each row is: id, symbol, type, time, ...
        symbols = set()
        for content in result.content:
            if hasattr(content, "text"):
                lines = content.text.strip().split("\n")
                if len(lines) <= 1:
                    continue
                # Skip header if present
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) >= 4:
                        # Second column should be symbol
                        symbol = parts[3].strip().upper()
                        if symbol:
                            symbols.add(symbol)
                logger.debug(f"Found {len(symbols)} open position symbols: {symbols}")
                return symbols
        
        return set()
    
    except asyncio.TimeoutError:
        logger.warning("Timeout fetching open positions from MT5 MCP (5s)")
        return set()
    except Exception as exc:
        logger.warning(f"Failed to fetch open positions: {exc}")
        return set()


async def _get_margin_level() -> float | None:
    """Fetch the account margin level percentage from MetaTrader.

    Returns:
        Margin level as a percentage (equity / margin * 100), or None if
        MT5 MCP is unavailable or the value cannot be parsed.
    """
    if not MT5_MCP_URL:
        logger.debug("MT5_MCP_URL not configured, cannot fetch margin level")
        return None

    try:
        from mcp_server.mt5_position_sizer import _get_mt5_client

        client = _get_mt5_client()
        result = await asyncio.wait_for(
            client.call_tool("get_account_info", {}),
            timeout=5.0,
        )

        for content in result.content:
            if hasattr(content, "text"):
                text = content.text.strip()
                try:
                    account_info = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse account info: {text}")
                    continue

                if not isinstance(account_info, dict):
                    continue

                margin_level = account_info.get("margin_level")
                if margin_level is None:
                    logger.debug(f"Account info missing margin_level: {account_info}")
                    return None

                try:
                    return float(margin_level)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid margin_level value: {margin_level!r}")
                    return None

        return None

    except asyncio.TimeoutError:
        logger.warning("Timeout fetching account info from MT5 MCP (5s)")
        return None
    except Exception as exc:
        logger.warning(f"Failed to fetch margin level: {exc}")
        return None


async def _validate_symbol(symbol: str) -> bool:
    """Validate that a symbol can be traded by calling get_symbol_info.
    
    Returns:
        True if symbol is valid and can be traded, False otherwise.
    """
    if not MT5_MCP_URL:
        logger.debug(f"MT5_MCP_URL not configured, cannot validate {symbol}")
        return True  # Assume valid if we can't check
    
    try:
        from mcp_server.mt5_position_sizer import _get_mt5_client
        
        client = _get_mt5_client()
        result = await asyncio.wait_for(
            client.call_tool(
                "get_symbol_price",
                {"symbol_name": symbol},
            ),
            timeout=3.0,
        )
        
        # If we got a response with content, assume the symbol is valid
        if result and result.content:
            for content in result.content:
                if hasattr(content, "text") and content.text.strip():
                    logger.debug(f"Symbol {symbol} validated successfully")
                    return True
        
        logger.warning(f"Symbol {symbol} validation returned empty response")
        return False
    
    except asyncio.TimeoutError:
        logger.warning(f"Timeout validating symbol {symbol} (3s)")
        return False
    except Exception as exc:
        logger.warning(f"Failed to validate symbol {symbol}: {exc}")
        return False


async def pick_market(
    max_positions: int = 10,
    minimum_margin_percent: float = 500.0,
) -> dict[str, Any]:
    """Pick a random market symbol that doesn't have an open position.

    This function:
    1. Checks if max_positions limit is reached (returns error if ≥ max_positions)
    2. Fetches all open MetaTrader positions and excludes their symbols
    3. Builds list of available symbols from open markets
    4. Randomly picks a symbol and validates it with get_symbol_info
    5. If validation fails, removes the symbol and tries again
    6. Returns the picked symbol with interval and recommended timeframe

    Args:
        max_positions: Maximum allowed open positions. Defaults to 10.
                      If current positions ≥ max_positions, returns error.
                      NOTE: This argument is NOT exposed to the MCP tool.
        minimum_margin_percent: Minimum acceptable account margin level (equity /
                      margin * 100). If the current margin level is less than or
                      or equal to this threshold, the function aborts. Defaults to
                      500.0 (the updated standard MT5 margin call level). Set to 0.0 to
                      disable the check.

    Returns:
        Dict with keys:
        - symbol: The picked market symbol
        - interval: Random interval (M15, M30, H1, H4, D1)
        - historical_data_timeframe: Recommended timeframe for historical data
        - current_time_utc: Current UTC time (ISO-8601)
        - current_time_local: Current local time (ISO-8601)

        Or error dict if:
        - Maximum positions reached
        - Margin level below the minimum threshold
        - No valid symbols available
    """
    if not MT5_MCP_URL:
        msg = "MetaTrader MCP server not configured; Abort immediately"
        logger.error(msg)
        return {"error": msg}

    now_utc = datetime.now(timezone.utc)

    # Step 1: Get open position symbols
    open_symbols = await _get_open_position_symbols()
    num_positions = len(open_symbols)
    allowed_additional_positions = max_positions - num_positions

    logger.info(f"Current open positions: {num_positions}, max_positions: {max_positions}")

    # Step 2: Check max_positions limit
    if num_positions >= max_positions:
        msg = f"Maximum positions ({max_positions}) reached; Abort immediately"
        logger.info(msg)
        return {"error": msg}

    # Step 2b: Check account margin level
    if minimum_margin_percent > 0:
        margin_level = await _get_margin_level()
        if margin_level is not None:
            logger.info(
                f"Account margin level: {margin_level:.2f}%, "
                f"minimum required: {minimum_margin_percent:.2f}%"
            )
            if margin_level <= minimum_margin_percent:
                msg = (
                    f"Account margin level ({margin_level:.2f}%) is at or below "
                    f"minimum threshold ({minimum_margin_percent:.2f}%); "
                    f"Abort immediately"
                )
                logger.info(msg)
                return {"error": msg}
        else:
            logger.warning(
                "Could not fetch margin level; proceeding without margin check"
            )
    
    # Step 3: Build available symbols list (only from open markets)
    available_symbols = []
    for category, symbols in SYMBOLS.items():
        if _is_market_open(category, now_utc):
            for sym in symbols:
                if sym not in open_symbols:
                    available_symbols.append(sym)
    
    if not available_symbols:
        msg = "No open markets available or all open symbols are excluded; Abort immediately"
        logger.warning(msg)
        return {"error": msg}
    
    logger.debug(f"Available symbols for trading: {len(available_symbols)}")
    
    # Step 4: Pick a symbol and validate it (with retry)
    random.shuffle(available_symbols)  # Randomize the list
    picked_symbol = None
    
    for symbol in available_symbols:
        is_valid = await _validate_symbol(symbol)
        if is_valid:
            picked_symbol = symbol
            break
        else:
            logger.info(f"Symbol {symbol} validation failed, skipping")
    
    if not picked_symbol:
        msg = "No valid symbols available to trade on; Abort immediately"
        logger.error(msg)
        return {"error": msg}
    
    # Step 5: Return result with random interval
    interval = random.choice(INTERVALS)
    result = {
        "symbol": picked_symbol,
        "interval": interval,
        "timeframe": _get_historical_timeframe(interval),
        # "num_open_positions": num_positions,
        # "allowed_additional_positions": allowed_additional_positions,
    }
    
    logger.info(f"Picked symbol: {picked_symbol}, interval: {interval}")
    return result

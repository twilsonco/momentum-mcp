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
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import pytz

load_dotenv()

logger = logging.getLogger(__name__)

TZ = "America/Denver"

# MT5 MCP server URL (from environment)
MT5_MCP_URL: str = os.getenv("MT5_MCP_URL", "").strip()

# Symbol catalog (same as market_picker.py)
SYMBOLS = {
    "Crypto": [
        "BCHUSD", "BITUSD", "BTCUSD", "DASHUSD", "EDOUSD", "EOSUSD", "ETCUSD",
        "ETHUSD", "ETPUSD", "IOTAUSD", "LTCUSD", "NEOUSD", "OMGUSD", "SANUSD", 
        "TRXUSD", "USDTUSD", "XMRUSD", "XRPUSD", "ZECUSD",
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

ALL_INTERVALS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "D5", "W1", "MN"]

INTERVALS = ["M15", "H1"]

# Market-specific trading hours (local time zones with DST support)
MARKET_HOURS = {
    # Indices - stock exchanges (Monday-Friday only)
    "AUS200": {
        "timezone": "Australia/Sydney",
        "open_hour": 10,
        "open_minute": 0,
        "close_hour": 16,
        "close_minute": 0,
        "weekdays_only": True,
    },
    "JPN225": {
        "timezone": "Asia/Tokyo",
        "open_hour": 9,
        "open_minute": 0,
        "close_hour": 15,
        "close_minute": 0,
        "weekdays_only": True,
    },
    "ESP35": {
        "timezone": "Europe/Madrid",
        "open_hour": 9,
        "open_minute": 0,
        "close_hour": 17,
        "close_minute": 30,
        "weekdays_only": True,
    },
    "FRA40": {
        "timezone": "Europe/Paris",
        "open_hour": 9,
        "open_minute": 0,
        "close_hour": 17,
        "close_minute": 30,
        "weekdays_only": True,
    },
    "GER30": {
        "timezone": "Europe/Berlin",
        "open_hour": 9,
        "open_minute": 0,
        "close_hour": 17,
        "close_minute": 30,
        "weekdays_only": True,
    },
    "EUSTX50": {
        "timezone": "Europe/Paris",
        "open_hour": 9,
        "open_minute": 0,
        "close_hour": 17,
        "close_minute": 30,
        "weekdays_only": True,
    },
    "UK100": {
        "timezone": "Europe/London",
        "open_hour": 8,
        "open_minute": 0,
        "close_hour": 16,
        "close_minute": 30,
        "weekdays_only": True,
    },
    "NAS100": {
        "timezone": "America/New_York",
        "open_hour": 9,
        "open_minute": 30,
        "close_hour": 16,
        "close_minute": 0,
        "weekdays_only": True,
    },
    "SPX500": {
        "timezone": "America/New_York",
        "open_hour": 9,
        "open_minute": 30,
        "close_hour": 16,
        "close_minute": 0,
        "weekdays_only": True,
    },
    "US30": {
        "timezone": "America/New_York",
        "open_hour": 9,
        "open_minute": 30,
        "close_hour": 16,
        "close_minute": 0,
        "weekdays_only": True,
    },
    # Futures/Energies - continuous trading with daily breaks
    "DOLLAR": {
        "timezone": "America/New_York",
        "continuous": True,
        "daily_break_start_hour": 17,
        "daily_break_start_minute": 0,
        "daily_break_end_hour": 18,
        "daily_break_end_minute": 0,
    },
    "UKOil": {
        "timezone": "America/New_York",
        "continuous": True,
        "daily_break_start_hour": 18,
        "daily_break_start_minute": 0,
        "daily_break_end_hour": 20,
        "daily_break_end_minute": 0,
    },
    "USOil": {
        "timezone": "America/New_York",
        "continuous": True,
        "daily_break_start_hour": 17,
        "daily_break_start_minute": 0,
        "daily_break_end_hour": 18,
        "daily_break_end_minute": 0,
    },
}


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


def _convert_local_time_to_utc(now_utc: datetime, tz_name: str, local_hour: int, local_minute: int) -> tuple[int, int]:
    """Convert local market time to UTC, accounting for DST.
    
    Args:
        now_utc: Current UTC time
        tz_name: Timezone name (e.g., "America/New_York")
        local_hour: Local hour (0-23)
        local_minute: Local minute (0-59)
    
    Returns:
        Tuple of (utc_hour, utc_minute)
    """
    try:
        tz = ZoneInfo(tz_name)
        
        # Create a naive datetime with the local time
        naive_time = now_utc.replace(hour=local_hour, minute=local_minute, second=0, microsecond=0, tzinfo=None)
        
        # Interpret as being in the target timezone
        local_time = naive_time.replace(tzinfo=tz)
        
        # Convert to UTC
        utc_time = local_time.astimezone(timezone.utc)
        
        return (utc_time.hour, utc_time.minute)
    except Exception as exc:
        logger.warning(f"Error converting time for {tz_name}: {exc}")
        return (local_hour, local_minute)


def _get_session_utc_times(now_utc: datetime, tz_name: str, local_open: int, local_close: int) -> tuple[int, int]:
    """Convert local market session times to UTC, accounting for DST.
    
    Args:
        now_utc: Current UTC time
        tz_name: Timezone name (e.g., "America/New_York")
        local_open: Local market open hour (0-23)
        local_close: Local market close hour (0-23)
    
    Returns:
        Tuple of (utc_open_hour, utc_close_hour)
    """
    utc_open, _ = _convert_local_time_to_utc(now_utc, tz_name, local_open, 0)
    utc_close, _ = _convert_local_time_to_utc(now_utc, tz_name, local_close, 0)
    return (utc_open, utc_close)


def _get_trading_sessions(now_utc: datetime) -> str:
    """Determine which forex trading sessions are active and their phases.
    
    Accounts for daylight savings time by converting local session times to UTC.
    All session checks are performed entirely in UTC.
    
    Returns a formatted string describing active sessions and their phases
    (beginning/middle/end), e.g. "end of Tokyo session and beginning of London session".
    """
    # Check for weekend closure first
    weekday = now_utc.weekday()  # Monday is 0, Sunday is 6
    hour = now_utc.hour
    
    # Global Weekend Closure: Friday 21:00 UTC -> Sunday 21:00 UTC
    if (weekday == 4 and hour >= 21) or (weekday == 5) or (weekday == 6 and hour < 21):
        return "markets closed"
    
    # Define sessions in local market hours (used only for conversion to UTC)
    sessions_local = [
        ("Sydney", "Australia/Sydney", 7, 16),      # 7 AM - 4 PM local
        ("Tokyo", "Asia/Tokyo", 9, 16),            # 9 AM - 4 PM local
        ("London", "Europe/London", 8, 17),        # 8 AM - 5 PM local
        ("New York", "America/New_York", 9, 16),   # 9 AM - 4 PM local
    ]
    
    # Convert all sessions to UTC (accounting for DST)
    sessions_utc = []
    for session_name, tz_name, local_open, local_close in sessions_local:
        utc_open, utc_close = _get_session_utc_times(now_utc, tz_name, local_open, local_close)
        sessions_utc.append((session_name, utc_open, utc_close))
    
    # Work entirely in UTC from here on
    active_sessions = []
    
    for session_name, utc_open, utc_close in sessions_utc:
        # Check if market is open in UTC
        is_active = False
        if utc_open < utc_close:
            # Session doesn't wrap around midnight
            is_active = utc_open <= hour < utc_close
        else:
            # Session wraps around midnight
            is_active = hour >= utc_open or hour < utc_close
        
        if is_active:
            # Calculate phase: beginning (0-33%), middle (33-67%), end (67-100%)
            if utc_open < utc_close:
                session_duration = utc_close - utc_open
                hours_into_session = hour - utc_open
            else:
                session_duration = (24 - utc_open) + utc_close
                if hour >= utc_open:
                    hours_into_session = hour - utc_open
                else:
                    hours_into_session = (24 - utc_open) + hour
            
            phase_percent = (hours_into_session / session_duration) * 100
            
            if phase_percent < 33:
                phase = "beginning"
            elif phase_percent < 67:
                phase = "middle"
            else:
                phase = "end"
            
            active_sessions.append(f"{phase} of {session_name} session")
    
    if not active_sessions:
        return "between trading sessions"
    
    return " and ".join(active_sessions)


def _is_market_open(symbol: str, now_utc: datetime) -> bool:
    """Check if a specific market/symbol is open at the given UTC time.
    
    Args:
        symbol: The trading symbol (e.g., "EURUSD", "AUS200", "DOLLAR")
        now_utc: Current UTC time
    
    Returns:
        True if the market is open, False otherwise.
    """
    weekday = now_utc.weekday()  # Monday is 0, Sunday is 6
    hour = now_utc.hour
    minute = now_utc.minute
    now_minutes = hour * 60 + minute
    
    # Check if symbol has specific market hours defined
    if symbol in MARKET_HOURS:
        hours = MARKET_HOURS[symbol]
        tz_name = hours.get("timezone")
        
        # Handle stock exchanges (weekdays only)
        if hours.get("weekdays_only"):
            if weekday >= 5:  # Saturday or Sunday
                return False
            
            # Convert session times from local to UTC
            open_utc_h, open_utc_m = _convert_local_time_to_utc(
                now_utc, tz_name, hours["open_hour"], hours["open_minute"]
            )
            close_utc_h, close_utc_m = _convert_local_time_to_utc(
                now_utc, tz_name, hours["close_hour"], hours["close_minute"]
            )
            
            open_minutes = open_utc_h * 60 + open_utc_m
            close_minutes = close_utc_h * 60 + close_utc_m
            
            if open_minutes < close_minutes:
                # Normal case: open and close on same day
                return open_minutes <= now_minutes < close_minutes
            else:
                # Wraps around midnight
                return now_minutes >= open_minutes or now_minutes < close_minutes
        
        # Handle continuous markets (Futures/Energies) with daily breaks
        if hours.get("continuous"):
            # Check daily break window
            if "daily_break_start_hour" in hours:
                break_start_utc_h, break_start_utc_m = _convert_local_time_to_utc(
                    now_utc, tz_name,
                    hours["daily_break_start_hour"],
                    hours["daily_break_start_minute"]
                )
                break_end_utc_h, break_end_utc_m = _convert_local_time_to_utc(
                    now_utc, tz_name,
                    hours["daily_break_end_hour"],
                    hours["daily_break_end_minute"]
                )
                
                break_start_minutes = break_start_utc_h * 60 + break_start_utc_m
                break_end_minutes = break_end_utc_h * 60 + break_end_utc_m
                
                # Check if we're in the break window
                if break_start_minutes < break_end_minutes:
                    # Normal case: break on same day
                    if break_start_minutes <= now_minutes < break_end_minutes:
                        return False
                else:
                    # Break wraps around midnight
                    if now_minutes >= break_start_minutes or now_minutes < break_end_minutes:
                        return False
            
            # Check for weekly closure (Friday close around 22:00 UTC)
            # Global Weekend Closure: Friday 22:00 UTC -> Sunday 22:00 UTC
            if (weekday == 4 and hour >= 22) or (weekday == 5) or (weekday == 6 and hour < 22):
                return False
            
            return True
    
    # Fallback for symbols not in MARKET_HOURS (Forex, Crypto, Metals)
    # Global Weekend Closure: Friday 21:00 UTC -> Sunday 21:00 UTC
    if (weekday == 4 and hour >= 21) or (weekday == 5) or (weekday == 6 and hour < 21):
        return False
    
    # Daily rollover break for some asset classes (conservative approach)
    # Block hours 21 and 22 UTC to safely cover daylight savings time shifts
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
    
    Here's the full content of symbol info, for reference:
    {
        "ask": 1875.47,
        "askhigh": 1940.04,
        "asklow": 1852.04,
        "bank": "",
        "basis": "",
        "bid": 1870.64,
        "bidhigh": 1934.76,
        "bidlow": 1847.06,
        "category": "",
        "chart_mode": 0,
        "currency_base": "USD",
        "currency_margin": "USD",
        "currency_profit": "USD",
        "custom": false,
        "description": "ETHEREUM v US DOLLAR",
        "digits": 2,
        "exchange": "",
        "expiration_mode": 15,
        "expiration_time": 0,
        "filling_mode": 1,
        "formula": "",
        "isin": "",
        "last": 0,
        "lasthigh": 0,
        "lastlow": 0,
        "margin_hedged": 0,
        "margin_hedged_use_leg": false,
        "margin_initial": 0,
        "margin_maintenance": 0,
        "n_fields": 96,
        "n_sequence_fields": 96,
        "n_unnamed_fields": 0,
        "name": "ETHUSD",
        "option_mode": 0,
        "option_right": 0,
        "option_strike": 0,
        "order_gtc_mode": 0,
        "order_mode": 127,
        "page": "",
        "path": "Cryptos\\ETHUSD",
        "point": 0.01,
        "price_change": -2.5647,
        "price_greeks_delta": 0,
        "price_greeks_gamma": 0,
        "price_greeks_omega": 0,
        "price_greeks_rho": 0,
        "price_greeks_theta": 0,
        "price_greeks_vega": 0,
        "price_sensitivity": 0,
        "price_theoretical": 0,
        "price_volatility": 0,
        "select": true,
        "session_aw": 0,
        "session_buy_orders": 0,
        "session_buy_orders_volume": 0,
        "session_close": 1919.88,
        "session_deals": 0,
        "session_interest": 0,
        "session_open": 1919.88,
        "session_price_limit_max": 0,
        "session_price_limit_min": 0,
        "session_price_settlement": 0,
        "session_sell_orders": 0,
        "session_sell_orders_volume": 0,
        "session_turnover": 0,
        "session_volume": 0,
        "spread": 483,
        "spread_float": true,
        "start_time": 0,
        "swap_long": -25,
        "swap_mode": 6,
        "swap_rollover3days": 7,
        "swap_short": -25,
        "ticks_bookdepth": 0,
        "time": 1785532534,
        "trade_accrued_interest": 0,
        "trade_calc_mode": 2,
        "trade_contract_size": 100,
        "trade_exemode": 2,
        "trade_face_value": 0,
        "trade_freeze_level": 0,
        "trade_liquidity_rate": 0,
        "trade_mode": 4,
        "trade_stops_level": 0,
        "trade_tick_size": 0.01,
        "trade_tick_value": 1,
        "trade_tick_value_loss": 1,
        "trade_tick_value_profit": 1,
        "visible": true,
        "volume": 0,
        "volume_limit": 0,
        "volume_max": 10,
        "volume_min": 0.01,
        "volume_real": 0,
        "volume_step": 0.01,
        "volumehigh": 0,
        "volumehigh_real": 0,
        "volumelow": 0,
        "volumelow_real": 0
    }
    
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
                "get_symbol_info",
                {"symbol_name": symbol},
            ),
            timeout=15.0,
        )
        
        # If we got a response with content, assume the symbol is valid
        if result and result.content:
            for content in result.content:
                if hasattr(content, "text") and content.text.strip():
                    try:
                        s = json.loads(content.text.strip())
                        print(s)
                        check_fields = ["ask", "bid", "trade_contract_size", "trade_tick_size", "trade_tick_value", "volume_step"]
                        missing_check_fields = [field for field in check_fields if field not in s or not s[field]]
                        if not missing_check_fields:
                            logger.debug(f"Symbol {symbol} validated successfully")
                            return True
                        else:
                            logger.warning(f"Symbol {symbol} validation failed: missing required fields: {missing_check_fields}")
                            return False
                    except json.JSONDecodeError:
                        logger.warning(f"get_symbol_info returned non-JSON for {symbol}: {content.text.strip()[:200]}")
                        return False
                    
        
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
    intervals: list[str] = INTERVALS,
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
        - trading_sessions: Current active trading session(s) with phase description
                           (e.g. "end of Tokyo session and beginning of London session")
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
    now_local = datetime.now(pytz.timezone(TZ))

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
        for sym in symbols:
            if sym not in open_symbols and _is_market_open(sym, now_utc):
                available_symbols.append(sym)
    
    if not available_symbols:
        msg = "No open markets available or all open symbols are excluded; Abort immediately"
        logger.warning(msg)
        return {"error": msg}
    
    logger.debug(f"Available symbols for trading: {len(available_symbols)}")
    
    # Step 4: Pick a symbol and validate it (with retry)
    random.shuffle(available_symbols)  # Randomize the list
    picked_symbol = None
    picked_market = None
    
    for symbol in available_symbols:
        is_valid = await _validate_symbol(symbol)
        if is_valid:
            picked_symbol = symbol
            picked_market = next((cat for cat, syms in SYMBOLS.items() if symbol in syms), None)
            break
        else:
            logger.info(f"Symbol {symbol} validation failed, skipping")
    
    if not picked_symbol:
        msg = "No valid symbols available to trade on; Abort immediately"
        logger.error(msg)
        return {"error": msg}
    
    # Step 5: Return result with random interval
    valid_intervals = [i for i in intervals if i in ALL_INTERVALS]
    if not valid_intervals:
        valid_intervals = INTERVALS
    interval = random.choice(valid_intervals)
    result = {
        "market": picked_market,
        "symbol": picked_symbol,
        "interval": interval,
        "timeframe": _get_historical_timeframe(interval),
        "trading_sessions": _get_trading_sessions(now_utc),
        "time_utc": now_utc,
        "time_local": now_local,
        # "num_open_positions": num_positions,
        # "allowed_additional_positions": allowed_additional_positions,
    }
    
    logger.info(f"Picked symbol: {picked_symbol}, interval: {interval}")
    return result

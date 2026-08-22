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
import csv
import io
import logging
import random
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import pytz

from mcp_server.charts import generate_chart as _generate_chart, ChartResult
from mcp_server.data import get_historical_data
from mcp_server.calculate_trade_setup import calculate_trade_setups
from mcp_server.mt5_position_sizer import calculate_mt5_position_size
from mcp_server.utils.mt5_mcp_server import (
    MT5_MCP_URL,
    call_mt5_tool,
    fetch_mt5_account_info,
    fetch_mt5_symbol_info,
)

load_dotenv()

# Silence the noisy per-request INFO logs emitted by httpx/httpcore (used
# internally by the MT5 MCP client). Keep WARNING+ so real errors still surface.
for _lib_logger in ("httpx", "httpcore"):
    logging.getLogger(_lib_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

TZ = "America/Denver"

ALLOWED_SYMBOL_MARKETS = [
    "Futures", "FX Crosses", "Energy", "Metals", "FX Majors", "Cryptos"
]

ALLOWED_CRYPTOS = [r".*USD"]

DISALLOWED_SYMBOLS = [
    "NEOUSD"
]

ALL_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",]

INTERVALS = ["15m", "1h"]

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

MARKET_ADJUSTMENTS = {
    "Cryptos": """- Frequent liquidity hunts break recent swing levels; use 2x to 2.5x ATR buffer on shorter timeframes.
- Validate spread tolerance strictly; spreads widen during low-volume periods.""",
    "FX Majors": """- Respect support and resistance cleanly during Asian session; breaks occur during London/New York overlap.
- Avoid placing SLs at round numbers where central banks defend.
- SL buffer: 1.5x ATR standard; increase to 2x ATR during high-impact news windows.""",
    "FX Crosses": """- Tighter trading ranges and lower volume; wider spreads during Asian/early European sessions.
- Prioritize entries during London/New York open for tighter fills.
- Commodity-linked minors spike on commodity price shocks; monitor correlation pairs.
- Use 1.5x to 2x ATR for SL buffer; be conservative on position sizing.""",
    "Metals": """- Move incredibly fast during London and New York session opens.
- Avoid tight SLs at obvious structural levels; use 1.5x to 2.5x ATR cushion, especially around major economic data.
- Silver exhibits higher volatility; increase buffer to 2x to 2.5x ATR.
- Spread tolerance critical (3–5 pips possible during quiet hours); tighten TP or abort if spread exceeds 15% of SL distance.""",
    "Indices": """- Highly correlated with equity market open times; avoid tight SLs at round numbers.
- Use 1.5x to 2x ATR on shorter timeframes; reduce to 1.2x ATR on daily+ charts for longer-term setups.
- Monitor overnight gaps; Monday opens often show directional reversals.""",
    "Futures": """- Sensitive to inventory reports and geopolitical shocks; avoid trading 30 minutes before scheduled releases.
- Use 2x to 3x ATR for SL buffer on shorter timeframes (3x ATR is the hard ceiling).
- Micro contracts allow tighter SLs (1.5x ATR) due to lower notional risk.
- Avoid trading 1–2 days before contract expiration due to liquidity migration.""",
    "Energies": """- Driven by OPEC+ decisions, geopolitical tensions, and strategic reserve releases.
- Correlate closely with the US Dollar; monitor DXY for directional bias.
- Extremely volatile on weather forecasts and inventory data; use 2.5x to 3x ATR buffer.
- Avoid trading 1 hour before scheduled reports; spreads widen 2–5 pips during liquid hours."""
}

def _get_historical_timeframe(interval: str) -> tuple[str, str]:
    """Map interval to recommended historical data timeframe."""
    if interval in ["1m", "2m", "5m"]:
        return ("1 day", "1d")
    elif interval in ["15m", "30m"]:
        return ("5 days", "5d")
    elif interval == "1h":
        return ("1 month", "1mo")
    elif interval == "4h":
        return ("3 months", "3mo")
    elif interval == "1d":
        return ("6 months", "6mo")
    elif interval == "5d":
        return ("2 years", "2y")
    elif interval == "1wk":
        return ("5 years", "5y")
    elif interval == "1mo":
        return ("max", "max")
    return ("1 month", "1mo")


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
        return "markets closed; Crypto open"
    
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


def _is_market_open(symbol: str, market: str, now_utc: datetime) -> bool:
    """Check if a specific market/symbol is open at the given UTC time.

    Args:
        symbol: The trading symbol (e.g., "EURUSD", "AUS200", "DOLLAR")
        market: The market category (e.g., "Indices", "Cryptos", "FX Crosses")
        now_utc: Current UTC time

    Returns:
        True if the market is open, False otherwise.
    """
    # Crypto markets are 24/7 and should always be considered open
    if market == "Cryptos":
        return True
    
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
            
            # Adjust hours for early Friday closure and late Monday opening
            adjusted_open_hour = hours["open_hour"]
            adjusted_open_minute = hours["open_minute"]
            adjusted_close_hour = hours["close_hour"]
            adjusted_close_minute = hours["close_minute"]
            
            # On Monday, open 3 hours later than usual
            if weekday == 0:
                adjusted_open_hour += 3
            
            # On Friday, close 3 hours earlier than usual
            if weekday == 4:
                adjusted_close_hour -= 3
            
            # Convert session times from local to UTC
            open_utc_h, open_utc_m = _convert_local_time_to_utc(
                now_utc, tz_name, adjusted_open_hour, adjusted_open_minute
            )
            close_utc_h, close_utc_m = _convert_local_time_to_utc(
                now_utc, tz_name, adjusted_close_hour, adjusted_close_minute
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
            
            # Global Weekend Closure with 2-hour buffers:
            # Close 2 hours early on Friday (19:00 UTC instead of 21:00 UTC)
            # Open 2 hours late on Monday (05:00 UTC instead of 03:00 UTC)
            if (weekday == 4 and hour >= 19) or (weekday == 5) or (weekday == 6) or (weekday == 0 and hour < 5):
                return False
            
            return True
    
    # Fallback for symbols not in MARKET_HOURS (Forex, Metals)

    # Global Weekend Closure with 2-hour buffers:
    # Close 2 hours early on Friday (19:00 UTC instead of 21:00 UTC)
    # Open 2 hours late on Monday (05:00 UTC instead of 03:00 UTC)
    if (weekday == 4 and hour >= 19) or (weekday == 5) or (weekday == 6) or (weekday == 0 and hour < 5):
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

    result = await call_mt5_tool("get_all_positions", {}, timeout=5.0)
    if not result or not getattr(result, "content", None):
        return set()

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


async def _get_open_position_comments() -> dict[str, str]:
    """Map position_id -> comment by scanning MT5 order history.

    ``get_all_positions`` does not include the order comment, so we pull it
    from the historical orders feed (which carries a ``position_id`` and a
    ``comment`` column) to enrich each open position with its original entry
    comment. Returns an empty dict if MT5 MCP is unavailable.

    Returns:
        Dict mapping position id (str) -> order comment.
    """
    result = await call_mt5_tool("get_orders", {}, timeout=10.0)
    comments: dict[str, str] = {}
    if not result or not getattr(result, "content", None):
        return comments

    for content in result.content:
        text = getattr(content, "text", "")
        rows = list(csv.reader(io.StringIO(text or "")))
        # Header: time_setup,ticket,...,position_id,...,symbol,comment,external_id
        header = [h.strip() for h in (rows[0] if rows else [])]
        try:
            pos_idx = header.index("position_id")
            comment_idx = header.index("comment")
        except ValueError:
            continue
        for row in rows[1:]:
            if len(row) <= max(pos_idx, comment_idx):
                continue
            pos_id = row[pos_idx].strip()
            comment = row[comment_idx].strip()
            if pos_id and not comments.get(pos_id):
                comments[pos_id] = comment

    logger.debug(f"Resolved {len(comments)} position comments from order history")
    return comments


async def get_open_positions(
    losing_positions_only: bool = False,
    winning_positions_only: bool = False,
    exclude_crypto: bool = False,
) -> dict[str, Any]:
    """Fetch currently open MetaTrader positions whose markets are actually open.

    Returns a list of dicts with ``symbol``, ``position_id`` (ticket),
    ``comment`` (entry order comment), and ``unrealized_profit`` (current
    floating P/L). Only positions in markets that are currently open for trading
    are returned. When ``losing_positions_only`` is True, only those with a
    negative unrealized profit are additionally included.

    Args:
        losing_positions_only: If True, return only positions that currently
            have an unrealized loss (negative floating profit). Defaults to False.
        winning_positions_only: If True, return only positions that currently
            have an unrealized profit (positive floating profit). Defaults to False.
            Note: if both this and ``losing_positions_only`` are set to True, a
            warning is logged and the direction filter is ignored.
        exclude_crypto: If True, exclude all crypto positions from the results,
            keeping only positions in non-crypto markets. Uses ``_fetch_mt5_symbols``
            to determine each symbol's market. Defaults to False.

    Returns:
        Dict with keys ``positions`` (list of position dicts) and ``count``.
        An empty list is returned if MT5 MCP is unavailable or there are no
        open/tradeable positions.
    """
    # If both direction filters are requested they contradict each other.
    # Warn and proceed as if neither were set (return all positions).
    if winning_positions_only and losing_positions_only:
        logger.warning(
            "Both winning_positions_only and losing_positions_only are True; "
            "ignoring the direction filter and returning all positions"
        )
        winning_positions_only = False
        losing_positions_only = False

    result = await call_mt5_tool("get_all_positions", {}, timeout=5.0)
    if not result or not getattr(result, "content", None):
        return {"positions": [], "count": 0}

    comments = await _get_open_position_comments()

    # Build symbol -> market map so we can determine which markets are open.
    symbols_by_market = await _fetch_mt5_symbols()
    symbol_to_market: dict[str, str] = {}
    if symbols_by_market:
        for market, syms in symbols_by_market.items():
            for s in syms:
                symbol_to_market[s.upper()] = market

    now_utc = datetime.now(timezone.utc)

    positions: list[dict[str, Any]] = []
    for content in result.content:
        text = getattr(content, "text", "")
        rows = list(csv.reader(io.StringIO(text or "")))
        # Header: ,id,time,symbol,type,volume,open,stop_loss,take_profit,profit
        header = [h.strip() for h in (rows[0] if rows else [])]
        try:
            id_idx = header.index("id")
            symbol_idx = header.index("symbol")
            profit_idx = header.index("profit")
        except ValueError:
            continue
        for row in rows[1:]:
            if len(row) <= max(id_idx, symbol_idx, profit_idx):
                continue

            pos_id = row[id_idx].strip()
            symbol = row[symbol_idx].strip().upper()

            # Skip positions whose market is not currently open for trading.
            # If the symbol isn't found in the MT5 list, assume its market is open.
            market = symbol_to_market.get(symbol)
            if market is None:
                logger.debug(f"Symbol {symbol} not found in MT5 list; assuming market open")
            elif not _is_market_open(symbol, market, now_utc):
                continue

            # Optionally exclude all crypto positions.
            if exclude_crypto and market == "Cryptos":
                logger.debug(f"Excluding crypto position {symbol}")
                continue

            try:
                profit = float(row[profit_idx])
            except (TypeError, ValueError):
                profit = None

            if losing_positions_only and (profit is None or profit >= 0):
                continue

            if winning_positions_only and (profit is None or profit <= 0):
                continue

            positions.append({
                "symbol": symbol,
                "position_id": pos_id,
                "comment": comments.get(pos_id, ""),
                "unrealized_profit": profit,
            })

    logger.info(
        f"Fetched {len(positions)} open positions "
        f"(losing_only={losing_positions_only}, winning_only={winning_positions_only})"
    )
    return {"positions": positions, "count": len(positions)}


async def _get_margin_level() -> float | None:
    """Fetch the account margin level percentage from MetaTrader.

    Returns:
        Margin level as a percentage (equity / margin * 100), or None if
        MT5 MCP is unavailable or the value cannot be parsed.
    """
    account_info = await fetch_mt5_account_info(timeout=5.0)
    if not isinstance(account_info, dict):
        return None

    margin_level = account_info.get("margin_level")
    if margin_level is None:
        logger.debug(f"Account info missing margin_level: {account_info}")
        return None

    try:
        return float(margin_level)
    except (TypeError, ValueError):
        logger.warning(f"Invalid margin_level value: {margin_level!r}")
        return None


_symbols_cache: dict[str, list[str]] | None = None


async def _fetch_mt5_symbols(timeout: float = 30.0) -> dict[str, list[str]] | None:
    """Fetch all available symbols from MT5 MCP and organize by market.

    Parses the market\\symbol format returned by get_symbols(fields=["path"]),
    filters by ALLOWED_SYMBOL_MARKETS and ALLOWED_CRYPTOS, and returns
    a dict with market names as keys and lists of symbols as values.

    Returns:
        Dict mapping market names to lists of symbols, or None on failure.
        Example: {"FX Crosses": ["EURUSD", "GBPUSD"], "Cryptos": ["BTCUSD", "ETHUSD"]}
    """
    global _symbols_cache

    if _symbols_cache is not None:
        return _symbols_cache

    result = await call_mt5_tool("get_symbols", {"fields": ["path"]}, timeout=timeout)
    if not result or not getattr(result, "content", None):
        logger.warning("MT5 MCP get_symbols returned no content")
        return None

    symbols_by_market: dict[str, list[str]] = {}
    total_parsed = 0
    total_filtered = 0
    
    for content in result.content:
        text = getattr(content, "text", "")
        if not text or not text.strip():
            continue
        
        # Parse "Market\\Symbol" format
        parts = text.strip().split("\\", 1)
        if len(parts) != 2:
            logger.debug(f"Unexpected symbol format (not Market\\Symbol): {text}")
            continue
        
        market = parts[0].strip()
        symbol = parts[1].strip()
        total_parsed += 1
        
        # Filter by allowed markets
        if market not in ALLOWED_SYMBOL_MARKETS:
            logger.debug(f"Symbol {symbol} market '{market}' not in allowed markets")
            continue
        
        # For Cryptos, apply regex filter
        if market == "Cryptos":
            if not any(re.match(pattern, symbol) for pattern in ALLOWED_CRYPTOS):
                logger.debug(f"Crypto symbol {symbol} does not match allowed patterns")
                continue
        
        total_filtered += 1
        if market not in symbols_by_market:
            symbols_by_market[market] = []
        symbols_by_market[market].append(symbol)
    
    if not symbols_by_market:
        logger.warning(f"No symbols available after filtering (parsed: {total_parsed}, passed filter: {total_filtered})")
        return None
    
    logger.info(f"Fetched {total_parsed} symbols, filtered to {total_filtered} across {len(symbols_by_market)} markets")
    for market, syms in symbols_by_market.items():
        logger.info(f"  {market}: {len(syms)} symbols")
    
    _symbols_cache = symbols_by_market
    return _symbols_cache


async def _validate_symbol(symbol: str) -> tuple[bool, dict]:
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
    s = await fetch_mt5_symbol_info(symbol, timeout=15.0)
    if isinstance(s, dict) and s:
        check_fields = ["ask", "bid", "trade_contract_size", "trade_tick_size", "trade_tick_value", "volume_step", "volume_max", "volume_min"]
        missing_check_fields = [field for field in check_fields if field not in s or not s[field]]
        if not missing_check_fields:
            logger.debug(f"Symbol {symbol} validated successfully")
            return True, s

        logger.warning(f"Symbol {symbol} validation failed: missing required fields: {missing_check_fields}")
        return False, s

    logger.warning(f"Symbol {symbol} validation returned empty response")
    return False, {}


async def pick_market(
    max_positions: int = 10,
    minimum_margin_percent: float = 500.0,
    intervals: list[str] = INTERVALS,
    generate_chart: bool = True,
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
    if minimum_margin_percent > 0 and num_positions > 0:
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
    
    # Step 3: Fetch all available symbols from MT5 MCP, organized by market and filtered
    symbols_by_market = await _fetch_mt5_symbols()
    if not symbols_by_market:
        msg = "Failed to fetch symbols from MT5 MCP; Abort immediately"
        logger.error(msg)
        return {"error": msg}

    # Step 4: Build available symbols list from all markets, excluding open positions
    # Structure: [(market, symbol), ...]
    available_candidates: list[tuple[str, str]] = []
    for market, symbols in symbols_by_market.items():
        for symbol in symbols:
            if symbol not in open_symbols and symbol.upper() not in DISALLOWED_SYMBOLS:
                available_candidates.append((market, symbol))
    
    if not available_candidates:
        msg = "All symbols are excluded (open positions or not tradeable); Abort immediately"
        logger.warning(msg)
        return {"error": msg}
    
    logger.debug(f"Available market/symbol candidates for trading: {len(available_candidates)}")
    
    # Step 5: Get random interval and associated timeframe
    valid_intervals = [i for i in intervals if i in ALL_INTERVALS]
    if not valid_intervals:
        valid_intervals = INTERVALS
    interval = random.choice(valid_intervals)
    timeframe = _get_historical_timeframe(interval)

    # Step 6: Pick a symbol and validate it (with retry)
    random.shuffle(available_candidates)  # Randomize the list
    picked_symbol = None
    picked_market = None
    
    for market, symbol in available_candidates:
        # Check if market is open
        if not _is_market_open(symbol, market, now_utc):
            logger.debug(f"Market {market} is closed, skipping {symbol}")
            continue
                
        is_valid, symbol_info = await _validate_symbol(symbol)
        if not is_valid:
            logger.info(f"Symbol {symbol} validation failed, skipping")
            continue
        
        try:
            records = await get_historical_data(symbol, interval=interval, period=timeframe[1])
            if not records or len(records) < 2:
                records = None
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            continue
        
        # Market is open and symbol is valid, proceed with trade setup
        trade_setups = await calculate_trade_setups(symbol, timeframe[1], interval, symbol_info=symbol_info, input_records=records)
        if "status" in trade_setups and "Abort" in trade_setups["status"]:
            logger.info(f"Trade setups for {symbol} indicate abort: {trade_setups['status']}")
            continue
        if generate_chart:
            try:
                chart_data = await _generate_chart(symbol, interval=interval, period=timeframe[1], input_records=records)
            except Exception as e:
                logger.error(f"Failed to generate chart for {symbol}: {e}")
                continue
        else:
            chart_data = None
        
        for setup_key, direction in (("long_buy_setup", "long"), ("short_sell_setup", "short")):
            if "Abort" not in trade_setups[setup_key].get("status", ""):
                position_size = await calculate_mt5_position_size(
                    symbol,
                    direction,
                    trade_setups[setup_key]["entry"],
                    trade_setups[setup_key]["stop_loss"]
                )
                print(position_size)
                trade_setups[setup_key]["position_risk"] = {
                    "position_lots": position_size.data["position_lots"],
                    "actual_risk": position_size.data["actual_risk"],
                    "actual_risk_pct": position_size.data["actual_risk_pct"]
                }
                tmp_chart_data = await _generate_chart(
                    symbol, interval=interval, period=timeframe[1], input_records=records,
                    entry_price=trade_setups[setup_key]["entry"],
                    stop_loss_price=trade_setups[setup_key]["stop_loss"],
                    take_profit_price=trade_setups[setup_key]["take_profit"]
                )
                trade_setups[setup_key][f"{direction}_chart_path"] = tmp_chart_data["path"]
        print(trade_setups)
        
        picked_symbol = symbol
        picked_market = market
        break
    
    if not picked_symbol:
        msg = "No valid symbols available to trade on; Abort immediately"
        logger.error(msg)
        return {"error": msg}

    # Step 7: Return 
    
    result = {
        "market": picked_market,
        "symbol": picked_symbol,
        "desc": symbol_info.get("description") if symbol_info else None,
        "interval": interval,
        "timeframe": timeframe,
        "trading_sessions": _get_trading_sessions(now_utc),
        "time_utc": now_utc,
        "time_local": now_local,
        "market_adjustment": MARKET_ADJUSTMENTS.get(picked_market, None),
        "symbol_data": {
            "spread": symbol_info.get("spread") if symbol_info else None,
            "ask": symbol_info.get("ask") if symbol_info else None,
            "bid": symbol_info.get("bid") if symbol_info else None,
            "trade_contract_size": symbol_info.get("trade_contract_size") if symbol_info else None,
            "trade_tick_size": symbol_info.get("trade_tick_size") if symbol_info else None,
            "trade_tick_value": symbol_info.get("trade_tick_value") if symbol_info else None,
            "volume_max": symbol_info.get("volume_max") if symbol_info else None,
            "volume_min": symbol_info.get("volume_min") if symbol_info else None,
            "volume_step": symbol_info.get("volume_step") if symbol_info else None
        },
        "chart_data": chart_data,
        # "num_open_positions": num_positions,
        # "allowed_additional_positions": allowed_additional_positions,
    }
    result |= trade_setups
    
    logger.info(f"Picked symbol: {picked_symbol}, interval: {interval}")
    return result

"""
Market data module with multi-source fallback.

Provides clean OHLCV data retrieval with ISO-formatted timestamps,
wrapped in async for non-blocking MCP tool calls.

Source priority (each tried in order until one succeeds):
  1. **yfinance** (default, always available)
  2. **MetaTrader MCP** (if ``MT5_MCP_URL`` env var is set)
  3. **TwelveData** (if ``TWELVEDATA_API_KEY`` env var is set)

Each source returns records in a unified format:
    ``{"date": ISO-8601, "open": float, "high": float, "low": float,
       "close": float, "volume": int, "source": str}``
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from mcp_server.utils.mt5_mcp_server import MT5_MCP_URL, call_mt5_tool, get_mt5_client

load_dotenv()

logger = logging.getLogger(__name__)

from mcp_server.cache import smart_cache

# yfinance is NOT thread-safe — concurrent downloads corrupt column dtypes
# (causes 'ufunc isnan not supported' errors). Serialize all downloads.
_yf_lock = threading.Lock()

# Valid periods and intervals accepted by yfinance
VALID_PERIODS = {
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
}
VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}

# ── Source Configuration ──────────────────────────────────────────────────────

TWELVEDATA_API_KEY: str = os.getenv("TWELVEDATA_API_KEY", "").strip()

# Interval mapping: yfinance → MetaTrader MCP timeframe codes
_INTERVAL_TO_MT5 = {
    "1m": "M1", "2m": "M2", "5m": "M5", "15m": "M15", "30m": "M30",
    "60m": "H1", "1h": "H1", "90m": "H1",
    "1d": "D1", "5d": "D1",
    "1wk": "W1", "1mo": "MN1", "3mo": "MN1",
}

# Interval mapping: yfinance → TwelveData interval codes
_INTERVAL_TO_TWELVE = {
    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "1h", "1h": "1h", "90m": "90min",
    "1d": "1day", "5d": "5day",
    "1wk": "1week", "1mo": "1month", "3mo": "1month",
}

# Approximate trading-day counts per period
_PERIOD_TRADING_DAYS = {
    "1d": 1, "5d": 5, "1mo": 22, "3mo": 65, "6mo": 130,
    "1y": 252, "2y": 504, "5y": 1260, "10y": 2520,
    "ytd": 200, "max": 5000,
}

# Bars per calendar day for intraday intervals.
#
# This module supports forex / metals / crypto via MT5 MCP (and TwelveData),
# which trade ~24h/day — NOT the US equity market's 6.5-hour session. Using a
# full-day bar count ensures we request enough candles to actually span the
# requested *period* for these instruments (e.g. a month of H1 forex needs
# ~500+ bars, not 154). Requesting more than exists is harmless: MT5 returns
# what's available and _date_range_covers_period() checks coverage separately.
_INTRADAY_BARS_PER_DAY = {
    "1m": 1440, "2m": 720, "5m": 288, "15m": 96,
    "30m": 48, "60m": 24, "1h": 23, "90m": 16,
}


def _period_to_count(period: str, interval: str) -> int:
    """Approximate number of bars needed to cover *period* at *interval*."""
    trading_days = _PERIOD_TRADING_DAYS.get(period, 65)
    bars_per_day = _INTRADAY_BARS_PER_DAY.get(interval)
    if bars_per_day:
        count = bars_per_day * trading_days
    else:
        count = trading_days
    return min(max(count, 1), 5000)


def _date_range_covers_period(records: list[dict[str, Any]], period: str) -> bool:
    """Check if the date range of records covers the requested period.
    
    For calendar-based periods (1d, 5d, etc.), verify that the bars span
    at least ~90% of the expected calendar days.
    """
    if not records or len(records) < 2:
        return False
    
    # Parse first and last timestamps
    from dateutil import parser
    try:
        first_dt = parser.parse(records[0]["date"])
        last_dt = parser.parse(records[-1]["date"])
    except Exception:
        return True  # If we can't parse, assume OK
    
    # Map period to expected calendar days
    period_days_map = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
        "ytd": 365, "max": 3650,
    }
    
    expected_days = period_days_map.get(period, 30)
    actual_days = (last_dt - first_dt).days
    
    # Accept if we cover at least 70% of expected calendar days
    return actual_days >= (expected_days * 0.70)


# Approximate maximum age (in seconds) for the most recent bar before a dataset
# is considered stale. The MT5 MCP server intermittently returns candles that are
# hours/days old even though fresher data exists; without this guard those stale
# bars get charted and cached, so charts silently miss the latest price action.
#
# We use a deliberately generous threshold (e.g. 4x the bar interval) because:
#   - During market closures / weekends the newest completed bar is naturally old.
#   - A just-forming intraday candle may lag "now" by up to one full bar length.
_MAX_BAR_AGE_SECONDS = {
    # intraday: allow a few bars of slack
    "1m": 4 * 60, "2m": 8 * 120, "5m": 6 * 300,
    "15m": 6 * 900, "30m": 6 * 1800, "60m": 6 * 3600, "90m": 3 * 5400,
    # daily and above: allow ~4 calendar days (covers weekends + holidays)
    "1d": 5 * 86400, "5d": 8 * 43200, "1wk": 10 * 604800,
    "1mo": 40 * 2592000, "3mo": 60 * 7776000,
}


def _is_data_fresh(records: list[dict[str, Any]], interval: str) -> bool:
    """Return True if the most recent bar is not stale relative to now.

    MT5 (and occasionally other sources) can return a dataset whose newest
    candle predates "now" by far more than one bar — e.g. Friday's last candle
    when Monday data already exists, or even several days back. Detecting this
    lets the caller retry / fall back instead of charting stale price action.

    The threshold is intentionally loose (see ``_MAX_BAR_AGE_SECONDS``) so that
    weekend and holiday closures don't trip it.
    """
    if not records:
        return False

    from dateutil import parser as _parser
    try:
        last_dt = _parser.parse(records[-1]["date"])
    except Exception:
        # Can't parse the timestamp — assume fresh rather than block on a format quirk.
        return True

    max_age = _MAX_BAR_AGE_SECONDS.get(interval, 5 * 86400)
    age = datetime.now(timezone.utc) - last_dt
    if age.total_seconds() <= max_age:
        return True

    logger.warning(
        "Stale data detected for interval=%s: newest bar %s is %.1fh old (max allowed %.1fh).",
        interval, records[-1]["date"], age.total_seconds() / 3600.0, max_age / 3600.0,
    )
    return False


def _is_source_available(source: str) -> bool:
    """Return True if the given source is configured and reachable."""
    if source == "yfinance":
        return True
    if source == "mt5_mcp":
        return bool(MT5_MCP_URL)
    if source == "twelvedata":
        return bool(TWELVEDATA_API_KEY)
    return False


@smart_cache(open_ttl=300, closed_ttl=3600)
async def get_historical_data(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
    fallback_for_incomplete_data: bool = True,
) -> list[dict[str, Any]]:
    """Fetch OHLCV historical data for any symbol.

    Tries multiple data sources in order until one succeeds:
      1. MetaTrader MCP (if ``MT5_MCP_URL`` is configured) — supports any MT5 symbol
      2. TwelveData (if ``TWELVEDATA_API_KEY`` is configured) — stocks, forex, crypto
      3. yfinance (default) — primarily stocks and crypto

    Works with stocks, forex, metals, indices, crypto, or any MT5-tradeable symbol.

    Args:
        ticker: Symbol name (e.g., ``"AAPL"``, ``"MSFT"``, ``"XAUUSD"``, ``"EURUSD"``).
            For MT5 MCP, use native MT5 symbol names. For yfinance/TwelveData, use appropriate
            ticker/crypto pair formats (e.g., EUR-USD, EURUSD, GC=F for gold).
        period: Lookback period. One of: ``1d``, ``5d``, ``1mo``, ``3mo``,
            ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``, ``ytd``, ``max``.
            Defaults to ``"3mo"``.
        interval: Bar interval. One of: ``1m``, ``2m``, ``5m``, ``15m``,
            ``30m``, ``60m``, ``90m``, ``1h``, ``1d``, ``5d``, ``1wk``,
            ``1mo``, ``3mo``. Defaults to ``"1d"``.
        fallback_for_incomplete_data: When ``True`` (default), if a source
            returns data that doesn't span the full requested *period*, the
            next source in the chain is tried. When ``False``, partial data
            is accepted as-is — useful for callers (e.g. trade setup
            calculators) that don't actually need the full requested window.

    Returns:
        A list of dicts with keys: ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``source``. Dates are ISO-8601 formatted
        strings.

    Raises:
        ValueError: If *period* or *interval* is invalid, or all configured
            data sources fail.
    """
    ticker = ticker.strip().upper()

    if period == "1M":
        period = "1mo"

    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Must be one of: {sorted(VALID_PERIODS)}"
        )
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Must be one of: {sorted(VALID_INTERVALS)}"
        )

    # Build the source chain — only include configured sources
    sources: list[tuple[str, Any]] = [
        ("mt5_mcp", _fetch_from_mt5),
        ("twelvedata", _fetch_from_twelvedata),
        ("yfinance", _fetch_from_yfinance),
    ]

    errors: list[str] = []
    
    for source_name, fetcher in sources:
        if not _is_source_available(source_name):
            logger.debug("Source %s not configured, skipping", source_name)
            continue
        try:
            if source_name == "yfinance":
                records = await asyncio.to_thread(fetcher, ticker, period, interval)
            else:
                records = await fetcher(ticker, period, interval)
            if records:
                # Check if data covers the requested period; if not, try next source
                if not _date_range_covers_period(records, period):
                    msg = (
                        f"Incomplete date range from {source_name}: got {len(records)} bars "
                        f"spanning only {(pd.to_datetime(records[-1]['date']) - pd.to_datetime(records[0]['date'])).days} days "
                        f"for {ticker} (requested {period})."
                    )
                    if fallback_for_incomplete_data:
                        msg += " Trying next source..."
                        logger.warning(msg)
                        errors.append(msg)
                        continue  # Try next source
                    # Caller is OK with partial data — accept it
                    logger.info(
                        "%s Accepting partial data from %s (fallback_for_incomplete_data=False).",
                        msg, source_name,
                    )
                    return records
                logger.info(
                    "Fetched %d bars for %s from %s (period=%s, interval=%s)",
                    len(records), ticker, source_name, period, interval,
                )
                return records
        except Exception as exc:
            msg = f"{source_name}: {exc}"
            logger.warning("Data source failed for %s — %s", ticker, msg)
            errors.append(msg)

    raise ValueError(
        f"All data sources failed or returned incomplete data for '{ticker}' "
        f"(period={period}, interval={interval}). "
        f"Errors: {'; '.join(errors)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: yfinance (primary)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_from_yfinance(
    ticker: str, period: str, interval: str
) -> list[dict[str, Any]]:
    """Fetch OHLCV from yfinance. Tries multiple ticker formats for forex/crypto pairs.
    
    For crypto (e.g., BTCUSD), tries formats: BTC-USD, BTCUSD, BTCUSD=X.
    For forex (e.g., EURUSD), tries formats: EURUSD=X, EUR-USD, EURUSD.
    """
    with _yf_lock:
        # Build list of ticker formats to try, ordered by likelihood of success
        ticker_formats = [ticker]

        # For 6-letter alphabetic tickers, try common variations
        if len(ticker) == 6 and ticker.isalpha():
            base = ticker[:3]
            quote = ticker[3:]

            # Detect likely crypto pairs (quote = USD/USDT/USDC/BTC/ETH)
            crypto_quotes = {"USD", "USDT", "USDC", "BTC", "ETH"}
            if quote in crypto_quotes:
                # Crypto on Yahoo Finance uses dash format: BTC-USD
                ticker_formats = [
                    f"{base}-{quote}",     # BTC-USD (yfinance crypto format)
                    ticker,                # BTCUSD (fallback)
                    f"{base}{quote}=X",   # BTCUSD=X (won't work for crypto but try anyway)
                ]
            else:
                # Forex pairs on Yahoo Finance use =X suffix: EURUSD=X
                ticker_formats = [
                    f"{base}{quote}=X",    # EURUSD=X (yfinance forex format)
                    f"{base}-{quote}",     # EUR-USD (alternative)
                    ticker,                # EURUSD (fallback)
                ]

        last_exc = None
        for fmt in ticker_formats:
            try:
                df = yf.download(
                    fmt,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if df is not None and not df.empty:
                    logger.debug("yfinance succeeded with format: %s", fmt)
                    break
            except Exception as exc:
                last_exc = exc
                logger.debug("yfinance format %s failed: %s", fmt, exc)
                continue
        else:
            # All formats failed
            logger.error("yfinance download failed for %s (all formats): %s", ticker, last_exc)
            raise ValueError(
                f"yfinance failed for '{ticker}' (tried formats: {', '.join(ticker_formats)}): {last_exc}"
            ) from last_exc

        if df is None or df.empty:
            raise ValueError(
                f"yfinance returned no data for '{ticker}' "
                f"(period={period}, interval={interval})"
            )

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        date_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        records.append(
            {
                "date": date_str,
                "open": _round(row.get("Open")),
                "high": _round(row.get("High")),
                "low": _round(row.get("Low")),
                "close": _round(row.get("Close")),
                "volume": _safe_int(row.get("Volume", 0)),
                "source": "yfinance",
            }
        )
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2: MetaTrader MCP (SSE)
# ═══════════════════════════════════════════════════════════════════════════════


async def _fetch_from_mt5(
    ticker: str, period: str, interval: str
) -> list[dict[str, Any]]:
    """Fetch OHLCV from the MetaTrader MCP server. Raises ValueError on failure.

    The MT5 MCP server intermittently returns stale candles (newest bar hours or
    days old even when fresher data exists). Because that staleness is transient,
    we retry a few times and only raise if every attempt comes back stale.
    """
    mt5_tf = _INTERVAL_TO_MT5.get(interval)
    if mt5_tf is None:
        raise ValueError(f"MetaTrader MCP does not support interval '{interval}'")

    count = _period_to_count(period, interval)

    attempts = 3
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        try:
            result = await call_mt5_tool(
                "get_candles_latest",
                {"symbol_name": ticker, "timeframe": mt5_tf, "count": count},
            )
        except Exception as exc:
            raise ValueError(f"MetaTrader MCP failed for '{ticker}': {exc}") from exc

        # Extract CSV text from the response
        csv_text: str | None = None
        for content in result.content:
            if hasattr(content, "text"):
                csv_text = content.text
                break
        if not csv_text:
            raise ValueError(f"MetaTrader MCP returned empty response for '{ticker}'")

        # Parse CSV — columns: ,time,open,high,low,close,tick_volume,spread,real_volume
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as exc:
            raise ValueError(f"MetaTrader MCP CSV parse failed for '{ticker}': {exc}") from exc

        if df is None or df.empty:
            raise ValueError(f"MetaTrader MCP returned no data for '{ticker}'")

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            # Normalize timestamp to ISO-8601
            ts = row.get("time")
            if pd.isna(ts):
                continue
            ts_str = str(ts)
            # Convert "2026-07-22 00:00:00+00:00" → "2026-07-22T00:00:00+00:00"
            if " " in ts_str and "T" not in ts_str:
                ts_str = ts_str.replace(" ", "T", 1)

            records.append(
                {
                    "date": ts_str,
                    "open": _round(row.get("open")),
                    "high": _round(row.get("high")),
                    "low": _round(row.get("low")),
                    "close": _round(row.get("close")),
                    "volume": _safe_int(row.get("tick_volume", 0)),
                    "source": "mt5_mcp",
                }
            )

        # Sort ascending by date (MT5 returns newest-first)
        records.sort(key=lambda r: r["date"])

        if _is_data_fresh(records, interval):
            return records

        last_error = (
            f"MetaTrader MCP returned stale data for '{ticker}' "
            f"(newest bar {records[-1]['date']}, attempt {attempt}/{attempts})"
        )
        logger.warning("%s — retrying...", last_error)

    raise ValueError(last_error or "MetaTrader MCP returned stale data")


# ═══════════════════════════════════════════════════════════════════════════════
# Source 3: TwelveData (REST)
# ═══════════════════════════════════════════════════════════════════════════════

def _twelvedata_symbol_formats(ticker: str) -> list[str]:
    """Return ordered list of TwelveData symbol formats to try for *ticker*.

    TwelveData expects:
      - Crypto pairs: ``BTC/USD`` (slash form)
      - Forex pairs: ``EUR/USD`` (slash form)
      - Stocks: ``AAPL`` (plain)

    For a 6-letter alphabetic ticker like ``BTCUSD`` or ``EURUSD`` we try the
    slash form first, then the plain form. For anything else we just return
    the ticker as-is.
    """
    if len(ticker) == 6 and ticker.isalpha():
        base = ticker[:3]
        quote = ticker[3:]
        return [
            f"{base}/{quote}",   # BTC/USD or EUR/USD (TwelveData canonical)
            ticker,              # BTCUSD / EURUSD (fallback)
        ]
    return [ticker]


async def _fetch_from_twelvedata(
    ticker: str, period: str, interval: str
) -> list[dict[str, Any]]:
    """Fetch OHLCV from TwelveData. Raises ValueError on failure.

    Tries multiple symbol formats (e.g. ``BTC/USD`` then ``BTCUSD``) until
    one returns data, so callers can pass an MT5-style or yfinance-style
    ticker without worrying about TwelveData's slash convention.
    """
    td_interval = _INTERVAL_TO_TWELVE.get(interval)
    if td_interval is None:
        raise ValueError(f"TwelveData does not support interval '{interval}'")

    outputsize = _period_to_count(period, interval)
    symbol_formats = _twelvedata_symbol_formats(ticker)

    def _fetch_one(symbol: str) -> pd.DataFrame:
        try:
            from twelvedata import TDClient
            td = TDClient(apikey=TWELVEDATA_API_KEY)
            df = td.time_series(
                symbol=symbol,
                interval=td_interval,
                outputsize=outputsize,
                order="ASC",
            ).as_pandas()
            if df is None or df.empty:
                raise ValueError(
                    f"TwelveData returned no data for '{symbol}' "
                    f"(period={period}, interval={interval})"
                )
            return df
        except Exception as exc:
            # Re-raise as ValueError so the fallback chain handles it uniformly
            raise ValueError(str(exc)) from exc

    last_exc: Exception | None = None
    df: pd.DataFrame | None = None
    for symbol in symbol_formats:
        try:
            df = await asyncio.to_thread(_fetch_one, symbol)
            logger.debug("TwelveData succeeded with symbol: %s", symbol)
            break
        except ValueError as exc:
            last_exc = exc
            logger.debug("TwelveData symbol %s failed: %s", symbol, exc)
            continue

    if df is None:
        assert last_exc is not None
        raise ValueError(
            f"TwelveData failed for '{ticker}' "
            f"(tried formats: {', '.join(symbol_formats)}): {last_exc}"
        ) from last_exc

    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        date_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        records.append(
            {
                "date": date_str,
                "open": _round(row.get("open")),
                "high": _round(row.get("high")),
                "low": _round(row.get("low")),
                "close": _round(row.get("close")),
                "volume": _safe_int(row.get("volume", 0)),
                "source": "twelvedata",
            }
        )
    return records


def _round(value: Any, decimals: int = 4) -> float | None:
    """Round a numeric value, returning None for NaN / missing."""
    try:
        f = float(value)
        if pd.isna(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    """Convert a value to int, handling Series/NaN gracefully."""
    try:
        if hasattr(value, "iloc"):
            value = value.iloc[0]
        f = float(value)
        if pd.isna(f):
            return 0
        return int(f)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Live Price — uses fast_info for real-time price instead of stale OHLCV close
# ---------------------------------------------------------------------------

@smart_cache(open_ttl=60, closed_ttl=300)
async def get_live_price(ticker: str) -> float:
    """Return the current/live price for a ticker.

    Uses ``yf.Ticker.fast_info.last_price`` which reflects the most recent
    trade, unlike daily OHLCV bars which only update after market close.

    During market hours this is the real-time price; after hours it equals
    the closing price.  Cached 60 s during market hours, 5 min when closed.

    Falls back to ``previous_close`` if ``last_price`` is unavailable.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Current price as a float.

    Raises:
        ValueError: If price cannot be determined.
    """
    ticker = ticker.strip().upper()

    def _fetch() -> float:
        with _yf_lock:
            try:
                info = yf.Ticker(ticker).fast_info
                price = getattr(info, "last_price", None)
                if price is None or price <= 0:
                    price = getattr(info, "previous_close", None)
                if price is None or price <= 0:
                    raise ValueError(f"No price available for '{ticker}'")
                return float(price)
            except Exception as exc:
                logger.error("Live price fetch failed for %s: %s", ticker, exc)
                raise ValueError(
                    f"Failed to get live price for '{ticker}': {exc}"
                ) from exc

    return await asyncio.to_thread(_fetch)


# ---------------------------------------------------------------------------
# Option Expirations — real chain dates with computed DTE
# ---------------------------------------------------------------------------

@smart_cache(open_ttl=300, closed_ttl=3600)
async def get_option_expirations(
    ticker: str,
    min_dte: int = 0,
    max_dte: int = 365,
) -> list[dict[str, Any]]:
    """Return real option expiration dates with days-to-expiry.

    Fetches expiration date strings from the yfinance option chain and
    computes the calendar-day DTE from today for each.

    Args:
        ticker: Stock ticker symbol.
        min_dte: Minimum DTE to include (default 0).
        max_dte: Maximum DTE to include (default 365).

    Returns:
        List of dicts, each with:
        - ``expiration``: ISO date string (``YYYY-MM-DD``)
        - ``dte``: calendar days to expiration (int)

        Sorted ascending by DTE.  Empty list if chain unavailable.
    """
    ticker = ticker.strip().upper()

    def _fetch() -> list[str]:
        try:
            return list(yf.Ticker(ticker).options)
        except Exception as exc:
            logger.warning("Option expirations fetch failed for %s: %s", ticker, exc)
            return []

    raw_dates = await asyncio.to_thread(_fetch)

    today = date.today()
    results: list[dict[str, Any]] = []
    for d_str in raw_dates:
        try:
            exp_date = datetime.strptime(d_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if min_dte <= dte <= max_dte:
            results.append({"expiration": d_str, "dte": dte})

    results.sort(key=lambda x: x["dte"])
    logger.info("Found %d expirations for %s (DTE %d–%d)", len(results), ticker, min_dte, max_dte)
    return results

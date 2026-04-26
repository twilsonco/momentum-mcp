"""
Market data module using yfinance.

Provides clean OHLCV data retrieval with ISO-formatted timestamps,
wrapped in async for non-blocking MCP tool calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

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


@smart_cache(open_ttl=300, closed_ttl=3600)
async def get_historical_data(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> list[dict[str, Any]]:
    """Fetch OHLCV historical data for a ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``, ``"MSFT"``).
        period: Lookback period. One of: ``1d``, ``5d``, ``1mo``, ``3mo``,
            ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``, ``ytd``, ``max``.
            Defaults to ``"3mo"``.
        interval: Bar interval. One of: ``1m``, ``2m``, ``5m``, ``15m``,
            ``30m``, ``60m``, ``90m``, ``1h``, ``1d``, ``5d``, ``1wk``,
            ``1mo``, ``3mo``. Defaults to ``"1d"``.

    Returns:
        A list of dicts with keys: ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume``. Dates are ISO-8601 formatted strings.

    Raises:
        ValueError: If *period* or *interval* is invalid, or the ticker
            returns no data.
    """
    ticker = ticker.strip().upper()

    if interval == "1M":
        interval = "1mo"
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

    def _download() -> pd.DataFrame:
        with _yf_lock:  # Serialize yfinance access (not thread-safe)
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
            except Exception as exc:
                logger.error("yfinance download failed for %s: %s", ticker, exc)
                raise ValueError(
                    f"Failed to fetch data for '{ticker}'. "
                    f"The ticker may be invalid or Yahoo Finance may be rate-limiting."
                ) from exc

            if df is None or df.empty:
                raise ValueError(
                    f"No data returned for ticker '{ticker}' "
                    f"(period={period}, interval={interval}). "
                    f"Verify the ticker symbol is correct."
                )
            return df

    df = await asyncio.to_thread(_download)

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Build clean JSON-serialisable output
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        date_str = (
            idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        )
        records.append(
            {
                "date": date_str,
                "open": _round(row.get("Open")),
                "high": _round(row.get("High")),
                "low": _round(row.get("Low")),
                "close": _round(row.get("Close")),
                "volume": _safe_int(row.get("Volume", 0)),
            }
        )

    logger.info(
        "Fetched %d bars for %s (period=%s, interval=%s)",
        len(records), ticker, period, interval,
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

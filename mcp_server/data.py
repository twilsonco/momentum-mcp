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
from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

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

MT5_MCP_URL: str = os.getenv("MT5_MCP_URL", "").strip()
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

# Bars per trading day for intraday intervals (6.5h = 390 min)
_INTRADAY_BARS_PER_DAY = {
    "1m": 390, "2m": 195, "5m": 78, "15m": 26,
    "30m": 13, "60m": 7, "1h": 7, "90m": 5,
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
) -> list[dict[str, Any]]:
    """Fetch OHLCV historical data for a ticker symbol.

    Tries multiple data sources in order until one succeeds:
      1. yfinance (default)
      2. MetaTrader MCP (if ``MT5_MCP_URL`` is set)
      3. TwelveData (if ``TWELVEDATA_API_KEY`` is set)

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
        ``close``, ``volume``, ``source``. Dates are ISO-8601 formatted
        strings.

    Raises:
        ValueError: If *period* or *interval* is invalid, or all configured
            data sources fail.
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

    # Build the source chain — only include configured sources
    sources: list[tuple[str, Any]] = [
        ("yfinance", _fetch_from_yfinance),
        ("mt5_mcp", _fetch_from_mt5),
        ("twelvedata", _fetch_from_twelvedata),
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
        f"All data sources failed for '{ticker}' "
        f"(period={period}, interval={interval}). Errors: {'; '.join(errors)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: yfinance (primary)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_from_yfinance(
    ticker: str, period: str, interval: str
) -> list[dict[str, Any]]:
    """Fetch OHLCV from yfinance. Raises ValueError on failure."""
    with _yf_lock:
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
                f"yfinance failed for '{ticker}': {exc}"
            ) from exc

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

class _MT5Client:
    """Lazy singleton MCP client for the MetaTrader server.

    Keeps a persistent SSE connection alive across calls.  Automatically
    reconnects on failure.
    """

    def __init__(self) -> None:
        self._cm: Any = None  # sse_client context manager
        self._session: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_connected(self) -> Any:
        """Return a live ClientSession, connecting if needed."""
        if self._session is not None:
            return self._session
        async with self._lock:
            if self._session is not None:
                return self._session
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            logger.info("MT5 MCP: connecting to %s", MT5_MCP_URL)
            self._cm = sse_client(MT5_MCP_URL)
            read, write = await self._cm.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            logger.info("MT5 MCP: connected and initialized")
            return self._session

    async def _disconnect(self) -> None:
        """Tear down the current connection (if any).

        Suppresses cleanup errors — the SSE client uses anyio task groups
        which can raise ``RuntimeError`` on Python 3.14+ when the cancel
        scope is exited from a different task than it was entered.  These
        are harmless cleanup artifacts.
        """
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except BaseException:
                pass
            self._session = None
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except BaseException:
                pass
            self._cm = None

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call a tool with one automatic reconnect on failure."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                session = await self._ensure_connected()
                return await session.call_tool(name, args)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "MT5 MCP call '%s' failed (attempt %d): %s",
                    name, attempt + 1, exc,
                )
                await self._disconnect()
        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        """Cleanly close the connection."""
        await self._disconnect()


_mt5_client: _MT5Client | None = None


def _get_mt5_client() -> _MT5Client:
    """Return the singleton MT5 client."""
    global _mt5_client
    if _mt5_client is None:
        _mt5_client = _MT5Client()
    return _mt5_client


async def _fetch_from_mt5(
    ticker: str, period: str, interval: str
) -> list[dict[str, Any]]:
    """Fetch OHLCV from the MetaTrader MCP server. Raises ValueError on failure."""
    mt5_tf = _INTERVAL_TO_MT5.get(interval)
    if mt5_tf is None:
        raise ValueError(f"MetaTrader MCP does not support interval '{interval}'")

    count = _period_to_count(period, interval)
    client = _get_mt5_client()

    try:
        result = await client.call_tool(
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
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Source 3: TwelveData (REST)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_from_twelvedata(
    ticker: str, period: str, interval: str
) -> list[dict[str, Any]]:
    """Fetch OHLCV from TwelveData. Raises ValueError on failure."""
    td_interval = _INTERVAL_TO_TWELVE.get(interval)
    if td_interval is None:
        raise ValueError(f"TwelveData does not support interval '{interval}'")

    outputsize = _period_to_count(period, interval)

    def _fetch() -> pd.DataFrame:
        try:
            from twelvedata import TDClient
            td = TDClient(apikey=TWELVEDATA_API_KEY)
            df = td.time_series(
                symbol=ticker,
                interval=td_interval,
                outputsize=outputsize,
                order="ASC",
            ).as_pandas()
            if df is None or df.empty:
                raise ValueError(
                    f"TwelveData returned no data for '{ticker}' "
                    f"(period={period}, interval={interval})"
                )
            return df
        except Exception as exc:
            # Re-raise as ValueError so the fallback chain handles it uniformly
            raise ValueError(str(exc)) from exc

    try:
        df = await asyncio.to_thread(_fetch)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"TwelveData failed for '{ticker}': {exc}") from exc

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

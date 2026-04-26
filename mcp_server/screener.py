"""
Stock screening module using tradingview-screener.

Provides preset-based stock screens that map to TradingView's scanner API,
returning filtered results with key metrics.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from tradingview_screener import Query, col

logger = logging.getLogger(__name__)

from mcp_server.cache import smart_cache

# ---------------------------------------------------------------------------
# Preset filter definitions
# ---------------------------------------------------------------------------
# Each preset is a callable that receives a Query and returns a filtered Query.

PRESET_FILTERS: dict[str, Any] = {
    "most_active": lambda q: q.order_by("volume", ascending=False),
    "new_highs": lambda q: q.where(col("High.All") == True),  # noqa: E712
    "new_lows": lambda q: q.where(col("Low.All") == True),  # noqa: E712
    "overbought": lambda q: q.where(col("RSI") > 70).order_by("RSI", ascending=False),
    "oversold": lambda q: q.where(col("RSI") < 30).order_by("RSI", ascending=True),
    "high_relative_volume": lambda q: q.where(
        col("relative_volume_10d_calc") > 2.0
    ).order_by("relative_volume_10d_calc", ascending=False),
    # --- Gap scanners ---
    "gap_up": lambda q: q.where(col("gap") > 2.0).order_by("gap", ascending=False),
    "gap_down": lambda q: q.where(col("gap") < -2.0).order_by("gap", ascending=True),
    # --- EMA Stack alignment (Michael's momentum stack) ---
    "bullish_ema_stack": lambda q: q.where(
        col("EMA20") > col("EMA50")
    ).where(
        col("EMA50") > col("EMA200")
    ).where(col("RSI") > 50).order_by("relative_volume_10d_calc", ascending=False),
    "bearish_ema_stack": lambda q: q.where(
        col("EMA20") < col("EMA50")
    ).where(
        col("EMA50") < col("EMA200")
    ).where(col("RSI") < 50).order_by("relative_volume_10d_calc", ascending=False),
    # --- Momentum ---
    "high_momentum": lambda q: q.where(
        col("ADX") > 25
    ).where(col("RSI") > 50).where(col("RSI") < 70).order_by("ADX", ascending=False),
    # --- Large cap undervalued ---
    "large_cap_undervalued": lambda q: q.where(
        col("market_cap_basic") > 10_000_000_000
    ).where(col("RSI") < 40).order_by("RSI", ascending=True),
    # --- Gainers / Losers ---
    "top_gainers": lambda q: q.where(
        col("change") > 0
    ).where(col("close") > 2).order_by("change", ascending=False),
    "biggest_losers": lambda q: q.where(
        col("change") < 0
    ).where(col("close") > 2).order_by("change", ascending=True),
    # --- Volatility ---
    "most_volatile": lambda q: q.order_by("Volatility.D", ascending=False),
    # --- Pre-market scanners (matching official TradingView queries) ---
    "pre_market_gainers": lambda q: (
        q.where(col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]))
        .where(col("is_primary") == True)  # noqa: E712
        .where(col("type") == "stock")
        .where(col("premarket_change") > 0)
        .where(col("premarket_change").not_empty())
        .where(col("active_symbol") == True)  # noqa: E712
        .order_by("premarket_change", ascending=False)
    ),
    "pre_market_losers": lambda q: (
        q.where(col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]))
        .where(col("is_primary") == True)  # noqa: E712
        .where(col("type") == "stock")
        .where(col("premarket_change") < 0)
        .where(col("premarket_change").not_empty())
        .where(col("active_symbol") == True)  # noqa: E712
        .order_by("premarket_change", ascending=True)
    ),
    "pre_market_active": lambda q: (
        q.where(col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]))
        .where(col("is_primary") == True)  # noqa: E712
        .where(col("type") == "stock")
        .where(col("premarket_volume").not_empty())
        .where(col("active_symbol") == True)  # noqa: E712
        .order_by("premarket_volume", ascending=False)
    ),
    "pre_market_gappers": lambda q: (
        q.where(col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]))
        .where(col("is_primary") == True)  # noqa: E712
        .where(col("type") == "stock")
        .where(col("premarket_gap").not_empty())
        .where(col("active_symbol") == True)  # noqa: E712
        .order_by("premarket_gap", ascending=False)
    ),
    # --- After-hours scanners ---
    "after_hours_gainers": lambda q: (
        q.where(col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]))
        .where(col("is_primary") == True)  # noqa: E712
        .where(col("type") == "stock")
        .where(col("postmarket_change") > 0)
        .where(col("postmarket_change").not_empty())
        .where(col("active_symbol") == True)  # noqa: E712
        .order_by("postmarket_change", ascending=False)
    ),
    "after_hours_losers": lambda q: (
        q.where(col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]))
        .where(col("is_primary") == True)  # noqa: E712
        .where(col("type") == "stock")
        .where(col("postmarket_change") < 0)
        .where(col("postmarket_change").not_empty())
        .where(col("active_symbol") == True)  # noqa: E712
        .order_by("postmarket_change", ascending=True)
    ),
    "after_hours_active": lambda q: (
        q.where(col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]))
        .where(col("is_primary") == True)  # noqa: E712
        .where(col("type") == "stock")
        .where(col("postmarket_volume").not_empty())
        .where(col("active_symbol") == True)  # noqa: E712
        .order_by("postmarket_volume", ascending=False)
    ),
}

# Columns we always request — every damn field that matters
_DEFAULT_COLUMNS = [
    # Price & Volume
    "name", "close", "open", "high", "low", "volume", "change", "change_abs",
    "change_from_open", "gap",
    "relative_volume_10d_calc",
    "average_volume_10d_calc", "average_volume_30d_calc", "average_volume_60d_calc",
    # Pre-market & After-hours
    "premarket_close", "premarket_change", "premarket_change_abs",
    "premarket_volume", "premarket_gap",
    "postmarket_close", "postmarket_change", "postmarket_change_abs",
    "postmarket_volume",
    # Consensus Recommendations
    "Recommend.All", "Recommend.MA", "Recommend.Other",
    # Moving Averages — EMAs
    "EMA5", "EMA10", "EMA20", "EMA30", "EMA50", "EMA100", "EMA200",
    # Moving Averages — SMAs
    "SMA5", "SMA10", "SMA20", "SMA30", "SMA50", "SMA100", "SMA200",
    # Hull MA & VWMA
    "HullMA9", "VWMA", "VWAP",
    # Oscillators
    "RSI", "RSI7", "Stoch.K", "Stoch.D", "Stoch.RSI.K", "Stoch.RSI.D",
    "CCI20", "W.R", "UO", "Mom", "AO", "ROC",
    # MACD
    "MACD.macd", "MACD.signal",
    # Trend
    "ADX", "ADX+DI", "ADX-DI",
    # Volatility
    "ATR", "BB.upper", "BB.lower", "BBPower", "Volatility.D",
    # Channels
    "KltChnl.upper", "KltChnl.lower",
    "DonchCh20.Upper", "DonchCh20.Lower",
    # Ichimoku
    "Ichimoku.BLine", "Ichimoku.CLine", "Ichimoku.Lead1", "Ichimoku.Lead2",
    # Aroon
    "Aroon.Up", "Aroon.Down",
    # Flow & Sentiment
    "MoneyFlow", "ChaikinMoneyFlow",
    # Parabolic SAR
    "P.SAR",
    # Fundamentals
    "market_cap_basic", "beta_1_year",
    # Performance
    "Perf.Y",
    # 52-week
    "High.All", "Low.All",
    # Pivot (Classic only — keep it sane)
    "Pivot.M.Classic.Middle", "Pivot.M.Classic.R1", "Pivot.M.Classic.S1",
]


@smart_cache(open_ttl=120, closed_ttl=1800)  # 2min open, 30min closed
async def run_stock_screen(
    preset: str = "most_active",
    market: str = "america",
    limit: int = 25,
    extra_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run a stock screen using TradingView's scanner API.

    Args:
        preset: Screen preset to use. One of:
            ``most_active``, ``new_highs``, ``new_lows``,
            ``overbought``, ``oversold``, ``high_relative_volume``,
            ``gap_up``, ``gap_down``, ``bullish_ema_stack``,
            ``bearish_ema_stack``, ``high_momentum``, ``large_cap_undervalued``,
            ``top_gainers``, ``biggest_losers``, ``most_volatile``,
            ``pre_market_gainers``, ``pre_market_losers``,
            ``pre_market_active``, ``pre_market_gappers``,
            ``after_hours_gainers``, ``after_hours_losers``,
            ``after_hours_active``.
        market: Market to scan. Defaults to ``"america"``.
        limit: Maximum number of results to return (1–100).
        extra_columns: Additional TradingView field names to include beyond defaults.

    Returns:
        A list of dicts with all available technical fields for each matching stock.

    Raises:
        ValueError: If *preset* is not recognised.
    """
    if preset not in PRESET_FILTERS:
        available = ", ".join(sorted(PRESET_FILTERS.keys()))
        raise ValueError(
            f"Unknown preset '{preset}'. Available presets: {available}"
        )

    limit = max(1, min(limit, 100))

    # Merge default + extra columns
    columns = list(_DEFAULT_COLUMNS)
    if extra_columns:
        for c in extra_columns:
            if c not in columns:
                columns.append(c)

    def _execute() -> list[dict[str, Any]]:
        query = Query().select(*columns).limit(limit)

        # Apply the market filter
        query = query.set_markets(market)

        # Apply preset-specific filters
        query = PRESET_FILTERS[preset](query)

        try:
            _, raw_rows = query.get_scanner_data()
        except Exception as exc:
            logger.error("Screener query failed for preset '%s': %s", preset, exc)
            return []

        if raw_rows is None or raw_rows.empty:
            logger.info("Screener returned 0 results for preset '%s'", preset)
            return []

        results: list[dict[str, Any]] = []
        for _, row in raw_rows.iterrows():
            ticker_raw = row.get("name", "")
            # TradingView returns tickers as "EXCHANGE:SYMBOL" — strip the exchange
            ticker = ticker_raw.split(":")[-1] if ":" in str(ticker_raw) else str(ticker_raw)

            # Include ALL returned columns (not just a hardcoded subset)
            entry: dict[str, Any] = {"ticker": ticker}
            for col_name in columns:
                if col_name == "name":
                    continue
                val = row.get(col_name)
                cleaned = _clean_value(val)
                if cleaned is not None:
                    entry[col_name] = cleaned

            results.append(entry)

        return results

    return await asyncio.to_thread(_execute)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None on failure.

    Handles NaN, Inf, and numpy scalar types that break JSON serialization.
    """
    try:
        f = float(value)
        if not math.isfinite(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert a value to int, returning None on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_value(value: Any) -> Any:
    """Sanitize a pandas/numpy value for JSON serialization."""
    if value is None:
        return None
    # numpy.nan passes `is not None` but is NaN
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float)):
        return _safe_float(value)
    # numpy.bool_ → Python bool
    try:
        import numpy as np
        if isinstance(value, (np.bool_, np.integer, np.floating)):
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, np.integer):
                return int(value)
            return _safe_float(float(value))
    except ImportError:
        pass
    return value


# ---------------------------------------------------------------------------
# Custom screen builder — LLM translates natural language to filters
# ---------------------------------------------------------------------------

# Operator map for custom filters
_OPS = {
    ">": lambda c, v: c > v,
    ">=": lambda c, v: c >= v,
    "<": lambda c, v: c < v,
    "<=": lambda c, v: c <= v,
    "==": lambda c, v: c == v,
    "!=": lambda c, v: c != v,
    "crosses_above": lambda c, v: c > v,  # simplified — true cross needs history
    "crosses_below": lambda c, v: c < v,
}


async def run_custom_screen(
    filters: list[dict[str, Any]],
    sort_by: str = "volume",
    sort_ascending: bool = False,
    market: str = "america",
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Run a custom stock screen with user-defined filter conditions.

    The LLM should translate natural language requests into structured filters.
    Each filter is a dict with: field, operator, value.

    Args:
        filters: List of filter conditions, each a dict with:
            - field: TradingView field name (e.g. 'RSI', 'EMA20', 'ADX', 'volume')
            - operator: '>', '>=', '<', '<=', '==', '!='
            - value: Numeric or string value to compare against.
              Can also be another field name (e.g. 'EMA50') for cross-field comparisons.
        sort_by: Field to sort results by. Default: 'volume'.
        sort_ascending: Sort direction. Default: False (descending).
        market: Market to scan. Default: 'america'.
        limit: Max results (1–100). Default: 25.

    Returns:
        A list of dicts with all available technical fields for matching stocks.

    Example filters:
        [
            {"field": "RSI", "operator": "<", "value": 30},
            {"field": "EMA20", "operator": ">", "value": "EMA50"},
            {"field": "volume", "operator": ">", "value": 1000000},
            {"field": "market_cap_basic", "operator": ">", "value": 1000000000}
        ]
    """
    limit = max(1, min(limit, 100))

    def _execute() -> list[dict[str, Any]]:
        query = Query().select(*_DEFAULT_COLUMNS).limit(limit)
        query = query.set_markets(market)

        # Apply each filter condition
        for f in filters:
            field = f.get("field", "")
            op = f.get("operator", ">")
            value = f.get("value")

            if not field or value is None:
                continue

            op_fn = _OPS.get(op)
            if not op_fn:
                logger.warning("Custom screen: unknown operator '%s', skipping", op)
                continue

            # Check if value is a field reference (string that's a known field)
            if isinstance(value, str) and not value.replace(".", "").replace("_", "").replace("+", "").replace("-", "").isdigit():
                # Cross-field comparison: e.g. EMA20 > EMA50
                try:
                    query = query.where(op_fn(col(field), col(value)))
                except Exception as e:
                    logger.warning("Custom screen: cross-field filter failed %s %s %s: %s",
                                   field, op, value, e)
            else:
                # Numeric comparison
                try:
                    num_val = float(value) if isinstance(value, str) else value
                    query = query.where(op_fn(col(field), num_val))
                except Exception as e:
                    logger.warning("Custom screen: filter failed %s %s %s: %s",
                                   field, op, value, e)

        # Apply sorting
        try:
            query = query.order_by(sort_by, ascending=sort_ascending)
        except Exception:
            pass  # fall back to default ordering

        try:
            _, raw_rows = query.get_scanner_data()
        except Exception as exc:
            logger.error("Custom screen query failed: %s", exc)
            return []

        if raw_rows is None or raw_rows.empty:
            return []

        results: list[dict[str, Any]] = []
        for _, row in raw_rows.iterrows():
            ticker_raw = row.get("name", "")
            ticker = ticker_raw.split(":")[-1] if ":" in str(ticker_raw) else str(ticker_raw)

            entry: dict[str, Any] = {"ticker": ticker}
            for col_name in _DEFAULT_COLUMNS:
                if col_name == "name":
                    continue
                val = row.get(col_name)
                cleaned = _clean_value(val)
                if cleaned is not None:
                    entry[col_name] = cleaned

            results.append(entry)

        return results

    return await asyncio.to_thread(_execute)

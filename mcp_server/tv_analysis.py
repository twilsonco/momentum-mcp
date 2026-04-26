"""
TradingView Technical Analysis — 26-Indicator Consensus.

Uses tradingview-ta to fetch real-time technical analysis summaries
for any ticker. Returns the aggregate buy/sell/neutral recommendation
from 26 technical indicators (all MAs + all oscillators).

This is like getting a second opinion from 26 analysts instantly.
"""

from __future__ import annotations

import logging
from typing import Any

from tradingview_ta import TA_Handler, Interval

logger = logging.getLogger(__name__)

from mcp_server.cache import smart_cache

# Map user-friendly interval names to tradingview-ta constants
INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
    "1M": Interval.INTERVAL_1_MONTH,
}


@smart_cache(open_ttl=300, closed_ttl=3600)
async def get_tv_analysis(
    ticker: str,
    interval: str = "1d",
    exchange: str = "NASDAQ",
) -> dict[str, Any]:
    """
    Get TradingView technical analysis consensus for a ticker.

    Returns the aggregate recommendation (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
    from 26 technical indicators, plus breakdowns by moving averages and oscillators.

    Args:
        ticker: Stock ticker symbol (e.g., NVDA, AAPL, TSLA).
        interval: Timeframe — 1m, 5m, 15m, 1h, 4h, 1d (default), 1w, 1M.
        exchange: Exchange name. Default: NASDAQ. Try NYSE, AMEX for others.
    """
    tv_interval = INTERVAL_MAP.get(interval, Interval.INTERVAL_1_DAY)

    # Try multiple exchanges if the first one fails
    exchanges_to_try = [exchange]
    if exchange == "NASDAQ":
        exchanges_to_try += ["NYSE", "AMEX"]
    elif exchange == "NYSE":
        exchanges_to_try += ["NASDAQ", "AMEX"]

    last_error = None
    for exch in exchanges_to_try:
        try:
            handler = TA_Handler(
                symbol=ticker.upper(),
                screener="america",
                exchange=exch,
                interval=tv_interval,
            )
            analysis = handler.get_analysis()

            # Build the response
            summary = analysis.summary
            ma_rec = analysis.moving_averages
            osc_rec = analysis.oscillators

            return {
                "ticker": ticker.upper(),
                "exchange": exch,
                "interval": interval,
                "recommendation": summary.get("RECOMMENDATION", "N/A"),
                "summary": {
                    "buy": summary.get("BUY", 0),
                    "sell": summary.get("SELL", 0),
                    "neutral": summary.get("NEUTRAL", 0),
                },
                "moving_averages": {
                    "recommendation": ma_rec.get("RECOMMENDATION", "N/A"),
                    "buy": ma_rec.get("BUY", 0),
                    "sell": ma_rec.get("SELL", 0),
                    "neutral": ma_rec.get("NEUTRAL", 0),
                },
                "oscillators": {
                    "recommendation": osc_rec.get("RECOMMENDATION", "N/A"),
                    "buy": osc_rec.get("BUY", 0),
                    "sell": osc_rec.get("SELL", 0),
                    "neutral": osc_rec.get("NEUTRAL", 0),
                },
                "indicators": {
                    "RSI": round(analysis.indicators.get("RSI", 0), 2),
                    "Stoch_K": round(analysis.indicators.get("Stoch.K", 0), 2),
                    "CCI": round(analysis.indicators.get("CCI20", 0), 2),
                    "ADX": round(analysis.indicators.get("ADX", 0), 2),
                    "MACD_signal": round(analysis.indicators.get("MACD.macd", 0), 4),
                    "ATR": round(analysis.indicators.get("ATR", 0), 2),
                    "BB_upper": round(analysis.indicators.get("BB.upper", 0), 2),
                    "BB_lower": round(analysis.indicators.get("BB.lower", 0), 2),
                    "EMA20": round(analysis.indicators.get("EMA20", 0), 2),
                    "EMA50": round(analysis.indicators.get("EMA50", 0), 2),
                    "EMA200": round(analysis.indicators.get("EMA200", 0), 2),
                    "SMA20": round(analysis.indicators.get("SMA20", 0), 2),
                    "SMA50": round(analysis.indicators.get("SMA50", 0), 2),
                    "SMA200": round(analysis.indicators.get("SMA200", 0), 2),
                    "close": round(analysis.indicators.get("close", 0), 2),
                    "volume": analysis.indicators.get("volume", 0),
                },
            }

        except Exception as e:
            last_error = e
            logger.debug("TV-TA failed for %s on %s: %s", ticker, exch, e)
            continue

    logger.warning("TV-TA: all exchanges failed for %s: %s", ticker, last_error)
    return {
        "ticker": ticker.upper(),
        "error": f"Could not fetch TradingView analysis: {last_error}",
    }

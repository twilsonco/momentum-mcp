"""
Watchlist — Ticker Management + On-Demand Scanning.

Flat JSON storage for a user's watchlist. Scan function runs
technicals + options analysis on each ticker and stores alerts.

Ready for background scheduling (cron/asyncio.Task) but currently
runs on-demand via API or conversational tool call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
ALERTS_FILE = DATA_DIR / "watchlist_alerts.json"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def _load_watchlist() -> list[str]:
    """Load watchlist from disk."""
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text())
            return data.get("tickers", [])
    except Exception as e:
        logger.warning("Failed to load watchlist: %s", e)
    return []


def _save_watchlist(tickers: list[str]) -> None:
    """Save watchlist to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps({
        "tickers": tickers,
        "updated_at": datetime.utcnow().isoformat(),
    }, indent=2))


def get_watchlist() -> dict[str, Any]:
    """Get the current watchlist.

    Returns:
        Dict with tickers list and count.
    """
    tickers = _load_watchlist()
    return {
        "tickers": tickers,
        "count": len(tickers),
    }


def add_to_watchlist(ticker: str) -> dict[str, Any]:
    """Add a ticker to the watchlist.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').

    Returns:
        Dict with updated watchlist and status.
    """
    ticker = ticker.strip().upper()
    tickers = _load_watchlist()

    if ticker in tickers:
        return {
            "status": "already_exists",
            "ticker": ticker,
            "tickers": tickers,
            "count": len(tickers),
        }

    tickers.append(ticker)
    tickers.sort()
    _save_watchlist(tickers)

    logger.info("Added %s to watchlist (%d tickers)", ticker, len(tickers))
    return {
        "status": "added",
        "ticker": ticker,
        "tickers": tickers,
        "count": len(tickers),
    }


def remove_from_watchlist(ticker: str) -> dict[str, Any]:
    """Remove a ticker from the watchlist.

    Args:
        ticker: Stock ticker symbol to remove.

    Returns:
        Dict with updated watchlist and status.
    """
    ticker = ticker.strip().upper()
    tickers = _load_watchlist()

    if ticker not in tickers:
        return {
            "status": "not_found",
            "ticker": ticker,
            "tickers": tickers,
            "count": len(tickers),
        }

    tickers.remove(ticker)
    _save_watchlist(tickers)

    logger.info("Removed %s from watchlist (%d tickers)", ticker, len(tickers))
    return {
        "status": "removed",
        "ticker": ticker,
        "tickers": tickers,
        "count": len(tickers),
    }


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

async def scan_watchlist(max_tickers: int = 20) -> dict[str, Any]:
    """Scan all watchlist tickers for setups and alerts.

    Runs analyze_technicals on each ticker in parallel, flags
    actionable setups (oversold RSI, bullish EMA cross, etc.),
    and stores alerts.

    Args:
        max_tickers: Max tickers to scan (default 20).

    Returns:
        Dict with scan results, alerts, and timing.
    """
    from mcp_server.technicals import analyze_technicals

    tickers = _load_watchlist()[:max_tickers]
    if not tickers:
        return {
            "status": "empty_watchlist",
            "message": "No tickers in watchlist. Add some first.",
            "alerts": [],
        }

    start = time.time()
    logger.info("Scanning watchlist: %s", tickers)

    # Run all scans in parallel
    tasks = [_scan_ticker(t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    alerts = []
    scan_results = []

    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            scan_results.append({"ticker": ticker, "error": str(result)})
            continue

        scan_results.append(result)
        # Extract alerts from the scan
        ticker_alerts = result.get("alerts", [])
        for alert in ticker_alerts:
            alerts.append({
                "ticker": ticker,
                "signal": alert,
                "timestamp": datetime.utcnow().isoformat(),
            })

    # Save alerts
    _save_alerts(alerts)

    elapsed = round(time.time() - start, 1)
    logger.info("Watchlist scan done: %d tickers, %d alerts, %.1fs", len(tickers), len(alerts), elapsed)

    return {
        "status": "complete",
        "tickers_scanned": len(tickers),
        "alerts_count": len(alerts),
        "alerts": alerts,
        "results": scan_results,
        "elapsed_seconds": elapsed,
    }


async def _scan_ticker(ticker: str) -> dict[str, Any]:
    """Scan a single ticker for actionable signals."""
    from mcp_server.technicals import analyze_technicals

    try:
        tech = await analyze_technicals(ticker)
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "alerts": []}

    alerts = []

    # --- RSI signals ---
    rsi = tech.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            alerts.append(f"🟢 RSI oversold at {rsi:.1f} — potential bounce setup")
        elif rsi > 70:
            alerts.append(f"🔴 RSI overbought at {rsi:.1f} — watch for reversal")

    # --- MACD signals ---
    macd_hist = tech.get("macd_histogram")
    if macd_hist is not None:
        if macd_hist > 0 and abs(macd_hist) < 0.5:
            alerts.append("🟢 MACD just crossed bullish")
        elif macd_hist < 0 and abs(macd_hist) < 0.5:
            alerts.append("🔴 MACD just crossed bearish")

    # --- EMA Stack alignment ---
    ema8 = tech.get("ema_8")
    ema21 = tech.get("ema_21")
    ema55 = tech.get("ema_55")
    if ema8 and ema21 and ema55:
        if ema8 > ema21 > ema55:
            alerts.append("🟢 Bullish EMA stack aligned (8 > 21 > 55)")
        elif ema8 < ema21 < ema55:
            alerts.append("🔴 Bearish EMA stack aligned (8 < 21 < 55)")

    # --- Price near support/resistance ---
    close = tech.get("close")
    bb_lower = tech.get("bb_lower")
    bb_upper = tech.get("bb_upper")
    if close and bb_lower and bb_upper:
        bb_width = bb_upper - bb_lower
        if bb_width > 0:
            if (close - bb_lower) / bb_width < 0.1:
                alerts.append(f"🟢 Price at lower Bollinger Band (${close:.2f}) — potential support")
            elif (bb_upper - close) / bb_width < 0.1:
                alerts.append(f"🔴 Price at upper Bollinger Band (${close:.2f}) — potential resistance")

    # --- ADX trend strength ---
    adx = tech.get("adx_14")
    if adx is not None:
        if adx > 40:
            alerts.append(f"⚡ Strong trend (ADX {adx:.1f}) — momentum play active")

    return {
        "ticker": ticker,
        "close": tech.get("close"),
        "rsi": rsi,
        "macd_histogram": macd_hist,
        "adx": adx,
        "alerts": alerts,
        "alert_count": len(alerts),
    }


# ---------------------------------------------------------------------------
# Alerts persistence
# ---------------------------------------------------------------------------

def _save_alerts(alerts: list[dict]) -> None:
    """Save alerts to disk, keeping the last 200."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load_alerts_raw()
    all_alerts = alerts + existing  # newest first
    all_alerts = all_alerts[:200]   # cap at 200
    ALERTS_FILE.write_text(json.dumps({
        "alerts": all_alerts,
        "updated_at": datetime.utcnow().isoformat(),
    }, indent=2))


def _load_alerts_raw() -> list[dict]:
    """Load raw alerts from disk."""
    try:
        if ALERTS_FILE.exists():
            data = json.loads(ALERTS_FILE.read_text())
            return data.get("alerts", [])
    except Exception:
        pass
    return []


def get_watchlist_alerts(limit: int = 50) -> dict[str, Any]:
    """Get stored watchlist alerts.

    Args:
        limit: Max alerts to return (default 50).

    Returns:
        Dict with alerts list and metadata.
    """
    alerts = _load_alerts_raw()[:limit]
    return {
        "alerts": alerts,
        "count": len(alerts),
        "total_stored": len(_load_alerts_raw()),
    }

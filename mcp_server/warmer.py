"""
warmer.py — Background cache warmer + Alpha Stream signal detector.

Pre-fetches technical analysis for popular tickers on a configurable
interval, so MCP users get instant cache hits.  After each warm cycle,
checks for RSI 70/30 crosses and logs them to data/alpha_signals.json.

Also warms the yfinance HTTP session at startup to avoid the ~5s
cold-start penalty on the very first request.

Usage:
    from mcp_server.warmer import start_warmer, stop_warmer, warm_yfinance_session
    await warm_yfinance_session()
    start_warmer()
    ...
    stop_warmer()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Most-requested tickers — ordered by expected frequency.
WARM_TICKERS = [
    # Mega-caps / index ETFs
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META",
    # High-retail-interest
    "AMD", "PLTR", "COIN", "SOFI", "MARA",
    # Volatility / bonds / gold
    "VIX", "TLT", "GLD",
    # Sector ETFs
    "XLF", "XLE", "XLK", "ARKK",
    # Meme / momentum favorites
    "GME", "AMC", "RIVN",
]

WARM_INTERVAL_SECONDS = 300  # 5 minutes (matches default open_ttl)

_warmer_task: asyncio.Task | None = None

# ── Alpha Stream Storage ─────────────────────────────────────────────────────

ALPHA_SIGNALS_PATH = Path("data/alpha_signals.json")


def _load_signals() -> list[dict]:
    """Load existing alpha signals from disk."""
    if not ALPHA_SIGNALS_PATH.exists():
        return []
    try:
        return json.loads(ALPHA_SIGNALS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_signals(signals: list[dict]) -> None:
    """Persist alpha signals to disk."""
    ALPHA_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALPHA_SIGNALS_PATH.write_text(json.dumps(signals, indent=2))


def _is_duplicate(signals: list[dict], ticker: str, signal_type: str, date: str) -> bool:
    """Check if this exact (ticker, type, date) triple already exists."""
    return any(
        s["ticker"] == ticker and s["type"] == signal_type and s["date"] == date
        for s in signals
    )


# ── Alpha Signals Query API ──────────────────────────────────────────────────

async def get_alpha_signals(
    ticker: str | None = None,
    signal_type: str | None = None,
    limit: int = 50,
) -> dict:
    """Retrieve recent alpha signals (RSI crosses, MACD crosses, volume spikes, etc.).

    Args:
        ticker: Optional ticker to filter by (e.g. 'NVDA').
        signal_type: Optional signal type filter (e.g. 'RSI_CROSS_UP_70', 'MACD_CROSS_UP',
                     'VOLUME_SPIKE', 'EMA_STACK_BREAKOUT', 'ADX_TREND_ENTRY').
        limit: Max signals to return (default 50).

    Returns:
        Dict with 'signals' list and 'total' count.
    """
    signals = _load_signals()

    if ticker:
        ticker_upper = ticker.strip().upper()
        signals = [s for s in signals if s["ticker"] == ticker_upper]
    if signal_type:
        signals = [s for s in signals if s["type"] == signal_type]

    # Newest first
    signals.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "signals": signals[:limit],
        "total": len(signals),
        "available_types": [
            "RSI_CROSS_UP_70", "RSI_CROSS_DOWN_70",
            "RSI_CROSS_UP_30", "RSI_CROSS_DOWN_30",
            "MACD_CROSS_UP", "MACD_CROSS_DOWN",
            "VOLUME_SPIKE", "EMA_STACK_BREAKOUT", "ADX_TREND_ENTRY",
        ],
    }


# ── Alpha Stream Signal Detection ────────────────────────────────────────────

ALPHA_THRESHOLDS = [
    # (signal_type, description, condition_fn taking data dict)
    # --- RSI 70/30 Crosses ---
    ("RSI_CROSS_UP_70", "RSI crossed ABOVE 70 — overbought entry",
     lambda d: d.get("rsi_14_prev") is not None and d.get("rsi_14") is not None
     and d["rsi_14_prev"] < 70 and d["rsi_14"] >= 70),

    ("RSI_CROSS_DOWN_70", "RSI crossed BELOW 70 — overbought exit / bearish reversal",
     lambda d: d.get("rsi_14_prev") is not None and d.get("rsi_14") is not None
     and d["rsi_14_prev"] >= 70 and d["rsi_14"] < 70),

    ("RSI_CROSS_UP_30", "RSI crossed ABOVE 30 — oversold exit / bullish reversal",
     lambda d: d.get("rsi_14_prev") is not None and d.get("rsi_14") is not None
     and d["rsi_14_prev"] < 30 and d["rsi_14"] >= 30),

    ("RSI_CROSS_DOWN_30", "RSI crossed BELOW 30 — oversold entry / bearish momentum",
     lambda d: d.get("rsi_14_prev") is not None and d.get("rsi_14") is not None
     and d["rsi_14_prev"] >= 30 and d["rsi_14"] < 30),

    # --- MACD Crosses ---
    ("MACD_CROSS_UP", "MACD crossed ABOVE signal line — bullish momentum",
     lambda d: d.get("macd_prev") is not None and d.get("macd_signal_prev") is not None
     and d.get("macd") is not None and d.get("macd_signal") is not None
     and d["macd_prev"] <= d["macd_signal_prev"] and d["macd"] > d["macd_signal"]),

    ("MACD_CROSS_DOWN", "MACD crossed BELOW signal line — bearish momentum",
     lambda d: d.get("macd_prev") is not None and d.get("macd_signal_prev") is not None
     and d.get("macd") is not None and d.get("macd_signal") is not None
     and d["macd_prev"] >= d["macd_signal_prev"] and d["macd"] < d["macd_signal"]),

    # --- Volume Spike ---
    ("VOLUME_SPIKE", "Volume > 200% of 20-day average — unusual activity",
     lambda d: d.get("volume") is not None and d.get("vol_sma_20") is not None
     and d["vol_sma_20"] > 0 and d["volume"] > d["vol_sma_20"] * 2.0),

    # --- EMA Stack Breakout (Michael's signature setup) ---
    ("EMA_STACK_BREAKOUT", "Price broke ABOVE entire EMA stack (8>21>34>55>89) — strong bullish",
     lambda d: (
         d.get("ema_stack_bullish") is True
         and d.get("close_prev") is not None and d.get("close") is not None
         and d.get("ema_89") is not None
         and d["close_prev"] <= d["ema_89"]  # was below the slowest EMA
         and d["close"] > d["ema_8"]          # now above the fastest EMA
     )),

    # --- ADX Trend Entry ---
    ("ADX_TREND_ENTRY", "ADX > 25 with fresh bullish DI cross — trend confirmed",
     lambda d: (
         d.get("adx_14") is not None and d["adx_14"] > 25
         and d.get("plus_di") is not None and d.get("minus_di") is not None
         and d.get("plus_di_prev") is not None and d.get("minus_di_prev") is not None
         and d["plus_di"] > d["minus_di"]            # DI+ currently above DI-
         and d["plus_di_prev"] <= d["minus_di_prev"]  # but wasn't yesterday
     )),
]


def _check_alpha_signals(ticker: str, technicals_data: dict) -> list[dict]:
    """Check for all technical threshold crosses and return any new signals."""
    if not isinstance(technicals_data, dict):
        return []

    price = technicals_data.get("close")
    date_str = technicals_data.get("date", "")

    # Normalize date to YYYY-MM-DD
    if "T" in str(date_str):
        date_str = str(date_str).split("T")[0]

    now_ts = datetime.now(timezone.utc).isoformat()
    new_signals = []

    for signal_type, description, condition_fn in ALPHA_THRESHOLDS:
        try:
            if condition_fn(technicals_data):
                # Build signal-specific value field
                value_map = {
                    "RSI_CROSS_UP_70": ("rsi_14", "rsi_14_prev"),
                    "RSI_CROSS_DOWN_70": ("rsi_14", "rsi_14_prev"),
                    "RSI_CROSS_UP_30": ("rsi_14", "rsi_14_prev"),
                    "RSI_CROSS_DOWN_30": ("rsi_14", "rsi_14_prev"),
                    "MACD_CROSS_UP": ("macd", "macd_prev"),
                    "MACD_CROSS_DOWN": ("macd", "macd_prev"),
                    "VOLUME_SPIKE": ("volume", "vol_sma_20"),
                    "EMA_STACK_BREAKOUT": ("close", "close_prev"),
                    "ADX_TREND_ENTRY": ("adx_14", "plus_di"),
                }
                val_key, prev_key = value_map.get(signal_type, ("close", "close_prev"))

                new_signals.append({
                    "ticker": ticker,
                    "type": signal_type,
                    "description": description,
                    "value": technicals_data.get(val_key),
                    "prev_value": technicals_data.get(prev_key),
                    "price": price,
                    "date": date_str,
                    "timestamp": now_ts,
                })
        except Exception as exc:
            logger.debug("Signal check %s failed for %s: %s", signal_type, ticker, exc)

    return new_signals


async def _process_alpha_signals(warm_results: dict[str, Any]) -> int:
    """Process all warm results for alpha signals. Returns count of new signals.

    Args:
        warm_results: Dict of {ticker: technicals_data_dict} from the warm cycle.
    """
    signals = _load_signals()
    new_count = 0

    for ticker, tech_data in warm_results.items():
        new_signals = _check_alpha_signals(ticker, tech_data)
        for sig in new_signals:
            if not _is_duplicate(signals, sig["ticker"], sig["type"], sig["date"]):
                signals.append(sig)
                new_count += 1
                logger.info(
                    "🔔 ALPHA SIGNAL: %s %s — RSI %.1f (prev %.1f) @ $%.2f",
                    sig["ticker"], sig["type"], sig["value"], sig["prev_value"], sig["price"],
                )

    if new_count > 0:
        # Keep last 500 signals max
        if len(signals) > 500:
            signals = signals[-500:]
        _save_signals(signals)
        logger.info("Alpha Stream: %d new signal(s) logged (%d total)", new_count, len(signals))

    return new_count


# ── yfinance Session Warming ─────────────────────────────────────────────────

async def warm_yfinance_session() -> None:
    """Make a cheap yfinance call to pre-establish the HTTP session.

    The first yfinance request pays ~5s for DNS + SSL + session setup.
    By making a trivial call at startup, all subsequent requests start
    at ~150-250ms instead of ~5,600ms.
    """
    import yfinance as yf

    def _warm():
        try:
            t = yf.Ticker("SPY")
            _ = t.fast_info  # Minimal network call — just needs session
            return True
        except Exception:
            return False

    start = time.monotonic()
    ok = await asyncio.to_thread(_warm)
    elapsed = (time.monotonic() - start) * 1000
    if ok:
        logger.info("yfinance session warmed in %.0fms (first-request cold start eliminated)", elapsed)
    else:
        logger.warning("yfinance session warm failed (%.0fms) — first request will be slow", elapsed)


# ── Background Warmer ────────────────────────────────────────────────────────

async def _warm_loop() -> None:
    """Periodically pre-fetch technicals for popular tickers + check alpha signals."""
    from mcp_server.cache import is_market_open
    from mcp_server.technicals import analyze_technicals

    # Wait a bit after startup before first warm cycle
    await asyncio.sleep(10)

    while True:
        try:
            market_open = is_market_open()
            interval = WARM_INTERVAL_SECONDS if market_open else WARM_INTERVAL_SECONDS * 6  # 30 min when closed

            start = time.monotonic()
            success = 0
            errors = 0
            warm_results: dict[str, Any] = {}

            # Warm in batches of 5 to be gentle on yfinance
            for i in range(0, len(WARM_TICKERS), 5):
                batch = WARM_TICKERS[i:i + 5]
                tasks = [_warm_ticker(ticker) for ticker in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for ticker, r in zip(batch, results):
                    if isinstance(r, Exception):
                        errors += 1
                    elif isinstance(r, dict):
                        warm_results[ticker] = r
                        success += 1
                    elif r is None:
                        errors += 1
                    else:
                        success += 1

                # Small delay between batches to avoid hammering yfinance
                if i + 5 < len(WARM_TICKERS):
                    await asyncio.sleep(2)

            # ── Alpha Stream: check for RSI crosses ──
            alpha_count = 0
            if warm_results:
                try:
                    alpha_count = await _process_alpha_signals(warm_results)
                except Exception as exc:
                    logger.error("Alpha Stream error: %s", exc)

            elapsed = (time.monotonic() - start) * 1000
            alpha_msg = f", 🔔 {alpha_count} new signals" if alpha_count else ""
            logger.info(
                "Cache warmer: %d/%d tickers refreshed in %.0fms (errors=%d, market=%s, next in %ds%s)",
                success, len(WARM_TICKERS), elapsed, errors,
                "open" if market_open else "closed", interval, alpha_msg,
            )

        except asyncio.CancelledError:
            logger.info("Cache warmer: shutting down")
            return
        except Exception as exc:
            logger.error("Cache warmer: unexpected error: %s", exc)

        await asyncio.sleep(interval)


async def _warm_ticker(ticker: str) -> dict | None:
    """Warm cache for a single ticker. Returns technicals data dict on success."""
    try:
        from mcp_server.technicals import analyze_technicals
        result = await analyze_technicals(ticker)

        # Extract the data dict for alpha signal checking
        if hasattr(result, 'data') and isinstance(result.data, dict):
            return result.data
        elif hasattr(result, 'status') and result.status == "success":
            return result.data
        elif isinstance(result, dict):
            return result
        return None
    except Exception as exc:
        # VIX and some ETFs may not have full technicals — that's fine
        logger.debug("Warmer skip %s: %s", ticker, str(exc)[:60])
        return None


def start_warmer() -> None:
    """Start the background cache warmer task."""
    global _warmer_task
    if _warmer_task is not None and not _warmer_task.done():
        logger.warning("Cache warmer already running")
        return
    _warmer_task = asyncio.create_task(_warm_loop(), name="cache-warmer")
    logger.info("Cache warmer started (%d tickers, every %ds)", len(WARM_TICKERS), WARM_INTERVAL_SECONDS)


def stop_warmer() -> None:
    """Stop the background cache warmer task."""
    global _warmer_task
    if _warmer_task and not _warmer_task.done():
        _warmer_task.cancel()
        logger.info("Cache warmer stopped")
    _warmer_task = None

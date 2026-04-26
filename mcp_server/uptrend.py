"""
Uptrend Analyzer — Measure market participation in structural uptrends.
Checks Price > EMA50 > EMA200 across all S&P sectors and major indices.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data

logger = logging.getLogger(__name__)

async def analyze_uptrend_participation() -> SignalResult:
    """Measure what % of major sectors and indices are in a confirmed uptrend (Price > EMA50 > EMA200)."""
    tickers = {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Small Caps", "DIA": "Dow Jones",
        "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
        "XLI": "Industrials", "XLY": "Discretionary", "XLP": "Staples", "XLU": "Utilities",
        "XLRE": "Real Estate", "XLB": "Materials", "XLC": "Communications"
    }
    
    async def _check_trend(t: str, label: str):
        try:
            records = await get_historical_data(t, period="2y", interval="1d")
            if not records: return t, None
            df = pd.DataFrame(records)
            # Use pandas-ta style EMAs or just pandas ewm
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            curr = df.iloc[-1]
            price = float(curr['close'])
            ema50 = float(curr['ema50'])
            ema200 = float(curr['ema200'])
            
            status = "No Uptrend"
            if price > ema50 > ema200:
                status = "Confirmed Uptrend"
            elif price > ema50:
                status = "Emerging Uptrend"
            elif price > ema200:
                status = "Support Test"
            
            return t, {
                "label": label,
                "price": round(price, 2),
                "ema50": round(ema50, 2),
                "ema200": round(ema200, 2),
                "status": status
            }
        except Exception as e:
            logger.debug(f"Trend check failed for {t}: {e}")
            return t, None

    tasks = [_check_trend(t, l) for t, l in tickers.items()]
    fetched = await asyncio.gather(*tasks)
    
    results = {t: r for t, r in fetched if r is not None}
    
    if not results:
        return SignalResult.error_msg("Failed to fetch trend data for any sectors.")
        
    uptrend_count = sum(1 for r in results.values() if r['status'] == "Confirmed Uptrend")
    total_valid = len(results)
    uptrend_pct = round((uptrend_count / total_valid) * 100, 1)
    
    verdict = "Bearish / Narrow"
    if uptrend_pct >= 80: verdict = "Broadly Bullish"
    elif uptrend_pct >= 60: verdict = "Healthy Participation"
    elif uptrend_pct >= 40: verdict = "Selective / Fragmented"
    elif uptrend_pct >= 20: verdict = "Weak / Topping"
    
    return SignalResult.success({
        "uptrend_pct": uptrend_pct,
        "verdict": verdict,
        "sectors": results,
        "summary": f"Uptrend Participation: {uptrend_pct}% of sectors/indices in confirmed uptrend. Market participation is {verdict}."
    })

"""
Market Environment Report — Cross-asset briefing (Equities, Bonds, Commodities, Dollar, Crypto).
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data

logger = logging.getLogger(__name__)

async def get_market_environment() -> SignalResult:
    """Structured cross-asset snapshot: Equities, Bonds, Commodities, Dollar, Crypto."""
    assets = {
        "Equities": {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Small Caps", "DIA": "Dow Jones"},
        "Fixed Income": {"TLT": "Long Bonds", "SHY": "Short Bonds"},
        "Commodities": {"GLD": "Gold", "SLV": "Silver", "USO": "Oil"},
        "Currencies": {"UUP": "US Dollar"},
        "Crypto": {"BTC-USD": "Bitcoin"}
    }
    
    async def _fetch_asset(ticker: str, label: str):
        try:
            records = await get_historical_data(ticker, period="3mo", interval="1d")
            if not records: return ticker, None
            df = pd.DataFrame(records)
            curr = float(df['close'].iloc[-1])
            
            # 1 month ago (approx 21 trading days)
            prev_1m = float(df['close'].iloc[-21]) if len(df) >= 21 else float(df['close'].iloc[0])
            # 3 months ago
            prev_3m = float(df['close'].iloc[0])
            
            return ticker, {
                "label": label,
                "price": round(curr, 2),
                "perf_1m_pct": round(((curr / prev_1m) - 1) * 100, 2),
                "perf_3m_pct": round(((curr / prev_3m) - 1) * 100, 2)
            }
        except Exception as e:
            logger.debug(f"Failed to fetch environment data for {ticker}: {e}")
            return ticker, None

    flat_assets = []
    for cat, items in assets.items():
        for t, l in items.items():
            flat_assets.append((t, l, cat))
            
    tasks = [_fetch_asset(t, l) for t, l, c in flat_assets]
    fetched = await asyncio.gather(*tasks)
    
    asset_data = dict(fetched)
    
    report = {}
    for t, l, cat in flat_assets:
        if cat not in report: report[cat] = []
        data = asset_data.get(t)
        if data:
            report[cat].append({"ticker": t, **data})
            
    # Rotation Logic (Equities vs Bonds)
    rotation = "Neutral"
    spy_data = asset_data.get("SPY")
    tlt_data = asset_data.get("TLT")
    
    if spy_data and tlt_data:
        spy_perf = spy_data.get("perf_1m_pct", 0)
        tlt_perf = tlt_data.get("perf_1m_pct", 0)
        if spy_perf > tlt_perf and spy_perf > 0:
            rotation = "Risk-On (Equities Outperforming Bonds)"
        elif tlt_perf > spy_perf and tlt_perf > 0:
            rotation = "Risk-Off (Bonds Outperforming Equities)"
        elif spy_perf < 0 and tlt_perf < 0:
            rotation = "De-leveraging (Both Equities & Bonds Down)"
    
    return SignalResult.success({
        "report": report,
        "rotation_signal": rotation,
        "summary": f"Market Environment Briefing complete. Current rotation: {rotation}."
    })

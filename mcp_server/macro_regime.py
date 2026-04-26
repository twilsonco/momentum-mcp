"""
Macro Regime Detector — Structural regime classification using cross-asset ratios.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data

logger = logging.getLogger(__name__)

async def detect_macro_regime(lookback: int = 90) -> SignalResult:
    """Detect structural market regime: Growth, Inflation, Deflation, or Goldilocks."""
    pairs = {
        "concentration": ("RSP", "SPY"),
        "yield_curve_proxy": ("TLT", "SHY"),
        "risk_appetite": ("GLD", "SPY"),
        "cyclical_vs_defensive": ("XLY", "XLP")
    }
    
    unique_tickers = set()
    for a, b in pairs.values():
        unique_tickers.add(a)
        unique_tickers.add(b)
        
    async def _fetch(t):
        try:
            records = await get_historical_data(t, period="6mo", interval="1d")
            if not records: return t, None
            df = pd.DataFrame(records)
            # Ensure price is numeric
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            return t, df['close']
        except Exception as e:
            logger.debug(f"Failed to fetch {t} for macro regime: {e}")
            return t, None

    tasks = [_fetch(t) for t in unique_tickers]
    fetched = await asyncio.gather(*tasks)
    data = dict(fetched)
    
    ratios = {}
    for name, (a, b) in pairs.items():
        if data.get(a) is not None and data.get(b) is not None:
            try:
                df_a = data[a].copy()
                df_b = data[b].copy()
                df_a.index = pd.to_datetime(df_a.index, utc=True).normalize()
                df_b.index = pd.to_datetime(df_b.index, utc=True).normalize()
                common = df_a.index.intersection(df_b.index)
                
                if len(common) < 0.8 * min(len(df_a), len(df_b)):
                    logger.warning(f"Insufficient data overlap for {name}")
                    continue
                    
                aligned_b = df_b.loc[common].replace(0, pd.NA)
                ratios[name] = df_a.loc[common] / aligned_b
            except Exception as e:
                logger.warning(f"Ratio computation failed for {name}: {e}")
                continue
            
    if not ratios:
        return SignalResult.error_msg("Failed to compute macro ratios due to missing data.")
        
    results = {}
    for name, series in ratios.items():
        if series.empty: continue
        curr = float(series.iloc[-1])
        # Compare to 1 month ago
        prev = float(series.iloc[-21]) if len(series) >= 21 else float(series.iloc[0])
        trend = "Rising" if curr > prev else "Falling"
        change_pct = round(((curr / prev) - 1) * 100, 2)
        results[name] = {"value": round(curr, 4), "trend": trend, "change_1m_pct": change_pct}
        
    # Regime Logic
    regime = "Transitioning"
    conc = results.get("concentration", {}).get("trend", "Stable")
    yield_c = results.get("yield_curve_proxy", {}).get("trend", "Stable")
    risk = results.get("risk_appetite", {}).get("trend", "Stable")
    cycl = results.get("cyclical_vs_defensive", {}).get("trend", "Stable")
    
    # Heuristic rules
    if cycl == "Rising" and risk == "Falling":
        if yield_c == "Falling":
            regime = "Goldilocks (Low rates, High growth)"
        else:
            regime = "Growth / Risk-On"
    elif risk == "Rising" and cycl == "Falling":
        regime = "Deflationary / Risk-Off"
    elif risk == "Rising" and yield_c == "Rising":
        regime = "Inflationary / Stress"
    elif conc == "Falling" and cycl == "Rising":
        regime = "Late Cycle / Narrow Leadership"
        
    summary = f"Macro Regime: {regime}. Risk appetite is {risk.lower()} and cyclicals vs staples is {cycl.lower()}."
    
    return SignalResult.success({
        "regime": regime,
        "ratios": results,
        "summary": summary
    })

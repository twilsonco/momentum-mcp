"""
Theme Detector — Identify trending market themes via thematic ETF clustering.
Tracks performance across AI, Biotech, Energy, Metals, and more.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data

logger = logging.getLogger(__name__)

async def detect_themes(lookback: int = 20) -> SignalResult:
    """Identify outperforming market themes by clustering thematic ETF performance."""
    themes = {
        "AI & Robotics": ["SMH", "ROBO", "BOTZ", "SOXX"],
        "Innovation & Tech": ["ARKK", "KWEB", "HACK", "CIBR"],
        "Biotech & Genomics": ["XBI", "IBB", "ARKG"],
        "Energy & Green": ["XLE", "XOP", "TAN", "LIT", "URA"],
        "Metals & Miners": ["GDX", "GDXJ", "XME", "SIL"],
        "Real Estate & Housing": ["XLRE", "XHB", "ITB"],
        "Travel & Retail": ["JETS", "XRT", "PEJ"]
    }
    
    unique_etfs = set()
    for etfs in themes.values():
        unique_etfs.update(etfs)
        
    async def _fetch_perf(t: str):
        try:
            records = await get_historical_data(t, period="1mo", interval="1d")
            if not records: return t, None
            df = pd.DataFrame(records)
            if df.empty: return t, None
            
            # Ensure close is numeric
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            curr = float(df['close'].iloc[-1])
            
            # 5 sessions ago
            prev_5d = float(df['close'].iloc[-6]) if len(df) >= 6 else float(df['close'].iloc[0])
            # 20 sessions ago (approx)
            prev_20d = float(df['close'].iloc[0])
            
            return t, {
                "perf_5d_pct": round(((curr / prev_5d) - 1) * 100, 2),
                "perf_20d_pct": round(((curr / prev_20d) - 1) * 100, 2)
            }
        except Exception as e:
            logger.debug(f"Failed to fetch theme ETF {t}: {e}")
            return t, None

    tasks = [_fetch_perf(t) for t in unique_etfs]
    fetched = await asyncio.gather(*tasks)
    etf_data = {t: r for t, r in fetched if r is not None}
    
    results = []
    for name, etfs in themes.items():
        valid_perfs_5d = [etf_data[t]['perf_5d_pct'] for t in etfs if t in etf_data]
        valid_perfs_20d = [etf_data[t]['perf_20d_pct'] for t in etfs if t in etf_data]
        
        if not valid_perfs_5d: continue
        
        avg_5d = sum(valid_perfs_5d) / len(valid_perfs_5d)
        avg_20d = sum(valid_perfs_20d) / len(valid_perfs_20d)
        
        momentum = "Neutral"
        if avg_5d > 2.0: momentum = "Strong Surge"
        elif avg_5d > 0.5: momentum = "Positive Momentum"
        elif avg_5d < -2.0: momentum = "Strong Fade"
        elif avg_5d < -0.5: momentum = "Weakening"
        
        results.append({
            "theme": name,
            "etfs": [t for t in etfs if t in etf_data],
            "avg_perf_5d_pct": round(avg_5d, 2),
            "avg_perf_20d_pct": round(avg_20d, 2),
            "momentum": momentum
        })
        
    if not results:
        return SignalResult.error_msg("No theme data could be fetched.")
        
    # Sort by 5d performance
    results.sort(key=lambda x: x['avg_perf_5d_pct'], reverse=True)
    
    leader = results[0]['theme']
    laggard = results[-1]['theme']
    
    summary = f"Theme Detection: {leader} is currently leading ({results[0]['avg_perf_5d_pct']}% 5d). {laggard} is lagging ({results[-1]['avg_perf_5d_pct']}% 5d)."
    
    return SignalResult.success({
        "themes": results,
        "leader": leader,
        "laggard": laggard,
        "summary": summary
    })

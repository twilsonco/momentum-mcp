"""
Pair Trade Screener — Statistical Arbitrage (Correlation & Z-Score).

Identifies mean-reversion opportunities between two correlated assets.
Answers: 'Is the spread between $KO and $PEP stretched?'
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any
import pandas as pd
import numpy as np

from mcp_server.cache import smart_cache
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data

logger = logging.getLogger(__name__)

@smart_cache(open_ttl=600, closed_ttl=3600)
async def analyze_pair(
    ticker_a: str,
    ticker_b: str,
    lookback: int = 60,
) -> SignalResult:
    """Analyze a pair of stocks for statistical arbitrage.
    Computes correlation, ratio, and Z-score of the spread."""
    try:
        ticker_a = ticker_a.strip().upper()
        ticker_b = ticker_b.strip().upper()
        
        # 1. Fetch 1y data for both
        data_a, data_b = await asyncio.gather(
            get_historical_data(ticker_a, period="1y", interval="1d"),
            get_historical_data(ticker_b, period="1y", interval="1d")
        )
        
        if len(data_a) < lookback or len(data_b) < lookback:
            return SignalResult.error_msg("Insufficient data for one or both tickers.")
        
        df_a = pd.DataFrame(data_a)
        df_b = pd.DataFrame(data_b)
        
        # Map keys to lowercase
        df_a.columns = [c.lower() for c in df_a.columns]
        df_b.columns = [c.lower() for c in df_b.columns]
        
        # Align dates
        df = pd.merge(
            df_a[['date', 'close']].rename(columns={'close': 'a'}),
            df_b[['date', 'close']].rename(columns={'close': 'b'}),
            on='date'
        )
        df['a'] = pd.to_numeric(df['a'], errors='coerce')
        df['b'] = pd.to_numeric(df['b'], errors='coerce')
        df.dropna(inplace=True)
        
        if len(df) < lookback:
            return SignalResult.error_msg("Insufficient overlapping data for the pair.")
        
        # 2. Correlation
        correlation = df['a'].corr(df['b'])
        
        # 3. Ratio and Z-Score
        df['ratio'] = df['a'] / df['b']
        avg_ratio = df['ratio'].rolling(lookback).mean()
        std_ratio = df['ratio'].rolling(lookback).std()
        df['zscore'] = (df['ratio'] - avg_ratio) / std_ratio
        
        current_z = df['zscore'].iloc[-1]
        current_ratio = df['ratio'].iloc[-1]
        
        # 4. Verdict
        if current_z > 2.0:
            verdict = "STRETCHED (HIGH)"
            action = f"Short {ticker_a} / Long {ticker_b}"
        elif current_z < -2.0:
            verdict = "STRETCHED (LOW)"
            action = f"Long {ticker_a} / Short {ticker_b}"
        elif abs(current_z) < 0.5:
            verdict = "MEAN REVERTED"
            action = "Neutral"
        else:
            verdict = "NORMAL"
            action = "None"

        summary = (
            f"Pair Analysis {ticker_a}/{ticker_b}:\n"
            f"- Status: {verdict}\n"
            f"- Current Z-Score: {current_z:.2f}\n"
            f"- Correlation (1y): {correlation:.2f}\n"
            f"- Recommendation: {action}"
        )

        return SignalResult.success({
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "correlation": round(correlation, 4),
            "current_ratio": round(current_ratio, 4),
            "zscore": round(current_z, 2),
            "lookback_days": lookback,
            "verdict": verdict,
            "action": action,
            "summary": summary
        })

    except Exception as e:
        logger.error(f"Pair analysis failed: {e}")
        return SignalResult.error_msg(str(e))

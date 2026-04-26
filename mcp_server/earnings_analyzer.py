"""
Earnings Trade Analyzer — Post-earnings reaction quality scoring.
Scores the "gap quality" based on price, volume, and fundamentals.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data
from mcp_server.fundamentals import get_fundamentals

logger = logging.getLogger(__name__)

async def analyze_recent_gap(ticker: str) -> SignalResult:
    """Score the most recent overnight gap reaction (0-100) based on gap size, volume, and follow-through."""
    ticker = ticker.strip().upper()
    try:
        # Fetch data
        hist_task = get_historical_data(ticker, period="3mo", interval="1d")
        fund_task = get_fundamentals(ticker)
        
        records, fund = await asyncio.gather(hist_task, fund_task)
        if not records or len(records) < 2:
            return SignalResult.error_msg("Insufficient historical data for earnings analysis.")
        
        df = pd.DataFrame(records)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        
        df['vol_20d_avg'] = df['volume'].rolling(20).mean()
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Basic metrics
        gap_pct = ((curr['open'] / prev['close']) - 1) * 100
        vol_ratio = curr['volume'] / curr['vol_20d_avg'] if curr['vol_20d_avg'] > 0 else 1.0
        
        # Scoring
        score = 0
        factors = {}
        
        # 1. Gap Size (20 pts)
        gap_score = 0
        if gap_pct >= 5.0: gap_score = 20
        elif gap_pct >= 3.0: gap_score = 15
        elif gap_pct >= 1.0: gap_score = 8
        elif gap_pct > 0: gap_score = 4
        score += gap_score
        factors["gap"] = {"value": f"{round(gap_pct, 2)}%", "score": gap_score}
        
        # 2. Volume Expansion (20 pts)
        vol_score = 0
        if vol_ratio >= 3.0: vol_score = 20
        elif vol_ratio >= 2.0: vol_score = 15
        elif vol_ratio >= 1.5: vol_score = 10
        elif vol_ratio > 1.0: vol_score = 5
        score += vol_score
        factors["volume"] = {"value": f"{round(vol_ratio, 2)}x", "score": vol_score}
        
        # 3. Price Hold (Intraday) (20 pts)
        day_range = curr['high'] - curr['low']
        hold_ratio = 0.5
        price_hold_score = 0
        if day_range > 0:
            hold_ratio = (curr['close'] - curr['low']) / day_range
            if hold_ratio >= 0.8: price_hold_score = 20
            elif hold_ratio >= 0.6: price_hold_score = 12
            elif hold_ratio >= 0.4: price_hold_score = 6
        score += price_hold_score
        factors["price_hold"] = {"value": f"{round(hold_ratio*100, 1)}% of range", "score": price_hold_score}
        
        # 4. Fundamental Quality (20 pts)
        fund_score = 0
        # Check revenueGrowth or earningsGrowth
        rev_growth = fund.get("revenueGrowth", 0)
        if rev_growth is None: rev_growth = 0
        
        if rev_growth > 0.25: fund_score = 20
        elif rev_growth > 0.10: fund_score = 15
        elif rev_growth > 0: fund_score = 10
        score += fund_score
        factors["fundamentals"] = {"rev_growth": f"{round(rev_growth*100, 1)}%", "score": fund_score}
        
        # 5. Technical Context (20 pts)
        tech_score = 0
        # Was it a new 3-month high?
        high_3m = df['close'].iloc[:-1].max()
        if curr['close'] > high_3m:
            tech_score = 20
        elif curr['close'] > df['close'].iloc[-21:-1].max():
            tech_score = 10
        score += tech_score
        factors["context"] = {"is_new_high": bool(curr['close'] > high_3m), "score": tech_score}
        
        grade = "F"
        if score >= 85: grade = "A+"
        elif score >= 75: grade = "A"
        elif score >= 60: grade = "B"
        elif score >= 45: grade = "C"
        elif score >= 30: grade = "D"
        
        return SignalResult.success({
            "ticker": ticker,
            "score": score,
            "grade": grade,
            "gap_pct": round(gap_pct, 2),
            "vol_ratio": round(vol_ratio, 2),
            "factors": factors,
            "summary": f"Earnings Trade Analysis for {ticker}: Grade {grade} (Score {score}/100). Gap: {round(gap_pct, 2)}% on {round(vol_ratio, 2)}x volume."
        })

    except Exception as e:
        logger.error(f"Earnings analysis failed for {ticker}: {e}")
        return SignalResult.error_msg(str(e))

"""
Market Breadth Analyzer — Quantitative breadth health scoring.
Uses ETF proxies and screener data to measure market participation.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data
from mcp_server.screener import run_stock_screen, run_custom_screen

logger = logging.getLogger(__name__)

async def analyze_breadth() -> SignalResult:
    """Compute a 100-point market breadth health score."""
    try:
        # 1. RSP vs SPY (20 pts)
        # 2. IWM vs SPY (20 pts)
        # 3. New Highs vs New Lows (20 pts)
        # 4. % of SPY above 50d MA (20 pts)
        # 5. VIX Term Structure (20 pts)
        
        async def _get_ratio_trend(a: str, b: str):
            try:
                data_a = await get_historical_data(a, period="3mo", interval="1d")
                data_b = await get_historical_data(b, period="3mo", interval="1d")
                if not data_a or not data_b: return 10
                df_a = pd.DataFrame(data_a)
                df_b = pd.DataFrame(data_b)
                # Align on last session
                curr_a = float(df_a['close'].iloc[-1])
                curr_b = float(df_b['close'].iloc[-1])
                # 1 month ago
                idx = -21 if len(df_a) >= 21 else 0
                prev_a = float(df_a['close'].iloc[idx])
                prev_b = float(df_b['close'].iloc[idx])
                
                curr_ratio = curr_a / curr_b
                prev_ratio = prev_a / prev_b
                if curr_ratio > prev_ratio * 1.01: return 20
                if curr_ratio < prev_ratio * 0.99: return 0
                return 10
            except Exception as e:
                logger.debug(f"Ratio trend failed for {a}/{b}: {e}")
                return 10

        async def _get_nh_nl_proxy():
            try:
                # Use EMA stack presets as proxies for New Highs vs New Lows participation
                bulls = await run_stock_screen(preset="bullish_ema_stack", limit=100)
                bears = await run_stock_screen(preset="bearish_ema_stack", limit=100)
                b_count = len(bulls)
                r_count = len(bears)
                
                if b_count > r_count * 2: return 20
                if b_count > r_count: return 15
                if r_count > b_count * 2: return 0
                if r_count > b_count: return 5
                return 10
            except Exception as e:
                logger.debug(f"NH/NL proxy failed: {e}")
                return 10

        async def _get_above_50d():
            try:
                # Direct check of components above 50d MA proxy
                res = await run_custom_screen(filters=[{"field": "close", "operation": ">", "value": "SMA50"}], limit=100)
                count = len(res)
                if count >= 80: return 20
                if count >= 60: return 16
                if count >= 40: return 12
                if count >= 20: return 6
                return 2
            except Exception as e:
                logger.debug(f"Above 50d proxy failed: {e}")
                return 10

        async def _get_vix_term():
            try:
                # VIX vs VIX3M (Contango = Healthy, Backwardation = Stress)
                vix_recs = await get_historical_data("^VIX", period="5d")
                vix3m_recs = await get_historical_data("^VIX3M", period="5d")
                if not vix_recs or not vix3m_recs: return 10
                
                vix_curr = float(vix_recs[-1]['close'])
                vix3m_curr = float(vix3m_recs[-1]['close'])
                
                # If VIX3M > VIX, markets are calm (contango)
                if vix3m_curr > vix_curr * 1.05: return 20
                if vix3m_curr > vix_curr: return 15
                if vix_curr > vix3m_curr * 1.1: return 0 # Deep backwardation
                return 5
            except Exception as e:
                logger.debug(f"VIX term failed: {e}")
                return 10

        results = await asyncio.gather(
            _get_ratio_trend("RSP", "SPY"),
            _get_ratio_trend("IWM", "SPY"),
            _get_nh_nl_proxy(),
            _get_above_50d(),
            _get_vix_term()
        )
        
        score = sum(results)
        score = min(100, max(0, score))
        
        verdict = "Neutral"
        if score >= 80: verdict = "Healthy"
        elif score >= 60: verdict = "Adequate"
        elif score >= 40: verdict = "Weakening"
        elif score >= 20: verdict = "Poor"
        else: verdict = "Severe Stress"
        
        return SignalResult.success({
            "score": score,
            "verdict": verdict,
            "components": {
                "rsp_vs_spy_trend": "Bullish" if results[0] == 20 else "Bearish" if results[0] == 0 else "Neutral",
                "iwm_vs_spy_trend": "Bullish" if results[1] == 20 else "Bearish" if results[1] == 0 else "Neutral",
                "nh_nl_score": results[2],
                "above_50d_proxy_score": results[3],
                "vix_term_score": results[4]
            },
            "summary": f"Market Breadth Health: {verdict} ({score}/100)."
        })

    except Exception as e:
        logger.error(f"Breadth analysis failed: {e}")
        return SignalResult.error_msg(str(e))

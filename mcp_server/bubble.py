"""
Bubble Detector — Minsky/Kindleberger euphoria scoring (0-15).
Measures extension, complacency, valuation, and speculative fever.
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

async def detect_bubble_risk() -> SignalResult:
    """Assess market euphoria and bubble risk (0-15 score)."""
    try:
        # 1. Price vs 200d SMA (SPY)
        # 2. VIX Complacency
        # 3. Valuation (SPY P/E)
        # 4. Speculative Volume (TQQQ vs SPY)
        # 5. Euphoria (Meme stock volume surge)
        
        async def _get_spy_ext():
            try:
                data = await get_historical_data("SPY", period="1y")
                if not data: return 1
                df = pd.DataFrame(data)
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['sma200'] = df['close'].rolling(200).mean()
                curr = df.iloc[-1]
                if pd.isna(curr['sma200']): return 1
                ext = (curr['close'] / curr['sma200']) - 1
                if ext > 0.15: return 3
                if ext > 0.10: return 2
                if ext > 0.05: return 1
                return 0
            except Exception: return 1

        async def _get_vix_comp():
            try:
                vix = await get_historical_data("^VIX", period="1mo")
                if not vix: return 1
                avg_vix = sum(float(d['close']) for d in vix) / len(vix)
                if avg_vix < 13: return 3
                if avg_vix < 15: return 2
                if avg_vix < 18: return 1
                return 0
            except Exception: return 1

        async def _get_valuation():
            try:
                fund = await get_fundamentals("SPY")
                pe = fund.get("trailingPE")
                if pe is None: pe = 22 # Normal fallback
                
                if pe > 32: return 3
                if pe > 28: return 2
                if pe > 24: return 1
                return 0
            except Exception: return 1

        async def _get_spec_vol():
            try:
                tqqq = await get_historical_data("TQQQ", period="1mo")
                spy = await get_historical_data("SPY", period="1mo")
                if not tqqq or not spy: return 1
                t_vol = sum(float(d['volume']) * float(d['close']) for d in tqqq)
                s_vol = sum(float(d['volume']) * float(d['close']) for d in spy)
                if s_vol == 0: return 1
                ratio = t_vol / s_vol
                if ratio > 0.08: return 3
                if ratio > 0.05: return 2
                if ratio > 0.03: return 1
                return 0
            except Exception: return 1

        async def _get_euphoria():
            try:
                # Use GME as the meme-euphoria proxy
                gme = await get_historical_data("GME", period="3mo")
                if not gme: return 1
                df = pd.DataFrame(gme)
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                curr_vol = df['volume'].iloc[-5:].mean()
                avg_vol = df['volume'].mean()
                if curr_vol > avg_vol * 4: return 3
                if curr_vol > avg_vol * 2.5: return 2
                if curr_vol > avg_vol * 1.5: return 1
                return 0
            except Exception: return 1

        results = await asyncio.gather(
            _get_spy_ext(),
            _get_vix_comp(),
            _get_valuation(),
            _get_spec_vol(),
            _get_euphoria()
        )
        
        score = sum(results)
        
        verdict = "Normal"
        if score >= 12: verdict = "Extreme Euphoria / Bubble"
        elif score >= 9: verdict = "Bubble Warning"
        elif score >= 6: verdict = "Elevated Risk"
        elif score >= 3: verdict = "Moderate Speculation"
        
        return SignalResult.success({
            "score": score,
            "verdict": verdict,
            "components": {
                "extension_score": results[0],
                "complacency_score": results[1],
                "valuation_score": results[2],
                "spec_volume_score": results[3],
                "euphoria_score": results[4]
            },
            "summary": f"Bubble Risk Assessment: {verdict} ({score}/15)."
        })

    except Exception as e:
        logger.error(f"Bubble detection failed: {e}")
        return SignalResult.error_msg(str(e))

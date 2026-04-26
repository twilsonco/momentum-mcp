"""
PEAD Screener — Post-Earnings Announcement Drift detection.

Identifies stocks that gapped up on earnings and are now providing low-risk
entries via pullbacks or consolidations.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any
import pandas as pd

from mcp_server.cache import smart_cache
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data
from mcp_server.traderdaddy import get_earnings_calendar

logger = logging.getLogger(__name__)

@smart_cache(open_ttl=600, closed_ttl=3600)
async def screen_pead(
    lookback_days: int = 10,
) -> SignalResult:
    """Screen for Post-Earnings Announcement Drift (PEAD) setups.
    Looks for recent earnings gaps followed by constructive consolidations or pullbacks."""
    try:
        # 1. Fetch recent earnings
        # Note: We'll check both 'current' and 'next' (which might contain recently past ones depending on API behavior)
        # But for now we use 'current' as our source of recent tickers.
        cal_res = await get_earnings_calendar(week="current")
        if cal_res.status != "success":
            return SignalResult.error_msg("Could not fetch earnings calendar.")
        
        # Structure is {'events': [...]}
        data = cal_res.data if isinstance(cal_res.data, dict) else {}
        tickers = [item["ticker"] for item in data.get("events", []) if "ticker" in item]
        
        if not tickers:
            # Try 'next'
            cal_res_next = await get_earnings_calendar(week="next")
            if cal_res_next.status == "success":
                data_next = cal_res_next.data if isinstance(cal_res_next.data, dict) else {}
                tickers = [item["ticker"] for item in data_next.get("events", []) if "ticker" in item]

        if not tickers:
            return SignalResult.error_msg("No recent earnings tickers found.")

        # Dedup
        tickers = list(set(tickers))

        results = []
        
        async def _analyze_pead(ticker: str):
            try:
                records = await get_historical_data(ticker, period="3mo", interval="1d")
                if len(records) < 20: return None
                
                df = pd.DataFrame(records)
                df.columns = [c.lower() for c in df.columns]
                for col in ["close", "open", "high", "low", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Find the gap day in the last 'lookback_days'
                gaps = []
                for i in range(1, lookback_days + 1):
                    idx = -i
                    if abs(idx) >= len(df): break
                    
                    prev_close = df["close"].iloc[idx-1]
                    curr_open = df["open"].iloc[idx]
                    if pd.isna(prev_close) or pd.isna(curr_open): continue
                    
                    gap = (curr_open / prev_close) - 1
                    gaps.append((gap, idx))
                
                if not gaps: return None
                
                # We want the largest POSITIVE gap
                max_gap, gap_pos = max(gaps, key=lambda x: x[0])
                
                # PEAD requires a significant positive gap (usually > 3%)
                if max_gap < 0.03: return None
                
                # Check volume on gap day (should be heavy)
                gap_vol = df["volume"].iloc[gap_pos]
                avg_vol_20 = df["volume"].rolling(20).mean().iloc[gap_pos]
                if pd.isna(avg_vol_20) or gap_vol < avg_vol_20 * 1.5:
                    return None
                
                # Current state analysis
                current_price = df["close"].iloc[-1]
                ema10 = df["close"].ewm(span=10).mean().iloc[-1]
                ema20 = df["close"].ewm(span=20).mean().iloc[-1]
                
                # 1. Pullback to EMA Setup
                near_ema10 = (current_price >= ema10 * 0.98) and (current_price <= ema10 * 1.02)
                near_ema20 = (current_price >= ema20 * 0.98) and (current_price <= ema20 * 1.03)
                
                # 2. High Tight Flag (consolidation near highs)
                gap_high = df["high"].iloc[gap_pos:].max()
                is_flag = (current_price >= gap_high * 0.94)
                
                if not (near_ema10 or near_ema20 or is_flag):
                    return None
                
                score = 40
                score += min(30, max_gap * 100 * 2) # Gap size weight
                if near_ema10: score += 15
                elif near_ema20: score += 20
                if is_flag: score += 10
                
                return {
                    "ticker": ticker,
                    "gap_pct": round(max_gap * 100, 2),
                    "days_since_gap": abs(gap_pos) - 1,
                    "current_price": round(current_price, 2),
                    "setup": "EMA Pullback" if (near_ema10 or near_ema20) else "Consolidation Flag",
                    "pead_score": round(score, 1)
                }

            except Exception as e:
                return None

        tasks = [_analyze_pead(t) for t in tickers]
        analyzed = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in analyzed if r is not None and not isinstance(r, Exception)]
        results.sort(key=lambda x: x["pead_score"], reverse=True)
        
        summary = f"PEAD Screen complete. Found {len(results)} setups from {len(tickers)} potential earners."
        if results:
            summary += f" Top setup: {results[0]['ticker']} ({results[0]['setup']})."

        return SignalResult.success({
            "results": results,
            "count": len(results),
            "summary": summary
        })

    except Exception as e:
        logger.error(f"PEAD screen failed: {e}")
        return SignalResult.error_msg(str(e))

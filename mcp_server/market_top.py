"""
Market Top Detector — O'Neil Distribution Days + Leading Stock Deterioration.

Detects when the market is topping out using:
1. Distribution day counting on major indices
2. Leading stock breakdown detection
3. Defensive sector rotation signal
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
import pandas as pd

from mcp_server.cache import smart_cache
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data
from mcp_server.screener import run_stock_screen
from mcp_server.traderdaddy import get_sector_flow

logger = logging.getLogger(__name__)

@smart_cache(open_ttl=1800, closed_ttl=3600)
async def detect_market_top() -> SignalResult:
    """Detect market topping signals using distribution days and leadership trends."""
    try:
        indices = ["SPY", "QQQ", "IWM"]
        index_data = {}
        
        # 1. Fetch index data
        for ticker in indices:
            try:
                records = await get_historical_data(ticker, period="3mo", interval="1d")
                df = pd.DataFrame(records)
                # Map keys to lowercase for consistency
                df.columns = [c.lower() for c in df.columns]
                for col in ["close", "high", "low", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                index_data[ticker] = df
            except Exception as e:
                logger.warning(f"Could not fetch data for {ticker}: {e}")

        if not index_data:
            return SignalResult.error_msg("Could not fetch index data for market top detection.")

        # 2. Distribution Day Count (last 25 sessions)
        dist_days = {}
        stalling_days = {}
        
        for ticker, df in index_data.items():
            df_recent = df.iloc[-26:] # Need 26 to get 25 diffs
            count = 0
            stall_count = 0
            
            for i in range(1, len(df_recent)):
                curr = df_recent.iloc[i]
                prev = df_recent.iloc[i-1]
                
                if pd.isna(curr["close"]) or pd.isna(prev["close"]): continue
                
                price_change = (curr["close"] - prev["close"]) / prev["close"]
                
                # Distribution day: price down >= 0.2% on higher volume than prior day
                if price_change <= -0.002 and curr["volume"] > prev["volume"]:
                    count += 1
                
                # Stalling day: Price up but closes in lower half of range on heavy volume
                price_range = curr["high"] - curr["low"]
                if price_change > 0 and price_range > 0:
                    if (curr["close"] - curr["low"]) < 0.5 * price_range and curr["volume"] > prev["volume"]:
                        stall_count += 1
            
            dist_days[ticker] = count
            stalling_days[ticker] = stall_count

        # 4. Leading Stock Health (current count)
        # Fetch current count of stocks in momentum/uptrend presets
        current_leaders = 0
        try:
            screen_momentum = await run_stock_screen(preset="high_momentum", limit=100)
            screen_ema = await run_stock_screen(preset="bullish_ema_stack", limit=100)
            unique_leaders = set([s["ticker"] for s in screen_momentum] + [s["ticker"] for s in screen_ema])
            current_leaders = len(unique_leaders)
        except Exception as e:
            logger.debug(f"Leader count failed: {e}")
            current_leaders = 40 # Fallback "healthy" number
            
        # Heuristic: declining leadership often precedes market tops
        # Without historical persistence, we use a baseline of 40
        deterioration_score = 0
        if current_leaders < 20:
            deterioration_score = 25
        elif current_leaders < 35:
            deterioration_score = 10

        # 5. Defensive Rotation (from sector flow)
        rotation_score = 0
        try:
            sector_res = await get_sector_flow(window="1d")
            if sector_res.status == "success":
                # Compare defensive sectors (Utilities, Staples, Health) vs Aggressive
                defensive_sectors = {"Utilities", "Consumer Staples", "Healthcare"}
                aggressive_sectors = {"Technology", "Consumer Discretionary", "Financials"}
                
                def_sent = 0
                agg_sent = 0
                for item in sector_res.data:
                    sector = item.get("sector")
                    sentiment = item.get("sentiment_score", 0)
                    if sector in defensive_sectors: def_sent += sentiment
                    if sector in aggressive_sectors: agg_sent += sentiment
                
                if def_sent > agg_sent and def_sent > 0:
                    rotation_score = 20
        except Exception:
            pass

        # 6. Composite Score (0-100)
        # Max distribution count across indices
        max_dist = max(dist_days.values()) if dist_days else 0
        dist_score = 0
        if max_dist >= 6: dist_score = 40
        elif max_dist == 5: dist_score = 30
        elif max_dist == 4: dist_score = 20
        
        max_stall = max(stalling_days.values()) if stalling_days else 0
        stall_score = min(15, max_stall * 3)
        
        total_score = dist_score + stall_score + deterioration_score + rotation_score
        
        # 7. Verdict
        if total_score >= 76: verdict = "DANGER"
        elif total_score >= 51: verdict = "DISTRIBUTION"
        elif total_score >= 26: verdict = "CAUTION"
        else: verdict = "HEALTHY"
        
        summary_parts = [
            f"Market Top Detection: {verdict} (Score: {total_score}/100)",
            f"Max Dist Days: {max_dist} (last 25 sessions)",
            f"Leadership: {current_leaders} stocks in momentum setups"
        ]
        if rotation_score > 0:
            summary_parts.append("⚠️ Defensive sector rotation detected in options flow.")

        return SignalResult.success({
            "score": total_score,
            "verdict": verdict,
            "distribution_days": dist_days,
            "stalling_days": stalling_days,
            "leader_count_now": current_leaders,
            "defensive_rotation_detected": rotation_score > 0,
            "summary": " | ".join(summary_parts)
        })

    except Exception as e:
        logger.error(f"Market top detection failed: {e}")
        return SignalResult.error_msg(str(e))

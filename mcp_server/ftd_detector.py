"""
FTD Detector — Follow-Through Day market bottom confirmation.

Identifies the start of a new bull market by detecting price/volume expansion
after a market correction.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any
import pandas as pd

from mcp_server.cache import smart_cache
from mcp_server.schema import SignalResult
from mcp_server.data import get_historical_data

logger = logging.getLogger(__name__)

@smart_cache(open_ttl=1800, closed_ttl=3600)
async def detect_ftd() -> SignalResult:
    """Detect Follow-Through Days (FTDs) on major indices to confirm market bottoms."""
    try:
        indices = ["SPY", "QQQ", "IWM"]
        results = {}
        
        for ticker in indices:
            try:
                records = await get_historical_data(ticker, period="6mo", interval="1d")
                df = pd.DataFrame(records)
                # Map keys to lowercase for consistency
                df.columns = [c.lower() for c in df.columns]
                for col in ["close", "high", "low", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Find the recent low (lowest point in the last 65 days)
                lookback = min(65, len(df))
                recent_df = df.iloc[-lookback:]
                low_idx_in_recent = recent_df["low"].idxmin()
                
                # Get the integer index relative to the full dataframe
                # In pandas, if index is RangeIndex, low_idx_in_recent is the integer index.
                # If it's a DatetimeIndex, we need to find the position.
                if isinstance(df.index, pd.RangeIndex):
                    low_pos = low_idx_in_recent
                else:
                    low_pos = df.index.get_loc(low_idx_in_recent)
                
                low_price = df["low"].iloc[low_pos]
                
                # Analyze rally since the low
                df_rally = df.iloc[low_pos:]
                
                undercut = False
                rally_days = 0
                ftd_found = False
                ftd_details = None
                
                for i in range(len(df_rally)):
                    curr = df_rally.iloc[i]
                    if curr["low"] < low_price:
                        # New low formed, reset rally count (simplified logic)
                        undercut = True
                        break
                    
                    rally_days += 1
                    
                    # FTD criteria: Day 4-10 of rally, Price up >= 1.5% on higher volume
                    if 4 <= rally_days <= 10:
                        if i > 0:
                            prev = df_rally.iloc[i-1]
                            price_change = (curr["close"] - prev["close"]) / prev["close"]
                            
                            if price_change >= 0.015 and curr["volume"] > prev["volume"]:
                                ftd_found = True
                                ftd_details = {
                                    "date": str(df_rally.index[i]),
                                    "price_change_pct": round(price_change * 100, 2),
                                    "rally_day": rally_days,
                                    "volume_expansion_pct": round((curr["volume"] / prev["volume"] - 1) * 100, 1)
                                }
                                break

                results[ticker] = {
                    "status": "FTD CONFIRMED" if ftd_found else "RALLY ATTEMPT" if not undercut else "UNDER CORRECTION",
                    "rally_days": rally_days,
                    "recent_low": round(low_price, 2),
                    "ftd": ftd_details
                }
            except Exception as e:
                logger.warning(f"FTD analysis failed for {ticker}: {e}")

        if not results:
            return SignalResult.error_msg("FTD detection failed for all indices.")

        # Global Summary
        confirmed = [t for t, r in results.items() if r["status"] == "FTD CONFIRMED"]
        if confirmed:
            summary = f"FTD Detected on {', '.join(confirmed)}! Market bottom confirmed."
        else:
            # Check if any are in rally attempt
            in_rally = [t for t, r in results.items() if r["status"] == "RALLY ATTEMPT"]
            if in_rally:
                summary = f"Market in Rally Attempt ({', '.join(in_rally)}). Waiting for Follow-Through Day (Day 4-10)."
            else:
                summary = "Market remains under correction. No valid rally attempt yet."

        return SignalResult.success({
            "indices": results,
            "summary": summary
        })

    except Exception as e:
        logger.error(f"FTD detection failed: {e}")
        return SignalResult.error_msg(str(e))

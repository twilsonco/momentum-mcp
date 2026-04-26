"""
VCP Screener — Minervini Volatility Contraction Pattern detection.

Identifies Stage 2 uptrends forming tight bases with contracting volatility
near breakout pivot points.
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
from mcp_server.screener import run_stock_screen

logger = logging.getLogger(__name__)

@smart_cache(open_ttl=600, closed_ttl=3600)
async def screen_vcp(
    tickers: list[str] | None = None,
    max_tickers: int = 50,
) -> SignalResult:
    """Screen for stocks forming a Volatility Contraction Pattern (VCP)."""
    try:
        # 1. Discover tickers if not provided
        if not tickers:
            screen_res = await run_stock_screen(preset="bullish_ema_stack", limit=max_tickers)
            tickers = [item["ticker"] for item in screen_res]
        
        if not tickers:
            return SignalResult.error_msg("No tickers provided or found via screen.")

        results = []
        
        async def _analyze_ticker(ticker: str):
            try:
                # Fetch 1y data
                records = await get_historical_data(ticker, period="1y", interval="1d")
                if len(records) < 200:
                    return None
                
                df = pd.DataFrame(records)
                # Map keys to lowercase for consistency
                df.columns = [c.lower() for c in df.columns]
                
                for col in ["close", "high", "low", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                df.dropna(subset=["close", "high", "low"], inplace=True)
                
                if len(df) < 200:
                    return None
                
                # 3. Minervini Trend Template
                current_price = df["close"].iloc[-1]
                sma50 = df["close"].rolling(50).mean().iloc[-1]
                sma150 = df["close"].rolling(150).mean().iloc[-1]
                sma200 = df["close"].rolling(200).mean().iloc[-1]
                
                # SMA 200 trending up (at least 1 month)
                # Check slope over last 20-40 days
                sma200_series = df["close"].rolling(200).mean()
                sma200_prev = sma200_series.iloc[-22] if len(sma200_series) > 22 else sma200_series.iloc[0]
                
                low_52w = df["low"].min()
                high_52w = df["high"].max()
                
                # Minervini's core 8 criteria (simplified)
                trend_template = (
                    current_price > sma150 > sma200
                    and current_price > sma50
                    and sma200 > sma200_prev
                    and current_price >= low_52w * 1.25 # At least 25% above low
                    and current_price >= high_52w * 0.75 # Within 25% of high
                )
                
                if not trend_template:
                    return None
                
                # 4. VCP Base Detection
                # A simplified approach to find contractions
                # We'll look at the last 100 days (typical base length)
                base_df = df.iloc[-100:]
                base_high = base_df["high"].max()
                base_low = base_df["low"].min()
                total_depth = (base_high - base_low) / base_high
                
                # We look for "T"s (Tightenings)
                # Divide the base into 3-4 segments and check the max volatility (high-low range) in each
                # This is a heuristic to detect contraction
                segment_size = 20
                if len(base_df) < 40: return None
                
                depths = []
                # Look backwards from now
                for i in range(0, 80, segment_size):
                    seg = base_df.iloc[-(i+segment_size) : (-i if i > 0 else None)]
                    if len(seg) < 5: continue
                    d = (seg["high"].max() - seg["low"].min()) / seg["high"].max()
                    depths.insert(0, d) # Keep chronological order
                
                # Ensure depths are contracting (roughly)
                valid_vcp = False
                if len(depths) >= 2:
                    # Last depth should be tighter than the first depth
                    if depths[-1] < depths[0] * 0.8:
                        valid_vcp = True
                
                if not valid_vcp:
                    return None

                # 5. Pivot Point & Scoring
                pivot_price = base_df["high"].iloc[-15:].max() # Last 3 weeks high
                pct_from_pivot = (pivot_price - current_price) / current_price
                
                avg_vol_50 = df["volume"].rolling(50).mean().iloc[-1]
                last_vol = df["volume"].iloc[-1]
                vol_dryup = last_vol / avg_vol_50 if avg_vol_50 > 0 else 1.0
                
                # Score components (0-100)
                score = 30 # Trend template passed
                score += min(20, (depths[0] - depths[-1]) * 100) # Improvement in tightness
                score += max(0, 20 - (depths[-1] * 100)) # Absolute tightness
                score += max(0, 15 * (1 - vol_dryup)) if vol_dryup < 1 else 0
                score += max(0, 15 * (1 - abs(pct_from_pivot)/0.05)) if abs(pct_from_pivot) < 0.05 else 0
                
                return {
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "pivot_price": round(pivot_price, 2),
                    "pct_from_pivot": round(pct_from_pivot * 100, 2),
                    "contractions": len(depths),
                    "contraction_depths": [round(d * 100, 1) for d in depths],
                    "vol_dryup_ratio": round(vol_dryup, 2),
                    "vcp_score": round(score, 1),
                    "trend_template_pass": True,
                    "stage": "Stage 2"
                }
            except Exception as e:
                logger.debug(f"VCP analysis failed for {ticker}: {e}")
                return None

        # Run analysis in parallel
        tasks = [_analyze_ticker(t) for t in tickers]
        analyzed = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = [r for r in analyzed if r is not None and not isinstance(r, Exception)]
        results.sort(key=lambda x: x["vcp_score"], reverse=True)
        
        summary = f"VCP Screen complete. Scanned {len(tickers)} tickers, found {len(results)} matching setups."
        if results:
            summary += f" Top candidate: {results[0]['ticker']} with score {results[0]['vcp_score']}."

        return SignalResult.success({
            "results": results,
            "count": len(results),
            "tickers_scanned": len(tickers),
            "summary": summary
        })

    except Exception as e:
        logger.error(f"VCP screen failed: {e}")
        return SignalResult.error_msg(str(e))

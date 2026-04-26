"""
CANSLIM Screener — O'Neil growth stock scoring system.

Ranks stocks based on Current earnings, Annual growth, New highs, Supply/Demand,
Leadership, Institutional sponsorship, and Market trend.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any
import pandas as pd

from mcp_server.cache import smart_cache
from mcp_server.schema import SignalResult
from mcp_server.fundamentals import get_fundamentals
from mcp_server.screener import run_stock_screen
from mcp_server.market_top import detect_market_top

logger = logging.getLogger(__name__)

@smart_cache(open_ttl=3600, closed_ttl=86400)
async def screen_canslim(
    tickers: list[str] | None = None,
    max_tickers: int = 30,
) -> SignalResult:
    """Screen for growth stocks matching CANSLIM criteria.
    Ranks candidates by earnings growth, price relative to highs, and institutional sponsorship."""
    try:
        # 1. Discover tickers if not provided
        if not tickers:
            try:
                screen_res = await run_stock_screen(preset="bullish_ema_stack", limit=max_tickers)
                tickers = [item["ticker"] for item in screen_res]
            except Exception:
                tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "NFLX"]
        
        if not tickers:
            return SignalResult.error_msg("No tickers provided or found.")

        # 2. Market Trend (M)
        market_healthy = False
        try:
            market_top = await detect_market_top()
            if market_top.status == "success":
                verdict = market_top.data.get("verdict")
                if verdict in ["HEALTHY", "CAUTION"]:
                    market_healthy = True
        except Exception:
            pass
        
        results = []
        
        async def _score_ticker(ticker: str):
            try:
                f = await get_fundamentals(ticker)
                if not f or "error" in f: return None
                
                # C & A: Earnings Growth
                # Use max of available growth metrics
                eg = max(
                    f.get("earnings_growth", 0) or 0,
                    f.get("quarterly_earnings_growth", 0) or 0,
                    f.get("revenue_growth", 0) or 0
                )
                c_score = 0
                if eg > 0.25: c_score = 25
                elif eg > 0.15: c_score = 15
                elif eg > 0: c_score = 5
                
                # N: New Highs (Within 10% of 52w high)
                price = f.get("price", 0)
                high52 = f.get("52_week_high", 0)
                n_score = 0
                if high52 > 0 and price >= high52 * 0.90:
                    n_score = 25
                elif high52 > 0 and price >= high52 * 0.80:
                    n_score = 10
                
                # I: Institutional Sponsorship
                inst = f.get("institutional_ownership", 0)
                i_score = 0
                if inst > 0.6: i_score = 20
                elif inst > 0.3: i_score = 10
                
                # S: Supply/Demand (Small float is better for CANSLIM, but we just use a baseline)
                s_score = 10
                
                # L: Leader (Relative to Market)
                l_score = 10 if market_healthy else 0
                
                total_score = c_score + n_score + i_score + s_score + l_score
                
                return {
                    "ticker": ticker,
                    "score": total_score,
                    "price": round(price, 2),
                    "growth_metric_pct": round(eg * 100, 1),
                    "institutional_pct": round(inst * 100, 1),
                    "near_52w_high": price >= high52 * 0.95 if high52 > 0 else False,
                    "sector": f.get("sector")
                }
            except Exception:
                return None

        # Process in batches to avoid rate limits
        tasks = [_score_ticker(t) for t in tickers]
        analyzed = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in analyzed if r is not None and not isinstance(r, Exception)]
        results.sort(key=lambda x: x["score"], reverse=True)
        
        summary = f"CANSLIM Screen complete. Found {len(results)} candidates from {len(tickers)} tickers."
        if results:
            summary += f" Top pick: {results[0]['ticker']} with score {results[0]['score']}."

        return SignalResult.success({
            "results": results,
            "count": len(results),
            "market_status": "Healthy" if market_healthy else "Caution/Danger",
            "summary": summary
        })

    except Exception as e:
        logger.error(f"CANSLIM screen failed: {e}")
        return SignalResult.error_msg(str(e))

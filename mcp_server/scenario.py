"""
Scenario Analyzer — Catalyst-driven projections and statistical price modeling.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.technicals import analyze_technicals
from mcp_server.fundamentals import get_fundamentals

logger = logging.getLogger(__name__)

async def analyze_scenario(
    ticker: str,
    catalyst: str,
    timeframe: str = "30d",
) -> SignalResult:
    """Generate bull/base/bear scenarios for a ticker based on a catalyst."""
    ticker = ticker.strip().upper()
    try:
        # Fetch technicals and fundamentals
        tech_task = analyze_technicals(ticker)
        fund_task = get_fundamentals(ticker)
        
        tech_res, f = await asyncio.gather(tech_task, fund_task)
        
        if tech_res.status != "success":
            return tech_res
            
        t = tech_res.data
        # Fundamentals returns dict directly in this codebase
        
        curr_price = t.get("close", 0)
        atr = t.get("atr_14", 0)
        
        if not curr_price or not atr:
            return SignalResult.error_msg("Incomplete data for scenario analysis.")
            
        # Scoring probability based on trend
        is_bullish = t.get("ema_stack_bullish", False)
        adx = t.get("adx_14", 0)
        
        # Bull Scenario: ~2.5 ATR extension
        bull_move = atr * 2.5
        bull_price = curr_price + bull_move
        bull_prob = 35 if is_bullish else 20
        if adx > 25 and is_bullish: bull_prob += 10
        
        # Bear Scenario: ~2.5 ATR rejection
        bear_move = atr * 2.5
        bear_price = curr_price - bear_move
        bear_prob = 20 if is_bullish else 35
        if adx > 25 and not is_bullish: bear_prob += 10
        
        # Base Scenario: Drift
        base_move = atr * 0.5 if is_bullish else -atr * 0.5
        base_price = curr_price + base_move
        base_prob = 100 - bull_prob - bear_prob
        
        scenarios = [
            {
                "name": "Bullish",
                "price_target": round(bull_price, 2),
                "move_pct": round((bull_move / curr_price) * 100, 1),
                "probability": f"{bull_prob}%",
                "rationale": f"Upside extension on {catalyst} if momentum carries to {round(bull_price, 2)}."
            },
            {
                "name": "Base Case",
                "price_target": round(base_price, 2),
                "move_pct": round((base_move / curr_price) * 100, 1),
                "probability": f"{base_prob}%",
                "rationale": f"Standard trend continuation following {catalyst}."
            },
            {
                "name": "Bearish",
                "price_target": round(bear_price, 2),
                "move_pct": round((-bear_move / curr_price) * 100, 1),
                "probability": f"{bear_prob}%",
                "rationale": f"Rejection or 'sell the news' reaction to {catalyst} down to {round(bear_price, 2)}."
            }
        ]
        
        return SignalResult.success({
            "ticker": ticker,
            "catalyst": catalyst,
            "current_price": curr_price,
            "atr": round(atr, 2),
            "scenarios": scenarios,
            "summary": f"Scenario analysis for {ticker} on {catalyst} completed. Base case target: {round(base_price, 2)} ({base_prob}% probability)."
        })

    except Exception as e:
        logger.error(f"Scenario analysis failed for {ticker}: {e}")
        return SignalResult.error_msg(str(e))

async def model_price_distribution(
    ticker: str,
    days_forward: int = 30,
) -> SignalResult:
    """Compute statistical price targets using historical volatility."""
    ticker = ticker.strip().upper()
    try:
        from mcp_server.data import get_historical_data
        df_raw = await get_historical_data(ticker, period="1y")
        if not df_raw:
            return SignalResult.error_msg("No historical data found.")
        
        df = pd.DataFrame(df_raw)
        df['returns'] = df['close'].pct_change()
        hist_vol = df['returns'].std() * (252 ** 0.5) # Annualized
        
        curr_price = float(df['close'].iloc[-1])
        
        # Periodic vol for days_forward
        period_vol = hist_vol * ((days_forward / 252) ** 0.5)
        
        # Standard Deviations
        targets = {
            "1_sigma_68pct": {
                "low": round(curr_price * (1 - period_vol), 2),
                "high": round(curr_price * (1 + period_vol), 2)
            },
            "2_sigma_95pct": {
                "low": round(curr_price * (1 - 2 * period_vol), 2),
                "high": round(curr_price * (1 + 2 * period_vol), 2)
            },
            "3_sigma_99pct": {
                "low": round(curr_price * (1 - 3 * period_vol), 2),
                "high": round(curr_price * (1 + 3 * period_vol), 2)
            }
        }
        
        return SignalResult.success({
            "ticker": ticker,
            "current_price": curr_price,
            "annualized_vol_pct": round(hist_vol * 100, 2),
            "days_forward": days_forward,
            "distribution": targets,
            "summary": f"Statistical distribution for {ticker} over {days_forward}d. 68% range: ${targets['1_sigma_68pct']['low']} - ${targets['1_sigma_68pct']['high']}."
        })
    except Exception as e:
        logger.error(f"Price distribution modeling failed for {ticker}: {e}")
        return SignalResult.error_msg(str(e))

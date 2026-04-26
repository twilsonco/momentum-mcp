"""
Risk Flags — Deterministic Trade Safety Validation.

Runs rule-based checks on every options recommendation to flag
potential risks. Zero LLM cost — pure logic.

Checks:
  - Earnings proximity (within 7 days of expiration)
  - VIX regime (elevated >25, panic >35)
  - Budget concentration (position >50% of budget)
  - 0DTE risk (DTE < 3)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def check_risk_flags(
    ticker: str,
    dte: int = 30,
    budget: float | None = None,
    position_cost: float | None = None,
) -> list[str]:
    """Run deterministic risk checks and return a list of warning flags.

    Args:
        ticker: Stock ticker symbol.
        dte: Days to expiration for the option.
        budget: User's stated budget (if provided).
        position_cost: Total cost of the proposed position.

    Returns:
        List of human-readable risk flag strings. Empty = no flags.
    """
    flags: list[str] = []

    # --- 1. 0DTE / ultra-short risk ---
    if dte <= 0:
        flags.append("🔴 0DTE — extreme gamma risk, position can go to zero in minutes")
    elif dte <= 3:
        flags.append("⚠️ Ultra-short DTE (<3 days) — rapid time decay, high gamma risk")

    # --- 2. VIX regime ---
    try:
        from mcp_server.data import get_live_price
        vix = await get_live_price("^VIX")
        if vix >= 35:
            flags.append(f"🔴 VIX at {vix:.1f} — panic regime. Premium is rich but risk of gap moves is extreme")
        elif vix >= 28:
            flags.append(f"⚠️ VIX at {vix:.1f} — elevated volatility. Size down, widen strikes")
        elif vix >= 22:
            flags.append(f"ℹ️ VIX at {vix:.1f} — above average, factor into position sizing")
    except Exception as e:
        logger.debug("VIX check failed: %s", e)

    # --- 3. Earnings proximity ---
    try:
        earnings_flags = await _check_earnings(ticker, dte)
        flags.extend(earnings_flags)
    except Exception as e:
        logger.debug("Earnings check failed for %s: %s", ticker, e)

    # --- 4. Budget concentration ---
    if budget and position_cost:
        pct = (position_cost / budget) * 100
        if pct > 80:
            flags.append(f"🔴 Position is {pct:.0f}% of your budget — extreme concentration risk")
        elif pct > 50:
            flags.append(f"⚠️ Position is {pct:.0f}% of your budget — consider sizing down")

    # --- 5. Institutional flow sentiment (TraderDaddy) ---
    try:
        from mcp_server.traderdaddy import get_ticker_flow
        flow = await get_ticker_flow(ticker, limit=5)
        if isinstance(flow, dict):
            trades = flow.get("trades", flow if isinstance(flow, list) else [])
        elif isinstance(flow, list):
            trades = flow
        else:
            trades = []
        if trades:
            bearish = sum(1 for t in trades if t.get("sentiment") == "bearish")
            bullish = sum(1 for t in trades if t.get("sentiment") == "bullish")
            if bearish > bullish and bearish >= 3:
                flags.append(f"⚠️ Institutional flow is {bearish}/{len(trades)} bearish on {ticker}")
    except Exception as e:
        logger.debug("Ticker flow check failed for %s: %s", ticker, e)

    # --- 6. GEX regime (TraderDaddy) ---
    try:
        from mcp_server.traderdaddy import get_ticker_gex
        gex = await get_ticker_gex(ticker)
        if isinstance(gex, dict) and not gex.get("error"):
            regime = gex.get("regime", "")
            if regime and "negative" in str(regime).lower():
                flags.append(f"⚠️ {ticker} in negative gamma — expect trending/acceleration moves")
    except Exception as e:
        logger.debug("GEX check failed for %s: %s", ticker, e)

    # --- 7. Pre-earnings institutional positioning (TraderDaddy) ---
    try:
        from mcp_server.traderdaddy import get_ticker_earnings_flow
        ef = await get_ticker_earnings_flow(ticker)
        if isinstance(ef, dict) and not ef.get("error"):
            flows = ef.get("flows", [])
            if flows:
                bearish_prem = sum(
                    t.get("premium", 0) for t in flows
                    if t.get("sentiment") == "bearish"
                )
                bullish_prem = sum(
                    t.get("premium", 0) for t in flows
                    if t.get("sentiment") == "bullish"
                )
                if bearish_prem > bullish_prem and bearish_prem > 100_000:
                    flags.append(
                        f"⚠️ Pre-earnings flow on {ticker} is bearish "
                        f"(${bearish_prem:,.0f} bearish vs ${bullish_prem:,.0f} bullish)"
                    )
    except Exception as e:
        logger.debug("Earnings flow check failed for %s: %s", ticker, e)

    return flags


async def _check_earnings(ticker: str, dte: int) -> list[str]:
    """Check if earnings fall within the option's life."""

    def _fetch_earnings() -> list[str]:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker.strip().upper())
            cal = stock.calendar
            if cal is None:
                return []

            # yfinance calendar can be a dict or DataFrame
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        earnings_date = ed[0]
                    elif hasattr(ed, "date"):
                        earnings_date = ed
            elif hasattr(cal, "loc"):
                try:
                    ed = cal.loc["Earnings Date"]
                    if hasattr(ed, "iloc"):
                        earnings_date = ed.iloc[0]
                    elif hasattr(ed, "date"):
                        earnings_date = ed
                except (KeyError, IndexError):
                    pass

            if earnings_date is None:
                return []

            # Convert to date
            if hasattr(earnings_date, "date"):
                if callable(earnings_date.date):
                    earnings_dt = earnings_date.date()
                else:
                    earnings_dt = earnings_date.date
            elif isinstance(earnings_date, str):
                earnings_dt = datetime.strptime(earnings_date[:10], "%Y-%m-%d").date()
            else:
                return []

            today = date.today()
            days_to_earnings = (earnings_dt - today).days

            flags = []
            if 0 <= days_to_earnings <= dte:
                if days_to_earnings <= 3:
                    flags.append(
                        f"🔴 EARNINGS in {days_to_earnings} day(s) ({earnings_dt}) — "
                        f"IV crush risk BEFORE expiration. Sellers beware of gap risk."
                    )
                else:
                    flags.append(
                        f"⚠️ Earnings on {earnings_dt} ({days_to_earnings}d out) — "
                        f"falls within this option's life. IV will spike then crush."
                    )
            elif 0 < days_to_earnings <= 7:
                flags.append(
                    f"ℹ️ Earnings on {earnings_dt} ({days_to_earnings}d out) — "
                    f"near-term but after expiration. Watch for pre-earnings IV ramp."
                )

            return flags
        except Exception as e:
            logger.debug("Earnings fetch failed for %s: %s", ticker, e)
            return []

    return await asyncio.to_thread(_fetch_earnings)

"""
Options Analysis Module — VoPR™ Engine for momentum-chat.

Provides institutional-grade options setup analysis:
  - Composite realized volatility (4-model: CC, Parkinson, Garman-Klass, Rogers-Satchell)
  - Volatility Risk Premium (VRP) ratio — when IV > RV, premium sellers have edge
  - Black-Scholes Greeks (Delta, Theta) for specific strikes
  - Budget-aware strike recommendations — "I have $X for puts"
  - A-F grading for setup quality

Usage: Sam calls analyze_options_setup(ticker, budget, option_type, dte)
and gets back graded strikes the user can actually afford to trade.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from mcp_server.data import get_historical_data, get_live_price, get_option_expirations
from mcp_server.risk import check_risk_flags
from mcp_server.schema import SignalResult
from mcp_server.options_greeks import (
    bs_put_price, bs_call_price, bs_delta, bs_theta,
    composite_rv
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logic & Grading
# ---------------------------------------------------------------------------

def _vol_regime(df: pd.DataFrame, period: int = 30) -> str:
    """Classify current volatility regime vs. longer-term baseline."""
    if len(df) < period * 2:
        return "UNKNOWN"
    recent = composite_rv(df, period)
    longer = composite_rv(df, period * 2)
    if recent == 0 or longer == 0:
        return "UNKNOWN"
    ratio = recent / longer
    if ratio > 1.15:
        return "RISING"
    elif ratio < 0.85:
        return "FALLING"
    return "HIGH" if recent > 0.30 else "LOW"


def _vrp(iv: float, rv: float) -> float:
    """VRP ratio — IV / RV_composite. >1.2 = rich IV = seller's edge."""
    if rv <= 0 or not math.isfinite(rv):
        return 0.0
    return round(iv / rv, 3)


def _vopr_grade(vrp_val: float, regime: str, delta: float | None) -> str:
    """A-F grade for a setup."""
    score = 0
    # VRP: 0-40 pts
    if vrp_val >= 1.5:
        score += 40
    elif vrp_val >= 1.2:
        score += 30
    elif vrp_val >= 1.0:
        score += 15
    # Regime: 0-30 pts
    score += {"LOW": 30, "FALLING": 25, "UNKNOWN": 15, "RISING": 5, "HIGH": 0}.get(regime, 10)
    # Delta: 0-30 pts
    if delta is not None:
        abs_d = abs(delta)
        if abs_d <= 0.10:
            score += 30
        elif abs_d <= 0.15:
            score += 25
        elif abs_d <= 0.20:
            score += 20
        elif abs_d <= 0.30:
            score += 10
    else:
        score += 15
    if score >= 80:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    return "F"


# ---------------------------------------------------------------------------
# Expiration date parser — handles multiple formats
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",    # 2026-04-17
    "%m/%d/%Y",    # 4/17/2026
    "%m/%d/%y",    # 4/17/26
    "%m-%d-%Y",    # 4-17-2026
    "%m-%d-%y",    # 4-17-26
    "%B %d, %Y",   # April 17, 2026
    "%b %d, %Y",   # Apr 17, 2026
    "%d %b %Y",    # 17 Apr 2026
]

def _parse_expiration(exp_str: str) -> date | None:
    """Parse a flexible expiration date string into a date object."""
    exp_str = exp_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(exp_str, fmt)
            # If 2-digit year parsed to < 2000, adjust (e.g. 26 → 2026)
            if dt.year < 100:
                dt = dt.replace(year=dt.year + 2000)
            return dt.date()
        except ValueError:
            continue
    logger.warning("Could not parse expiration date: '%s'", exp_str)
    return None


# ---------------------------------------------------------------------------
# OHLCV → DataFrame helper
# ---------------------------------------------------------------------------

def _to_ohlcv_df(records: list[dict]) -> pd.DataFrame:
    """Convert get_historical_data() list to mplfinance-compatible DataFrame."""
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"}, inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main Tool: analyze_options_setup
# ---------------------------------------------------------------------------

async def analyze_options_setup(
    ticker: str,
    option_type: str = "put",
    dte: int = 30,
    expiration: str | None = None,
    strike: float | None = None,
    budget: float | None = None,
    contracts: int = 1,
    risk_free_rate: float = 0.05,
    iv_override: float | None = None,
) -> SignalResult:
    """Analyze option setup quality using the VoPR™ engine."""
    try:
        ticker = ticker.strip().upper()
        is_put = option_type.lower() == "put"

        # --- Compute DTE from expiration date if provided ---
        expiration_date_str = None
        if expiration:
            exp_date = _parse_expiration(expiration)
            if exp_date:
                dte = max((exp_date - date.today()).days, 1)
                expiration_date_str = exp_date.isoformat()
            else:
                logger.warning("Could not parse expiration '%s', using dte=%d", expiration, dte)

        # Fetch 3 months of OHLCV for vol calculations
        records = await get_historical_data(ticker, period="3mo", interval="1d")
        if len(records) < 20:
            return SignalResult.error_msg(f"Not enough data for {ticker}: got {len(records)} bars")

        df = await asyncio.to_thread(_to_ohlcv_df, records)

        # Use live price for current S
        try:
            S = await get_live_price(ticker)
        except Exception:
            S = float(df["Close"].iloc[-1])

        # --------------- Volatility Calculations ---------------
        rv = composite_rv(df)
        regime = _vol_regime(df)

        # IV estimate
        if iv_override:
            iv = iv_override
        else:
            if regime in ("LOW", "FALLING"):
                iv = rv * 1.25
            elif regime in ("HIGH", "RISING"):
                iv = rv * 1.40
            else:
                iv = rv * 1.30

        vrp_val = _vrp(iv, rv)
        T = max(dte / 365.0, 1 / 365.0)

        # --------------- Strike Candidates ---------------
        if is_put:
            otm_pcts = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
            strikes = [round(S * (1 - pct), 2) for pct in otm_pcts]
        else:
            otm_pcts = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
            strikes = [round(S * (1 + pct), 2) for pct in otm_pcts]

        strikes = [round(k) for k in strikes]
        if strike is not None:
            strikes.append(round(float(strike)))
        strikes = sorted(set(strikes), reverse=is_put)

        candidate_results = []
        for K in strikes:
            if is_put:
                prem = bs_put_price(S, K, T, risk_free_rate, iv)
            else:
                prem = bs_call_price(S, K, T, risk_free_rate, iv)

            delta = bs_delta(S, K, T, risk_free_rate, iv, is_put=is_put)
            theta = bs_theta(S, K, T, risk_free_rate, iv, is_put=is_put)
            grade = _vopr_grade(vrp_val, regime, delta)

            if prem is None or prem <= 0:
                continue

            cost_per_contract = round(prem * 100, 2)
            total_cost = round(cost_per_contract * contracts, 2)
            affordable = (budget is None) or (total_cost <= budget)
            pct_otm = round(abs(S - K) / S * 100, 1)

            roc_basis = K if is_put else S
            roc_pct = round((prem / roc_basis) * 100, 2) if roc_basis > 0 else 0.0

            candidate_results.append({
                "strike": K,
                "pct_otm": pct_otm,
                "premium_est": round(prem, 4),
                "cost_per_contract": cost_per_contract,
                "total_cost": total_cost if budget else None,
                "delta": round(delta, 3) if delta is not None else None,
                "theta_daily": round(theta, 4) if theta is not None else None,
                "roc_pct": roc_pct,
                "roc_period_days": dte,
                "grade": grade,
                "affordable": affordable if budget else None,
            })

        grade_order = {"A": 0, "B": 1, "C": 2, "F": 3}
        candidate_results.sort(key=lambda x: (grade_order.get(x["grade"], 4), -x["pct_otm"]))

        affordable_candidates = [c for c in candidate_results if c.get("affordable") is not False]
        best = affordable_candidates[0] if affordable_candidates else candidate_results[0] if candidate_results else None

        # --------------- Plain-English Summary ---------------
        regime_labels = {
            "LOW": "🟢 calm/compressing (ideal for premium sellers)",
            "FALLING": "🟢 falling (great entry window)",
            "RISING": "🔴 rising (size down or wait)",
            "HIGH": "🔴 elevated (be cautious)",
            "UNKNOWN": "⚪ indeterminate (limited history)",
        }

        vrp_label = ""
        if vrp_val >= 1.2:
            vrp_label = "✅ IV is RICH vs. realized vol — premium sellers have an edge"
        elif vrp_val >= 1.0:
            vrp_label = "⚠️ IV is fair — marginal edge for sellers"
        else:
            vrp_label = "❌ IV is CHEAP vs. realized vol — avoid selling premium here"

        summary_parts = [
            f"{ticker} @ ${S:.2f} | {option_type.upper()} | {dte} DTE",
            f"Realized Vol (composite): {rv*100:.1f}% | IV Est: {iv*100:.1f}% | VRP: {vrp_val:.2f}",
            vrp_label,
            f"Vol Regime: {regime_labels.get(regime, regime)}",
        ]
        if best:
            b = best
            summary_parts.append(
                f"Best Strike: ${b['strike']} ({b['pct_otm']}% OTM) — "
                f"Premium ~${b['premium_est']:.2f}/share (${b['cost_per_contract']:.0f}/contract) — "
                f"Delta {b['delta']} — Daily Theta {b['theta_daily']} — "
                f"RoC {b['roc_pct']:.1f}% ({dte}d) — Grade: {b['grade']}"
            )
            if budget:
                if b["affordable"]:
                    summary_parts.append(f"✅ Fits your ${budget:.0f} budget")
                else:
                    summary_parts.append(f"⚠️ Cheapest available strike is ${candidate_results[-1]['total_cost']} — over your ${budget:.0f} budget")

        best_cost = best["total_cost"] if best else None
        risk_flags = await check_risk_flags(
            ticker=ticker, dte=dte, budget=budget, position_cost=best_cost,
        )

        data = {
            "ticker": ticker,
            "price": S,
            "option_type": option_type.lower(),
            "dte": dte,
            "realized_vol_pct": round(rv * 100, 2),
            "implied_vol_est_pct": round(iv * 100, 2),
            "vol_regime": regime,
            "vrp_ratio": vrp_val,
            "grade_context": "sell",
            "candidate_strikes": candidate_results,
            "best_strike": best,
            "setup_summary": "\n".join(summary_parts),
            "budget": budget,
            "contracts": contracts,
            "expiration": expiration_date_str,
            "risk_flags": risk_flags,
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error("Options analysis failed for %s: %s", ticker, e)
        return SignalResult.error_msg(str(e))


# ---------------------------------------------------------------------------
# Composite Scoring Helpers
# ---------------------------------------------------------------------------

def _sell_score(
    annualized_roc: float,
    grade: str,
    theta_daily: float | None,
    premium: float,
    delta: float | None,
) -> float:
    """Composite score (0-100) for a sell-side candidate."""
    score = 0.0
    if annualized_roc > 0:
        score += min(30, (annualized_roc / 80.0) * 30)
    grade_pts = {"A": 25, "B": 18, "C": 10, "F": 0}
    score += grade_pts.get(grade, 0)
    if theta_daily is not None and premium > 0:
        eff = abs(theta_daily) / premium
        score += min(20, eff * 200)
    if delta is not None:
        abs_d = abs(delta)
        if 0.15 <= abs_d <= 0.25:
            score += 15
        elif 0.10 <= abs_d <= 0.30:
            score += 10
        elif abs_d <= 0.10:
            score += 5
        else:
            score += 3
    if premium >= 1.00:
        score += 10
    elif premium >= 0.50:
        score += 7
    elif premium >= 0.20:
        score += 4
    elif premium >= 0.10:
        score += 2
    return round(score, 1)


def _buy_score(
    delta: float | None,
    dte: int,
    vrp: float,
    direction_strength: float,
) -> float:
    """Composite score (0-100) for a buy-side candidate."""
    score = 0.0
    score += direction_strength * 30
    if delta is not None:
        abs_d = abs(delta)
        if 0.30 <= abs_d <= 0.60:
            score += 25
        elif 0.20 <= abs_d <= 0.70:
            score += 15
        elif abs_d > 0.70:
            score += 10
        else:
            score += 5
    if 30 <= dte <= 60:
        score += 20
    elif 21 <= dte < 30:
        score += 14
    elif 60 < dte <= 90:
        score += 12
    elif dte < 21:
        score += 5
    if vrp < 0.9:
        score += 15
    elif vrp < 1.0:
        score += 12
    elif vrp < 1.2:
        score += 6
    else:
        score += 2
    score += min(10, direction_strength * 12)
    return round(score, 1)


# ---------------------------------------------------------------------------
# Tool: find_best_to_sell
# ---------------------------------------------------------------------------

async def find_best_to_sell(
    ticker: str,
    budget: float | None = None,
    risk_free_rate: float = 0.05,
    iv_override: float | None = None,
) -> SignalResult:
    """Find the optimal put and call contracts to SELL across 7-45 DTE."""
    try:
        ticker = ticker.strip().upper()

        records = await get_historical_data(ticker, period="3mo", interval="1d")
        if len(records) < 20:
            return SignalResult.error_msg(f"Not enough data for {ticker}: got {len(records)} bars")

        df = await asyncio.to_thread(_to_ohlcv_df, records)

        try:
            S = await get_live_price(ticker)
        except Exception:
            S = float(df["Close"].iloc[-1])

        rv = composite_rv(df)
        regime = _vol_regime(df)

        if iv_override:
            iv = iv_override
        else:
            iv = rv * (1.40 if regime in ("HIGH", "RISING") else 1.25 if regime in ("LOW", "FALLING") else 1.30)

        vrp_val = _vrp(iv, rv)

        try:
            expirations = await get_option_expirations(ticker, min_dte=5, max_dte=50)
            dte_list = [e["dte"] for e in expirations] if expirations else [7, 14, 21, 30, 45]
        except Exception:
            dte_list = [7, 14, 21, 30, 45]
        otm_pcts = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]

        all_puts: list[dict] = []
        all_calls: list[dict] = []

        for dte in dte_list:
            T = max(dte / 365.0, 1 / 365.0)

            put_strikes = sorted(set(round(S * (1 - p)) for p in otm_pcts), reverse=True)
            for K in put_strikes:
                prem = bs_put_price(S, K, T, risk_free_rate, iv)
                if prem is None or prem < 0.10:
                    continue
                delta = bs_delta(S, K, T, risk_free_rate, iv, is_put=True)
                theta = bs_theta(S, K, T, risk_free_rate, iv, is_put=True)
                grade = _vopr_grade(vrp_val, regime, delta)
                pct_otm = round(abs(S - K) / S * 100, 1)

                roc_pct = round((prem / K) * 100, 2) if K > 0 else 0.0
                annualized_roc = round(roc_pct * (365 / max(dte, 1)), 1)

                cost = round(K * 100, 2)
                affordable = budget is None or cost <= budget

                composite = _sell_score(annualized_roc, grade, theta, prem, delta)

                all_puts.append({
                    "strike": K, "dte": dte, "pct_otm": pct_otm,
                    "premium_est": round(prem, 4),
                    "cost_per_contract": round(prem * 100, 2),
                    "cash_secured_req": cost,
                    "delta": round(delta, 3) if delta else None,
                    "theta_daily": round(theta, 4) if theta else None,
                    "roc_pct": roc_pct,
                    "annualized_roc_pct": annualized_roc,
                    "grade": grade,
                    "composite_score": composite,
                    "affordable": affordable,
                })

            call_strikes = sorted(set(round(S * (1 + p)) for p in otm_pcts))
            for K in call_strikes:
                prem = bs_call_price(S, K, T, risk_free_rate, iv)
                if prem is None or prem < 0.10:
                    continue
                delta = bs_delta(S, K, T, risk_free_rate, iv, is_put=False)
                theta = bs_theta(S, K, T, risk_free_rate, iv, is_put=False)
                grade = _vopr_grade(vrp_val, regime, delta)
                pct_otm = round(abs(S - K) / S * 100, 1)

                roc_pct = round((prem / S) * 100, 2) if S > 0 else 0.0
                annualized_roc = round(roc_pct * (365 / max(dte, 1)), 1)

                cost_per = round(prem * 100, 2)
                affordable = budget is None or cost_per <= budget

                composite = _sell_score(annualized_roc, grade, theta, prem, delta)

                all_calls.append({
                    "strike": K, "dte": dte, "pct_otm": pct_otm,
                    "premium_est": round(prem, 4),
                    "cost_per_contract": cost_per,
                    "delta": round(delta, 3) if delta else None,
                    "theta_daily": round(theta, 4) if theta else None,
                    "roc_pct": roc_pct,
                    "annualized_roc_pct": annualized_roc,
                    "grade": grade,
                    "composite_score": composite,
                    "affordable": affordable,
                })

        if budget:
            affordable_puts = [p for p in all_puts if p["affordable"]]
            affordable_calls = [c for c in all_calls if c["affordable"]]
        else:
            affordable_puts = all_puts
            affordable_calls = all_calls

        affordable_puts.sort(key=lambda x: x["composite_score"], reverse=True)
        affordable_calls.sort(key=lambda x: x["composite_score"], reverse=True)

        top_puts = affordable_puts[:3]
        top_calls = affordable_calls[:3]

        regime_labels = {
            "LOW": "calm/compressing", "FALLING": "falling",
            "RISING": "rising", "HIGH": "elevated", "UNKNOWN": "indeterminate",
        }
        vrp_label = "RICH ✅" if vrp_val >= 1.2 else "FAIR ⚠️" if vrp_val >= 1.0 else "CHEAP ❌"

        summary_lines = [
            f"🔍 SELL SCANNER — {ticker} @ ${S:.2f}",
            f"RV: {rv*100:.1f}% | IV Est: {iv*100:.1f}% | VRP: {vrp_val:.2f} ({vrp_label})",
            f"Vol Regime: {regime_labels.get(regime, regime)}",
            f"Scanned {len(all_puts)} put combos + {len(all_calls)} call combos across {len(dte_list)} DTEs",
            "",
        ]

        if top_puts:
            bp = top_puts[0]
            summary_lines.append(
                f"🏆 Best PUT to sell: ${bp['strike']} ({bp['pct_otm']}% OTM, {bp['dte']} DTE) — "
                f"~${bp['premium_est']:.2f}/share — Delta {bp['delta']} — "
                f"RoC {bp['roc_pct']}% ({bp['annualized_roc_pct']}% ann.) — "
                f"Grade {bp['grade']} — Score {bp['composite_score']}"
            )

        if top_calls:
            bc = top_calls[0]
            summary_lines.append(
                f"🏆 Best CALL to sell: ${bc['strike']} ({bc['pct_otm']}% OTM, {bc['dte']} DTE) — "
                f"~${bc['premium_est']:.2f}/share — Delta {bc['delta']} — "
                f"RoC {bc['roc_pct']}% ({bc['annualized_roc_pct']}% ann.) — "
                f"Grade {bc['grade']} — Score {bc['composite_score']}"
            )

        min_dte = 30
        for picks in (top_puts, top_calls):
            for p in picks:
                if p.get("dte") and p["dte"] < min_dte:
                    min_dte = p["dte"]
        risk_flags = await check_risk_flags(
            ticker=ticker, dte=min_dte, budget=budget,
        )

        data = {
            "ticker": ticker,
            "price": S,
            "realized_vol_pct": round(rv * 100, 2),
            "implied_vol_est_pct": round(iv * 100, 2),
            "vol_regime": regime,
            "vrp_ratio": vrp_val,
            "top_puts": top_puts,
            "top_calls": top_calls,
            "total_candidates_scanned": len(all_puts) + len(all_calls),
            "setup_summary": "\n".join(summary_lines),
            "budget": budget,
            "risk_flags": risk_flags,
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error("Find best to sell failed for %s: %s", ticker, e)
        return SignalResult.error_msg(str(e))


# ---------------------------------------------------------------------------
# Tool: find_best_to_buy
# ---------------------------------------------------------------------------

async def find_best_to_buy(
    ticker: str,
    budget: float | None = None,
    option_type: str | None = None,
    risk_free_rate: float = 0.05,
    iv_override: float | None = None,
) -> SignalResult:
    """Find the optimal directional contract to BUY based on technicals."""
    try:
        from mcp_server.technicals import analyze_technicals

        ticker = ticker.strip().upper()

        tech_res = await analyze_technicals(ticker)
        if tech_res.status == "error":
            logger.warning("Technicals failed for %s: %s", ticker, tech_res.error)
            tech = {}
        else:
            tech = tech_res.data

        rsi = tech.get("rsi_14", 50)
        adx = tech.get("adx_14", 0)
        macd_hist = tech.get("macd_histogram", 0)
        ema_8 = tech.get("ema_8")
        ema_21 = tech.get("ema_21")
        ema_34 = tech.get("ema_34")
        ema_55 = tech.get("ema_55")

        bias = 0.0
        reasons = []

        if rsi < 30:
            bias += 0.25
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 40:
            bias += 0.10
            reasons.append(f"RSI leaning oversold ({rsi:.0f})")
        elif rsi > 70:
            bias -= 0.25
            reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            bias -= 0.10
            reasons.append(f"RSI leaning overbought ({rsi:.0f})")

        if ema_8 and ema_21 and ema_34 and ema_55:
            if ema_8 > ema_21 > ema_34 > ema_55:
                bias += 0.35
                reasons.append("EMA stack bullish (8>21>34>55)")
            elif ema_8 < ema_21 < ema_34 < ema_55:
                bias -= 0.35
                reasons.append("EMA stack bearish (8<21<34<55)")
            elif ema_8 > ema_21:
                bias += 0.15
                reasons.append("Short EMAs bullish")
            elif ema_8 < ema_21:
                bias -= 0.15
                reasons.append("Short EMAs bearish")

        if macd_hist > 0:
            bias += 0.20
            reasons.append("MACD histogram positive")
        elif macd_hist < 0:
            bias -= 0.20
            reasons.append("MACD histogram negative")

        if adx > 25:
            bias *= 1.3
            reasons.append(f"ADX trending ({adx:.0f}) — amplified")

        bias = max(-1.0, min(1.0, bias))
        direction_strength = abs(bias)
        is_bullish = bias > 0

        user_override = option_type and option_type.lower() in ("call", "put")
        if user_override:
            forced_type = option_type.lower()
            if forced_type == "call":
                direction_label = "BULLISH (user override)"
                is_put = False
            else:
                direction_label = "BEARISH (user override)"
                is_put = True
            reasons.append(f"Direction overridden by user: {forced_type}")
        else:
            if direction_strength < 0.15:
                direction_label = "NEUTRAL"
                is_put = bias < 0
            elif is_bullish:
                direction_label = "BULLISH"
                is_put = False
            else:
                direction_label = "BEARISH"
                is_put = True

        records = await get_historical_data(ticker, period="3mo", interval="1d")
        if len(records) < 20:
            return SignalResult.error_msg(f"Not enough data for {ticker}: got {len(records)} bars")

        df = await asyncio.to_thread(_to_ohlcv_df, records)

        try:
            S = await get_live_price(ticker)
        except Exception:
            S = float(df["Close"].iloc[-1])

        rv = composite_rv(df)
        regime = _vol_regime(df)

        if iv_override:
            iv = iv_override
        else:
            iv = rv * (1.40 if regime in ("HIGH", "RISING") else 1.25 if regime in ("LOW", "FALLING") else 1.30)

        vrp_val = _vrp(iv, rv)

        try:
            expirations = await get_option_expirations(ticker, min_dte=14, max_dte=75)
            dte_list = [e["dte"] for e in expirations] if expirations else [21, 30, 45, 60]
        except Exception:
            dte_list = [21, 30, 45, 60]
        moneyness_pcts = [-0.05, -0.02, 0, 0.02, 0.05, 0.08]

        candidates: list[dict] = []

        for dte in dte_list:
            T = max(dte / 365.0, 1 / 365.0)

            for m_pct in moneyness_pcts:
                if is_put:
                    K = round(S * (1 + m_pct))
                else:
                    K = round(S * (1 - m_pct))

                if K <= 0:
                    continue

                if is_put:
                    prem = bs_put_price(S, K, T, risk_free_rate, iv)
                else:
                    prem = bs_call_price(S, K, T, risk_free_rate, iv)

                if prem is None or prem < 0.05:
                    continue

                delta = bs_delta(S, K, T, risk_free_rate, iv, is_put=is_put)
                theta = bs_theta(S, K, T, risk_free_rate, iv, is_put=is_put)

                cost_per_contract = round(prem * 100, 2)
                affordable = budget is None or cost_per_contract <= budget

                pct_otm = round((S - K) / S * 100, 1) if is_put else round((K - S) / S * 100, 1)
                moneyness = "ITM" if pct_otm < 0 else "ATM" if pct_otm == 0 else "OTM"

                composite = _buy_score(delta, dte, vrp_val, direction_strength)

                candidates.append({
                    "strike": K, "dte": dte, "option_type": "put" if is_put else "call",
                    "pct_from_price": round(abs(S - K) / S * 100, 1),
                    "moneyness": moneyness,
                    "premium_est": round(prem, 4),
                    "cost_per_contract": cost_per_contract,
                    "delta": round(delta, 3) if delta else None,
                    "theta_daily": round(theta, 4) if theta else None,
                    "composite_score": composite,
                    "affordable": affordable,
                })

        if budget:
            candidates = [c for c in candidates if c["affordable"]]
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        top_3 = candidates[:3]

        confidence_label = (
            "HIGH" if direction_strength > 0.6 else
            "MODERATE" if direction_strength > 0.3 else
            "LOW"
        )

        final_type = "PUT" if is_put else "CALL"
        summary_lines = [
            f"🎯 BUY SCANNER — {ticker} @ ${S:.2f}",
            f"Direction: {direction_label} ({confidence_label} confidence, bias={bias:+.2f})",
            f"Signals: {' | '.join(reasons) if reasons else 'Mixed/indeterminate'}",
            f"RV: {rv*100:.1f}% | IV Est: {iv*100:.1f}% | VRP: {vrp_val:.2f}",
            f"Recommendation: BUY {final_type}S",
            "",
        ]

        if top_3:
            t = top_3[0]
            summary_lines.append(
                f"🏆 Best {final_type} to buy: ${t['strike']} ({t['moneyness']}, {t['dte']} DTE) — "
                f"~${t['premium_est']:.2f}/share (${t['cost_per_contract']}/contract) — "
                f"Delta {t['delta']} — Score {t['composite_score']}"
            )

        best_dte = top_3[0]["dte"] if top_3 else 30
        best_cost = top_3[0].get("cost_per_contract") if top_3 else None
        risk_flags = await check_risk_flags(
            ticker=ticker, dte=best_dte, budget=budget, position_cost=best_cost,
        )

        data = {
            "ticker": ticker,
            "price": S,
            "direction": direction_label,
            "direction_bias": round(bias, 3),
            "confidence": confidence_label,
            "direction_strength": round(direction_strength, 3),
            "direction_reasons": reasons,
            "option_type": final_type.lower(),
            "realized_vol_pct": round(rv * 100, 2),
            "implied_vol_est_pct": round(iv * 100, 2),
            "vol_regime": regime,
            "vrp_ratio": vrp_val,
            "top_contracts": top_3,
            "total_candidates_scanned": len(candidates),
            "setup_summary": "\n".join(summary_lines),
            "budget": budget,
            "risk_flags": risk_flags,
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error("Find best to buy failed for %s: %s", ticker, e)
        return SignalResult.error_msg(str(e))


# ---------------------------------------------------------------------------
# Tool: sweep_setups — "What should I trade today?"
# ---------------------------------------------------------------------------

async def sweep_setups(
    tickers: list[str] | None = None,
    budget: float | None = None,
    max_tickers: int = 10,
) -> SignalResult:
    """Scan multiple tickers and rank the best options opportunities."""
    try:
        from mcp_server.screener import run_stock_screen

        max_tickers = min(max_tickers, 10)

        if not tickers:
            try:
                screen_result = await run_stock_screen(preset="most_active", limit=max_tickers)
                tickers = [r.get("ticker", "") for r in screen_result.get("results", []) if r.get("ticker")]
                tickers = tickers[:max_tickers]
            except Exception:
                tickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
        else:
            tickers = [t.strip().upper() for t in tickers[:max_tickers]]

        if not tickers:
            return SignalResult.error_msg("No tickers to scan")

        async def _scan_ticker(ticker: str) -> dict:
            result = {"ticker": ticker, "sell": None, "buy": None, "error": None}
            try:
                sell_res, buy_res = await asyncio.gather(
                    find_best_to_sell(ticker, budget=budget),
                    find_best_to_buy(ticker, budget=budget),
                    return_exceptions=True,
                )
                if not isinstance(sell_res, Exception) and sell_res.status == "success":
                    result["sell"] = sell_res.data
                if not isinstance(buy_res, Exception) and buy_res.status == "success":
                    result["buy"] = buy_res.data
            except Exception as e:
                result["error"] = str(e)
            return result

        ticker_results = await asyncio.gather(*[_scan_ticker(t) for t in tickers])

        opportunities: list[dict] = []

        for tr in ticker_results:
            ticker = tr["ticker"]
            if tr.get("sell"):
                sell = tr["sell"]
                for p in sell.get("top_puts", [])[:1]:
                    opportunities.append({
                        "ticker": ticker, "action": "SELL", "option_type": "PUT",
                        "strike": p["strike"], "dte": p["dte"],
                        "premium_est": p["premium_est"],
                        "delta": p.get("delta"), "grade": p.get("grade"),
                        "composite_score": p["composite_score"],
                        "rationale": f"VRP {sell.get('vrp_ratio', 0):.2f} | {sell.get('vol_regime', '')} vol | Grade {p.get('grade')}",
                    })
                for c in sell.get("top_calls", [])[:1]:
                    opportunities.append({
                        "ticker": ticker, "action": "SELL", "option_type": "CALL",
                        "strike": c["strike"], "dte": c["dte"],
                        "premium_est": c["premium_est"],
                        "delta": c.get("delta"), "grade": c.get("grade"),
                        "composite_score": c["composite_score"],
                        "rationale": f"VRP {sell.get('vrp_ratio', 0):.2f} | {sell.get('vol_regime', '')} vol | Grade {c.get('grade')}",
                    })
            if tr.get("buy"):
                buy = tr["buy"]
                for t in buy.get("top_contracts", [])[:1]:
                    opportunities.append({
                        "ticker": ticker, "action": "BUY",
                        "option_type": t.get("option_type", "call").upper(),
                        "strike": t["strike"], "dte": t["dte"],
                        "premium_est": t["premium_est"],
                        "delta": t.get("delta"),
                        "composite_score": t["composite_score"],
                        "rationale": f"{buy.get('direction', 'NEUTRAL')} ({buy.get('confidence', 'LOW')})",
                    })

        opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
        top_10 = opportunities[:10]

        summary_lines = [
            f"📋 OPPORTUNITY BOARD — {len(tickers)} tickers scanned",
            f"Found {len(opportunities)} setups, showing top {len(top_10)}:",
            "",
        ]
        for i, opp in enumerate(top_10, 1):
            action_emoji = "💰" if opp["action"] == "SELL" else "🎯"
            summary_lines.append(
                f"#{i}  {action_emoji} {opp['ticker']}  "
                f"${opp['strike']}{opp['option_type'][0]} {opp['action']}  "
                f"{opp['dte']}DTE  Score {opp['composite_score']}  "
                f"— {opp.get('rationale', '')}"
            )

        data = {
            "tickers_scanned": tickers,
            "total_opportunities": len(opportunities),
            "opportunities": top_10,
            "setup_summary": "\n".join(summary_lines),
            "budget": budget,
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error("Sweep setups failed: %s", e)
        return SignalResult.error_msg(str(e))


# ---------------------------------------------------------------------------
# Tool: find_best_straddle
# ---------------------------------------------------------------------------

async def find_best_straddle(
    ticker: str,
    budget: float,
    contracts: int = 1,
    risk_free_rate: float = 0.05,
    iv_override: float | None = None,
) -> SignalResult:
    """Find the optimal options straddle (long call + long put) within a budget."""
    try:
        ticker = ticker.strip().upper()

        records = await get_historical_data(ticker, period="3mo", interval="1d")
        if len(records) < 20:
            return SignalResult.error_msg(f"Not enough data for {ticker}: got {len(records)} bars")

        df = await asyncio.to_thread(_to_ohlcv_df, records)

        try:
            S = await get_live_price(ticker)
        except Exception:
            S = float(df["Close"].iloc[-1])

        rv = composite_rv(df)
        regime = _vol_regime(df)

        if iv_override:
            iv = iv_override
        else:
            iv = rv * (1.40 if regime in ("HIGH", "RISING") else 1.25 if regime in ("LOW", "FALLING") else 1.30)
            
        vrp_val = _vrp(iv, rv)

        try:
            expirations = await get_option_expirations(ticker, min_dte=14, max_dte=75)
            dte_list = [e["dte"] for e in expirations] if expirations else [14, 21, 30, 45, 60]
        except Exception:
            dte_list = [14, 21, 30, 45, 60]
            
        moneyness_pcts = [0, 0.02, -0.02]

        candidates: list[dict] = []

        for dte in dte_list:
            T = max(dte / 365.0, 1 / 365.0)

            for m_pct in moneyness_pcts:
                call_strike = round(S * (1 + abs(m_pct)))
                put_strike = round(S * (1 - abs(m_pct)))

                if call_strike <= 0 or put_strike <= 0:
                    continue

                call_prem = bs_call_price(S, call_strike, T, risk_free_rate, iv)
                put_prem = bs_put_price(S, put_strike, T, risk_free_rate, iv)

                if call_prem is None or put_prem is None or call_prem < 0.05 or put_prem < 0.05:
                    continue

                total_premium = call_prem + put_prem
                cost_per_straddle = round(total_premium * 100, 2)
                total_trade_cost = cost_per_straddle * contracts
                
                affordable = budget is None or total_trade_cost <= budget
                setup_type = "Straddle" if m_pct == 0 else "Strangle"

                score_vrp = 1.0 / (vrp_val + 0.1)
                score_dte = dte / 60.0
                composite = round((score_vrp * 0.7) + (score_dte * 0.3), 3)

                candidates.append({
                    "setup_type": setup_type,
                    "dte": dte,
                    "call_strike": call_strike,
                    "put_strike": put_strike,
                    "total_premium": round(total_premium, 4),
                    "cost_per_straddle": cost_per_straddle,
                    "total_trade_cost": total_trade_cost,
                    "composite_score": composite,
                    "affordable": affordable,
                })

        if budget:
            candidates = [c for c in candidates if c["affordable"]]
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        top_3 = candidates[:3]

        summary_lines = [
            f"🎯 STRADDLE/STRANGLE SCANNER — {ticker} @ ${S:.2f}",
            f"Budget: ${budget} for {contracts} contract(s)",
            f"RV: {rv*100:.1f}% | IV Est: {iv*100:.1f}% | VRP Ratio: {vrp_val:.2f}",
            f"{'⚠️ Warning: Options are theoretically overpriced' if vrp_val > 1.3 else '✅ Favorable volatility pricing'}",
            "",
        ]

        if not top_3:
            summary_lines.append(f"❌ No straddles found safely under ${budget} budget limit.")
        else:
            t = top_3[0]
            summary_lines.append(
                f"🏆 Best {t['setup_type']}: {t['dte']} DTE (Call ${t['call_strike']} / Put ${t['put_strike']}) — "
                f"~${t['total_premium']:.2f}/share (${t['total_trade_cost']}/total) — Score {t['composite_score']}"
            )

        best_dte = top_3[0]["dte"] if top_3 else 30
        best_cost = top_3[0]["cost_per_straddle"] if top_3 else None
        risk_flags = await check_risk_flags(
            ticker=ticker, dte=best_dte, budget=budget, position_cost=best_cost,
        )

        data = {
            "ticker": ticker,
            "price": S,
            "realized_vol_pct": round(rv * 100, 2),
            "implied_vol_est_pct": round(iv * 100, 2),
            "vol_regime": regime,
            "vrp_ratio": vrp_val,
            "top_candidates": top_3,
            "total_candidates_scanned": len(candidates),
            "setup_summary": "\n".join(summary_lines),
            "budget": budget,
            "risk_flags": risk_flags,
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error("Find best straddle failed: %s", e)
        return SignalResult.error_msg(str(e))


# ---------------------------------------------------------------------------
# Tool: simulate_options_trade
# ---------------------------------------------------------------------------

async def simulate_options_trade(
    ticker: str,
    strike: float,
    option_type: str,
    dte: int,
    target_price: float,
    days_held: int,
    current_price: float | None = None,
    iv_override: float | None = None,
    contracts: int = 1,
) -> SignalResult:
    """Project future options PnL by calculating Black-Scholes premium shift."""
    try:
        ticker = ticker.strip().upper()
        option_type = option_type.lower()
        is_put = option_type == "put"

        if current_price is None:
            try:
                S = await get_live_price(ticker)
            except Exception:
                records = await get_historical_data(ticker, period="1mo", interval="1d")
                S = float(records[-1]["close"]) if records else 100.0
        else:
            S = current_price

        if iv_override is None:
            try:
                records = await get_historical_data(ticker, period="3mo", interval="1d")
                df = await asyncio.to_thread(_to_ohlcv_df, records)
                rv = composite_rv(df)
                iv = rv * 1.30
            except Exception:
                iv = 0.40
        else:
            iv = iv_override

        T_current = max(dte / 365.0, 1 / 365.0)
        r = 0.05

        if is_put:
            current_premium = bs_put_price(S, strike, T_current, r, iv)
        else:
            current_premium = bs_call_price(S, strike, T_current, r, iv)
        
        future_dte = dte - days_held
        T_future = max(future_dte / 365.0, 0.0001)
        
        if future_dte <= 0:
            if is_put:
                future_premium = max(0.0, strike - target_price)
            else:
                future_premium = max(0.0, target_price - strike)
        else:
            if is_put:
                future_premium = bs_put_price(target_price, strike, T_future, r, iv)
            else:
                future_premium = bs_call_price(target_price, strike, T_future, r, iv)

        if current_premium is None or future_premium is None:
            return SignalResult.error_msg("Black-Scholes calculation failed.")

        pnl_per_share = future_premium - current_premium
        pnl_total = pnl_per_share * 100 * contracts
        initial_cost = current_premium * 100 * contracts
        roi_pct = (pnl_per_share / current_premium) * 100 if current_premium > 0 else 0

        data = {
            "ticker": ticker,
            "option": f"${strike} {option_type.upper()}",
            "scenario": f"Buy at {S:.2f}, hold {days_held} days, stock goes to {target_price:.2f}",
            "current_dte": dte,
            "future_dte": max(0, future_dte),
            "iv_used_pct": round(iv * 100, 1),
            "current_premium": round(current_premium, 2),
            "future_premium": round(future_premium, 2),
            "pnl_per_contract": round(pnl_per_share * 100, 2),
            "total_pnl": round(pnl_total, 2),
            "total_cost": round(initial_cost, 2),
            "roi_pct": round(roi_pct, 1),
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error("Simulate options trade failed: %s", e)
        return SignalResult.error_msg(str(e))

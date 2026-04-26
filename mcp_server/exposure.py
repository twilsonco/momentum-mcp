"""
Exposure Coach — Rollup meta-signal for capital deployment.
Synthesizes VIX, flow, distribution days, GEX, and rally status.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any
from mcp_server.schema import SignalResult
from mcp_server.macro import get_macro_context
from mcp_server.market_top import detect_market_top
from mcp_server.ftd_detector import detect_ftd

logger = logging.getLogger(__name__)

async def get_exposure_recommendation() -> SignalResult:
    """Synthesize market signals into a capital deployment recommendation (0-100%)."""
    try:
        # Fetch signals in parallel
        # Note: macro.get_macro_context() handles errors internally
        results = await asyncio.gather(
            get_macro_context(),
            detect_market_top(),
            detect_ftd(),
            return_exceptions=True
        )
        
        # Handle exceptions gracefully
        macro = results[0] if not isinstance(results[0], Exception) else {"vix": 20, "flow_score": 0, "gex_regime": "neutral"}
        top = results[1] if not isinstance(results[1], Exception) else type('obj', (object,), {'status': 'error', 'data': {}})()
        ftd = results[2] if not isinstance(results[2], Exception) else type('obj', (object,), {'status': 'error', 'data': {}})()

        
        score = 0
        details = {}
        
        # 1. VIX Regime (20 pts)
        vix = macro.get("vix", 20)
        vix_score = 0
        if vix < 14: vix_score = 20
        elif vix < 18: vix_score = 16
        elif vix < 22: vix_score = 12
        elif vix < 28: vix_score = 6
        else: vix_score = 0
        score += vix_score
        details["vix"] = {"value": vix, "score": vix_score}
        
        # 2. Flow Score (20 pts)
        flow_val = macro.get("flow_score", 0)
        flow_score = 0
        if flow_val >= 4: flow_score = 20
        elif flow_val >= 1: flow_score = 15
        elif flow_val >= -1: flow_score = 10
        elif flow_val >= -3: flow_score = 5
        else: flow_score = 0
        score += flow_score
        details["flow"] = {"value": flow_val, "score": flow_score}
        
        # 3. Market Top (Distribution Days) (20 pts)
        dist_days_dict = {}
        if top.status == "success":
            dist_days_dict = top.data.get("distribution_days", {})
        
        # distribution_days might be a dict {'SPY': 4, ...} or a single int depending on implementation
        if isinstance(dist_days_dict, dict):
            dist_days = max([int(v) for v in dist_days_dict.values() if v is not None], default=0)
        else:
            try:
                dist_days = int(dist_days_dict) if dist_days_dict is not None else 0
            except (ValueError, TypeError):
                dist_days = 0
        
        top_score = 0
        if dist_days <= 1: top_score = 20
        elif dist_days <= 3: top_score = 15
        elif dist_days <= 5: top_score = 8
        else: top_score = 0
        score += top_score
        details["distribution_days"] = {"value": dist_days, "score": top_score}
        
        # 4. GEX Regime (20 pts)
        gex = macro.get("gex_regime", "neutral").lower()
        gex_score = 10
        if "positive" in gex: gex_score = 20
        elif "negative" in gex: gex_score = 0
        score += gex_score
        details["gex"] = {"value": gex, "score": gex_score}
        
        # 5. Market Trend (FTD/Rally) (20 pts)
        ftd_score = 0
        trend_status = "Unknown"
        if ftd.status == "success":
            trend_status = ftd.data.get("summary", "")
            summary_up = trend_status.upper()
            if "CONFIRMED" in summary_up: ftd_score = 20
            elif "RALLY" in summary_up: ftd_score = 12
            elif "CORRECTION" in summary_up: ftd_score = 4
        score += ftd_score
        details["trend"] = {"status": trend_status, "score": ftd_score}
        
        # Exposure Ceiling
        ceiling = "0-10%"
        verdict = "CASH HEAVY"
        if score >= 85: 
            ceiling = "80-100%"
            verdict = "AGGRESSIVE LONG"
        elif score >= 65:
            ceiling = "50-80%"
            verdict = "MODERATE EXPOSURE"
        elif score >= 45:
            ceiling = "30-50%"
            verdict = "REDUCED EXPOSURE"
        elif score >= 25:
            ceiling = "10-30%"
            verdict = "DEFENSIVE"
            
        summary = f"Exposure Recommendation: {verdict} ({ceiling}). Meta-score: {score}/100."
        
        return SignalResult.success({
            "score": score,
            "exposure_ceiling": ceiling,
            "verdict": verdict,
            "details": details,
            "summary": summary
        })

    except Exception as e:
        logger.error(f"Exposure calculation failed: {e}")
        return SignalResult.error_msg(str(e))

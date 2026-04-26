"""
Macro Context — Aggregated Market Regime Summary.

Combines VIX, market pulse (flow score), GEX, and put/call ratios
into a structured regime dict that gets injected as system context.

Zero extra LLM cost — just richer context on every turn.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp_server.cache import smart_cache

logger = logging.getLogger(__name__)

# Cache the full macro context for 5 minutes
_macro_cache: dict[str, Any] | None = None
_macro_cache_ts: float = 0
_MACRO_TTL = 300  # 5 minutes


async def get_macro_context() -> dict[str, Any]:
    """Aggregate multiple data sources into a market regime summary.

    Fetches VIX, market pulse, GEX overview, and put/call ratios
    in parallel and combines into a single structured dict.

    Returns:
        Dict with regime classifications and a human-readable summary.
        Partial data is OK — each source fails gracefully.
    """
    global _macro_cache, _macro_cache_ts

    # Return cached if fresh
    if _macro_cache and (time.time() - _macro_cache_ts) < _MACRO_TTL:
        return _macro_cache

    import asyncio

    # Fire all fetches in parallel
    vix_task = asyncio.create_task(_fetch_vix())
    pulse_task = asyncio.create_task(_fetch_pulse())
    gex_task = asyncio.create_task(_fetch_gex())
    pcr_task = asyncio.create_task(_fetch_pcr())
    alerts_task = asyncio.create_task(_fetch_alerts_summary())

    vix_data, pulse_data, gex_data, pcr_data, alerts_data = await asyncio.gather(
        vix_task, pulse_task, gex_task, pcr_task, alerts_task,
        return_exceptions=True,
    )

    # Handle exceptions from gather
    if isinstance(vix_data, Exception):
        vix_data = {}
    if isinstance(pulse_data, Exception):
        pulse_data = {}
    if isinstance(gex_data, Exception):
        gex_data = {}
    if isinstance(pcr_data, Exception):
        pcr_data = {}
    if isinstance(alerts_data, Exception):
        alerts_data = {}

    # --- Build regime summary ---
    result: dict[str, Any] = {"timestamp": time.time()}

    # VIX
    vix = vix_data.get("vix")
    if vix is not None:
        result["vix"] = round(vix, 2)
        if vix >= 35:
            result["vix_regime"] = "panic"
        elif vix >= 28:
            result["vix_regime"] = "elevated"
        elif vix >= 20:
            result["vix_regime"] = "above_average"
        elif vix >= 14:
            result["vix_regime"] = "normal"
        else:
            result["vix_regime"] = "complacent"

    # Market Pulse (flow score)
    flow_score = pulse_data.get("flow_score")
    if flow_score is not None:
        result["flow_score"] = flow_score
        if flow_score >= 5:
            result["flow_regime"] = "extremely_bullish"
        elif flow_score >= 2:
            result["flow_regime"] = "bullish"
        elif flow_score >= -1:
            result["flow_regime"] = "neutral"
        elif flow_score >= -4:
            result["flow_regime"] = "bearish"
        else:
            result["flow_regime"] = "extremely_bearish"

    pulse_summary = pulse_data.get("summary")
    if pulse_summary:
        result["pulse_summary"] = pulse_summary

    # GEX
    gex_regime = gex_data.get("regime")
    if gex_regime:
        result["gex_regime"] = gex_regime
    gex_flip = gex_data.get("flip_level")
    if gex_flip:
        result["gex_flip_level"] = gex_flip

    # Put/Call Ratio
    pcr = pcr_data.get("ratio")
    if pcr is not None:
        result["pc_ratio"] = round(pcr, 3)
        if pcr < 0.7:
            result["pc_signal"] = "complacent (contrarian bearish)"
        elif pcr <= 1.0:
            result["pc_signal"] = "normal"
        else:
            result["pc_signal"] = "fear (contrarian bullish)"

    # AI Alerts summary
    alert_count = alerts_data.get("total_count")
    if alert_count is not None:
        result["alert_count"] = alert_count
    top_alert = alerts_data.get("top_alert")
    if top_alert:
        result["top_alert"] = top_alert
    high_count = alerts_data.get("high_count", 0)
    critical_count = alerts_data.get("critical_count", 0)
    if high_count or critical_count:
        result["urgent_alerts"] = high_count + critical_count

    # Build human-readable summary
    result["summary"] = _build_summary(result)

    # Cache it
    _macro_cache = result
    _macro_cache_ts = time.time()

    logger.info("Macro context refreshed: %s", result.get("summary", "no summary"))
    return result


def format_macro_for_injection(ctx: dict[str, Any]) -> str:
    """Format macro context as a system message for LLM injection."""
    parts = ["[MACRO CONTEXT — auto-injected, not from user]"]

    vix = ctx.get("vix")
    vix_regime = ctx.get("vix_regime", "unknown")
    if vix:
        parts.append(f"VIX: {vix} ({vix_regime})")

    flow = ctx.get("flow_score")
    flow_regime = ctx.get("flow_regime")
    if flow is not None:
        parts.append(f"Flow Score: {flow:+d}/7 ({flow_regime})")

    gex = ctx.get("gex_regime")
    flip = ctx.get("gex_flip_level")
    if gex:
        gex_str = f"GEX: {gex}"
        if flip:
            gex_str += f" (flip @ {flip})"
        parts.append(gex_str)

    pcr = ctx.get("pc_ratio")
    pc_signal = ctx.get("pc_signal")
    if pcr:
        parts.append(f"P/C Ratio: {pcr:.2f} ({pc_signal})")

    summary = ctx.get("summary")
    if summary:
        parts.append(f"Regime: {summary}")

    urgent = ctx.get("urgent_alerts", 0)
    if urgent:
        parts.append(f"🚨 {urgent} urgent alert(s)")
    top_alert = ctx.get("top_alert")
    if top_alert:
        parts.append(f"Top: {top_alert}")

    return " | ".join(parts)


def _build_summary(ctx: dict[str, Any]) -> str:
    """Build a one-line regime summary from macro data."""
    signals = []

    vix_regime = ctx.get("vix_regime")
    if vix_regime in ("panic", "elevated"):
        signals.append(f"VIX {vix_regime}")

    flow_regime = ctx.get("flow_regime")
    if flow_regime and flow_regime != "neutral":
        signals.append(f"flow {flow_regime}")

    gex_regime = ctx.get("gex_regime")
    if gex_regime and "negative" in str(gex_regime).lower():
        signals.append("negative GEX (trending)")
    elif gex_regime and "positive" in str(gex_regime).lower():
        signals.append("positive GEX (pinning)")

    pc_signal = ctx.get("pc_signal")
    if pc_signal and "complacent" in pc_signal:
        signals.append("low P/C (complacent)")
    elif pc_signal and "fear" in pc_signal:
        signals.append("high P/C (fear)")

    if not signals:
        return "Market conditions nominal"

    # Assess overall bias
    bearish_count = sum(1 for s in signals if any(w in s for w in ["panic", "elevated", "bearish", "negative", "fear"]))
    bullish_count = sum(1 for s in signals if any(w in s for w in ["bullish", "positive", "complacent"]))

    if bearish_count > bullish_count:
        prefix = "⚠️ Risk-off"
    elif bullish_count > bearish_count:
        prefix = "🟢 Risk-on"
    else:
        prefix = "⚡ Mixed"

    return f"{prefix}: {', '.join(signals)}"


# ---------------------------------------------------------------------------
# Individual data fetchers (each gracefully returns {} on failure)
# ---------------------------------------------------------------------------

async def _fetch_vix() -> dict:
    try:
        from mcp_server.data import get_live_price
        vix = await get_live_price("^VIX")
        return {"vix": vix}
    except Exception as e:
        logger.debug("VIX fetch failed: %s", e)
        return {}


async def _fetch_pulse() -> dict:
    try:
        from mcp_server.traderdaddy import get_market_pulse
        pulse = await get_market_pulse()
        if isinstance(pulse, dict):
            return {
                "flow_score": pulse.get("score") or pulse.get("flow_score"),
                "summary": pulse.get("summary") or pulse.get("ai_summary"),
            }
        return {}
    except Exception as e:
        logger.debug("Market pulse fetch failed: %s", e)
        return {}


async def _fetch_gex() -> dict:
    try:
        from mcp_server.traderdaddy import get_gex_overview
        gex = await get_gex_overview()
        if isinstance(gex, dict):
            return {
                "regime": gex.get("regime") or gex.get("gex_regime"),
                "flip_level": gex.get("flip_level") or gex.get("gex_flip"),
            }
        return {}
    except Exception as e:
        logger.debug("GEX fetch failed: %s", e)
        return {}


async def _fetch_pcr() -> dict:
    try:
        from mcp_server.traderdaddy import get_put_call_ratios
        pcr = await get_put_call_ratios()
        if isinstance(pcr, dict):
            spy = pcr.get("SPY") or pcr.get("spy") or pcr
            ratio = spy.get("put_call_ratio") or spy.get("ratio") or spy.get("pc_ratio")
            if ratio:
                return {"ratio": float(ratio)}
        return {}
    except Exception as e:
        logger.debug("P/C ratio fetch failed: %s", e)
        return {}


async def _fetch_alerts_summary() -> dict:
    try:
        from mcp_server.traderdaddy import get_alerts_summary
        data = await get_alerts_summary()
        if isinstance(data, dict):
            # Extract counts and top alert
            result = {}
            total = data.get("total") or data.get("total_count")
            if total is not None:
                result["total_count"] = total
            # Count high/critical
            by_severity = data.get("by_severity") or data.get("severity_counts") or {}
            result["high_count"] = by_severity.get("high", 0)
            result["critical_count"] = by_severity.get("critical", 0)
            # Top alert preview
            top_alerts = data.get("top_alerts") or data.get("alerts") or []
            if top_alerts and isinstance(top_alerts, list):
                top = top_alerts[0]
                msg = top.get("message") or top.get("title") or top.get("description", "")
                result["top_alert"] = msg[:100]  # truncate
            return result
        return {}
    except Exception as e:
        logger.debug("Alerts summary fetch failed: %s", e)
        return {}

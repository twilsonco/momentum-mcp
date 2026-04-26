"""
Alpha Cards — Shareable Trade Setup Cards.

Generates sleek, branded HTML cards for trade setups that users
can screenshot to share on Discord/Twitter/etc.

Design: Full black + Grafana-style gradient fills, gauge bars,
glow accents. Professional fintech terminal aesthetic.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_alpha_card(
    ticker: str,
    technicals: dict[str, Any] | None = None,
    tv_analysis: dict[str, Any] | None = None,
    sam_take: str = "",
) -> str:
    """Generate a shareable HTML Alpha Card for a ticker."""
    # ── Extract technicals data ──
    price = "—"
    rsi_val = "—"
    macd_hist = "—"
    adx_val = "—"
    atr_val = "—"
    cci_val = "—"
    willr_val = "—"
    stoch_k = "—"
    stoch_d = "—"
    bb_upper = "—"
    bb_mid = "—"
    bb_lower = "—"
    ema_stack = {}
    sma_levels = {}

    if technicals:
        price = f"${technicals.get('close', '?')}"
        rsi_val = _fmt(technicals.get("rsi_14"), 1)
        macd_hist = _fmt(technicals.get("macd_histogram"), 4)
        adx_val = _fmt(technicals.get("adx_14"), 1)
        atr_val = _fmt(technicals.get("atr_14"), 2)
        cci_val = _fmt(technicals.get("cci_20"), 1)
        willr_val = _fmt(technicals.get("williams_r_14"), 1)
        stoch_k = _fmt(technicals.get("stoch_k"), 1)
        stoch_d = _fmt(technicals.get("stoch_d"), 1)
        bb_upper = _fmt(technicals.get("bb_upper"), 2)
        bb_mid = _fmt(technicals.get("bb_mid"), 2)
        bb_lower = _fmt(technicals.get("bb_lower"), 2)

        for period in [8, 21, 34, 55, 89]:
            ema_stack[period] = _fmt(technicals.get(f"ema_{period}"), 2)
        for period in [50, 100, 200]:
            sma_levels[period] = _fmt(technicals.get(f"sma_{period}"), 2)

    # ── TradingView consensus ──
    tv_rec = "—"
    tv_ma_rec = "—"
    tv_osc_rec = "—"
    tv_buy = tv_sell = tv_neutral = 0

    if tv_analysis and "recommendation" in tv_analysis:
        tv_rec = tv_analysis["recommendation"]
        tv_summary = tv_analysis.get("summary", {})
        tv_buy = tv_summary.get("buy", 0)
        tv_sell = tv_summary.get("sell", 0)
        tv_neutral = tv_summary.get("neutral", 0)
        tv_ma_rec = tv_analysis.get("moving_averages", {}).get("recommendation", "—")
        tv_osc_rec = tv_analysis.get("oscillators", {}).get("recommendation", "—")

        indicators = tv_analysis.get("indicators", {})
        if price in ("—", "$?", "$None"):
            price = f"${indicators.get('close', '?')}"
        if rsi_val == "—":
            tv_rsi = indicators.get("RSI")
            if tv_rsi is not None:
                rsi_val = f"{round(float(tv_rsi), 1)}"

    rec_color = _rec_color(tv_rec)
    rsi_color = _rsi_color(rsi_val)
    ema_status_text, ema_status_color = _ema_stack_status(technicals) if technicals else ("—", "#555")
    adx_label, adx_color = _adx_label(adx_val)

    # ── Gauge bars (RSI: 0-100, Stoch: 0-100, CCI: -200 to 200) ──
    rsi_pct = _to_pct(rsi_val, 0, 100)
    stoch_pct = _to_pct(stoch_k, 0, 100)

    # ── Consensus bar (proportional) ──
    total_votes = tv_buy + tv_sell + tv_neutral
    buy_pct = round(tv_buy / total_votes * 100) if total_votes else 33
    neutral_pct = round(tv_neutral / total_votes * 100) if total_votes else 34
    sell_pct = 100 - buy_pct - neutral_pct

    # ── EMA stack cells ──
    ema_colors = {8: "#00d4ff", 21: "#22c55e", 34: "#eab308", 55: "#f97316", 89: "#ef4444"}
    ema_cells = ""
    for period in [8, 21, 34, 55, 89]:
        val = ema_stack.get(period, "—")
        color = ema_colors[period]
        ema_cells += f'''<div style="text-align:center;flex:1">
            <div style="font-size:8px;color:{color};font-weight:700;text-transform:uppercase;letter-spacing:1px">{period}</div>
            <div style="font-size:11px;font-family:'Fira Code',monospace;color:#ddd;margin-top:2px">{val}</div>
        </div>'''

    # ── SMA cells ──
    sma_cells = ""
    for period in [50, 100, 200]:
        val = sma_levels.get(period, "—")
        sma_cells += f'''<div style="text-align:center;flex:1">
            <div style="font-size:8px;color:#666;font-weight:700;letter-spacing:1px">SMA {period}</div>
            <div style="font-size:11px;font-family:'Fira Code',monospace;color:#bbb;margin-top:2px">{val}</div>
        </div>'''

    # ── Sam's take ──
    if not sam_take:
        sam_take = "No comment yet — ask Sam for her take."
    sam_take_html = sam_take.replace("\n", "<br>")

    # ── Assemble ──
    html = f"""
<div style="
    background:#000;
    border:1px solid #1a1a1a;
    border-radius:16px;
    padding:28px;
    font-family:'Outfit','Inter',-apple-system,sans-serif;
    color:#e0e0e0;
    max-width:600px;
    box-shadow:0 0 60px rgba(0,0,0,0.8),inset 0 1px 0 rgba(255,255,255,0.03);
    position:relative;
    overflow:hidden;
">
    <!-- Subtle top gradient accent line -->
    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00d4ff,#22c55e,#eab308,#f97316,#ef4444);border-radius:16px 16px 0 0"></div>

    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid #161616">
        <div>
            <span style="font-size:28px;font-weight:800;background:linear-gradient(135deg,#f0b400,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:1px">{ticker.upper()}</span>
            <span style="font-size:16px;color:#888;margin-left:14px;font-family:'Fira Code',monospace">{price}</span>
        </div>
        <div style="
            background:linear-gradient(135deg,{rec_color}18,{rec_color}08);
            border:1px solid {rec_color}66;border-radius:8px;
            padding:5px 14px;font-size:12px;font-weight:700;color:{rec_color};
            letter-spacing:0.8px;text-shadow:0 0 12px {rec_color}44;
        ">{tv_rec.replace('_', ' ')}</div>
    </div>

    <!-- Consensus -->
    <div style="margin-bottom:18px">
        <div style="font-size:9px;color:#555;font-weight:700;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase">26-Indicator Consensus</div>
        <!-- Gradient consensus bar -->
        <div style="display:flex;height:6px;border-radius:3px;overflow:hidden;margin-bottom:10px;background:#111">
            <div style="width:{buy_pct}%;background:linear-gradient(90deg,#00ff41,#22c55e)"></div>
            <div style="width:{neutral_pct}%;background:#333"></div>
            <div style="width:{sell_pct}%;background:linear-gradient(90deg,#ef4444,#e53935)"></div>
        </div>
        <div style="display:flex;gap:6px">
            <div style="flex:1;background:linear-gradient(180deg,#0a1a0a,#050f05);border:1px solid #0a2a0a;border-radius:8px;padding:8px;text-align:center">
                <div style="font-size:22px;font-weight:800;color:#00ff41;text-shadow:0 0 20px #00ff4133">{tv_buy}</div>
                <div style="font-size:8px;color:#555;font-weight:600;letter-spacing:1px;margin-top:2px">BUY</div>
            </div>
            <div style="flex:1;background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;padding:8px;text-align:center">
                <div style="font-size:22px;font-weight:800;color:#666">{tv_neutral}</div>
                <div style="font-size:8px;color:#555;font-weight:600;letter-spacing:1px;margin-top:2px">NEUTRAL</div>
            </div>
            <div style="flex:1;background:linear-gradient(180deg,#1a0a0a,#0f0505);border:1px solid #2a0a0a;border-radius:8px;padding:8px;text-align:center">
                <div style="font-size:22px;font-weight:800;color:#ef4444;text-shadow:0 0 20px #ef444433">{tv_sell}</div>
                <div style="font-size:8px;color:#555;font-weight:600;letter-spacing:1px;margin-top:2px">SELL</div>
            </div>
        </div>
        <div style="display:flex;gap:6px;margin-top:6px">
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:6px;padding:4px 8px;text-align:center">
                <span style="font-size:8px;color:#444">MAs </span>
                <span style="font-size:10px;font-weight:700;color:{_rec_color(tv_ma_rec)}">{tv_ma_rec.replace('_', ' ')}</span>
            </div>
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:6px;padding:4px 8px;text-align:center">
                <span style="font-size:8px;color:#444">Oscillators </span>
                <span style="font-size:10px;font-weight:700;color:{_rec_color(tv_osc_rec)}">{tv_osc_rec.replace('_', ' ')}</span>
            </div>
        </div>
    </div>

    <!-- Core Indicators with gauge fills -->
    <div style="margin-bottom:18px">
        <div style="font-size:9px;color:#555;font-weight:700;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase">Core Indicators</div>
        <div style="display:flex;gap:6px">
            <!-- RSI with gauge -->
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:10px;text-align:center;position:relative;overflow:hidden">
                <div style="position:absolute;bottom:0;left:0;right:0;height:{rsi_pct}%;background:linear-gradient(0deg,{rsi_color}08,{rsi_color}02);transition:height 0.3s"></div>
                <div style="position:relative;z-index:1">
                    <div style="font-size:8px;color:#555;font-weight:700;letter-spacing:1px">RSI</div>
                    <div style="font-size:22px;font-weight:800;color:{rsi_color};text-shadow:0 0 15px {rsi_color}22;margin:4px 0 2px">{rsi_val}</div>
                    <div style="font-size:7px;color:#555;font-weight:600;letter-spacing:0.5px">{_rsi_label(rsi_val)}</div>
                </div>
            </div>
            <!-- MACD -->
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:8px;color:#555;font-weight:700;letter-spacing:1px">MACD</div>
                <div style="font-size:14px;font-weight:700;color:{'#00ff41' if not str(macd_hist).startswith('-') else '#ef4444'};font-family:'Fira Code',monospace;margin:4px 0 2px">{macd_hist}</div>
                <div style="font-size:7px;color:#555;font-weight:600">{'▲ BULL' if not str(macd_hist).startswith('-') else '▼ BEAR'}</div>
            </div>
            <!-- ADX -->
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:8px;color:#555;font-weight:700;letter-spacing:1px">ADX</div>
                <div style="font-size:22px;font-weight:800;color:{adx_color};margin:4px 0 2px">{adx_val}</div>
                <div style="font-size:7px;color:#555;font-weight:600">{adx_label.upper()}</div>
            </div>
            <!-- ATR -->
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:8px;color:#555;font-weight:700;letter-spacing:1px">ATR</div>
                <div style="font-size:18px;font-weight:700;color:#aaa;font-family:'Fira Code',monospace;margin:4px 0 2px">{atr_val}</div>
                <div style="font-size:7px;color:#555;font-weight:600">VOLATILITY</div>
            </div>
        </div>
    </div>

    <!-- Momentum with gauge fills -->
    <div style="margin-bottom:18px">
        <div style="font-size:9px;color:#555;font-weight:700;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase">Momentum</div>
        <div style="display:flex;gap:6px">
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:8px;text-align:center;position:relative;overflow:hidden">
                <div style="position:absolute;bottom:0;left:0;right:0;height:{stoch_pct}%;background:linear-gradient(0deg,#eab30808,#eab30802)"></div>
                <div style="position:relative;z-index:1">
                    <div style="font-size:8px;color:#555;font-weight:700;letter-spacing:1px">STOCH %K/%D</div>
                    <div style="font-size:13px;font-weight:700;color:#ddd;font-family:'Fira Code',monospace;margin-top:4px">{stoch_k}<span style="color:#444"> / </span>{stoch_d}</div>
                </div>
            </div>
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:8px;text-align:center">
                <div style="font-size:8px;color:#555;font-weight:700;letter-spacing:1px">CCI(20)</div>
                <div style="font-size:13px;font-weight:700;color:{_cci_color(cci_val)};font-family:'Fira Code',monospace;margin-top:4px">{cci_val}</div>
            </div>
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:8px;text-align:center">
                <div style="font-size:8px;color:#555;font-weight:700;letter-spacing:1px">WILLIAMS %R</div>
                <div style="font-size:13px;font-weight:700;color:#ddd;font-family:'Fira Code',monospace;margin-top:4px">{willr_val}</div>
            </div>
        </div>
    </div>

    <!-- EMA Stack with gradient spine -->
    <div style="margin-bottom:18px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div style="font-size:9px;color:#555;font-weight:700;letter-spacing:2px;text-transform:uppercase">EMA Stack</div>
            <div style="font-size:9px;font-weight:700;color:{ema_status_color};background:{ema_status_color}11;border:1px solid {ema_status_color}33;border-radius:4px;padding:2px 8px;text-shadow:0 0 8px {ema_status_color}22">{ema_status_text}</div>
        </div>
        <!-- Rainbow gradient spine behind EMAs -->
        <div style="background:#080808;border:1px solid #161616;border-radius:8px;padding:10px;position:relative;overflow:hidden">
            <div style="position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00d4ff,#22c55e,#eab308,#f97316,#ef4444)"></div>
            <div style="display:flex;gap:4px">
                {ema_cells}
            </div>
        </div>
    </div>

    <!-- Structure: SMA + Bollinger -->
    <div style="margin-bottom:18px">
        <div style="font-size:9px;color:#555;font-weight:700;letter-spacing:2px;margin-bottom:8px;text-transform:uppercase">Structure</div>
        <div style="display:flex;gap:6px;margin-bottom:6px">
            <div style="flex:1;background:#080808;border:1px solid #161616;border-radius:8px;padding:8px;display:flex;gap:4px">
                {sma_cells}
            </div>
        </div>
        <div style="display:flex;gap:4px;background:#080808;border:1px solid #161616;border-radius:8px;padding:8px">
            <div style="text-align:center;flex:1">
                <div style="font-size:8px;color:#ef4444;font-weight:700;letter-spacing:1px">BB ↑</div>
                <div style="font-size:11px;font-family:'Fira Code',monospace;color:#bbb;margin-top:2px">{bb_upper}</div>
            </div>
            <div style="text-align:center;flex:1">
                <div style="font-size:8px;color:#666;font-weight:700;letter-spacing:1px">MID</div>
                <div style="font-size:11px;font-family:'Fira Code',monospace;color:#bbb;margin-top:2px">{bb_mid}</div>
            </div>
            <div style="text-align:center;flex:1">
                <div style="font-size:8px;color:#00ff41;font-weight:700;letter-spacing:1px">BB ↓</div>
                <div style="font-size:11px;font-family:'Fira Code',monospace;color:#bbb;margin-top:2px">{bb_lower}</div>
            </div>
        </div>
    </div>

    <!-- Sam's Take -->
    <div style="
        background:linear-gradient(135deg,#0a0a00,#080800);
        border-left:3px solid;
        border-image:linear-gradient(180deg,#f0b400,#ff6b35) 1;
        border-radius:0 10px 10px 0;
        padding:16px 18px;
        margin-bottom:16px;
    ">
        <div style="font-size:9px;font-weight:700;letter-spacing:2px;margin-bottom:8px;background:linear-gradient(90deg,#f0b400,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent">👻 SAM'S TAKE</div>
        <div style="font-size:13px;color:#ccc;line-height:1.7">{sam_take_html}</div>
    </div>

    <!-- Footer -->
    <div style="text-align:center;font-size:8px;color:#333;letter-spacing:2px;font-weight:600">
        ALPHA CARD · TRADERLADY × TRADERDADDY PRO
    </div>
</div>
"""
    return html


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(val: Any, decimals: int = 2) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _to_pct(val: str, low: float, high: float) -> int:
    """Convert a string value to a percentage within a range."""
    try:
        v = float(val)
        return max(0, min(100, int((v - low) / (high - low) * 100)))
    except (ValueError, TypeError):
        return 0


def _rec_color(rec: str) -> str:
    rec_upper = str(rec).upper()
    if "STRONG_BUY" in rec_upper:
        return "#00ff41"
    if "BUY" in rec_upper:
        return "#4caf50"
    if "STRONG_SELL" in rec_upper:
        return "#ef4444"
    if "SELL" in rec_upper:
        return "#f0b400"
    return "#666"


def _rsi_color(val: str) -> str:
    try:
        v = float(val)
        if v > 70:
            return "#ef4444"
        if v < 30:
            return "#00ff41"
        return "#eab308"
    except (ValueError, TypeError):
        return "#666"


def _rsi_label(val: str) -> str:
    try:
        v = float(val)
        if v > 70:
            return "OVERBOUGHT"
        if v < 30:
            return "OVERSOLD"
        if v > 55:
            return "WARM"
        if v < 45:
            return "COOL"
        return "NEUTRAL"
    except (ValueError, TypeError):
        return ""


def _cci_color(val: str) -> str:
    try:
        v = float(val)
        if v > 100:
            return "#ef4444"
        if v < -100:
            return "#00ff41"
        return "#ddd"
    except (ValueError, TypeError):
        return "#ddd"


def _adx_label(val: str) -> tuple[str, str]:
    try:
        v = float(val)
        if v > 40:
            return ("Strong Trend", "#00ff41")
        if v > 25:
            return ("Trending", "#4caf50")
        if v > 20:
            return ("Weak Trend", "#eab308")
        return ("No Trend", "#666")
    except (ValueError, TypeError):
        return ("—", "#666")


def _ema_stack_status(technicals: dict) -> tuple[str, str]:
    emas = []
    for period in [8, 21, 34, 55, 89]:
        val = technicals.get(f"ema_{period}")
        if val is not None:
            emas.append((period, float(val)))

    if len(emas) < 3:
        return ("INSUFFICIENT DATA", "#555")

    bullish = all(emas[i][1] > emas[i+1][1] for i in range(len(emas)-1))
    bearish = all(emas[i][1] < emas[i+1][1] for i in range(len(emas)-1))

    if bullish:
        return ("BULLISH STACK ↑", "#00ff41")
    if bearish:
        return ("BEARISH STACK ↓", "#ef4444")
    return ("MIXED", "#eab308")

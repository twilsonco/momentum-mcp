"""
Chart generation module using mplfinance.

Renders candlestick + volume charts with optional EMA overlays as static
PNGs and returns both the file path and a base64-encoded string.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import matplotlib
matplotlib.use("Agg")  # Headless rendering — must be set before importing pyplot

import mplfinance as mpf  # noqa: E402
import pandas as pd  # noqa: E402

from mcp_server.data import get_historical_data  # noqa: E402
from mcp.server.fastmcp.utilities.types import Image  # noqa: E402


class TradeLevels(TypedDict, total=False):
    """Optional trade position overlay metadata."""

    direction: str | None  # "LONG" | "SHORT" | None
    entry_price: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    risk_reward_ratio: float | None
    risk: float | None
    reward: float | None


class ChartResult(TypedDict):
    """Return shape for ``generate_chart`` when ``return_image=False``.
    
    All fields except 'trade' are always present. The 'trade' field is only
    included when trade position parameters (entry_price, stop_loss_price,
    take_profit_price) are provided to the chart generation function.
    """

    ticker: str
    period: str
    interval: str
    bars: int
    emas: list[int]
    path: str
    trade: NotRequired[TradeLevels]


logger = logging.getLogger(__name__)

# Default output directory for saved charts
CHARTS_DIR = Path("./charts")

# Clean dark style for chart rendering
_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#22c55e",
        down="#ef4444",
        wick={"up": "#22c55e", "down": "#ef4444"},
        edge={"up": "#22c55e", "down": "#ef4444"},
        volume={"up": "#22c55e80", "down": "#ef444480"},
    ),
    facecolor="#0f0f0f",
    figcolor="#0f0f0f",
    gridcolor="#858585",
    gridstyle="--",
    y_on_right=True,
    rc={
        "font.size": 11,
        "axes.labelcolor": "#e1e1e1",
        "xtick.color": "#C9C9C9",
        "ytick.color": "#C9C9C9",
    },
)

# EMA overlay configuration: (period, color, label)
_EMA_STACK = [
    (8,  "#00d4ff", "EMA 8"),   # cyan
    (21, "#22c55e", "EMA 21"),  # green
    (34, "#eab308", "EMA 34"),  # yellow
    (55, "#f97316", "EMA 55"),  # orange
    (89, "#ef4444", "EMA 89"),  # red
]


async def generate_chart(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    style: str = "dark",
    show_emas: bool = True,
    show_volume: bool = False,
    entry_price: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
    return_image: bool = False,
) -> ChartResult | Image:
    """Generate a candlestick chart with EMA overlays for a ticker symbol.

    Fetches OHLCV data, renders a candlestick chart with optional volume panel
    and stacked EMA overlays (8/21/34/55/89) using ``mplfinance``, saves
    the PNG to the ``./charts/`` directory, and returns either:

    - A FastMCP ``Image`` object (when ``return_image=True``) that the
      MCP transport serializes as proper ``ImageContent`` so an AI agent
      can actually *see* the chart.
    - A JSON dict with metadata + the on-disk path (when
      ``return_image=False``, the default). The base64 string is no
      longer included — it was useless to the agent (a VLM cannot
      "see" a base64 string stuffed in a JSON field) and wasted tokens.

    Optionally overlays a trading position: horizontal lines for entry,
    stop loss, and take profit, auto-detected direction (LONG/SHORT),
    and risk:reward ratio.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        period: Lookback period (e.g. ``"3mo"``, ``"1y"``).
            Defaults to ``"6mo"``.
        interval: Bar interval (e.g. ``"1d"``, ``"1h"``).
            Defaults to ``"1d"``.
        style: Chart colour theme. Currently only ``"dark"`` is
            supported. Reserved for future expansion.
        show_emas: Whether to overlay the EMA stack (8/21/34/55/89).
            Defaults to ``True``.
        show_volume: Whether to display the volume panel below the chart.
            Defaults to ``False``.
        entry_price: Optional entry price for a trade position. Drawn
            as a solid blue horizontal line. When provided together
            with ``stop_loss_price`` and/or ``take_profit_price``, the
            trade direction (LONG/SHORT) and risk:reward ratio are
            auto-detected and displayed on the chart.
        stop_loss_price: Optional stop loss price. Drawn as a dashed
            red horizontal line. Requires ``entry_price``.
        take_profit_price: Optional take profit price. Drawn as a
            dashed green horizontal line. Requires ``entry_price``.
        return_image: If ``True``, return a FastMCP ``Image`` object
            that the MCP transport delivers as ``ImageContent`` so the
            calling agent can actually view the chart. If ``False``
            (default), return a JSON dict with metadata and the file
            path so the agent can reference it by URL/path.

    Returns:
        Either a FastMCP ``Image`` (when ``return_image=True``) or a
        dict with:

        - ``ticker`` — The symbol charted.
        - ``period`` — The period used.
        - ``interval`` — The interval used.
        - ``bars`` — Number of bars rendered.
        - ``emas`` — List of EMA periods overlaid (e.g. [8, 21, 34, 55, 89]).
        - ``path`` — Absolute path to the saved PNG file.
        - ``trade`` — (only when trade levels are provided) Dict with
          ``direction``, ``entry_price``, ``stop_loss_price``,
          ``take_profit_price``, ``risk_reward_ratio``, ``risk``, and
          ``reward``.

    Raises:
        ValueError: If the ticker is invalid, returns no data, or if
            ``stop_loss_price``/``take_profit_price`` are provided
            without ``entry_price``.
    """
    ticker = ticker.strip().upper()

    # Fetch data
    records = await get_historical_data(ticker, period=period, interval=interval, fallback_for_incomplete_data=False)

    if len(records) < 5:
        raise ValueError(
            f"Not enough data to chart '{ticker}': got {len(records)} bars."
        )

    # Validate trade level inputs
    if (stop_loss_price is not None or take_profit_price is not None) and entry_price is None:
        raise ValueError(
            "entry_price is required when stop_loss_price or take_profit_price is provided."
        )
    
    if (stop_loss_price and stop_loss_price <= 0):
        raise ValueError("stop_loss_price must be positive.")
    if (take_profit_price and take_profit_price <= 0):
        raise ValueError("take_profit_price must be positive.")
    if (entry_price and entry_price <= 0):
        raise ValueError("entry_price must be positive.")

    # Compute trade levels (price, color, linestyle, label)
    trade_levels: list[tuple[float, str, str, str]] = []
    direction: str | None = None
    risk_reward: float | None = None

    if entry_price is not None:
        trade_levels.append((entry_price, "#3b82f6", "-", f"Entry ${entry_price:.2f}"))

        if stop_loss_price is not None and take_profit_price is not None:
            # Auto-detect direction from price relationships
            if stop_loss_price < entry_price < take_profit_price:
                direction = "LONG"
            elif stop_loss_price > entry_price > take_profit_price:
                direction = "SHORT"
            else:
                direction = "LONG"  # default when ambiguous

            risk = abs(entry_price - stop_loss_price)
            reward = abs(take_profit_price - entry_price)
            risk_reward = reward / risk if risk > 0 else None

        if stop_loss_price is not None:
            trade_levels.append((stop_loss_price, "#ef4444", "--", f"Stop ${stop_loss_price:.2f}"))

        if take_profit_price is not None:
            trade_levels.append((take_profit_price, "#22c55e", "--", f"TP ${take_profit_price:.2f}"))

    # Build DataFrame in mplfinance-expected format
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    # Coerce to numeric (yfinance occasionally returns strings)
    for col_name in ("Open", "High", "Low", "Close", "Volume"):
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # Build EMA overlay lines
    ema_plots: list[mpf.make_addplot] = []
    ema_periods_used: list[int] = []

    if show_emas:
        for ema_len, color, _label in _EMA_STACK:
            if len(df) >= ema_len:
                ema_series = df["Close"].ewm(span=ema_len, adjust=False).mean()
                ema_plots.append(
                    mpf.make_addplot(
                        ema_series,
                        color=color,
                        width=1.2,
                        panel=0,
                    )
                )
                ema_periods_used.append(ema_len)

    # Render in a background thread
    def _render() -> tuple[str, bytes]:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

        # Build filename — add trade-level suffix so different setups
        # for the same ticker/period/interval don't overwrite each other
        suffix_parts: list[str] = []
        if entry_price is not None:
            suffix_parts.append(f"e{entry_price:.2f}")
        if stop_loss_price is not None:
            suffix_parts.append(f"s{stop_loss_price:.2f}")
        if take_profit_price is not None:
            suffix_parts.append(f"t{take_profit_price:.2f}")
        if suffix_parts:
            filename = f"{ticker}_{period}_{interval}_{'_'.join(suffix_parts)}.png"
        else:
            filename = f"{ticker}_{period}_{interval}.png"
        filepath = CHARTS_DIR / filename

        plot_kwargs: dict[str, Any] = {
            "type": "candle",
            "style": _STYLE,
            "volume": show_volume,
            "figsize": (14, 8),
            "tight_layout": True,
            "warn_too_much_data": 500,
            "returnfig": True,  # Get fig/axes back for legend
        }

        if ema_plots:
            plot_kwargs["addplot"] = ema_plots

        fig, axes = mpf.plot(df, **plot_kwargs)

        # Expand price-axis y-limits to ensure trade-level lines are
        # visible with padding. Without this, an entry/TP/stop placed
        # outside the candle range (e.g. a TP above recent highs) would
        # be drawn but clipped off the chart.
        if trade_levels:
            trade_prices = [price for price, _, _, _ in trade_levels]
            current_lo, current_hi = axes[0].get_ylim()
            price_range = current_hi - current_lo
            # Use 2% of candle range OR 5% of trade-level range,
            # whichever is larger, so the lines aren't flush against
            # the chart edge.
            trade_range = max(trade_prices) - min(trade_prices)
            pad = max(price_range * 0.02, trade_range * 0.05)
            new_lo = min(current_lo, min(trade_prices) - pad)
            new_hi = max(current_hi, max(trade_prices) + pad)
            axes[0].set_ylim(new_lo, new_hi)

        # Build legend handles (EMA + trade levels)
        import matplotlib.lines as mlines
        legend_handles: list[mlines.Line2D] = []

        if ema_periods_used:
            for ema_len, color, label in _EMA_STACK:
                if ema_len in ema_periods_used:
                    legend_handles.append(
                        mlines.Line2D([], [], color=color, linewidth=1.2, label=label)
                    )

        # Draw trade levels on price panel (axes[0])
        if trade_levels:
            for price, color, linestyle, label in trade_levels:
                axes[0].axhline(
                    y=price, color=color, linestyle=linestyle,
                    linewidth=1.2, alpha=0.8,
                )
                # Price label on the right side of the chart
                axes[0].text(
                    0.99, price, f" ${price:.2f} ",
                    transform=axes[0].get_yaxis_transform(),
                    color=color, fontsize=7, fontweight="bold",
                    ha="right", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="#1a1a1a",
                        edgecolor=color,
                        alpha=0.85,
                    ),
                )
                legend_handles.append(
                    mlines.Line2D(
                        [], [], color=color, linestyle=linestyle,
                        linewidth=1.2, label=label,
                    )
                )

            # Trade summary annotation (direction + R:R) in upper-right
            if direction:
                if risk_reward is not None:
                    summary = f"{direction}  |  R:R 1:{risk_reward:.1f}"
                else:
                    summary = direction
                axes[0].text(
                    0.98, 0.98, summary,
                    transform=axes[0].transAxes,
                    color="#ffffff", fontsize=10, fontweight="bold",
                    ha="right", va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="#1a1a1a",
                        edgecolor="#3b82f6",
                        alpha=0.9,
                    ),
                )

        # Add legend if we have any handles
        if legend_handles:
            axes[0].legend(
                handles=legend_handles,
                loc="upper left",
                fontsize=8,
                framealpha=0.3,
                facecolor="#1a1a1a",
                edgecolor="#333333",
                labelcolor="#cccccc",
            )

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        raw_bytes = buf.read()
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Write the same bytes to disk for download
        filepath.write_bytes(raw_bytes)

        return str(filepath.resolve()), raw_bytes

    path, raw_bytes = await asyncio.to_thread(_render)

    logger.info("Chart saved: %s (%d bars, EMAs: %s)", path, len(df), ema_periods_used)

    if return_image:
        # Return as a FastMCP Image so the MCP transport serializes it
        # as proper ImageContent — the agent can actually see the chart.
        return Image(data=raw_bytes, format="png")

    result: ChartResult = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "bars": len(df),
        "emas": ema_periods_used,
        "path": path,
    }

    if trade_levels:
        result["trade"] = {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "risk_reward_ratio": round(risk_reward, 2) if risk_reward is not None else None,
            "risk": round(abs(entry_price - stop_loss_price), 2)
                if entry_price is not None and stop_loss_price is not None else None,
            "reward": round(abs(take_profit_price - entry_price), 2)
                if entry_price is not None and take_profit_price is not None else None,
        }

    return result

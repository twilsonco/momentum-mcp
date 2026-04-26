"""
GARCH Volatility Forecasting Module.

Provides forward-looking volatility estimates using GARCH(1,1) models
to complement the backward-looking composite RV estimator in options.py.

The blend weight between GARCH forecast and composite RV was determined
by backtesting against 10 tickers across 3 years — see calibrate_blend()
for the methodology.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress arch convergence warnings during production use
warnings.filterwarnings("ignore", category=UserWarning, module="arch")


def garch_forecast(
    df: pd.DataFrame,
    horizon: int = 5,
    period: int = 252,
) -> dict[str, Any]:
    """Forecast annualized volatility using GARCH(1,1).

    Args:
        df: OHLCV DataFrame with a 'Close' column.
        horizon: Forecast horizon in trading days (default 5 = 1 week).
        period: Number of recent bars to fit on (default 252 = 1 year).

    Returns:
        dict with:
        - garch_vol_ann: annualized forecasted volatility (decimal, e.g. 0.32)
        - garch_vol_pct: same as percentage (e.g. 32.0)
        - persistence: alpha + beta (closer to 1 = more persistent shocks)
        - converged: whether the model fit converged
        - error: error message if fit failed
    """
    try:
        from arch import arch_model
    except ImportError:
        return {"error": "arch library not installed", "garch_vol_ann": None}

    try:
        close = df["Close"].dropna().tail(period)
        if len(close) < 60:
            return {"error": f"Need 60+ bars, got {len(close)}", "garch_vol_ann": None}

        # Log returns in percentage (arch expects percentage-scale returns)
        log_returns = np.log(close / close.shift(1)).dropna() * 100

        # Fit GARCH(1,1) with Student-t distribution (fatter tails than normal)
        model = arch_model(
            log_returns,
            vol="Garch",
            p=1, q=1,
            dist="t",
            rescale=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)

        # Forecast variance over horizon
        forecast = result.forecast(horizon=horizon)
        # Mean of forecasted daily variances over the horizon
        daily_var = forecast.variance.iloc[-1].mean()

        # Annualize: sqrt(252) * daily_vol
        daily_vol = math.sqrt(daily_var) / 100  # back to decimal from percentage
        ann_vol = daily_vol * math.sqrt(252)

        # Model persistence: alpha1 + beta1 (closer to 1 = shocks persist longer)
        params = result.params
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)
        persistence = alpha + beta

        return {
            "garch_vol_ann": round(ann_vol, 4),
            "garch_vol_pct": round(ann_vol * 100, 2),
            "persistence": round(persistence, 4),
            "converged": result.convergence_flag == 0,
            "error": None,
        }

    except Exception as e:
        logger.warning("GARCH forecast failed: %s", e)
        return {"error": str(e), "garch_vol_ann": None}


def blended_vol_forecast(
    composite_rv: float,
    garch_result: dict,
    blend_weight: float = 0.40,
) -> float:
    """Blend composite RV with GARCH forecast.

    Args:
        composite_rv: Backward-looking composite realized vol (decimal).
        garch_result: Output from garch_forecast().
        blend_weight: Weight given to GARCH forecast (0-1).
                      Determined by calibrate_blend() backtest.
                      Default 0.40 = placeholder until calibration runs.

    Returns:
        Blended annualized volatility estimate (decimal).
    """
    garch_vol = garch_result.get("garch_vol_ann")

    # If GARCH failed, fall back to pure composite RV
    if garch_vol is None or not garch_result.get("converged", False):
        return composite_rv

    # Sanity bounds: if GARCH is wildly different (>3x or <0.3x), reduce its weight
    if composite_rv > 0:
        ratio = garch_vol / composite_rv
        if ratio > 3.0 or ratio < 0.3:
            blend_weight *= 0.5  # halve GARCH weight when it disagrees strongly
            logger.debug("GARCH/RV ratio %.2f — reducing GARCH weight to %.2f", ratio, blend_weight)

    blended = (blend_weight * garch_vol) + ((1 - blend_weight) * composite_rv)
    return round(blended, 4)


# ── Calibration Backtest ─────────────────────────────────────────────────────

async def calibrate_blend(
    tickers: list[str] | None = None,
    period: str = "3y",
    forward_window: int = 21,
) -> dict[str, Any]:
    """Backtest different GARCH/RV blend weights to find the optimal one.

    Methodology:
    1. For each ticker, roll through history in 252-bar windows.
    2. At each point, compute: composite RV, GARCH forecast, and various blends.
    3. Compare each forecast to the REALIZED vol over the next `forward_window` days.
    4. Score by mean absolute error (MAE) — lower = better forecast.
    5. The blend weight with lowest average MAE across all tickers wins.

    Args:
        tickers: List of tickers to test on. Default: diverse set of 10.
        period: Historical period to fetch. Default: '3y'.
        forward_window: Days ahead to measure realized vol (default 21 = 1 month).

    Returns:
        dict with optimal_weight, results per ticker, and MAE comparison.
    """
    import asyncio
    from mcp_server.data import get_historical_data
    from mcp_server.options import _composite_rv, _to_ohlcv_df

    if not tickers:
        # Diverse set: mega-cap, mid-cap, volatile, stable, ETF
        tickers = ["SPY", "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "META", "AMZN", "XOM", "JPM"]

    blend_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_errors: dict[float, list[float]] = {w: [] for w in blend_weights}
    ticker_results: list[dict] = []

    for ticker in tickers:
        try:
            records = await get_historical_data(ticker, period=period, interval="1d")
            if len(records) < 300:
                logger.warning("Skipping %s: only %d bars", ticker, len(records))
                continue

            df = _to_ohlcv_df(records)
            n_obs = len(df)
            step = 21  # roll forward 21 bars at a time
            n_windows = 0
            ticker_errors: dict[float, list[float]] = {w: [] for w in blend_weights}

            for start in range(252, n_obs - forward_window, step):
                window_df = df.iloc[start - 252:start]
                future_df = df.iloc[start:start + forward_window]

                if len(window_df) < 252 or len(future_df) < forward_window:
                    continue

                # Backward-looking composite RV
                rv = _composite_rv(window_df)
                if rv <= 0:
                    continue

                # GARCH forecast
                garch = garch_forecast(window_df, horizon=forward_window)
                garch_vol = garch.get("garch_vol_ann")

                # Actual realized vol over the forward window
                future_returns = np.log(future_df["Close"] / future_df["Close"].shift(1)).dropna()
                actual_vol = float(future_returns.std() * math.sqrt(252))

                if actual_vol <= 0:
                    continue

                n_windows += 1

                # Score each blend weight
                for w in blend_weights:
                    if garch_vol is not None and garch.get("converged"):
                        forecast = (w * garch_vol) + ((1 - w) * rv)
                    else:
                        forecast = rv  # fallback
                    error = abs(forecast - actual_vol)
                    ticker_errors[w].append(error)
                    all_errors[w].append(error)

            # Per-ticker results
            if n_windows > 0:
                best_w = min(blend_weights, key=lambda w: np.mean(ticker_errors[w]) if ticker_errors[w] else 999)
                ticker_results.append({
                    "ticker": ticker,
                    "windows_tested": n_windows,
                    "best_weight": best_w,
                    "best_mae": round(np.mean(ticker_errors[best_w]), 4),
                    "pure_rv_mae": round(np.mean(ticker_errors[0.0]), 4) if ticker_errors[0.0] else None,
                    "pure_garch_mae": round(np.mean(ticker_errors[1.0]), 4) if ticker_errors[1.0] else None,
                })

        except Exception as e:
            logger.warning("Calibration failed for %s: %s", ticker, e)
            continue

    # Global optimal weight
    if not all_errors[0.0]:
        return {"error": "No valid calibration windows found"}

    weight_scores = {}
    for w in blend_weights:
        if all_errors[w]:
            weight_scores[w] = round(np.mean(all_errors[w]), 4)

    optimal_weight = min(weight_scores, key=weight_scores.get)

    return {
        "optimal_weight": optimal_weight,
        "weight_scores": weight_scores,  # MAE per blend weight
        "ticker_results": ticker_results,
        "total_windows": sum(len(v) for v in all_errors.values()) // len(blend_weights),
        "pure_rv_mae": weight_scores.get(0.0),
        "pure_garch_mae": weight_scores.get(1.0),
        "improvement_pct": round(
            (1 - weight_scores[optimal_weight] / weight_scores[0.0]) * 100, 1
        ) if weight_scores.get(0.0) else None,
        "recommendation": (
            f"Use GARCH weight = {optimal_weight}. "
            f"Reduces vol forecast error by {round((1 - weight_scores[optimal_weight] / weight_scores[0.0]) * 100, 1)}% "
            f"vs pure RV."
        ) if weight_scores.get(0.0) else "Insufficient data for recommendation.",
    }

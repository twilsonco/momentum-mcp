from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
from asyncio.log import logger
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import gaussian_kde, median_abs_deviation
from sklearn.cluster import DBSCAN
from typing import Any
from mcp_server.data import get_historical_data
from mcp_server.technicals import _extract_last
from mcp_server.utils.mt5_mcp_server import fetch_mt5_symbol_info

async def _get_symbol_info(symbol: str) -> dict:
    """Validate that a symbol can be traded by calling get_symbol_info.

    Returns:
        Dictionary containing symbol info if valid, empty dict otherwise.
    """
    symbol_info = await fetch_mt5_symbol_info(symbol, timeout=15.0)
    if isinstance(symbol_info, dict) and symbol_info:
        logger.debug(f"Symbol {symbol} validated successfully")
        return symbol_info

    logger.warning(f"Symbol {symbol} validation returned empty response")
    return {}

def _evaluate_strategy(
    df: pd.DataFrame,
    entry_price: float,
    direction: str,
    spread: int,
    strategy: str,
    atr_period: int = 14,
    max_spread_factor_of_sl_dist: float = 0.15,
    max_sl_atr_factor: float = 3.0,
    max_tp_sl_factor: float = 2.5,
    digits: int = 5,
    strategy_params: dict[str, Any] | None = None,
) -> dict:
    """Run a single SL/TP strategy and apply the spread + distance-limit checks.

    This is factored out so `calculate_trade_setup` can iterate over strategies
    in preference order. Returns either a "Valid" setup or an abort dict.

    After a strategy proposes levels, two hard caps are enforced uniformly:
      - SL distance must not exceed ``max_sl_atr_factor`` x ATR.
      - TP distance must not exceed ``max_tp_sl_factor`` x SL distance.
    Both clamp the proposed level inward toward entry rather than aborting,
    keeping risk bounded regardless of how wide a structural level sits.
    """
    # Determine SL and TP levels using the selected strategy, forwarding any
    # per-strategy tuning parameters (unknown keys are swallowed by **kwargs).
    setup = _STRATEGIES[strategy](
        df,
        entry_price=entry_price,
        direction=direction,
        atr_period=atr_period,
        digits=digits,
        **(strategy_params or {}),
    )

    # Propagate any abort from level determination (e.g. insufficient data).
    if setup["status"] != "Valid":
        return {
            "status": setup.get("reason", "Abort"),
            "reason": setup.get("reason", ""),
        }

    proposed_sl = setup["stop_loss"]
    sl_distance = setup["sl_distance"]
    final_tp = setup["take_profit"]
    target_rr = float(setup["risk_reward_ratio"].split(":")[1])
    atr = setup["atr"]

    # Cap 1 — Stop Loss distance must not exceed a factor of ATR.
    max_sl_dist = max_sl_atr_factor * atr
    if sl_distance > max_sl_dist:
        logger.info(
            f"SL distance ({sl_distance:.5f}) exceeds cap "
            f"{max_sl_atr_factor}x ATR ({max_sl_dist:.5f}); clamping SL."
        )
        sl_distance = max_sl_dist
        proposed_sl = (
            entry_price - sl_distance if direction.lower() == 'long'
            else entry_price + sl_distance
        )

    # Cap 2 — Take Profit distance must not exceed a factor of the (capped) SL.
    tp_distance = abs(final_tp - entry_price)
    max_tp_atr_factor = 3.0  # Max TP distance cannot exceed 4x ATR
    max_allowed_tp_dist = min(max_tp_sl_factor * sl_distance, max_tp_atr_factor * atr)

    if tp_distance > max_allowed_tp_dist:
        logger.info(f"TP distance ({tp_distance:.5f}) exceeds cap; clamping TP.")
        final_tp = (
            entry_price + max_allowed_tp_dist if direction.lower() == 'long'
            else entry_price - max_allowed_tp_dist
        )

    # Recompute the discrete RR from the (possibly capped) distances.
    tp_distance = abs(final_tp - entry_price)

    # Guard against a non-finite/zero SL distance (e.g. ATR was NaN). Dividing
    # by it would emit "RuntimeWarning: invalid value encountered in scalar
    # divide" and yield an unusable setup, so abort instead.
    if not np.isfinite(sl_distance) or sl_distance <= 0:
        logger.warning(
            f"Invalid SL distance ({sl_distance}) after capping; "
            f"invalidating {direction} trade."
        )
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": (
                f"Invalid SL distance ({sl_distance}); cannot compute a valid "
                f"{direction} risk/reward setup."
            ),
        }

    target_rr = round(min(tp_distance / sl_distance, 3.0), 2)

    proposed_sl = round(proposed_sl, digits)
    final_tp = round(final_tp, digits)
    sl_distance = round(sl_distance, digits)

    # Check Spread Limit
    # Spread must be strictly less than `max_spread_factor_of_sl_dist` of the SL distance
    if spread >= (max_spread_factor_of_sl_dist * sl_distance):
        logger.warning(f"High spread. Spread ({spread:.5f}) >= {max_spread_factor_of_sl_dist*100:.0f}% of SL Distance ({sl_distance:.5f} @ {proposed_sl:.5f})")
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"High spread. Spread ({spread:.5f}) >= {max_spread_factor_of_sl_dist*100:.0f}% of SL Distance ({sl_distance:.5f} @ {proposed_sl:.5f})"
        }

    return {
        "status": "Valid",
        "entry": entry_price,
        "stop_loss": proposed_sl,
        "take_profit": final_tp,
        "risk_reward_ratio": f"1:{target_rr}",
        "sl_distance": sl_distance,
        "atr": atr
    }


def calculate_trade_setup(df: pd.DataFrame, entry_price: float, direction: str, spread: int, atr_period=14, max_spread_factor_of_sl_dist: float = 0.15, max_sl_atr_factor: float = 3.0, max_tp_sl_factor: float = 2.5, digits: int = 5, strategy: str | None = "auto", strategy_params: dict[str, Any] | None = None):
    """
    Calculates deterministic SL and TP based on ATR, recent swings, and spread limits.

    When ``strategy`` is a specific name (e.g. "swings"), only that strategy is
    tried. When it is ``"auto"`` (the default) or ``None``, strategies are tried in
    decreasing order of preference (`_STRATEGY_PREFERENCE`: KDE > DBSCAN > Swings);
    the first one to produce a valid setup wins, otherwise the trade aborts.

    Distance caps applied uniformly after any strategy proposes levels:
      - SL distance is capped at ``max_sl_atr_factor`` x ATR (default 3x).
      - TP distance is capped at ``max_tp_sl_factor`` x SL distance (default 2.5x).

    Parameters:
    df (pd.DataFrame): Price data with a DatetimeIndex and ['High', 'Low', 'Close']
    entry_price (float): Current price for trade entry
    direction (str): 'long' or 'short'
    spread (float): Current spread distance (in price terms, e.g., 0.0002 for forex)
    atr_period (int): Period for ATR calculation (default 14)
    max_sl_atr_factor (float): Max SL distance as a multiple of ATR.
    max_tp_sl_factor (float): Max TP distance as a multiple of the SL distance.
    strategy (str | None): SL/TP method: "auto"/None to cascade through all
        strategies in preference order, or a specific name from _STRATEGIES.
    strategy_params (dict | None): Optional per-strategy tuning overrides forwarded
        to the selected level-determination function(s). Keys not applicable to a
        given strategy are ignored. See each strategy's docstring for its params.
    """

    logger.info(f"Calculating trade setup for entry_price={entry_price}, direction={direction}, spread={spread}, atr_period={atr_period}, max_spread_factor_of_sl_dist={max_spread_factor_of_sl_dist}, max_sl_atr_factor={max_sl_atr_factor}, max_tp_sl_factor={max_tp_sl_factor}, digits={digits}, strategy={strategy}")

    # Resolve which strategies to try, in order.
    if strategy is None or strategy == "auto":
        candidates = _STRATEGY_PREFERENCE
    else:
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Valid options: {list(_STRATEGIES)}")
        candidates = [strategy]

    # Try each candidate until one yields a valid setup. All operate on the same
    # `df`, so no additional data fetching is required between attempts.
    for strat in candidates:
        result = _evaluate_strategy(
            df,
            entry_price=entry_price,
            direction=direction,
            spread=spread,
            strategy=strat,
            atr_period=atr_period,
            max_spread_factor_of_sl_dist=max_spread_factor_of_sl_dist,
            max_sl_atr_factor=max_sl_atr_factor,
            max_tp_sl_factor=max_tp_sl_factor,
            digits=digits,
            strategy_params=strategy_params,
        )
        if result["status"] == "Valid":
            logger.info(f"Strategy '{strat}' produced a valid {direction} setup.")
            return result
        logger.info(f"Strategy '{strat}' did not produce a valid {direction} setup; trying next.")

    # No strategy produced a valid setup — abort in the requested direction.
    reason = (
        f"No SL/TP strategy ({', '.join(candidates)}) yielded a valid "
        f"{direction} trade setup."
    )
    logger.warning(reason)
    return {
        "status": f"Abort: Do not open {direction} position",
        "reason": reason,
    }


def _atr_series(df: pd.DataFrame, atr_period: int = 14):
    """Compute the ATR series for a price DataFrame using ``ta.atr``.

    Single source of truth for ATR so no caller re-implements or duplicates
    the calculation. Returns the raw pandas Series from ``pandas_ta``.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    return ta.atr(high, low, close, length=atr_period)


def _calculate_atr(df: pd.DataFrame, atr_period: int = 14) -> float:
    """Calculate the current ATR value from a price DataFrame."""
    return _extract_last(_atr_series(df, atr_period=atr_period))


def _apply_target_rr(
    entry_price: float,
    direction: str,
    sl_distance: float,
    barrier_distance: float,
) -> tuple[float, float]:
    """Enforce a discrete risk-to-reward ratio (1, 2, or 3).

    Returns:
        Tuple of (final_tp, target_rr). If the structural barrier doesn't allow
        at least 1:1, forces a 1:1 target.
    """
    # Ratios available: 1, 2, or 3
    # Guard against a non-finite/zero SL distance (e.g. ATR was NaN). Dividing
    # by it would emit "RuntimeWarning: invalid value encountered in scalar
    # divide"; force the safest discrete target instead.
    if not np.isfinite(sl_distance) or sl_distance <= 0:
        return entry_price, 1

    if barrier_distance < sl_distance:
        # Force 1:1 if structural barrier doesn't allow it
        target_rr = 1
    else:
        # Find the highest discrete RR (1, 2, or 3) that fits before the barrier
        target_rr = round(min(barrier_distance / sl_distance, 3.0), 2)

    if direction.lower() == 'long':
        final_tp = entry_price + (sl_distance * target_rr)
    else:
        final_tp = entry_price - (sl_distance * target_rr)

    return final_tp, target_rr


def _determine_sl_tp_from_swings(
    df: pd.DataFrame,
    entry_price: float,
    direction: str,
    atr_period: int = 14,
    digits: int = 5,
    **kwargs: Any,
) -> dict:
    """Determine SL and TP levels using the default ATR/swing strategy.

    This is one implementation of level determination. Alternative strategies
    (e.g. fixed-percentage, volatility-based, or support/resistance methods)
    can be added as separate functions with an identical signature and return
    contract so they are drop-in replacements for `calculate_trade_setup`.

    Returns:
        dict: On success contains status "Valid" plus stop_loss, take_profit,
              risk_reward_ratio (as "1:x"), sl_distance, and atr. On failure it
              returns a non-"Valid" status with a human-readable reason.
    """
    
    # 1. Calculate ATR (Current)
    current_atr = _calculate_atr(df, atr_period=atr_period)

    # Define bar-based windows based on inferred interval.
    # We prefer a 48-bar lookback for the swing low/high, but will accept
    # as little as 24 bars of history. Less than 24 bars → abort.
    # Using bar-based windows (rather than time-based) avoids issues with
    # markets that close on weekends/holidays, where the actual time gap
    # can be much larger than the number of bars available.

    # Infer the typical bar interval from the DataFrame index. The median
    # is robust to weekend/holiday gaps because most consecutive bars are
    # the regular interval during trading hours.
    if len(df) >= 2:
        deltas = df.index.to_series().diff().dropna()
        bar_interval = deltas.median()
    else:
        bar_interval = pd.Timedelta(hours=1)  # fallback

    bars_per_24h = max(1, int(pd.Timedelta(hours=24) / bar_interval))
    bars_per_48h = max(1, int(pd.Timedelta(hours=48) / bar_interval))

    # Fail if we don't have at least 24 hours worth of bars
    if len(df) < bars_per_24h:
        logger.warning(f"Insufficient data: len(df)={len(df)}, bars_per_24h={bars_per_24h}, bar_interval={bar_interval}")
        return {
            "status": "Abort",
            "reason": (
                f"Insufficient data: less than 24 hours of history available "
                f"({len(df)} bars, need at least {bars_per_24h} for interval {bar_interval})."
            ),
        }

    # Swing window: prefer the 48h-to-24h slice, but extend back to the
    # oldest bar if we don't have a full 48h of bars.
    swing_start_idx = max(0, len(df) - bars_per_48h)
    swing_end_idx = len(df) - bars_per_24h
    window_24_to_48 = df.iloc[swing_start_idx:swing_end_idx]

    # Barrier window: last 48h worth of bars (naturally handles < 48h case
    # by capping at the available bar count).
    barrier_bars_count = min(bars_per_48h, len(df))
    window_last_48 = df.iloc[len(df) - barrier_bars_count:]

    if window_24_to_48.empty or window_last_48.empty:
        logger.warning(f"Insufficient data for lookback windows. window_24_to_48 empty: {window_24_to_48.empty}, window_last_48 empty: {window_last_48.empty}, len(df)={len(df)}, bars_per_24h={bars_per_24h}, bars_per_48h={bars_per_48h}")
        return {"status": "Abort", "reason": "Insufficient data for lookback windows."}

    # 2. Determine Stop Loss (SL)
    if direction.lower() == 'long':
        swing_low = window_24_to_48['Low'].min()
        proposed_sl = swing_low - (1.5 * current_atr)
        sl_distance = entry_price - proposed_sl
        
        logger.info(f"Proposed SL: {proposed_sl}, SL Distance: {sl_distance}, Current ATR: {current_atr}")
        
        # Guard: for a long trade, SL must be below entry. If the swing-based
        # SL ended up above entry (price has dropped below the prior swing low),
        # fall back to a minimum ATR-based SL distance.
        if sl_distance < (1.5 * current_atr):
            proposed_sl = entry_price - (1.5 * current_atr)
            sl_distance = 1.5 * current_atr
            logger.info(f"SL was above entry; using minimum ATR-based SL. Proposed SL: {proposed_sl}, SL Distance: {sl_distance}")
        
        # Constraint: SL distance must never exceed 3 * ATR
        if sl_distance > (3 * current_atr):
            proposed_sl = entry_price - (3 * current_atr)
            sl_distance = 3 * current_atr
            
    elif direction.lower() == 'short':
        swing_high = window_24_to_48['High'].max()
        proposed_sl = swing_high + (1.5 * current_atr)
        sl_distance = proposed_sl - entry_price
        
        logger.info(f"Proposed SL: {proposed_sl}, SL Distance: {sl_distance}, Current ATR: {current_atr}")
        
        # Guard: for a short trade, SL must be above entry. If the swing-based
        # SL ended up below entry (price has risen above the prior swing high),
        # fall back to a minimum ATR-based SL distance.
        if sl_distance < (1.5 * current_atr):
            proposed_sl = entry_price + (1.5 * current_atr)
            sl_distance = 1.5 * current_atr
            logger.info(f"SL was below entry; using minimum ATR-based SL. Proposed SL: {proposed_sl}, SL Distance: {sl_distance}")
        
        # Constraint: SL distance must never exceed 3 * ATR
        if sl_distance > (3 * current_atr):
            proposed_sl = entry_price + (3 * current_atr)
            sl_distance = 3 * current_atr

    else:
        raise ValueError("Direction must be 'long' or 'short'")

    proposed_sl = round(proposed_sl, digits)
    sl_distance = round(sl_distance, digits)
    
    logger.info(f"Adjusted Proposed SL: {proposed_sl}, SL Distance: {sl_distance}, Current ATR: {current_atr}")

    # 3. Find Take Profit Barrier using 95th/5th percentile to reject outlier wicks
    if direction.lower() == 'long':
        barrier_price = float(np.percentile(window_last_48['High'], 95))
        barrier_distance = barrier_price - entry_price
    else:
        barrier_price = float(np.percentile(window_last_48['Low'], 5))
        barrier_distance = entry_price - barrier_price
        
    logger.info(f"Barrier Price: {barrier_price}, Barrier Distance: {barrier_distance}")

    if not np.isfinite(sl_distance) or sl_distance <= 0 \
            or not np.isfinite(barrier_distance) or barrier_distance <= 0:
        logger.warning(f"Invalid SL or Barrier distance. SL Distance: {sl_distance}, Barrier Distance: {barrier_distance}")
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"Invalid SL ({sl_distance}) or Barrier ({barrier_distance}) distance"
        }
    
    # 4. Enforce Risk-to-Reward (RR)
    final_tp, target_rr = _apply_target_rr(
        entry_price, direction, sl_distance, barrier_distance
    )
    
    logger.info(f"Target RR: {target_rr}, SL Distance: {sl_distance}, Barrier Distance: {barrier_distance}")
            
    if final_tp <= 0 or proposed_sl <= 0:
        logger.warning(f"Invalid final TP ({final_tp}) or proposed SL ({proposed_sl})")
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"Invalid final TP ({final_tp}) or proposed SL ({proposed_sl})"
        }
    
    final_tp = round(final_tp, digits)

    return {
        "status": "Valid",
        "stop_loss": proposed_sl,
        "take_profit": final_tp,
        "risk_reward_ratio": f"1:{target_rr}",
        "sl_distance": sl_distance,
        "atr": current_atr
    }


def _cluster_levels(levels: np.ndarray, min_gap: float) -> np.ndarray:
    """Merge peaks that are closer than ``min_gap`` into a representative level.

    Raw KDE peak detection on a fine price grid can return many micro-peaks
    clustered around the same consolidation zone. This merges neighbours within
    ``min_gap`` and returns their mean as a single structural level.
    """
    if len(levels) == 0:
        return np.array([])
    levels = np.sort(np.asarray(levels, dtype=float))
    clusters: list[list[float]] = [[levels[0]]]
    for lv in levels[1:]:
        if lv - clusters[-1][-1] <= min_gap:
            clusters[-1].append(float(lv))
        else:
            clusters.append([float(lv)])
    return np.array([np.mean(c) for c in clusters])


def _get_vw_kde_sr_levels(
    df: pd.DataFrame,
    resolution: int = 1000,
    bw_frac: float = 0.10,
    prominence_frac: float = 0.05,
) -> np.ndarray:
    """Calculate Support/Resistance levels using Volume-Weighted KDE.

    Typical Price (High+Low+Close)/3 is weighted by volume so that high-volume
    consolidation areas produce density spikes while low-volume outlier wicks
    are suppressed. Prominent peaks in the resulting curve represent High
    Volume Nodes (HVNs).

    Bandwidth is specified as a fraction of the price standard deviation, giving
    an *absolute* bandwidth that behaves consistently across instruments with
    very different price scales (e.g. EURUSD at ~1.15 vs AAPL at ~200). Detected
    peaks are then clustered so nearby micro-peaks collapse into single true
    structural levels.

    Args:
        df: DataFrame containing 'High', 'Low', 'Close', and 'Volume'.
        resolution: Number of points in the price grid.
        bw_frac: KDE bandwidth as a fraction of the typical-price standard
            deviation. Smaller = more sensitive (more micro-levels); larger =
            smoother (fewer, broader consolidation zones).
        prominence_frac: Minimum peak prominence as a fraction of max density.

    Returns:
        Array of price levels representing prominent High Volume Nodes, or an
        empty array if no peaks meet the prominence threshold.
    """
    # Typical Price captures intra-bar action; volume acts as its weight so
    # high-volume areas dominate the distribution.
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    prices = np.asarray(typical_price, dtype=float)
    weights = np.asarray(df['Volume'], dtype=float)

    if len(prices) < 2 or weights.sum() <= 0:
        return np.array([])

    # Winsorize volume weights so 1-2 news candles do not dominate the PDF
    median_vol = float(np.nanmedian(weights))
    capped_weights = np.clip(weights, 0, max(median_vol * 3.0, 1.0))

    bw_abs = float(np.nanstd(prices)) * bw_frac
    if not np.isfinite(bw_abs) or bw_abs <= 0:
        return np.array([])

    kde = gaussian_kde(prices, weights=capped_weights, bw_method=bw_abs)
    
    # Restrict the grid to the 2nd and 98th percentiles to avoid stretching across isolated wicks
    p_min = float(np.nanpercentile(df['Low'], 2))
    p_max = float(np.nanpercentile(df['High'], 98))
    
    if p_max <= p_min:
        p_min, p_max = float(np.nanmin(df['Low'])), float(np.nanmax(df['High']))

    price_grid = np.linspace(p_min, p_max, resolution)
    pdf = kde(price_grid)

    prominence_threshold = pdf.max() * prominence_frac
    peaks, _ = find_peaks(pdf, prominence=prominence_threshold)
    raw_levels = price_grid[peaks]

    return _cluster_levels(raw_levels, min_gap=bw_abs)


def _determine_sl_tp_from_vw_kde(
    df: pd.DataFrame,
    entry_price: float,
    direction: str,
    atr_period: int = 14,
    digits: int = 5,
    resolution: int = 1000,
    bw_frac: float = 0.10,
    prominence_frac: float = 0.05,
    **kwargs: Any,
) -> dict:
    """Determine SL and TP levels using Volume-Weighted KDE High Volume Nodes.

    This is an alternative level-determination strategy that shares the same
    signature and return contract as `_determine_sl_tp_from_swings`, making it a
    drop-in replacement for use by `calculate_trade_setup`.

    For a long, the nearest prominent HVN below entry acts as support (SL) and
    the nearest above as resistance (TP). For a short these are reversed. If no
    valid structural node exists on either side of the current price, the trade
    is invalidated.

    Tunable parameters (via ``strategy_params``):
        resolution: Number of points in the KDE price grid.
        bw_frac: KDE bandwidth as a fraction of typical-price std dev. Smaller =
            more sensitive; larger = smoother/fewer levels.
        prominence_frac: Minimum peak prominence as a fraction of max density.

    Returns:
        dict: On success contains status "Valid" plus stop_loss, take_profit,
              risk_reward_ratio (as "1:x"), sl_distance, and atr. On failure it
              returns a non-"Valid" status with a human-readable reason.
    """
    current_atr = _calculate_atr(df, atr_period=atr_period)

    sr_levels = _get_vw_kde_sr_levels(
        df,
        resolution=resolution,
        bw_frac=bw_frac,
        prominence_frac=prominence_frac,
    )
    if len(sr_levels) == 0:
        logger.warning("VW-KDE produced no prominent High Volume Nodes; invalidating trade.")
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": (
                "No prominent volume nodes found (flat/wide distribution). "
                "Trend is unbacked by structural support/resistance."
            ),
        }

    levels_below = sr_levels[sr_levels < entry_price]
    levels_above = sr_levels[sr_levels > entry_price]

    if direction.lower() == 'long':
        # Long needs support below (SL) and resistance above (TP).
        if len(levels_below) == 0 or len(levels_above) == 0:
            return {
                "status": f"Abort: Do not open {direction} position",
                "reason": (
                    "No structural node on both sides of price; "
                    "cannot place a valid SL and TP."
                ),
            }
        proposed_sl = levels_below[-1]  # nearest level below
        barrier_price = levels_above[0]  # nearest level above
    elif direction.lower() == 'short':
        # Short needs resistance above (SL) and support below (TP).
        if len(levels_above) == 0 or len(levels_below) == 0:
            return {
                "status": f"Abort: Do not open {direction} position",
                "reason": (
                    "No structural node on both sides of price; "
                    "cannot place a valid SL and TP."
                ),
            }
        proposed_sl = levels_above[0]  # nearest level above
        barrier_price = levels_below[-1]  # nearest level below
    else:
        raise ValueError("Direction must be 'long' or 'short'")

    sl_distance = abs(entry_price - proposed_sl)
    barrier_distance = abs(barrier_price - entry_price)

    if not np.isfinite(sl_distance) or sl_distance <= 0 \
            or not np.isfinite(barrier_distance) or barrier_distance <= 0:
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"Invalid SL ({sl_distance}) or Barrier ({barrier_distance}) distance",
        }

    final_tp, target_rr = _apply_target_rr(
        entry_price, direction, sl_distance, barrier_distance
    )

    if final_tp <= 0 or proposed_sl <= 0:
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"Invalid final TP ({final_tp}) or proposed SL ({proposed_sl})",
        }

    return {
        "status": "Valid",
        "stop_loss": round(proposed_sl, digits),
        "take_profit": round(final_tp, digits),
        "risk_reward_ratio": f"1:{target_rr}",
        "sl_distance": round(sl_distance, digits),
        "atr": current_atr,
    }


def _extract_extrema(df: pd.DataFrame, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Extract local swing highs and lows with their candle volumes.

    A peak is only a valid swing high if it is higher than the ``order`` candles
    to its immediate left and right (and symmetrically for swing lows). The price
    stored for a maximum is the bar's High; for a minimum, the bar's Low. Each is
    paired with that candle's total volume so downstream clustering can weight by
    liquidity.

    Args:
        df: DataFrame containing 'High', 'Low', and 'Volume'.
        order: Number of candles to each side required to confirm an extremum.
            Higher values filter noise but require more history.

    Returns:
        Tuple of (prices, volumes) arrays aligned by index. Empty if no extrema
        were found or the DataFrame is too short for the requested ``order``.
    """
    n = len(df)
    # argrelextrema needs at least 2*order+1 samples to find any extremum.
    if n < (2 * order + 1):
        return np.array([]), np.array([])

    highs = df['High'].to_numpy(dtype=float)
    lows = df['Low'].to_numpy(dtype=float)
    volumes = df['Volume'].to_numpy(dtype=float)

    local_max_idx = argrelextrema(highs, np.greater, order=order)[0]
    local_min_idx = argrelextrema(lows, np.less, order=order)[0]

    # A single index cannot be both a max and a min; union them for iteration.
    extrema_idx = np.sort(np.unique(np.concatenate((local_max_idx, local_min_idx))))

    prices = []
    vols = []
    for idx in extrema_idx:
        if idx in local_max_idx:
            prices.append(highs[idx])
        else:
            prices.append(lows[idx])
        vols.append(volumes[idx])

    return np.asarray(prices), np.asarray(vols)


def _dbscan_sr_levels(
    df: pd.DataFrame,
    method: str = "atr",
    k: float = 0.2,
    min_samples: int = 3,
    order: int = 5,
    atr_series=None,
) -> np.ndarray:
    """Compute volume-weighted Support/Resistance levels via DBSCAN clustering.

    Local swing highs/lows are extracted from the OHLCV data, clustered with
    ``sklearn.cluster.DBSCAN`` using a volatility-adaptive ``eps``, and each valid
    cluster is collapsed to its volume-weighted average price. This pulls an S/R
    line toward the specific touch that saw the heaviest trading volume rather than
    the geometric center of the cluster.

    Args:
        df: DataFrame containing 'High', 'Low', 'Close', and 'Volume'.
        method: ``"atr"`` (eps = k * mean ATR over the window) or ``"mad"``
            (eps = k * median absolute deviation of the extracted swing prices).
        k: Coefficient multiplier for the eps threshold. Typically 0.1-0.3.
        min_samples: Minimum number of touches required to form a valid cluster;
            isolated wicks are labelled as noise (-1) and discarded.
        order: Sensitivity for swing-point extraction (see ``_extract_extrema``).
        atr_series: Optional precomputed ATR series (from ``_atr_series``). When
            provided it is reused instead of recomputing ATR, avoiding a redundant
            calculation when the caller already has one.

    Returns:
        Sorted array of volume-weighted S/R levels, or an empty array if there is
        insufficient data / no valid clusters were found.
    """
    prices, volumes = _extract_extrema(df, order=order)
    if len(prices) < min_samples:
        return np.array([])

    # 1. Dynamically derive the DBSCAN eps threshold from market volatility.
    if method == "atr":
        if atr_series is None:
            atr_series = _atr_series(df, atr_period=14)
        baseline_volatility = float(atr_series.dropna().mean())
        eps = k * baseline_volatility
    elif method == "mad":
        # scale=1 returns the raw unscaled MAD (robust to outlier wicks).
        mad = median_abs_deviation(prices, scale=1)
        eps = k * float(mad)
    else:
        raise ValueError("Method must be 'atr' or 'mad'")

    # Guard against a perfectly flat historical range producing zero eps.
    eps = max(eps, 1e-5)

    # 2. Cluster the extremum prices with DBSCAN (needs a 2D column vector).
    price_matrix = prices.reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(price_matrix)

    # 3. Collapse each valid cluster to its volume-weighted average price.
    sr_levels: list[float] = []
    for label in np.unique(labels):
        if label == -1:
            continue  # noise (isolated wicks) — not a structural level
        mask = labels == label
        cluster_prices = prices[mask]
        cluster_volumes = volumes[mask]

        total_volume = float(cluster_volumes.sum())
        if total_volume > 0:
            vw_price = float(np.sum(cluster_prices * cluster_volumes) / total_volume)
        else:
            # Fallback to a simple mean when volume data is missing/zero.
            vw_price = float(np.mean(cluster_prices))
        sr_levels.append(vw_price)

    return np.sort(sr_levels)


def _determine_sl_tp_from_dbscan(
    df: pd.DataFrame,
    entry_price: float,
    direction: str,
    atr_period: int = 14,
    digits: int = 5,
    method: str = "atr",
    k: float = 0.2,
    min_samples: int = 3,
    order: int = 5,
    **kwargs: Any,
) -> dict:
    """Determine SL and TP levels using DBSCAN volume-weighted S/R clusters.

    This is an alternative level-determination strategy that shares the same
    signature and return contract as `_determine_sl_tp_from_swings`, making it a
    drop-in replacement for use by `calculate_trade_setup`.

    Local swing points are clustered with DBSCAN (eps scaled to volatility) into
    structural S/R levels weighted by volume. For a long, the nearest support
    cluster below entry acts as SL and the nearest resistance above as TP; these
    are reversed for a short. If no valid structural level exists on either side of
    price, the trade is invalidated (downward/upward price discovery or breakout).

    Tunable parameters (via ``strategy_params``):
        method: ``"atr"`` or ``"mad"`` — how eps is derived from volatility.
        k: Coefficient multiplier for the eps threshold. Typically 0.1-0.3.
        min_samples: Minimum touches required to form a valid cluster.
        order: Sensitivity of swing-point extraction (higher = fewer, more
            significant swings).

    Returns:
        dict: On success contains status "Valid" plus stop_loss, take_profit,
              risk_reward_ratio (as "1:x"), sl_distance, and atr. On failure it
              returns a non-"Valid" status with a human-readable reason.
    """
    # Compute ATR once and reuse it both for the current value (below) and as
    # the DBSCAN eps baseline, so we never call ta.atr twice on the same data.
    atr_series = _atr_series(df, atr_period=atr_period)
    current_atr = _extract_last(atr_series)

    sr_levels = _dbscan_sr_levels(
        df,
        method=method,
        k=k,
        min_samples=min_samples,
        order=order,
        atr_series=atr_series,
    )
    if len(sr_levels) == 0:
        logger.warning("DBSCAN produced no valid S/R clusters; invalidating trade.")
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": (
                "No structural volume-weighted support/resistance found "
                "(insufficient swings or all points labelled as noise)."
            ),
        }

    supports = sr_levels[sr_levels < entry_price]
    resistances = sr_levels[sr_levels > entry_price]

    if direction.lower() == 'long':
        # Long needs support below (SL) and resistance above (TP).
        if len(supports) == 0:
            return {
                "status": f"Abort: Do not open {direction} position",
                "reason": (
                    "No structural support below for Stop Loss. "
                    "Market is in downward price discovery."
                ),
            }
        if len(resistances) == 0:
            return {
                "status": f"Abort: Do not open {direction} position",
                "reason": (
                    "No structural resistance above for Take Profit. "
                    "Market is in a blue-sky breakout."
                ),
            }
        proposed_sl = supports[-1]     # closest support below
        barrier_price = resistances[0]  # closest resistance above
    elif direction.lower() == 'short':
        # Short needs resistance above (SL) and support below (TP).
        if len(resistances) == 0:
            return {
                "status": f"Abort: Do not open {direction} position",
                "reason": (
                    "No structural resistance above for Stop Loss. "
                    "Market is in upward price discovery."
                ),
            }
        if len(supports) == 0:
            return {
                "status": f"Abort: Do not open {direction} position",
                "reason": (
                    "No structural support below for Take Profit. "
                    "Market is in freefall."
                ),
            }
        proposed_sl = resistances[0]   # closest resistance above
        barrier_price = supports[-1]    # closest support below
    else:
        raise ValueError("Direction must be 'long' or 'short'")

    sl_distance = abs(entry_price - proposed_sl)
    barrier_distance = abs(barrier_price - entry_price)

    if not np.isfinite(sl_distance) or sl_distance <= 0 \
            or not np.isfinite(barrier_distance) or barrier_distance <= 0:
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"Invalid SL ({sl_distance}) or Barrier ({barrier_distance}) distance",
        }

    final_tp, target_rr = _apply_target_rr(
        entry_price, direction, sl_distance, barrier_distance
    )

    if final_tp <= 0 or proposed_sl <= 0:
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"Invalid final TP ({final_tp}) or proposed SL ({proposed_sl})",
        }

    return {
        "status": "Valid",
        "stop_loss": round(proposed_sl, digits),
        "take_profit": round(final_tp, digits),
        "risk_reward_ratio": f"1:{target_rr}",
        "sl_distance": round(sl_distance, digits),
        "atr": current_atr,
    }


# Registry of available SL/TP determination strategies. Each maps a strategy
# name to the function that computes stop_loss/take_profit levels.
_STRATEGIES = {
    "swings": _determine_sl_tp_from_swings,
    "vw_kde": _determine_sl_tp_from_vw_kde,
    "dbscan": _determine_sl_tp_from_dbscan,
}

# Order in which strategies are tried when `strategy="auto"` (the default).
# Strategies earlier in the list are preferred; the first one that yields a
# valid trade setup wins. Add new entries here to include them in the cascade.
_STRATEGY_PREFERENCE = ["vw_kde", "dbscan", "swings"]


async def calculate_trade_setups(ticker: str, period: str, interval: str, symbol_info: dict[str, Any] | None = None, strategy: str | None = "auto", input_records: list[dict[str, Any]] | None = None, strategy_params: dict[str, Any] | None = None) -> dict[str, Any]:
    
    ticker = ticker.strip().upper()
    
    # Fetch data — we don't need the full requested period, so don't waste
    # time falling back to other sources just because the date range is short.
    if input_records is not None:
        records = input_records
    else:
        records = await get_historical_data(
            ticker,
            period=period,
            interval=interval,
            fallback_for_incomplete_data=False,
        )
    if symbol_info is None:
        symbol_info = await _get_symbol_info(ticker)
    
    if symbol_info is None:
        raise ValueError(f"Failed to fetch symbol info for {ticker}")

    # Validate symbol info
    required_fields = ["ask", "bid", "digits"]
    if not symbol_info or any(field not in symbol_info or not symbol_info[field] for field in required_fields):
        raise ValueError(f"Symbol {ticker} is missing required fields: {required_fields}")
    
    if len(records) < 5:
        raise ValueError(
            f"Not enough data to chart '{ticker}': got {len(records)} bars."
        )
    
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
    
    ret = {}
    
    spread = symbol_info["ask"] - symbol_info["bid"]
    
    # Calculate long setup
    logger.info(f"Calculating long setup for {ticker} (strategy={strategy})")
    long_setup = calculate_trade_setup(df, symbol_info["bid"], "long", spread, atr_period=14, digits=symbol_info["digits"], strategy=strategy, strategy_params=strategy_params)
    ret["long_buy_setup"] = long_setup
    
    # Calculate short setup
    logger.info(f"Calculating short setup for {ticker} (strategy={strategy})")
    short_setup = calculate_trade_setup(df, symbol_info["ask"], "short", spread, atr_period=14, digits=symbol_info["digits"], strategy=strategy, strategy_params=strategy_params)
    ret["short_sell_setup"] = short_setup
    
    if all("Abort" in setup.get("status", "") for setup in (long_setup, short_setup)):
        ret = {"status": "Abort: Do not open any position"}

    return ret
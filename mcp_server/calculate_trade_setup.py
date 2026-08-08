from __future__ import annotations

from asyncio.log import logger
import pandas as pd
import pandas_ta as ta
import numpy as np
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

def calculate_trade_setup(df: pd.DataFrame, entry_price: float, direction: str, spread: int, atr_period=14, max_spread_factor_of_sl_dist: float = 0.15, digits: int = 5):
    """
    Calculates deterministic SL and TP based on ATR, recent swings, and spread limits.
    
    Parameters:
    df (pd.DataFrame): Price data with a DatetimeIndex and ['High', 'Low', 'Close']
    entry_price (float): Current price for trade entry
    direction (str): 'long' or 'short'
    spread (float): Current spread distance (in price terms, e.g., 0.0002 for forex)
    atr_period (int): Period for ATR calculation (default 14)
    """
    
    logger.info(f"Calculating trade setup for entry_price={entry_price}, direction={direction}, spread={spread}, atr_period={atr_period}, max_spread_factor_of_sl_dist={max_spread_factor_of_sl_dist}, digits={digits}")
    
    # 1. Calculate ATR (Current)
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    current_atr = _extract_last(ta.atr(high, low, close, length=14))
    last_timestamp = df.index[-1]
    
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

    # 3. Find Take Profit (TP) Structural Barrier
    # Deterministic proxy: Highest high (longs) or lowest low (shorts) in the last 48 hours
    if direction.lower() == 'long':
        barrier_price = window_last_48['High'].max()
        barrier_distance = barrier_price - entry_price
    else:
        barrier_price = window_last_48['Low'].min()
        barrier_distance = entry_price - barrier_price
        
    logger.info(f"Barrier Price: {barrier_price}, Barrier Distance: {barrier_distance}")

    # 4. Enforce Risk-to-Reward (RR)
    # Ratios available: 1, 2, or 3
    if barrier_distance < sl_distance:
        # Force 1:1 if structural barrier doesn't allow it
        target_rr = 1
    else:
        # Find the highest discrete RR (1, 2, or 3) that fits before the barrier
        target_rr = round(min(barrier_distance / sl_distance, 3.0), 2)
    
    logger.info(f"Target RR: {target_rr}, SL Distance: {sl_distance}, Barrier Distance: {barrier_distance}")
            
    if direction.lower() == 'long':
        final_tp = entry_price + (sl_distance * target_rr)
    else:
        final_tp = entry_price - (sl_distance * target_rr)
    
    if final_tp <= 0 or proposed_sl <= 0:
        logger.warning(f"Invalid final TP ({final_tp}) or proposed SL ({proposed_sl})")
        return {
            "status": f"Abort: Do not open {direction} position",
            "reason": f"Invalid final TP ({final_tp}) or proposed SL ({proposed_sl})"
        }
    
    final_tp = round(final_tp, digits)

    # 5. Check Spread Limit
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
        "atr": current_atr
    }



async def calculate_trade_setups(ticker: str, period: str, interval: str, symbol_info: dict[str, Any] | None = None) -> dict[str, Any]:
    
    ticker = ticker.strip().upper()
    
    # Fetch data — we don't need the full requested period, so don't waste
    # time falling back to other sources just because the date range is short.
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
    logger.info(f"Calculating long setup for {ticker}")
    long_setup = calculate_trade_setup(df, symbol_info["bid"], "long", spread, atr_period=14, digits=symbol_info["digits"])
    ret["long_buy_setup"] = long_setup
    
    # Calculate short setup
    logger.info(f"Calculating short setup for {ticker}")
    short_setup = calculate_trade_setup(df, symbol_info["ask"], "short", spread, atr_period=14, digits=symbol_info["digits"])
    ret["short_sell_setup"] = short_setup
    
    if all("Abort" in setup.get("status", "") for setup in (long_setup, short_setup)):
        ret = {"status": "Abort: Do not open any position"}

    return ret
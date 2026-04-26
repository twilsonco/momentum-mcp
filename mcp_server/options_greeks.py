import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Any

# ---------------------------------------------------------------------------
# Black-Scholes Primitives
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Return (d1, d2) for Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float | None:
    """Black-Scholes put price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return None
    disc = math.exp(-r * T)
    return float(K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float | None:
    """Black-Scholes call price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return None
    disc = math.exp(-r * T)
    return float(S * norm.cdf(d1) - K * disc * norm.cdf(d2))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_put: bool = True) -> float | None:
    """Delta — put returns [-1, 0], call returns [0, 1]."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return None
    if is_put:
        return float(norm.cdf(d1) - 1.0)
    return float(norm.cdf(d1))


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, is_put: bool = True) -> float | None:
    """Daily theta in $ per share. Negative = you earn (as a seller)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return None
    disc = math.exp(-r * T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if is_put:
        # Put theta includes carry benefit
        annual_theta = term1 + r * K * disc * norm.cdf(-d2)
    else:
        annual_theta = term1 - r * K * disc * norm.cdf(d2)
    return float(annual_theta / 365)


# ---------------------------------------------------------------------------
# Composite Realized Volatility (4-model ensemble)
# ---------------------------------------------------------------------------

def cc_vol(df: pd.DataFrame, period: int = 30) -> float:
    """Close-to-Close — baseline, least efficient."""
    closes = df["Close"].tail(period + 1)
    if len(closes) < 2:
        return 0.0
    log_ret = np.log(closes / closes.shift(1)).dropna()
    return float(log_ret.std() * math.sqrt(252))


def parkinson_vol(df: pd.DataFrame, period: int = 30) -> float:
    """Parkinson (1980) — uses High-Low range. ~5x more efficient."""
    d = df.tail(period)
    if len(d) < 2:
        return 0.0
    hl = np.log(d["High"] / d["Low"])
    variance = (1 / (4 * len(d) * math.log(2))) * (hl**2).sum()
    return float(math.sqrt(max(variance, 0) * 252))


def garman_klass_vol(df: pd.DataFrame, period: int = 30) -> float:
    """Garman-Klass (1980) — OHLC. Max efficiency for classical bars."""
    d = df.tail(period)
    if len(d) < 2:
        return 0.0
    hl2 = np.log(d["High"] / d["Low"])**2
    co2 = np.log(d["Close"] / d["Open"])**2
    variance = (1 / len(d)) * (0.5 * hl2 - (2 * math.log(2) - 1) * co2).sum()
    return float(math.sqrt(max(variance, 0) * 252))


def rogers_satchell_vol(df: pd.DataFrame, period: int = 30) -> float:
    """Rogers-Satchell (1991) — drift-unbiased, great for trending stocks."""
    d = df.tail(period)
    if len(d) < 2:
        return 0.0
    hc = np.log(d["High"] / d["Close"])
    ho = np.log(d["High"] / d["Open"])
    lc = np.log(d["Low"] / d["Close"])
    lo = np.log(d["Low"] / d["Open"])
    variance = (1 / len(d)) * (hc * ho + lc * lo).sum()
    return float(math.sqrt(max(variance, 0) * 252))


def composite_rv(df: pd.DataFrame, period: int = 30) -> float:
    """Weighted blend: CC=0.15, Parkinson=0.25, GK=0.35, RS=0.25."""
    weights = [0.15, 0.25, 0.35, 0.25]
    fns = [cc_vol, parkinson_vol, garman_klass_vol, rogers_satchell_vol]
    valid = [(w, fn(df, period)) for w, fn in zip(weights, fns)]
    valid = [(w, v) for w, v in valid if v > 0 and math.isfinite(v)]
    if not valid:
        return 0.0
    total_w = sum(w for w, _ in valid)
    return sum(w * v for w, v in valid) / total_w

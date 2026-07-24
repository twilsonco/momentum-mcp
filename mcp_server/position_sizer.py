"""
Position sizing module — Fixed Fractional, ATR-based, and Kelly Criterion.

Answers: "How many shares/contracts should I buy?"
Integrates with MetaTrader MCP server (if configured) to fetch contract sizes
and symbol information for more accurate Forex/CFD position sizing.
"""
from __future__ import annotations
import asyncio
import logging
import math
from typing import Any

from mcp_server.schema import SignalResult
from mcp_server.data import get_live_price
from mcp_server.technicals import analyze_technicals
from mcp_server.metatrader_client import get_symbol_contract_size_from_mt5

logger = logging.getLogger(__name__)

async def calculate_position_size(
    ticker: str,
    account_size: float,
    risk_pct: float = 1.0,          # % of account to risk per trade
    entry_price: float | None = None,
    stop_price: float | None = None,
    max_position_pct: float = 10.0, # max % of account in one position
    max_sector_pct: float = 25.0,   # max % of account in one sector (placeholder)
    method: str = "fixed_fractional", # or "atr" or "kelly"
    win_rate: float | None = 0.5,  # for Kelly default
    avg_win: float | None = 2.0,   # for Kelly default
    avg_loss: float | None = 1.0,  # for Kelly default
    mt5_symbol: str | None = None,  # MetaTrader symbol (e.g., "EURUSD") if different from ticker
) -> SignalResult:
    """Calculate risk-based position size using Fixed Fractional, ATR, or Kelly methods.
    Answers 'how many shares/contracts should I buy?' given account size and risk tolerance.
    
    If mt5_symbol is provided and MetaTrader MCP server is configured, fetches contract size
    and symbol info from MetaTrader for more accurate Forex/CFD position sizing.
    
    Args:
        ticker: Stock ticker or symbol name (e.g., "AAPL", "EURUSD").
        account_size: Total account size in account currency.
        risk_pct: Percentage of account to risk per trade (default 1%).
        entry_price: Entry price (fetched live if None).
        stop_price: Stop loss price (calculated from ATR if None).
        max_position_pct: Maximum position size as % of account (default 10%).
        max_sector_pct: Maximum sector exposure (placeholder, default 25%).
        method: Position sizing method - "fixed_fractional", "atr", or "kelly" (default fixed_fractional).
        win_rate: Win rate for Kelly criterion (default 0.5).
        avg_win: Average win size for Kelly (default 2.0).
        avg_loss: Average loss size for Kelly (default 1.0).
        mt5_symbol: MetaTrader symbol name if different from ticker (enables MT5 integration).
        
    Returns:
        SignalResult with position size recommendation and detailed metrics.
    """
    try:
        ticker = ticker.strip().upper()
        mt5_symbol_to_use = (mt5_symbol or ticker).strip().upper()
        
        # Fetch MetaTrader contract size if symbol is provided and MT5 is configured
        contract_size = None
        if mt5_symbol is not None:  # Only fetch if explicitly provided
            try:
                contract_size = await get_symbol_contract_size_from_mt5(mt5_symbol_to_use)
            except Exception as e:
                logger.warning(f"Could not fetch contract size from MT5 for {mt5_symbol_to_use}: {e}")
        
        # 1. Fetch live price if entry_price is None
        if entry_price is None:
            try:
                entry_price = await get_live_price(ticker)
            except Exception as e:
                return SignalResult.error_msg(f"Could not fetch live price for {ticker}: {e}")
        
        # 2. Get ATR(14) if needed (for stop_price or atr method)
        atr_14 = None
        if stop_price is None or method == "atr":
            tech_res = await analyze_technicals(ticker)
            if tech_res.status == "success":
                atr_14 = tech_res.data.get("atr_14")
            
        if stop_price is None:
            if atr_14:
                stop_price = entry_price - (atr_14 * 2)
            else:
                # Fallback stop if ATR fails: 5% below entry
                stop_price = entry_price * 0.95
                logger.warning(f"ATR unavailable for {ticker}, using 5% fallback stop.")

        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return SignalResult.error_msg(f"Invalid setup for {ticker}: Entry ({entry_price}) <= Stop ({stop_price})")

        risk_amount = account_size * (risk_pct / 100)
        
        # 3. Compute methods
        # Fixed Fractional
        ff_shares = int(risk_amount / risk_per_share)
        ff_value = ff_shares * entry_price
        
        # ATR-based
        if atr_14:
            atr_shares = int(risk_amount / (atr_14 * 2))
            atr_value = atr_shares * entry_price
        else:
            atr_shares = ff_shares
            atr_value = ff_value
            
        # Kelly
        kelly_fraction = None
        kelly_shares = 0
        kelly_value = 0
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            # Kelly % = (bp - q) / b  where b = odds (avg_win/avg_loss), p = win_rate, q = loss_rate
            b = avg_win / avg_loss if avg_loss != 0 else 1.0
            p = win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b if b != 0 else 0
            if kelly_fraction > 0:
                kelly_value = account_size * kelly_fraction
                kelly_shares = int(kelly_value / entry_price)
            else:
                kelly_fraction = 0
                kelly_value = 0
                kelly_shares = 0

        # Select primary method
        if method == "atr":
            shares = atr_shares
            position_value = atr_value
        elif method == "kelly":
            shares = kelly_shares
            position_value = kelly_value
        else:
            shares = ff_shares
            position_value = ff_value

        # 6. Apply constraints
        max_position_value = account_size * (max_position_pct / 100)
        max_shares = int(max_position_value / entry_price)
        
        constraints_applied = False
        if shares > max_shares:
            shares = max_shares
            position_value = shares * entry_price
            constraints_applied = True
        
        # Adjust to contract size if available from MetaTrader
        mt5_contract_adjustment = False
        if contract_size and contract_size > 0:
            # Validate contract_size is reasonable (not too small)
            if contract_size < 0.001:
                logger.warning(f"Contract size {contract_size} for {ticker} seems too small, skipping adjustment")
            else:
                # Round shares to nearest contract multiple (for Forex/CFD trading)
                # For both integer contract sizes (e.g., 100) and fractional ones (e.g., 0.001)
                num_contracts = round(shares / contract_size)
                if num_contracts <= 0:
                    num_contracts = 1  # Minimum 1 contract
                # Use round() instead of int() to avoid truncation of fractional contract sizes
                contract_adjusted_shares = round(num_contracts * contract_size)
                # Ensure the result is a positive integer
                contract_adjusted_shares = max(1, int(contract_adjusted_shares) if contract_adjusted_shares >= 1 else 1)
                
                if contract_adjusted_shares != shares:
                    mt5_contract_adjustment = True
                    shares = contract_adjusted_shares
                    position_value = shares * entry_price

        # Summary
        summary = (
            f"Position size for {ticker} @ ${entry_price:.2f}:\n"
            f"- Recommended: {shares} shares (~${position_value:,.2f})\n"
            f"- Method: {method.replace('_', ' ').title()}\n"
            f"- Risk Amount: ${risk_amount:,.2f} ({risk_pct}% of account)\n"
            f"- Stop Loss: ${stop_price:.2f} ({abs(entry_price-stop_price)/entry_price*100:.1f}% risk per share)"
        )
        if constraints_applied:
            summary += f"\n- ⚠️ Capped by {max_position_pct}% max position constraint."
        if mt5_contract_adjustment:
            summary += f"\n- ✓ Adjusted to {contract_size} contract size (MetaTrader)."

        data = {
            "ticker": ticker,
            "entry_price": round(entry_price, 2),
            "stop_price": round(stop_price, 2),
            "risk_per_share": round(risk_per_share, 2),
            "method": method,
            "account_size": account_size,
            "risk_pct": risk_pct,
            "risk_amount": round(risk_amount, 2),
            "recommended_shares": shares,
            "recommended_contracts": shares // 100,
            "position_value": round(position_value, 2),
            "position_pct_of_account": round((position_value / account_size) * 100, 2),
            "constraints": {
                "max_position_shares": max_shares,
                "max_position_binding": constraints_applied,
                "max_sector_binding": False,
            },
            "metatrader": {
                "mt5_symbol": mt5_symbol_to_use if mt5_symbol is not None else None,
                "contract_size": round(contract_size, 4) if contract_size else None,
                "contract_adjusted": mt5_contract_adjustment,
            },
            "methods_compared": {
                "fixed_fractional": {"shares": ff_shares, "value": round(ff_value, 2)},
                "atr_based": {"shares": atr_shares, "value": round(atr_value, 2), "atr_14": round(atr_14, 4) if atr_14 else None},
                "kelly": {"shares": kelly_shares, "value": round(kelly_value, 2), "kelly_fraction": round(kelly_fraction, 4)} if kelly_fraction is not None else None,
            },
            "summary": summary,
        }
        return SignalResult.success(data)

    except Exception as e:
        logger.error(f"Position sizing failed for {ticker}: {e}")
        return SignalResult.error_msg(str(e))

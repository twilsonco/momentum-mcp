"""
TraderDaddy Pro Agent API Client

Async HTTP client for the TraderDaddy Agent API endpoints.
Handles JWT authentication, token refresh, and rate-limit retries.

All routes live under /api/agent/ and require superuser JWT.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

from mcp_server.cache import smart_cache
from mcp_server.schema import SignalResult

# ---------------------------------------------------------------------------
# Singleton HTTP Client
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return a shared httpx client (created once, reused across calls)."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        logger.info("TraderDaddy: singleton HTTP client created")
    return _http_client


# ---------------------------------------------------------------------------
# Circuit Breaker (3-state: CLOSED / OPEN / HALF_OPEN)
#
# CLOSED:    Normal operation. Failures increment counter.
# OPEN:      All requests blocked. Backoff timer running.
# HALF_OPEN: After backoff expires, allow ONE probe request through.
#            If success → CLOSED. If fail → OPEN with doubled backoff.
# ---------------------------------------------------------------------------

_circuit = {
    "state": "CLOSED",          # CLOSED | OPEN | HALF_OPEN
    "failures": 0,
    "last_failure": 0.0,
    "open_until": 0.0,
    "backoff_multiplier": 1,
}
CIRCUIT_FAIL_THRESHOLD = 3      # failures before opening
CIRCUIT_BASE_DURATION = 60      # seconds (first backoff)
CIRCUIT_MAX_DURATION = 600      # 10 minutes max


def _circuit_check() -> str | None:
    """Return error message if circuit is OPEN, None if request can proceed.

    When the backoff timer expires, transitions to HALF_OPEN to allow
    one probe request through.
    """
    now = time.time()

    if _circuit["state"] == "CLOSED":
        return None

    if _circuit["state"] == "HALF_OPEN":
        # Already in half-open — one probe is in flight
        return None

    # state == "OPEN"
    if now >= _circuit["open_until"]:
        # Backoff expired → transition to HALF_OPEN, let one request through
        _circuit["state"] = "HALF_OPEN"
        logger.info("TraderDaddy: circuit breaker → HALF_OPEN (testing with probe request)")
        return None

    remaining = int(_circuit["open_until"] - now)
    return (
        f"TraderDaddy API is temporarily unavailable "
        f"(circuit OPEN, auto-retry in {remaining}s). Sam's got the rest covered."
    )


def _circuit_record_failure():
    """Record a failure. Opens circuit with exponential backoff if threshold reached."""
    now = time.time()

    if _circuit["state"] == "HALF_OPEN":
        # Probe failed → go back to OPEN with DOUBLED backoff
        _circuit["backoff_multiplier"] = min(_circuit["backoff_multiplier"] * 2, 16)
        duration = min(CIRCUIT_BASE_DURATION * _circuit["backoff_multiplier"], CIRCUIT_MAX_DURATION)
        _circuit["state"] = "OPEN"
        _circuit["open_until"] = now + duration
        _circuit["last_failure"] = now
        logger.warning(
            "TraderDaddy: HALF_OPEN probe FAILED → OPEN for %ds (multiplier=%d)",
            duration, _circuit["backoff_multiplier"],
        )
        return

    # CLOSED state — count failures
    _circuit["failures"] += 1
    _circuit["last_failure"] = now

    if _circuit["failures"] >= CIRCUIT_FAIL_THRESHOLD:
        duration = min(CIRCUIT_BASE_DURATION * _circuit["backoff_multiplier"], CIRCUIT_MAX_DURATION)
        _circuit["state"] = "OPEN"
        _circuit["open_until"] = now + duration
        logger.warning(
            "TraderDaddy: circuit breaker CLOSED → OPEN for %ds (%d failures, multiplier=%d)",
            duration, _circuit["failures"], _circuit["backoff_multiplier"],
        )


def _circuit_record_success():
    """Record a success. Closes the circuit and decays the backoff multiplier."""
    if _circuit["state"] == "HALF_OPEN":
        # Probe succeeded → fully close the circuit
        _circuit["state"] = "CLOSED"
        _circuit["failures"] = 0
        _circuit["backoff_multiplier"] = max(1, _circuit["backoff_multiplier"] // 2)
        logger.info(
            "TraderDaddy: HALF_OPEN probe succeeded → CLOSED (multiplier=%d)",
            _circuit["backoff_multiplier"],
        )
        return

    # Normal success in CLOSED state — decay counters
    if _circuit["failures"] > 0:
        _circuit["failures"] -= 1
    if _circuit["backoff_multiplier"] > 1:
        _circuit["backoff_multiplier"] = max(1, _circuit["backoff_multiplier"] // 2)


# ---------------------------------------------------------------------------
# JWT Token Manager
# ---------------------------------------------------------------------------

_token_cache: dict[str, Any] = {
    "access_token": None,
    "refresh_token": None,
    "expires_at": 0,
}


async def _login() -> str:
    """Authenticate and cache JWT tokens."""
    client = _get_client()
    base = os.getenv("TRADERDADDY_API_URL", "").rstrip("/")
    email = os.getenv("TRADERDADDY_EMAIL", "")
    password = os.getenv("TRADERDADDY_PASSWORD", "")

    if not all([base, email, password]):
        raise RuntimeError(
            "TraderDaddy credentials not configured. "
            "Set TRADERDADDY_API_URL, TRADERDADDY_EMAIL, and TRADERDADDY_PASSWORD in .env"
        )

    resp = await client.post(
        f"{base}/api/auth/login",
        json={"email": email, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    _token_cache["access_token"] = data["access_token"]
    _token_cache["refresh_token"] = data.get("refresh_token")
    # JWT exp is 30 min — refresh at 25 min to be safe
    _token_cache["expires_at"] = time.time() + 25 * 60

    logger.info("TraderDaddy: authenticated successfully")
    return _token_cache["access_token"]


async def _get_token() -> str:
    """Return a valid access token, refreshing if needed."""
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["access_token"]
    return await _login()


async def _agent_get(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    retry_on_429: bool = True,
) -> SignalResult:
    """Make an authenticated GET to /api/agent/<path>."""
    # Circuit breaker check
    circuit_err = _circuit_check()
    if circuit_err:
        return SignalResult.error_msg(circuit_err)

    base = os.getenv("TRADERDADDY_API_URL", "").rstrip("/")
    if not base:
        return SignalResult.error_msg("TRADERDADDY_API_URL not set in .env")

    client = _get_client()
    try:
        token = await _get_token()
        url = f"{base}/api/agent/{path.lstrip('/')}"

        resp = await client.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Token expired mid-session — re-auth once
        if resp.status_code == 401:
            logger.warning("TraderDaddy: 401 — re-authenticating")
            token = await _login()
            resp = await client.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )

        # Rate limited — wait and retry once
        if resp.status_code == 429 and retry_on_429:
            retry_after = resp.json().get("retryAfter", 60)
            wait = min(retry_after, 65)  # cap at ~1 min
            logger.warning("TraderDaddy: rate limited, waiting %ds", wait)
            await asyncio.sleep(wait)
            return await _agent_get(path, params, retry_on_429=False)

        resp.raise_for_status()
        _circuit_record_success()
        return SignalResult.success(resp.json())

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        _circuit_record_failure()
        logger.error("TraderDaddy API error: %s", exc)
        return SignalResult.error_msg(f"TraderDaddy API unreachable: {exc}")


# ---------------------------------------------------------------------------
# Tool Functions (8 endpoints)
# ---------------------------------------------------------------------------


@smart_cache(open_ttl=120, closed_ttl=1800)  # 2min open, 30min closed
async def get_market_pulse() -> SignalResult:
    """
    Get the current market sentiment pulse.

    Returns an AI-generated summary sentence, weighted sentiment score (-7 to +7),
    call/put premium stats, and sector breakdown.
    """
    return await _agent_get("market-pulse")


@smart_cache(open_ttl=120, closed_ttl=1800)
async def get_market_stats() -> SignalResult:
    """
    Get market-wide statistics: put/call ratios, sentiment indicators,
    largest unusual trade in 24h.
    """
    return await _agent_get("market-stats")


@smart_cache(open_ttl=120, closed_ttl=1800)
async def get_put_call_ratios(ticker: str | None = None) -> SignalResult:
    """
    Get put/call ratios for SPY, QQQ, and IWM — or for a specific ticker.
    """
    path = f"put-call-ratios/{ticker.upper()}" if ticker else "put-call-ratios"
    return await _agent_get(path)


@smart_cache(open_ttl=180, closed_ttl=3600)  # 3min open, 1hr closed
async def get_sector_flow(window: str = "1d") -> SignalResult:
    """
    Get sector-by-sector options flow with sentiment scores.

    Args:
        window: Time window — '1h', '4h', '1d', or '1w'. Default: '1d'.
    """
    return await _agent_get("sectors/flow", {"window": window})


async def get_unusual_activity(
    ticker: str | None = None,
    sentiment: str = "all",
    type: str = "all",
    time_frame: str = "today",
    min_premium: int = 20000,
    min_score: int = 70,
    page_size: int = 25,
) -> SignalResult:
    """
    Get unusual options activity feed — the core institutional flow data.

    Args:
        ticker: Filter by ticker symbol (e.g. 'NVDA'). Default: all tickers.
        sentiment: 'bullish', 'bearish', or 'all'. Default: 'all'.
        type: 'call', 'put', or 'all'. Default: 'all'.
        time_frame: 'hour', 'today', 'yesterday', '3days', 'week', 'month'. Default: 'today'.
        min_premium: Minimum premium in dollars. Default: 20000.
        min_score: Minimum unusual score (0-100). Default: 70.
        page_size: Results per page (max 500). Default: 25.
    """
    params: dict[str, Any] = {
        "sentiment": sentiment,
        "type": type,
        "timeFrame": time_frame,
        "minPremium": min_premium,
        "minScore": min_score,
        "pageSize": page_size,
    }
    if ticker:
        params["ticker"] = ticker.upper()
    return await _agent_get("unusual-activity", params)


async def get_signals(
    signal_type: str = "all",
    ticker: str | None = None,
    timeframe: str = "daily",
    page_size: int = 25,
) -> SignalResult:
    """
    Get breakout and continuation signals with technical indicator data.

    Args:
        signal_type: 'breakout', 'continuation', or 'all'. Default: 'all'.
        ticker: Filter by ticker. Default: all.
        timeframe: 'daily' or 'weekly'. Default: 'daily'.
        page_size: Results per page. Default: 25.
    """
    params: dict[str, Any] = {
        "signalType": signal_type,
        "timeframe": timeframe,
        "pageSize": page_size,
    }
    if ticker:
        params["ticker"] = ticker.upper()
    return await _agent_get("signals", params)


async def get_politician_trades(
    ticker: str | None = None,
    party: str | None = None,
    trade_type: str | None = None,
    days: int = 90,
    limit: int = 25,
) -> SignalResult:
    """
    Get congressional stock trading disclosures.

    Args:
        ticker: Filter by stock ticker. Default: all.
        party: 'Democrat', 'Republican', or None for all. Default: all.
        trade_type: 'buy', 'sell', or None for all. Default: all.
        days: Number of days to look back. Default: 90.
        limit: Results per page. Default: 25.
    """
    params: dict[str, Any] = {"days": days, "limit": limit}
    if ticker:
        params["ticker"] = ticker.upper()
    if party:
        params["party"] = party
    if trade_type:
        params["trade_type"] = trade_type
    return await _agent_get("politician-trades", params)


@smart_cache(open_ttl=300, closed_ttl=3600)  # 5min — GEX updates less frequently
async def get_gex_overview() -> SignalResult:
    """
    Get Gamma Exposure (GEX) market overview for SPY, QQQ, and IWM.

    Returns total gamma, net gamma by strike, market bias (LONG_GAMMA/SHORT_GAMMA/NEUTRAL),
    and interpretation of each symbol's gamma profile.
    """
    return await _agent_get("gex/market/overview")


@smart_cache(open_ttl=3600, closed_ttl=86400)  # 1hr open, 24hr closed — rarely changes
async def get_earnings_calendar(week: str = "current") -> SignalResult:
    """
    Get the weekly earnings calendar — which companies report this week.

    Args:
        week: 'current' or 'next'. Default: 'current'.
    """
    params: dict[str, Any] = {}
    if week == "next":
        params["week"] = "next"
    return await _agent_get("earnings", params)


async def get_earnings_flow(days: int = 7) -> SignalResult:
    """
    Get pre-earnings options flow analysis — institutional positioning ahead of earnings.

    Shows which upcoming earnings have bullish vs bearish flow, total premium,
    and consensus sentiment.

    Args:
        days: Days ahead to look. Default: 7. Max: 30.
    """
    return await _agent_get("earnings-flow/flow", {"days": min(days, 30)})


# ---------------------------------------------------------------------------
# NEW Endpoints — Alerts, Flow, Sectors, GEX, Signals, Politicians, Earnings
# ---------------------------------------------------------------------------

# ── Alerts ────────────────────────────────────────────────────────────────────

async def get_ai_alerts(
    alert_type: str | None = None,
    severity: str | None = None,
    limit: int = 15,
) -> SignalResult:
    """
    Get AI-generated trading alerts — unusual flow, GEX breaches, sector
    shifts, and breakouts.  Filters by type and/or severity.

    Args:
        alert_type: Filter by alert type (use get_alert_types for values).
        severity: Minimum severity — 'low', 'medium', 'high', 'critical'.
        limit: Max alerts to return (max 50). Default: 15.
    """
    params: dict[str, Any] = {"limit": min(limit, 50)}
    if alert_type:
        params["type"] = alert_type
    if severity:
        params["severity"] = severity
    result = await _agent_get("alerts", params)

    if result.status == "error":
        return result

    data = result.data
    # Trim: keep only essential fields per alert
    if isinstance(data, list):
        result.data = [_trim_alert(a) for a in data[:limit]]
    elif isinstance(data, dict) and "alerts" in data:
        result.data["alerts"] = [_trim_alert(a) for a in data["alerts"][:limit]]
    return result


def _trim_alert(alert: dict) -> dict:
    """Keep only token-efficient fields from an alert."""
    keys = ["type", "severity", "ticker", "message", "title",
            "description", "score", "timestamp", "direction"]
    return {k: alert[k] for k in keys if k in alert}


@smart_cache(open_ttl=120, closed_ttl=1800)
async def get_alerts_summary() -> SignalResult:
    """
    Lightweight alert dashboard: counts by type and severity,
    plus top 5 highest-priority alerts. Use this for quick overviews
    instead of the full alerts list.
    """
    return await _agent_get("alerts/summary")


async def get_alert_types() -> SignalResult:
    """Get all valid alert types with descriptions (internal/schema discovery)."""
    return await _agent_get("alerts/types")


async def refresh_alerts() -> SignalResult:
    """Force-rerun alert generation (admin). Returns new count + timestamp."""
    return await _agent_post("alerts/refresh")


async def get_options_listings() -> SignalResult:
    """CBOE new options contract listings — newly listable equity options."""
    return await _agent_get("alerts/options-listings")


# ── Flow Summary ──────────────────────────────────────────────────────────────

@smart_cache(open_ttl=120, closed_ttl=1800)
async def get_flow_summary() -> SignalResult:
    """
    Per-ticker aggregated flow: bullish vs bearish premium totals,
    trade counts, and net sentiment. Ranked by activity.

    Use this to find WHICH tickers have the heaviest institutional flow.
    For individual trade details, use get_ticker_flow(ticker) instead.
    """
    result = await _agent_get("flow/summary")
    if result.status == "error":
        return result

    data = result.data
    # Trim: keep top 15 tickers, essential fields only
    if isinstance(data, list):
        result.data = [_trim_flow_ticker(t) for t in data[:15]]
    elif isinstance(data, dict) and "tickers" in data:
        result.data["tickers"] = [_trim_flow_ticker(t) for t in data["tickers"][:15]]
    return result


def _trim_flow_ticker(t: dict) -> dict:
    """Keep essential flow summary fields per ticker."""
    keys = ["ticker", "bullish_premium", "bearish_premium", "net_premium",
            "bullish_count", "bearish_count", "net_sentiment", "total_trades",
            "total_premium", "score"]
    return {k: t[k] for k in keys if k in t}


async def get_ticker_flow(ticker: str, limit: int = 10) -> SignalResult:
    """
    Full trade list for a single ticker — each flow with strike, expiry,
    premium, sentiment, and score. More granular than the summary.

    Args:
        ticker: Stock ticker symbol (e.g. NVDA).
        limit: Max trades to return. Default: 10.
    """
    result = await _agent_get(f"flow/ticker/{ticker.upper()}")
    if result.status == "error":
        return result

    data = result.data
    # Trim: cap trades and keep essential fields
    if isinstance(data, list):
        result.data = [_trim_flow_trade(t) for t in data[:limit]]
    elif isinstance(data, dict) and "trades" in data:
        result.data["trades"] = [_trim_flow_trade(t) for t in data["trades"][:limit]]
    return result


def _trim_flow_trade(t: dict) -> dict:
    """Keep essential fields per individual trade."""
    keys = ["ticker", "type", "strike", "expiry", "dte", "premium",
            "sentiment", "score", "volume", "open_interest", "side"]
    return {k: t[k] for k in keys if k in t}


# ── Enhanced Sectors ──────────────────────────────────────────────────────────

@smart_cache(open_ttl=180, closed_ttl=3600)
async def get_sector_heatmap() -> SignalResult:
    """
    Heatmap-ready sector strength data based on net options flow.
    Shows relative rotation between sector ETFs (XLK, XLF, XLE, etc.).
    """
    return await _agent_get("sectors/heatmap")


@smart_cache(open_ttl=300, closed_ttl=3600)
async def get_sector_analysis() -> SignalResult:
    """
    Narrative sector analysis: leading/lagging sectors, momentum shifts,
    and divergences. Use this when user asks 'which sectors are hot?'
    or 'where is money rotating?'
    """
    return await _agent_get("sectors/analysis")


# ── Enhanced GEX ──────────────────────────────────────────────────────────────

async def get_ticker_gex(symbol: str) -> SignalResult:
    """
    Gamma Exposure (GEX) levels for a specific symbol — key support/resistance
    levels based on market maker gamma positioning.

    Args:
        symbol: Ticker symbol (e.g. SPX, AAPL, TSLA).
    """
    result = await _agent_get(f"gex/{symbol.upper()}")
    if result.status == "error":
        return result

    data = result.data
    # Trim: keep regime + key levels only
    if isinstance(data, dict):
        keys = ["symbol", "regime", "net_gamma", "flip_level",
                "key_levels", "call_wall", "put_wall", "max_gamma_strike",
                "bias", "interpretation", "timestamp"]
        trimmed = {k: data[k] for k in keys if k in data}
        # If key_levels is a long list, keep top 5
        if "key_levels" in trimmed and isinstance(trimmed["key_levels"], list):
            trimmed["key_levels"] = trimmed["key_levels"][:5]
        result.data = trimmed
    return result


async def get_ticker_gex_historical(symbol: str) -> SignalResult:
    """
    Historical GEX data for a symbol (internal — for backtesting/analysis).

    Args:
        symbol: Ticker symbol.
    """
    return await _agent_get(f"gex/{symbol.upper()}/historical")


# ── Enhanced Signals ──────────────────────────────────────────────────────────

async def get_signal_stats() -> SignalResult:
    """Signal detection statistics (internal — counts and distributions)."""
    return await _agent_get("signals/stats")


async def get_signal_detail(signal_id: str) -> SignalResult:
    """Get a single signal's full detail (internal).

    Args:
        signal_id: Signal ID.
    """
    return await _agent_get(f"signals/{signal_id}")


async def get_signal_chart(signal_id: str) -> SignalResult:
    """Get chart data for a specific signal (internal).

    Args:
        signal_id: Signal ID.
    """
    return await _agent_get(f"signals/{signal_id}/chart-data")


# ── Enhanced Politicians ──────────────────────────────────────────────────────

async def get_politician_stats() -> SignalResult:
    """Summary statistics for politician trading activity (internal)."""
    return await _agent_get("politician-trades/stats")


async def get_politician_trades_by_ticker(ticker: str) -> SignalResult:
    """
    Congressional stock trades filtered by ticker — see if politicians
    are buying or selling a specific stock.

    Args:
        ticker: Stock ticker symbol (e.g. NVDA).
    """
    result = await _agent_get(f"politician-trades/by-ticker/{ticker.upper()}")
    if result.status == "error":
        return result

    data = result.data
    # Trim: keep top 10 trades, essential fields
    if isinstance(data, list):
        result.data = [_trim_politician(t) for t in data[:10]]
    elif isinstance(data, dict) and "trades" in data:
        result.data["trades"] = [_trim_politician(t) for t in data["trades"][:10]]
    return result


def _trim_politician(t: dict) -> dict:
    """Keep essential politician trade fields."""
    keys = ["politician", "party", "ticker", "type", "amount",
            "transaction_date", "disclosure_date", "chamber"]
    return {k: t[k] for k in keys if k in t}


# ── Enhanced Earnings ─────────────────────────────────────────────────────────

async def get_ticker_earnings_flow(symbol: str) -> SignalResult:
    """
    Pre-earnings options positioning for a single symbol — shows institutional
    bets placed ahead of earnings (bullish vs bearish flow, premium, sentiment).

    Args:
        symbol: Ticker symbol (e.g. NVDA).
    """
    result = await _agent_get(f"earnings-flow/flow/{symbol.upper()}")
    if result.status == "error":
        return result

    data = result.data
    # Trim: keep top 10 flows, essential fields
    if isinstance(data, list):
        result.data = [_trim_flow_trade(t) for t in data[:10]]
    elif isinstance(data, dict) and "flows" in data:
        result.data["flows"] = [_trim_flow_trade(t) for t in data["flows"][:10]]
    return result


# ── Institutional ─────────────────────────────────────────────────────────────

@smart_cache(open_ttl=300, closed_ttl=3600)
async def get_most_institutional() -> SignalResult:
    """
    Top 10 most actively traded tickers by institutional flow count.
    Excludes SPY, QQQ, GLD, SLV, and MAG7 to surface non-obvious activity.
    Use when user asks 'what are institutions trading?' or 'smart money activity'.
    """
    return await _agent_get("most-institutionally-traded-tickers/aggregated")


# ── Combined Ticker Endpoint ──────────────────────────────────────────────────

async def get_ticker_data(symbol: str) -> SignalResult:
    """
    Combined ticker data bundle — options chain + technicals + GEX in one call.

    This mirrors TraderDaddy's /api/agent/ticker/{symbol} endpoint used by
    the TraderLady chat widget. Reduces round-trips when Sam needs a full
    picture of a ticker.

    Args:
        symbol: Ticker symbol (e.g. AAPL, TSLA).
    """
    result = await _agent_get(f"ticker/{symbol.upper()}")
    if result.status == "error":
        return result

    data = result.data
    # Trim: the combined endpoint returns a LOT of data.
    # Keep the essential fields to stay within token budget.
    if isinstance(data, dict):
        # Cap strikes list if present
        for key in ["topStrikes", "top_strikes", "strikes"]:
            if key in data and isinstance(data[key], list):
                data[key] = data[key][:10]
        # Cap chain data if present
        for key in ["chain", "options_chain", "optionsChain"]:
            if key in data and isinstance(data[key], list):
                data[key] = data[key][:20]
    return result


# ── Admin: Market Stats Refresh ───────────────────────────────────────────────

async def refresh_market_stats() -> SignalResult:
    """Force-invalidate and recompute market stats cache (admin)."""
    return await _agent_get("market-stats/refresh")


# ---------------------------------------------------------------------------
# POST helper (for refresh/admin endpoints)
# ---------------------------------------------------------------------------

async def _agent_post(
    path: str,
    data: dict[str, Any] | None = None,
) -> SignalResult:
    """Make an authenticated POST to /api/agent/<path>."""
    circuit_err = _circuit_check()
    if circuit_err:
        return SignalResult.error_msg(circuit_err)

    base = os.getenv("TRADERDADDY_API_URL", "").rstrip("/")
    if not base:
        return SignalResult.error_msg("TRADERDADDY_API_URL not set in .env")

    client = _get_client()
    try:
        token = await _get_token()
        url = f"{base}/api/agent/{path.lstrip('/')}"

        resp = await client.post(
            url,
            json=data or {},
            headers={"Authorization": f"Bearer {token}"},
        )

        if resp.status_code == 401:
            token = await _login()
            resp = await client.post(
                url,
                json=data or {},
                headers={"Authorization": f"Bearer {token}"},
            )

        resp.raise_for_status()
        _circuit_record_success()
        return SignalResult.success(resp.json())

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        _circuit_record_failure()
        logger.error("TraderDaddy POST error: %s", exc)
        return SignalResult.error_msg(f"TraderDaddy API unreachable: {exc}")



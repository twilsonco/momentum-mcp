from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MT5_MCP_URL: str = os.getenv("MT5_MCP_URL", "").strip()


class MT5MCPClient:
    """Lazy singleton MCP client for the MetaTrader server.

    Keeps a persistent SSE connection alive across calls and reconnects
    automatically on failure.
    """

    def __init__(self) -> None:
        self._cm: Any = None
        self._session: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_connected(self) -> Any:
        """Return a live ClientSession, connecting if needed."""
        if self._session is not None:
            return self._session

        async with self._lock:
            if self._session is not None:
                return self._session

            from mcp import ClientSession
            from mcp.client.sse import sse_client

            logger.info("MT5 MCP: connecting to %s", MT5_MCP_URL)
            try:
                self._cm = sse_client(MT5_MCP_URL)
                read, write = await asyncio.wait_for(self._cm.__aenter__(), timeout=10.0)
                self._session = ClientSession(read, write)
                await asyncio.wait_for(self._session.__aenter__(), timeout=10.0)
                await asyncio.wait_for(self._session.initialize(), timeout=10.0)
                logger.info("MT5 MCP: connected and initialized")
                return self._session
            except asyncio.TimeoutError as exc:
                logger.error("MT5 MCP connection timeout (10s)")
                self._session = None
                raise RuntimeError("MT5 MCP connection timeout") from exc

    async def disconnect(self) -> None:
        """Tear down the current connection safely.

        Always resets internal state so the next call reconnects from scratch.
        """
        session = self._session
        cm = self._cm
        self._session = None
        self._cm = None

        if session is not None:
            try:
                await asyncio.wait_for(session.__aexit__(None, None, None), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("MT5 MCP: session.__aexit__ timed out (2s)")
            except RuntimeError as exc:
                if "cancel scope" in str(exc):
                    logger.debug("MT5 MCP: suppressed cancel scope error during session cleanup")
                else:
                    logger.warning("MT5 MCP: session cleanup RuntimeError: %s", exc)
            except Exception as exc:
                logger.warning("MT5 MCP: session cleanup error: %s", exc)

        if cm is not None:
            try:
                await asyncio.wait_for(cm.__aexit__(None, None, None), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("MT5 MCP: sse_client.__aexit__ timed out (2s)")
            except RuntimeError as exc:
                if "cancel scope" in str(exc):
                    logger.debug("MT5 MCP: suppressed cancel scope error during sse_client cleanup")
                else:
                    logger.warning("MT5 MCP: sse_client cleanup RuntimeError: %s", exc)
            except Exception as exc:
                logger.warning("MT5 MCP: sse_client cleanup error: %s", exc)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call a tool with one automatic reconnect on failure."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                session = await self._ensure_connected()
                return await asyncio.wait_for(session.call_tool(name, args), timeout=60.0)
            except asyncio.TimeoutError:
                last_exc = TimeoutError(f"MT5 MCP tool '{name}' timeout (60s)")
                logger.warning("MT5 MCP call '%s' timeout (attempt %d)", name, attempt + 1)
                await self.disconnect()
            except Exception as exc:
                last_exc = exc
                logger.warning("MT5 MCP call '%s' failed (attempt %d): %s", name, attempt + 1, exc)
                await self.disconnect()

        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        await self.disconnect()


_mt5_client: MT5MCPClient | None = None


def is_mt5_configured() -> bool:
    return bool(MT5_MCP_URL)


def get_mt5_client() -> MT5MCPClient:
    global _mt5_client
    if _mt5_client is None:
        _mt5_client = MT5MCPClient()
    return _mt5_client


async def disconnect_mt5_client() -> None:
    await get_mt5_client().disconnect()


async def call_mt5_tool(name: str, args: dict[str, Any], timeout: float = 60.0) -> Any:
    """Call an MT5 MCP tool and always close the MT5 connection afterward."""
    client = get_mt5_client()
    try:
        return await asyncio.wait_for(client.call_tool(name, args), timeout=timeout)
    finally:
        # Keep lifecycle fully encapsulated in the helper: one call in,
        # one connection cycle out.
        try:
            await client.disconnect()
        except Exception as exc:
            logger.debug("MT5 disconnect after '%s' failed: %s", name, exc)


async def fetch_mt5_tool_json(name: str, args: dict[str, Any], timeout: float = 60.0) -> dict[str, Any] | None:
    """Call an MT5 tool and parse the first JSON text content payload.

    Returns ``None`` on any failure (timeout, connection error, parse error,
    empty response). Callers can simply check for ``None`` without try/except.
    """
    try:
        result = await call_mt5_tool(name, args, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("MT5 MCP tool '%s' timeout (%.1fs)", name, timeout)
        return None
    except Exception as exc:
        logger.warning("MT5 MCP tool '%s' failed: %s", name, exc)
        return None

    if not result or not getattr(result, "content", None):
        return None

    for content in result.content:
        text = getattr(content, "text", "")
        if not text or not text.strip():
            continue
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.debug("%s returned non-JSON content", name)
    return None


async def fetch_mt5_account_info(timeout: float = 5.0) -> dict[str, Any] | None:
    """Fetch account info dict from MT5 MCP."""
    return await fetch_mt5_tool_json("get_account_info", {}, timeout=timeout)


async def fetch_mt5_symbol_info(symbol: str, timeout: float = 15.0) -> dict[str, Any] | None:
    """Fetch symbol info dict from MT5 MCP."""
    return await fetch_mt5_tool_json("get_symbol_info", {"symbol_name": symbol}, timeout=timeout)

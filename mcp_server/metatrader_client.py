"""
MetaTrader MCP Server client — optional integration for Forex/CFD symbol data.

If the MetaTrader MCP server is configured and running, this module provides
async wrappers to fetch symbol information (contract size, price, etc.) for
improved position sizing calculations across assets (Forex, commodities, etc.).

Configuration via environment variables:
- METATRADER_MCP_URL: Base URL to the MetaTrader MCP server (e.g., http://localhost:8080)
- METATRADER_MCP_ENABLED: Set to "true" to enable (default: auto-detect from URL env var)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Configuration
METATRADER_MCP_URL = os.getenv("METATRADER_MCP_URL", "").strip()
METATRADER_MCP_ENABLED = os.getenv("METATRADER_MCP_ENABLED", "true" if METATRADER_MCP_URL else "false").lower() == "true"


class MetaTraderMCPClient:
    """Async client to call MetaTrader MCP server tools via SSE transport.
    
    Supports both context manager usage (automatic session lifecycle management)
    and persistent session usage (for reuse across multiple calls).
    """
    
    # Class-level persistent session to avoid recreating connections
    _persistent_session: Optional[aiohttp.ClientSession] = None
    
    def __init__(self, base_url: str, use_persistent_session: bool = False):
        """Initialize the MCP client.
        
        Args:
            base_url: Base URL of the MetaTrader MCP server (e.g., http://localhost:8080)
            use_persistent_session: If True, use a persistent session shared across instances.
                                   If False, use a temporary session (for context manager use).
        """
        self.base_url = base_url.rstrip("/")
        self.use_persistent_session = use_persistent_session
        self.session: Optional[aiohttp.ClientSession] = None
        self._owns_session = False  # Track if this instance created the session
    
    async def __aenter__(self):
        """Context manager entry."""
        if not self.use_persistent_session:
            self.session = aiohttp.ClientSession()
            self._owns_session = True
        else:
            await self._ensure_persistent_session()
            self.session = self._persistent_session
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Only close session if we created it (not using persistent session)
        if self._owns_session and self.session:
            await self.session.close()
            self._owns_session = False
    
    @classmethod
    async def _ensure_persistent_session(cls):
        """Ensure the persistent session exists."""
        if cls._persistent_session is None or cls._persistent_session.closed:
            cls._persistent_session = aiohttp.ClientSession()
    
    @classmethod
    async def close_persistent_session(cls):
        """Close the persistent session. Call this during app shutdown."""
        if cls._persistent_session and not cls._persistent_session.closed:
            await cls._persistent_session.close()
            cls._persistent_session = None
    
    async def _call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool on the MetaTrader MCP server via JSON-RPC.
        
        Args:
            tool_name: Name of the tool to call.
            **kwargs: Tool parameters.
            
        Returns:
            The tool result.
            
        Raises:
            ValueError: If the tool call fails or server is not reachable.
        """
        if not self.session:
            raise ValueError("Session not initialized. Use async context manager: async with MetaTraderMCPClient(...) as client")
        
        # Build JSON-RPC request
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs,
            },
            "id": 1,
        }
        
        try:
            # Try to call the tool via the MCP server's SSE endpoint
            url = f"{self.base_url}/mcp/sse"
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    raise ValueError(f"MetaTrader MCP server returned status {resp.status}")
                data = await resp.json()
                
                # Check for errors in JSON-RPC response
                if "error" in data:
                    raise ValueError(f"MetaTrader MCP tool error: {data['error']}")
                
                # Extract result
                if "result" in data:
                    return data["result"]
                raise ValueError(f"No result in MetaTrader MCP response: {data}")
        
        except asyncio.TimeoutError:
            raise ValueError(f"MetaTrader MCP server timeout (no response within 10s)")
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to connect to MetaTrader MCP server at {self.base_url}: {e}")
        except Exception as e:
            raise ValueError(f"MetaTrader MCP call failed: {e}")
    
    async def get_symbol_contract_size(self, symbol: str) -> float:
        """Get the contract size (trade size unit) for a symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD", "AAPL").
            
        Returns:
            Contract size as a float.
            
        Raises:
            ValueError: If the symbol not found, contract size is invalid, or call fails.
        """
        result = await self._call_tool("get_symbol_contract_size", symbol_name=symbol)
        if isinstance(result, (int, float)):
            contract_size = float(result)
            if contract_size <= 0:
                raise ValueError(f"Invalid contract size for {symbol}: {contract_size} (must be positive)")
            return contract_size
        raise ValueError(f"Unexpected contract size response for {symbol}: {result}")
    
    async def get_symbol_price(self, symbol: str) -> dict[str, float]:
        """Get the latest price info for a symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD").
            
        Returns:
            Dict with price info (bid, ask, etc.).
            
        Raises:
            ValueError: If call fails.
        """
        result = await self._call_tool("get_symbol_price", symbol_name=symbol)
        if isinstance(result, dict):
            return result
        raise ValueError(f"Unexpected price response for {symbol}: {result}")


async def get_symbol_contract_size_from_mt5(symbol: str) -> Optional[float]:
    """Fetch contract size from MetaTrader MCP server if available.
    
    This is a convenience wrapper that handles configuration and error handling.
    Uses a persistent session to avoid creating/closing connections on every call.
    Returns None if the server is not configured or unreachable.
    
    Args:
        symbol: Symbol name to look up.
        
    Returns:
        Contract size if available, None otherwise.
    """
    if not METATRADER_MCP_ENABLED or not METATRADER_MCP_URL:
        logger.debug("MetaTrader MCP not configured (METATRADER_MCP_URL not set)")
        return None
    
    try:
        async with MetaTraderMCPClient(METATRADER_MCP_URL, use_persistent_session=True) as client:
            contract_size = await client.get_symbol_contract_size(symbol)
            logger.info(f"Fetched contract size for {symbol} from MetaTrader MCP: {contract_size}")
            return contract_size
    except Exception as e:
        logger.warning(f"Failed to fetch contract size for {symbol} from MetaTrader MCP: {e}")
        return None


async def get_symbol_price_from_mt5(symbol: str) -> Optional[dict[str, float]]:
    """Fetch price info from MetaTrader MCP server if available.
    
    This is a convenience wrapper that handles configuration and error handling.
    Uses a persistent session to avoid creating/closing connections on every call.
    Returns None if the server is not configured or unreachable.
    
    Args:
        symbol: Symbol name to look up.
        
    Returns:
        Price dict if available, None otherwise.
    """
    if not METATRADER_MCP_ENABLED or not METATRADER_MCP_URL:
        logger.debug("MetaTrader MCP not configured (METATRADER_MCP_URL not set)")
        return None
    
    try:
        async with MetaTraderMCPClient(METATRADER_MCP_URL, use_persistent_session=True) as client:
            price_info = await client.get_symbol_price(symbol)
            logger.info(f"Fetched price info for {symbol} from MetaTrader MCP")
            return price_info
    except Exception as e:
        logger.warning(f"Failed to fetch price for {symbol} from MetaTrader MCP: {e}")
        return None

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
    """Async client to call MetaTrader MCP server tools via SSE transport."""
    
    def __init__(self, base_url: str):
        """Initialize the MCP client.
        
        Args:
            base_url: Base URL of the MetaTrader MCP server (e.g., http://localhost:8080)
        """
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.session:
            await self.session.close()
    
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
            raise ValueError("Session not initialized. Use context manager: async with MetaTraderMCPClient(...) as client:")
        
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
        async with MetaTraderMCPClient(METATRADER_MCP_URL) as client:
            contract_size = await client.get_symbol_contract_size(symbol)
            logger.info(f"Fetched contract size for {symbol} from MetaTrader MCP: {contract_size}")
            return contract_size
    except Exception as e:
        logger.warning(f"Failed to fetch contract size for {symbol} from MetaTrader MCP: {e}")
        return None


async def get_symbol_price_from_mt5(symbol: str) -> Optional[dict[str, float]]:
    """Fetch price info from MetaTrader MCP server if available.
    
    This is a convenience wrapper that handles configuration and error handling.
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
        async with MetaTraderMCPClient(METATRADER_MCP_URL) as client:
            price_info = await client.get_symbol_price(symbol)
            logger.info(f"Fetched price info for {symbol} from MetaTrader MCP")
            return price_info
    except Exception as e:
        logger.warning(f"Failed to fetch price for {symbol} from MetaTrader MCP: {e}")
        return None

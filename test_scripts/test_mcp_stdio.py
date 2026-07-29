"""Test script to reproduce generate_chart hang via MCP stdio transport.

This simulates how the agent calls the tool via the MCP server, as opposed
to the direct function call in test_trade_chart.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

# Server config matching the Hermes config
# The venv lives at the project root, not in test_scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_PYTHON = PROJECT_ROOT / ".venv/bin/python"
SERVER_MODULE = "mcp_server.server"
SERVER_ENV = {
    "MCP_TRANSPORT": "stdio",
    "PYTHONPATH": str(PROJECT_ROOT),
    "MT5_MCP_URL": "http://10.0.1.105:8080/sse",
}


class MCPClient:
    """Minimal MCP client that communicates via stdio."""
    
    def __init__(self):
        self.process = None
        self.request_id = 0
        
    async def start(self) -> None:
        """Start the MCP server process."""
        print(f"Starting MCP server: {SERVER_PYTHON} -m {SERVER_MODULE}")
        self.process = await asyncio.create_subprocess_exec(
            str(SERVER_PYTHON),
            "-m",
            SERVER_MODULE,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**dict(os.environ), **SERVER_ENV},
        )
        print(f"Server started (PID: {self.process.pid})")
        
        # Initialize the connection
        await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        })
        
        # Read initialize response
        await self._read_response()
        
        # Send initialized notification
        await self._send_notification({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })
        
        print("✓ MCP connection initialized")
    
    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool and return the response."""
        print(f"\n→ Calling tool: {tool_name}")
        print(f"  Arguments: {json.dumps(arguments, indent=2)}")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        await self._send_request(request)
        response = await self._read_response()
        
        print(f"← Response received")
        return response
    
    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            print("\nStopping MCP server...")
            self.process.terminate()
            await self.process.wait()
            print("✓ Server stopped")
    
    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: dict) -> None:
        """Send a JSON-RPC request to the server."""
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
    
    async def _send_notification(self, notification: dict) -> None:
        """Send a JSON-RPC notification to the server."""
        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
    
    async def _read_response(self) -> dict:
        """Read a JSON-RPC response from the server."""
        line = await self.process.stdout.readline()
        if not line:
            raise EOFError("Server closed connection")
        return json.loads(line.decode())


async def test_generate_chart_basic():
    """Test basic chart generation (no trade levels)."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic chart (no trade levels)")
    print("=" * 70)
    
    client = MCPClient()
    try:
        await client.start()
        
        result = await asyncio.wait_for(
            client.call_tool("generate_chart", {
                "ticker": "AAPL",
                "period": "3mo",
                "interval": "1d",
            }),
            timeout=30.0  # 30 second timeout
        )
        
        print("✓ Test passed!")
        print(f"Result: {json.dumps(result, indent=2)}")
        
    except asyncio.TimeoutError:
        print("✗ Test FAILED: Timeout after 30 seconds")
        raise
    finally:
        await client.stop()


async def test_generate_chart_with_trade_levels():
    """Test chart generation with trade levels (LONG setup)."""
    print("\n" + "=" * 70)
    print("TEST 2: Chart with trade levels (LONG setup)")
    print("=" * 70)
    
    client = MCPClient()
    try:
        await client.start()
        
        result = await asyncio.wait_for(
            client.call_tool("generate_chart", {
                "ticker": "EURUSD",
                "period": "5d",
                "interval": "1h",
                "entry_price": 1.137,
                "stop_loss_price": 1.134,
                "take_profit_price": 1.140,
            }),
            timeout=30.0
        )
        
        print("✓ Test passed!")
        print(f"Result: {json.dumps(result, indent=2)}")
        
    except asyncio.TimeoutError:
        print("✗ Test FAILED: Timeout after 30 seconds")
        raise
    finally:
        await client.stop()


async def test_other_tools():
    """Test that other tools work fine (baseline)."""
    print("\n" + "=" * 70)
    print("TEST 3: Baseline - other tools (analyze_technicals)")
    print("=" * 70)
    
    client = MCPClient()
    try:
        await client.start()
        
        result = await asyncio.wait_for(
            client.call_tool("analyze_technicals", {
                "ticker": "AAPL",
                "period": "1y",
                "interval": "1d",
            }),
            timeout=30.0
        )
        
        print("✓ Test passed!")
        # The 'text' field is a JSON string, not a dict — parse it first
        text = result.get('result', {}).get('content', [{}])[0].get('text', '{}')
        parsed = json.loads(text) if isinstance(text, str) else text
        print(f"Result keys: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}")
        
    except asyncio.TimeoutError:
        print("✗ Test FAILED: Timeout after 30 seconds")
        raise
    finally:
        await client.stop()


async def main():
    """Run all tests."""
    import os
    os.environ.update(SERVER_ENV)
    
    print("=" * 70)
    print("MCP STDIO Transport Test Suite")
    print("=" * 70)
    print(f"Server: {SERVER_PYTHON}")
    print(f"Module: {SERVER_MODULE}")
    print(f"MT5 URL: {SERVER_ENV['MT5_MCP_URL']}")
    
    tests = [
        ("Other tools (baseline)", test_other_tools),
        ("Basic chart", test_generate_chart_basic),
        ("Chart with trade levels", test_generate_chart_with_trade_levels),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test '{name}' failed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

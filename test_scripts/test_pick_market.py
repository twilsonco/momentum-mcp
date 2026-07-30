#!/usr/bin/env python3
"""Standalone script that calls the pick_market tool via MCP stdio transport.

Returns {"wakeAgent": false} if any value in the result contains "Abort",
otherwise returns the tool output as JSON.
"""

import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path

# Server config
PROJECT_ROOT = Path("/Users/haiiro/NoSync/momentum-mcp")
SERVER_PYTHON = PROJECT_ROOT / ".venv/bin/python"
SERVER_MODULE = "mcp_server.server"
SERVER_ENV = {
    "MCP_TRANSPORT": "stdio",
    "PYTHONPATH": str(PROJECT_ROOT),
    "MT5_MCP_URL": os.getenv("MT5_MCP_URL", "http://10.0.1.105:8080/sse"),
}

MAX_POSITIONS = 15


class MCPClient:
    """Minimal MCP client that communicates via stdio."""

    def __init__(self):
        self.process = None
        self.request_id = 0

    async def start(self) -> None:
        """Start the MCP server process and initialize the connection."""
        self.process = await asyncio.create_subprocess_exec(
            str(SERVER_PYTHON),
            "-m",
            SERVER_MODULE,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**dict(os.environ), **SERVER_ENV},
        )

        # Initialize
        await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pick-market-client", "version": "1.0.0"},
            },
        })
        await self._read_response()

        # Initialized notification
        await self._send_notification({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool and return the response."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        await self._send_request(request)
        return await self._read_response()

    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    async def _send_request(self, request: dict) -> None:
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()

    async def _send_notification(self, notification: dict) -> None:
        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()

    async def _read_response(self) -> dict:
        line = await self.process.stdout.readline()
        if not line:
            raise EOFError("Server closed connection")
        return json.loads(line.decode())


def _contains_abort(obj) -> bool:
    """Check if any value in the result contains 'Abort'."""
    if isinstance(obj, dict):
        for value in obj.values():
            if _contains_abort(value):
                return True
    elif isinstance(obj, str) and "Abort" in obj:
        return True
    elif isinstance(obj, list):
        for item in obj:
            if _contains_abort(item):
                return True
    return False


async def main() -> dict:
    client = MCPClient()
    try:
        await client.start()

        response = await asyncio.wait_for(
            client.call_tool("pick_market", {"max_positions": MAX_POSITIONS}),
            timeout=30.0,
        )

        # Extract the text content from the MCP response
        content_list = response.get("result", {}).get("content", [])
        if content_list:
            text = content_list[0].get("text", "{}")
            result = json.loads(text) if isinstance(text, str) else text
        else:
            result = response.get("result", {})

        if _contains_abort(result):
            return {"wakeAgent": False}

        return result

    except asyncio.TimeoutError:
        return {"wakeAgent": False, "error": "pick_market timed out (30s)"}
    finally:
        await client.stop()


if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result))
    sys.exit(0)

#!/usr/bin/env python3
"""Standalone script that calls the pick_market tool via MCP stdio transport.

Returns {"wakeAgent": false} if any value in the result contains "Abort",
otherwise returns the tool output as JSON.
"""

import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import subprocess
from pathlib import Path

# Server config
PROJECT_ROOT = Path("/Users/haiiro/NoSync/momentum-mcp")
SERVER_PYTHON = PROJECT_ROOT / ".venv/bin/python"
SERVER_MODULE = "mcp_server.server"

MAX_POSITIONS = 15


def _setup_logging() -> logging.Logger:
    """Configure a rotating file logger for this script in logs/.

    Writes to logs/market_precheck.log (20MB cap, 3 backups) so the market
    picker output and lifecycle events are captured separately from the MCP
    server's own momentum.log.
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("market_precheck")
    if not getattr(logger, "_file_handler_attached", False):
        handler = RotatingFileHandler(
            str(log_dir / "market_precheck.log"),
            maxBytes=20 * 1024 * 1024,  # 20 MB
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._file_handler_attached = True

    return logger


logger = _setup_logging()
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
        logger.info("MCP server process started (pid=%s)", self.process.pid)

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
            logger.info("Stopping MCP server (pid=%s)", self.process.pid)
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
    """Check if the result dict is a pure error response (no market data)."""
    return isinstance(obj, dict) and len(obj) == 1 and 'error' in obj


async def main() -> dict:
    logger.info("market_precheck starting (max_positions=%s)", MAX_POSITIONS)
    
    # Clean up charts directory (use same logic as charts.py, but standalone)
    charts_dir_env = os.getenv("CHARTS_DIR")
    if charts_dir_env:
        charts_dir = Path(charts_dir_env)
    else:
        charts_dir = PROJECT_ROOT / "charts"
    
    if charts_dir.exists():
        import shutil
        try:
            shutil.rmtree(charts_dir)
            logger.info("Cleaned up charts directory: %s", charts_dir)
        except Exception as e:
            logger.warning("Failed to clean up charts directory: %s", e)
    
    client = MCPClient()
    try:
        await client.start()

        response = await asyncio.wait_for(
            client.call_tool("pick_market", {"max_positions": MAX_POSITIONS}),
            timeout=120.0,
        )

        # Extract the text content from the MCP response
        content_list = response.get("result", {}).get("content", [])
        try:
            if content_list:
                text = content_list[0].get("text", "{}")
                result = json.loads(text) if isinstance(text, str) else text
            else:
                result = response.get("result", {})
        except json.JSONDecodeError:
            result = {"error": f"Invalid JSON in response: {text}"}

        # Log the full market picker output for later inspection.
        logger.info("pick_market raw result:\n%s",
                    json.dumps(result, indent=2, default=str))

        if _contains_abort(result):
            logger.warning("Market picker returned error: %s", result)
            return {"wakeAgent": False}

        logger.info("pick_market completed successfully")
        return result

    except asyncio.TimeoutError:
        logger.error("pick_market timed out (120s)")
        return {"wakeAgent": False, "error": "pick_market timed out (120s)"}
    finally:
        await client.stop()


if __name__ == "__main__":
    result = asyncio.run(main())
    logger.info("market_precheck finished, wakeAgent=%s", result.get("wakeAgent"))
    print(json.dumps(result))
    sys.exit(0)

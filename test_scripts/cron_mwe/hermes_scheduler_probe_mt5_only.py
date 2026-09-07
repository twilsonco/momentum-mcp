#!/usr/bin/env python3
"""Hermes Scheduler Probe — Layer 1: real MT5 network path only.

Purpose
-------
The minimal MWE probes (hermes_scheduler_probe*.py) both SUCCEED on Hermes's
scheduled cron runs, which rules out a general scheduler bug with async,
subprocess spawning, and local sockets. This probe isolates the NEXT layer:

  Does connectivity to the *real* MT5 server (10.0.1.105:8080) work when run
  automatically by the Hermes scheduler?

It deliberately does NOT spawn an MCP subprocess and does NOT call any tool.
It only performs raw socket / network operations against the real target, so we
can answer cleanly:

    - If this FAILS on schedule but works manually -> it's a network/environment
      difference in how Hermes runs scheduled jobs (routing/proxy/firewall).
    - If this SUCCEEDS on schedule too          -> the problem is downstream,
      i.e. MCP server startup or pick_market execution.

Standard library only, no external dependencies.
"""

import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import socket
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "logs", "hermes_scheduler_probe_mt5_only.log")

# The real MT5 MCP server endpoint (same as market_precheck.py's default).
MT5_HOST = "10.0.1.105"
MT5_PORT = 8080

CONNECT_TIMEOUT = 8          # seconds per connect attempt
MAX_ATTEMPTS = 3
RETRY_DELAY = 2              # seconds between attempts


def _setup_logging() -> logging.Logger:
    log_dir = os.path.dirname(LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("hermes_scheduler_probe_mt5_only")
    if not getattr(logger, "_file_handler_attached", False):
        handler = RotatingFileHandler(
            LOG_PATH,
            maxBytes=20 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(console)

        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._file_handler_attached = True

    return logger


logger = _setup_logging()


def log_environment() -> None:
    logger.info("=== Environment & System ===")
    logger.info("Python: %s", sys.executable)
    logger.info("Version: %s (%s)", sys.version.split()[0], sys.platform)
    logger.info("CWD: %s", os.getcwd())
    logger.info("USER=%s HOME=%s", os.getenv("USER"), os.getenv("HOME"))

    env_vars_to_check = {
        "PATH": "Command search path",
        "SHELL": "Shell",
        "TERM": "Terminal type",
        "LANG": "Language/locale",
        "LC_ALL": "Locale override",
        "PYTHONPATH": "Python module search path",
        "SSH_AUTH_SOCK": "SSH auth socket",
        "http_proxy": "HTTP proxy (lowercase)",
        "https_proxy": "HTTPS proxy (lowercase)",
        "HTTP_PROXY": "HTTP proxy (uppercase)",
        "HTTPS_PROXY": "HTTPS proxy (uppercase)",
        "NO_PROXY": "Proxy exclusions",
        "no_proxy": "Proxy exclusions (lowercase)",
    }
    for var, description in env_vars_to_check.items():
        val = os.getenv(var)
        if val is None:
            logger.info("Env: %s (%s) = [NOT SET]", var, description)
        elif len(val) > 80:
            logger.info("Env: %s (%s) = <value, %d chars>", var, description, len(val))
        else:
            logger.info("Env: %s (%s) = %s", var, description, val)

    try:
        import locale
        logger.info("locale.getpreferredencoding()=%r default_locale=%s",
                    locale.getpreferredencoding(False), locale.getdefaultlocale())
    except Exception as e:
        logger.warning("Could not read locale info: %s", e)
    logger.info("=== End Environment ===")


def log_network_diagnostics() -> None:
    """Detailed network diagnostics for the real MT5 target (no subprocess/MCP)."""
    import ipaddress

    logger.info("=== Network Diagnostics for %s:%d ===", MT5_HOST, MT5_PORT)

    # 1) DNS / host resolution
    try:
        infos = socket.getaddrinfo(MT5_HOST, MT5_PORT,
                                   proto=socket.IPPROTO_TCP)
        logger.info("getaddrinfo(%s): %d results", MT5_HOST, len(infos))
        for fam, socktype, proto, canonname, sa in infos[:4]:
            logger.info("  family=%s type=%s -> %r",
                        socket.AddressFamily(fam).name,
                        socket.SocketKind(socktype).name if hasattr(socket, "SocketKind") else socktype,
                        sa)
    except Exception as e:
        logger.error("getaddrinfo(%s) FAILED: %s", MT5_HOST, e)

    # 2) Is the target on a directly-connected subnet?
    try:
        ip = socket.gethostbyname(MT5_HOST)
        addr = ipaddress.ip_address(ip)
        logger.info("Target IP: %s (is_private=%s)", ip, addr.is_private)
    except Exception as e:
        logger.error("Could not resolve target IP: %s", e)

    # 3) Routing table lines mentioning the relevant subnet
    try:
        out = subprocess.run(
            "netstat -rn 2>/dev/null | grep '10\\.0\\.1' || true",
            shell=True, capture_output=True, text=True, timeout=4,
        )
        if out.stdout.strip():
            logger.info("Routes for 10.0.1/24:")
            for line in out.stdout.splitlines()[:8]:
                if line.strip():
                    logger.info("  %s", line)
        else:
            logger.warning("No route found matching 10.0.1.x")
    except Exception as e:
        logger.warning("Could not inspect routes: %s", e)

    # 4) Sync (blocking) connect test — independent of asyncio.
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT)
        result = sock.connect_ex((MT5_HOST, MT5_PORT))
        sock.close()
        if result == 0:
            logger.info("✓ sync connect %s:%d OK", MT5_HOST, MT5_PORT)
        else:
            import errno
            name = errno.errorcode.get(result, "UNKNOWN")
            logger.warning("✗ sync connect %s:%d ERROR_CODE_%d (%s)",
                           MT5_HOST, MT5_PORT, result, name)
    except Exception as e:
        logger.error("✗ sync connect exception: %r", e)

    # 5) Compare against a well-known reachable host to see if it's target-specific.
    for probe_host, probe_port in [("8.8.8.8", 53), ("1.1.1.1", 443)]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(4)
            result = sock.connect_ex((probe_host, probe_port))
            sock.close()
            if result == 0:
                logger.info("✓ control connect %s:%d OK", probe_host, probe_port)
            else:
                import errno
                name = errno.errorcode.get(result, "UNKNOWN")
                logger.warning("✗ control connect %s:%d ERROR_CODE_%d (%s)",
                               probe_host, probe_port, result, name)
        except Exception as e:
            logger.warning("control connect %s:%d exception: %r", probe_host, probe_port, e)

    logger.info("=== End Network ===")


def _sync_probe() -> bool:
    """Blocking socket connect to the real MT5 server with retries."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECT_TIMEOUT)
            result = sock.connect_ex((MT5_HOST, MT5_PORT))
            sock.close()
            if result == 0:
                logger.info("✓ Attempt %d/%d: connected to %s:%d",
                            attempt, MAX_ATTEMPTS, MT5_HOST, MT5_PORT)
                return True
            import errno
            name = errno.errorcode.get(result, "UNKNOWN")
            logger.warning("✗ Attempt %d/%d: connect_ex=%d (%s)",
                           attempt, MAX_ATTEMPTS, result, name)
        except Exception as e:
            logger.error("Attempt %d/%d exception: %r", attempt, MAX_ATTEMPTS, e)
        if attempt < MAX_ATTEMPTS:
            import time
            time.sleep(RETRY_DELAY)
    return False


async def _async_probe() -> bool:
    """Async connect to the real MT5 server with retries."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(MT5_HOST, MT5_PORT),
                timeout=CONNECT_TIMEOUT,
            )
            logger.info("✓ Async attempt %d/%d: connected to %s:%d",
                        attempt, MAX_ATTEMPTS, MT5_HOST, MT5_PORT)
            writer.close()
            await writer.wait_closed()
            return True
        except asyncio.TimeoutError:
            logger.warning("✗ Async attempt %d/%d timed out (%ss)",
                           attempt, MAX_ATTEMPTS, CONNECT_TIMEOUT)
        except Exception as e:
            import errno
            if isinstance(e, OSError):
                name = errno.errorcode.get(getattr(e, "errno", None), "UNKNOWN")
                logger.warning("✗ Async attempt %d/%d: %s (errno=%s)",
                               attempt, MAX_ATTEMPTS, e.strerror or e,
                               getattr(e, "errno", name))
            else:
                logger.error("Async attempt %d/%d exception: %r",
                             attempt, MAX_ATTEMPTS, e)
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(RETRY_DELAY)
    return False


async def main() -> dict:
    logger.info("hermes_scheduler_probe_mt5_only starting")
    log_environment()
    log_network_diagnostics()

    sync_ok = _sync_probe()
    async_ok = await _async_probe()

    if sync_ok and async_ok:
        logger.info("PROBE RESULT: SUCCESS (both sync & async connect to MT5)")
        return {"status": "success", "target": f"{MT5_HOST}:{MT5_PORT}",
                "sync": True, "async": True}
    else:
        logger.error("PROBE RESULT: FAILURE (sync=%s async=%s)",
                     sync_ok, async_ok)
        return {"status": "failure", "target": f"{MT5_HOST}:{MT5_PORT}",
                "sync": sync_ok, "async": async_ok}


if __name__ == "__main__":
    result = asyncio.run(main())
    logger.info("probe finished: %s", json.dumps(result))
    sys.exit(0 if result.get("status") == "success" else 1)

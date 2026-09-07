#!/usr/bin/env python3
"""LAN reachability probe for Hermes cron jobs on macOS 15+ (stdlib only).

Reproduces https://github.com/NousResearch/hermes-agent/issues/71206 from the
cron angle: scheduled cron fires get EHOSTUNREACH (errno 65) on LAN targets
while manual triggers of the same job succeed, because the responsible process
differs (gateway vs terminal).

SETUP: edit LAN_HOST / LAN_PORT below to any LAN device reachable from your
machine (router, NAS, Home Assistant, vLLM box, ...), then attach this script
to a Hermes cron job and compare scheduled vs manual output.

Exit 0 = LAN reachable, 1 = blocked.
"""

import ctypes
import errno
import os
import socket
import subprocess
import sys
import time

# --- EDIT THESE: any LAN host reachable from your machine -------------------
LAN_HOST = "10.0.1.105"   # e.g. your router, NAS, Home Assistant, vLLM box
LAN_PORT = 8080              # a TCP port that is definitely open on that host
# -----------------------------------------------------------------------------

# Log next to this script (cron's working directory is unreliable).
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "logs", "lan_probe.log")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # logging must never mask the probe result


def responsible_pid() -> int:
    """macOS TCC attribution: which process's permissions apply to us."""
    try:
        return ctypes.CDLL(None).responsibility_get_pid_responsible_for_pid(os.getpid())
    except Exception:
        return -1


def proc(pid: int) -> str:
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip()[:100] or "?"
    except Exception:
        return "?"


def connect(host: str, port: int, timeout: float = 5.0) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port))
    finally:
        sock.close()


def main() -> int:
    me, resp = os.getpid(), responsible_pid()
    log("--- lan_probe run ---")
    log(f"python={sys.executable} cwd={os.getcwd()}")
    log(f"self pid={me} cmd={proc(me)!r}")
    log(f"responsible pid={resp} cmd={proc(resp)!r}")

    rc_lan = connect(LAN_HOST, LAN_PORT)
    rc_www = connect("8.8.8.8", 53)
    log(f"LAN  {LAN_HOST}:{LAN_PORT} -> {rc_lan} "
        f"({'OK' if rc_lan == 0 else errno.errorcode.get(rc_lan, rc_lan)})")
    log(f"inet 8.8.8.8:53          -> {rc_www} "
        f"({'OK' if rc_www == 0 else errno.errorcode.get(rc_www, rc_www)})")

    if rc_lan == 65 and rc_www == 0:  # EHOSTUNREACH on LAN, internet fine
        log("SIGNATURE: LAN blocked while internet OK -> macOS Local Network "
            "privacy silent denial (see hermes-agent#71206)")
    log(f"RESULT: {'SUCCESS' if rc_lan == 0 else 'FAILURE'}")
    return 0 if rc_lan == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

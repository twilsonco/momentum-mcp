#!/usr/bin/env python3
"""Test script to debug get_symbols with different parameters."""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/haiiro/NoSync/momentum-mcp")
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from mcp_server.utils.mt5_mcp_server import call_mt5_tool

async def main():
    """Call get_symbols with different parameters and see what comes back."""
    
    # Test 1: No parameters
    print("=" * 80)
    print("Test 1: get_symbols with no args")
    print("=" * 80)
    try:
        result = await call_mt5_tool("get_symbols", {}, timeout=30.0)
        if hasattr(result, "content") and result.content:
            # Just show first 3 items
            for i, content in enumerate(result.content[:3]):
                if hasattr(content, "text"):
                    print(f"Item {i}: {repr(content.text[:100])}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: With fields=["path"]
    print("\n" + "=" * 80)
    print("Test 2: get_symbols with fields=['path']")
    print("=" * 80)
    try:
        result = await call_mt5_tool("get_symbols", {"fields": ["path"]}, timeout=30.0)
        if hasattr(result, "content") and result.content:
            # Just show first 3 items
            for i, content in enumerate(result.content[:3]):
                if hasattr(content, "text"):
                    print(f"Item {i}: {repr(content.text[:100])}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Try get_all_symbols
    print("\n" + "=" * 80)
    print("Test 3: get_all_symbols with no args")
    print("=" * 80)
    try:
        result = await call_mt5_tool("get_all_symbols", {}, timeout=30.0)
        if hasattr(result, "content") and result.content:
            # Just show first 3 items
            for i, content in enumerate(result.content[:3]):
                if hasattr(content, "text"):
                    text = content.text[:200] if content.text else "(empty)"
                    print(f"Item {i}: {repr(text)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

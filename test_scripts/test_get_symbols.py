#!/usr/bin/env python3
"""Test script to debug get_symbols response format."""

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
    """Call get_symbols and print the raw response."""
    try:
        result = await call_mt5_tool("get_symbols", {"fields": ["path"]}, timeout=30.0)
        print("Result object:", result)
        print("Result type:", type(result))
        
        if hasattr(result, "content"):
            print(f"Has content attribute: {len(result.content)} items")
            for i, content in enumerate(result.content):
                print(f"\nContent item {i}:")
                print(f"  Type: {type(content)}")
                print(f"  Dir: {[x for x in dir(content) if not x.startswith('_')]}")
                if hasattr(content, "text"):
                    text = content.text[:500] if content.text else "(empty)"
                    print(f"  Text (first 500 chars): {text}")
                    # Show first few lines
                    if content.text:
                        lines = content.text.strip().split("\n")[:5]
                        print(f"  First {len(lines)} lines:")
                        for line in lines:
                            print(f"    {repr(line)}")
        else:
            print("No content attribute")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

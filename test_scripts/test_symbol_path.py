#!/usr/bin/env python3
"""Test script to check symbol_info structure for path field."""

import asyncio
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path("/Users/haiiro/NoSync/momentum-mcp")
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from mcp_server.utils.mt5_mcp_server import fetch_mt5_symbol_info

async def main():
    """Fetch symbol info for a few different symbols and check for path field."""
    
    test_symbols = [
        "EURUSD",   # FX major
        "ETHUSD",   # Crypto
        "XAUUSD",   # Metals/Precious
        "NAS100",   # Index
    ]
    
    for symbol in test_symbols:
        print(f"\n{'='*80}")
        print(f"Symbol: {symbol}")
        print('='*80)
        try:
            info = await fetch_mt5_symbol_info(symbol, timeout=10.0)
            if info:
                # Print the path field if it exists
                if "path" in info:
                    print(f"Path: {info['path']}")
                else:
                    print("No 'path' field in symbol_info")
                    # Show all available keys
                    print(f"Available fields: {list(info.keys())}")
            else:
                print(f"No info returned for {symbol}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

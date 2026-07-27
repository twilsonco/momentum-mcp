"""Quick test to verify generate_chart completes without hanging."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make sure we import the local mcp_server package
sys.path.insert(0, str(Path(__file__).parent))

# Set up environment before importing
import os
os.environ["MT5_MCP_URL"] = "http://10.0.1.105:8080/sse"

from mcp_server.server import generate_chart


async def test_with_timeout():
    """Test that generate_chart completes within reasonable time."""
    print("Testing generate_chart (30 second timeout)...")
    
    try:
        result = await asyncio.wait_for(
            generate_chart(
                ticker="AAPL",
                period="1mo",
                interval="1d",
            ),
            timeout=30.0
        )
        print(f"✓ SUCCESS: Chart generated in < 30 seconds")
        print(f"  Result: {result}")
        return True
        
    except asyncio.TimeoutError:
        print("✗ FAILED: Timeout after 30 seconds (tool is hanging)")
        return False
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 70)
    print("Quick Hang Test - generate_chart tool")
    print("=" * 70)
    
    success = await test_with_timeout()
    
    print("\n" + "=" * 70)
    if success:
        print("PASS: Tool completes without hanging")
        sys.exit(0)
    else:
        print("FAIL: Tool hangs or errors")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

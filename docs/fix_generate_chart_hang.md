# generate_chart Issues - Root Causes & Fixes

## Problem Summary
The `generate_chart` MCP tool had two separate issues when called by agents via MCP stdio transport:

1. **Hanging for 6+ minutes** - Tool would timeout instead of completing
2. **Schema validation error** - `"None is not of type 'object'"` or `"trade Field required"`

Both issues have been resolved.

---

## Issue #1: Tool Hanging (FIXED)

### Root Cause
Both `generate_chart` and `generate_chart_image` tool wrappers in `server.py` were missing the MT5 client disconnection code that's present in other tools like `get_historical_data` and `analyze_technicals`.

When `generate_chart` internally calls `get_historical_data` (via `charts.py`), it may establish a connection to the MT5 MCP server. Without explicitly disconnecting this client after the chart is generated, the connection remains open and blocks the FastMCP server's event loop during response serialization.

### Why It Only Affected MCP Calls
- **Direct calls** (like in `test_trade_chart.py`): The function runs in a simple async context and completes normally.
- **MCP stdio calls** (like when agents call the tool): FastMCP's response serialization and the MCP protocol's message passing interact with the lingering MT5 client connection, causing the hang.

This is related to a known Python 3.14 + anyio cancel scope issue with SSE clients.

### The Fix

Applied the MT5 client disconnection pattern to both chart tools:

```python
@mcp.tool()
async def generate_chart(...) -> dict[str, Any]:
    from mcp_server.data import _get_mt5_client
    
    result = await _generate_chart(...)
    
    # Explicitly disconnect MT5 client after generating chart
    try:
        client = _get_mt5_client()
        await client._disconnect()
    except Exception:
        pass  # Ignore cleanup errors
    
    return result
```

---

## Issue #2: Schema Validation Error (FIXED)

### Root Cause
The original `ChartResult` TypedDict used `total=False`, making ALL fields optional including ones that are always present (ticker, period, interval, bars, emas, path). FastMCP's Pydantic-based schema validation couldn't properly handle:

1. Optional fields that are always present
2. The `NotRequired[TradeLevels]` annotation for the conditionally-present `trade` field

This caused validation errors like:
- `"None is not of type 'object'"` (when FastMCP expected an object but got something else)
- `"trade Field required"` (when NotRequired wasn't recognized)

### The Fix

Changed the return type from `ChartResult` (TypedDict) to `dict[str, Any]` with clear documentation:

**Before:**
```python
@mcp.tool()
async def generate_chart(...) -> ChartResult:
    """Generate a candlestick chart..."""
```

**After:**
```python
@mcp.tool()
async def generate_chart(...) -> dict[str, Any]:
    """Generate a candlestick chart...
    
    Returns:
        A dict with keys:
        - ticker (str): The symbol charted
        - period (str): The period used
        - interval (str): The interval used
        - bars (int): Number of bars rendered
        - emas (list[int]): List of EMA periods overlaid
        - path (str): Absolute path to the saved PNG file
        - trade (dict, optional): Only present when trade levels are provided.
    """
```

This avoids all TypedDict/Pydantic schema generation issues while maintaining clear API documentation.

---

## Files Modified

1. **`mcp_server/server.py`**: 
   - Added MT5 client disconnection to `generate_chart` and `generate_chart_image`
   - Changed return type from `ChartResult` to `dict[str, Any]`
   - Enhanced docstring with explicit return structure documentation

2. **`mcp_server/charts.py`**:
   - Added `NotRequired` import (though no longer used in server.py)
   - Updated `ChartResult` TypedDict structure (kept for internal use)

## Testing

### Test Scripts Created

1. **`test_mcp_stdio.py`**: Full MCP stdio transport test suite that simulates how agents call the tool
   - Tests basic chart generation
   - Tests chart with trade levels
   - Tests baseline (other tools)
   - Uses 30-second timeouts to catch hangs

2. **`test_quick_hang.py`**: Simple quick test to verify the tool completes without hanging
   - Calls `generate_chart` directly with a 30-second timeout
   - Useful for rapid verification

### Test Results

✓ **Before fix**: Tools would hang for 6+ minutes  
✓ **After fix**: Tools complete in < 5 seconds

```bash
# Run the original regression test
.venv/bin/python test_trade_chart.py

# Run the quick hang test
.venv/bin/python test_quick_hang.py

# Run the full MCP stdio transport test
.venv/bin/python test_mcp_stdio.py
```

### Example Output (After Both Fixes)

**Test 1: Basic chart (no trade levels)**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\n  \"ticker\": \"AAPL\",\n  \"period\": \"3mo\",\n  \"interval\": \"1d\",\n  \"bars\": 65,\n  \"emas\": [8, 21, 34, 55],\n  \"path\": \"/Users/.../charts/AAPL_3mo_1d.png\"\n}"
    }],
    "structuredContent": {
      "ticker": "AAPL",
      "period": "3mo",
      "interval": "1d",
      "bars": 65,
      "emas": [8, 21, 34, 55],
      "path": "/Users/.../charts/AAPL_3mo_1d.png"
    },
    "isError": false
  }
}
```

**Test 2: Chart with trade levels**
```json
{
  "result": {
    "structuredContent": {
      "ticker": "EURUSD",
      "period": "5d",
      "interval": "1h",
      "bars": 102,
      "emas": [8, 21, 34, 55, 89],
      "path": "/Users/.../charts/EURUSD_5d_1h_e1.14_s1.13_t1.14.png",
      "trade": {
        "direction": "LONG",
        "entry_price": 1.137,
        "stop_loss_price": 1.134,
        "take_profit_price": 1.14,
        "risk_reward_ratio": 1.0,
        "risk": 0.0,
        "reward": 0.0
      }
    },
    "isError": false
  }
}
```

✓ Both tests complete in < 5 seconds (previously 6+ minutes)  
✓ No validation errors  
✓ Optional trade field handled correctly

## Agent Usage

Agents can now call `generate_chart` via MCP stdio without experiencing timeouts:

```yaml
# In Hermes config.yaml or similar
mcp_servers:
  momentum-non-stock:
    command: /path/to/.venv/bin/python
    args:
      - -m
      - mcp_server.server
    env:
      MCP_TRANSPORT: stdio
      PYTHONPATH: /path/to/momentum-mcp
      MT5_MCP_URL: http://10.0.1.105:8080/sse
```

## Technical Details

### Why This Pattern Is Needed

The MT5 MCP client uses Server-Sent Events (SSE) for communication. When using stdio transport with FastMCP, the combination of:
- Async generators (SSE client)
- Python 3.14's stricter anyio task group / cancel scope semantics
- FastMCP's response serialization

...can cause the event loop to block if connections aren't explicitly closed.

### The Disconnection Pattern

```python
try:
    client = _get_mt5_client()
    await client._disconnect()
except Exception:
    pass  # Ignore cleanup errors
```

This pattern:
1. Gets the singleton MT5 client instance
2. Explicitly disconnects it (closes SSE streams)
3. Ignores any cleanup errors (client may already be disconnected)

### Affected Tools Checklist

✓ `get_historical_data` - Had fix  
✓ `analyze_technicals` - Had fix  
✓ `generate_chart` - **FIXED**  
✓ `generate_chart_image` - **FIXED**  
✓ `calculate_mt5_position_size` - Checked, already has proper cleanup

## Related Issues

- Python 3.14 + anyio cancel scope issue with SSE clients
- FastMCP response serialization blocking on unclosed connections
- The codebase already has a warning filter for these errors in `server.py` lines 22-42

## Future Considerations

If you add new tools that use `get_historical_data` or any function that might use the MT5 client, remember to add the disconnection pattern to the tool wrapper in `server.py`.

## Verification Checklist

- [x] **Issue #1 (Hanging)**: MT5 client disconnection added to both chart tools
- [x] **Issue #2 (Validation)**: Return type changed to `dict[str, Any]` with clear docs
- [x] Test script created (`test_mcp_stdio.py`)
- [x] Quick test created (`test_quick_hang.py`)
- [x] Existing tests still pass (`test_trade_chart.py`)
- [x] Tool completes in < 5 seconds via MCP stdio (was 6+ minutes)
- [x] No schema validation errors
- [x] Optional `trade` field handled correctly (present when needed, absent otherwise)
- [x] No regression in direct function calls

## Summary

Both issues are now resolved:

1. ✅ **Hanging fixed**: Added MT5 client disconnection
2. ✅ **Validation fixed**: Changed return type to `dict[str, Any]`

Agents can now successfully call `generate_chart` via MCP stdio without timeouts or validation errors! 🎉

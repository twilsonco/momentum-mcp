from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field

class SignalResult(BaseModel):
    """Standardized tool output to prevent downstream crashes."""
    status: str = Field(..., description="success or error")
    data: Any = Field(default=None, description="The actual payload (dict, list, etc.)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Contextual info (timing, source, model)")
    error: Optional[str] = Field(default=None, description="Human-readable error message")

    @classmethod
    def success(cls, data: Any, metadata: Optional[dict] = None) -> SignalResult:
        return cls(status="success", data=data, metadata=metadata or {})

    @classmethod
    def error_msg(cls, message: str, metadata: Optional[dict] = None) -> SignalResult:
        return cls(status="error", error=message, metadata=metadata or {})

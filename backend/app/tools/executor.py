"""Tool execution engine and registry.

Follows design doc §3 & §11:
- ToolExecutor manages a registry of tools.
- Applies execution timeout (TOOL_TIMEOUT_S, default 5).
- run() NEVER raises; returns {"error": "...", "code": "TOOL_ERROR"} on error or timeout.
- Exposes tool specifications for LLM consumption.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from backend.app.adapters.protocols import Tool, ToolSpec
from backend.app.config import settings
from backend.app.tools.order_tool import OrderTool
from backend.app.tools.time_tool import TimeTool


class ToolExecutor:
    """Executes registered tools safely with timeouts and error wrapping."""

    def __init__(self, timeout_s: Optional[float] = None) -> None:
        self.timeout_s: float = timeout_s if timeout_s is not None else settings.TOOL_TIMEOUT_S
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance by its spec name."""
        self._tools[tool.spec.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Lookup a registered tool by name."""
        return self._tools.get(name)

    def get_specs(self) -> List[ToolSpec]:
        """Return specifications of all registered tools for the LLM."""
        return [tool.spec for tool in self._tools.values()]

    async def run(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool by name with arguments.

        Guaranteed NEVER to raise: returns a standard error dict on failure or timeout.
        """
        tool_args = args or {}
        tool = self._tools.get(name)

        if tool is None:
            return {
                "error": f"Tool '{name}' not found",
                "code": "TOOL_ERROR",
            }

        try:
            result = await asyncio.wait_for(
                tool.run(tool_args),
                timeout=self.timeout_s,
            )
            return result
        except asyncio.TimeoutError:
            return {
                "error": f"Tool '{name}' execution timed out after {self.timeout_s}s",
                "code": "TOOL_ERROR",
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "code": "TOOL_ERROR",
            }


def create_default_tool_executor() -> ToolExecutor:
    """Factory creating a ToolExecutor pre-configured with default tools."""
    executor = ToolExecutor()
    executor.register(TimeTool())
    executor.register(OrderTool())
    return executor


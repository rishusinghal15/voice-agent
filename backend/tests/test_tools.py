"""Unit tests for tools (time_tool, order_tool) and ToolExecutor."""

import asyncio
from typing import Any, Dict

import pytest

from backend.app.adapters.protocols import Tool, ToolSpec
from backend.app.tools.executor import ToolExecutor, create_default_tool_executor
from backend.app.tools.order_tool import (
    MOCK_ORDERS,
    OrderNotFoundError,
    OrderTool,
    get_order_status,
)
from backend.app.tools.time_tool import TimeTool, get_current_time


# ============================================================================
# Time Tool Tests
# ============================================================================

def test_get_current_time_valid():
    """Verify get_current_time returns valid ISO and human strings for IANA timezones."""
    for tz in ["UTC", "Asia/Tokyo", "America/New_York", "Europe/London"]:
        res = get_current_time(tz)
        assert "iso" in res
        assert "human" in res
        assert len(res["iso"]) > 0
        assert len(res["human"]) > 0


def test_get_current_time_invalid():
    """Verify invalid timezones raise ValueError."""
    for invalid_tz in ["Invalid/Timezone", "Mars/Crater", "", "1234"]:
        with pytest.raises(ValueError):
            get_current_time(invalid_tz)


def test_time_tool_instance():
    """Verify TimeTool conforms to Tool protocol and executes correctly."""
    async def _run():
        tool = TimeTool()
        assert tool.spec.name == "get_current_time"
        assert "timezone" in tool.spec.parameters["properties"]

        res = await tool.run({"timezone": "UTC"})
        assert "iso" in res
        assert "human" in res

    asyncio.run(_run())


# ============================================================================
# Order Tool Tests
# ============================================================================

def test_get_order_status_valid():
    """Verify get_order_status returns expected mock order data."""
    for order_id in ["ord_101", "ord_102", "ord_103"]:
        order = get_order_status(order_id)
        assert order["order_id"] == order_id
        assert "status" in order
        assert "items" in order
        assert "delivery_date" in order


def test_get_order_status_not_found():
    """Verify unknown order IDs raise OrderNotFoundError."""
    for unknown_id in ["ord_999", "nonexistent", ""]:
        with pytest.raises(OrderNotFoundError):
            get_order_status(unknown_id)


def test_order_tool_instance():
    """Verify OrderTool conforms to Tool protocol and executes correctly."""
    async def _run():
        tool = OrderTool()
        assert tool.spec.name == "get_order_status"
        res = await tool.run({"order_id": "ord_101"})
        assert res["order_id"] == "ord_101"
        assert res["status"] == "shipped"

    asyncio.run(_run())


# ============================================================================
# Tool Executor Tests
# ============================================================================

class SlowTool:
    """Mock tool that sleeps longer than timeout."""
    spec = ToolSpec(name="slow_tool", description="A slow tool")

    async def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.5)
        return {"result": "done"}


class CrashingTool:
    """Mock tool that raises an unexpected exception."""
    spec = ToolSpec(name="crash_tool", description="A crashing tool")

    async def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("Unexpected failure in tool code")


def test_executor_success():
    """Verify ToolExecutor runs registered tools successfully."""
    async def _run():
        executor = create_default_tool_executor()
        specs = executor.get_specs()
        assert len(specs) == 2
        spec_names = {s.name for s in specs}
        assert "get_current_time" in spec_names
        assert "get_order_status" in spec_names

        # Run time tool
        time_res = await executor.run("get_current_time", {"timezone": "Asia/Tokyo"})
        assert "iso" in time_res
        assert "code" not in time_res

        # Run order tool
        order_res = await executor.run("get_order_status", {"order_id": "ord_102"})
        assert order_res["status"] == "processing"

    asyncio.run(_run())


def test_executor_never_raises_on_unknown_tool():
    """Verify executor returns TOOL_ERROR and never raises for unknown tool."""
    async def _run():
        executor = ToolExecutor()
        res = await executor.run("unknown_tool", {})
        assert res["code"] == "TOOL_ERROR"
        assert "error" in res

    asyncio.run(_run())


def test_executor_never_raises_on_tool_failure():
    """Verify executor wraps tool exceptions in TOOL_ERROR without raising."""
    async def _run():
        executor = ToolExecutor()
        executor.register(CrashingTool())
        executor.register(TimeTool())
        executor.register(OrderTool())

        # Crashing tool
        crash_res = await executor.run("crash_tool", {})
        assert crash_res["code"] == "TOOL_ERROR"
        assert "Unexpected failure" in crash_res["error"]

        # Tool with invalid argument (e.g. invalid timezone)
        time_err = await executor.run("get_current_time", {"timezone": "Bad/Tz"})
        assert time_err["code"] == "TOOL_ERROR"
        assert "Invalid timezone" in time_err["error"]

        # Tool with unknown order id
        order_err = await executor.run("get_order_status", {"order_id": "ord_missing"})
        assert order_err["code"] == "TOOL_ERROR"
        assert "not found" in order_err["error"]

    asyncio.run(_run())


def test_executor_timeout_path():
    """Verify executor enforces timeout and returns TOOL_ERROR without raising."""
    async def _run():
        executor = ToolExecutor(timeout_s=0.05)
        executor.register(SlowTool())

        res = await executor.run("slow_tool", {})
        assert res["code"] == "TOOL_ERROR"
        assert "timed out" in res["error"]

    asyncio.run(_run())


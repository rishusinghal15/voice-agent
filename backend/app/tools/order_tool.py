"""Order status tool implementation.

Follows design doc §11:
- get_order_status(order_id) from a mock dict (3 orders)
- unknown id raises a not-found error
"""

from __future__ import annotations

from typing import Any, Dict

from backend.app.adapters.protocols import Tool, ToolSpec


class OrderNotFoundError(KeyError):
    """Raised when an order is not found in the orders repository."""

    def __init__(self, order_id: str) -> None:
        super().__init__(f"Order '{order_id}' not found")
        self.order_id = order_id


# Mock orders dictionary containing 3 predefined orders
MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "ord_101": {
        "order_id": "ord_101",
        "status": "shipped",
        "items": ["Wireless Noise-Canceling Headphones"],
        "delivery_date": "2026-11-03",
    },
    "ord_102": {
        "order_id": "ord_102",
        "status": "processing",
        "items": ["Ergonomic Mechanical Keyboard", "Wrist Rest"],
        "delivery_date": "2026-11-06",
    },
    "ord_103": {
        "order_id": "ord_103",
        "status": "delivered",
        "items": ["USB-C Fast Charging Cable"],
        "delivery_date": "2026-10-25",
    },
}


def get_order_status(order_id: str) -> Dict[str, Any]:
    """Retrieve details and shipping status for an order.

    Raises:
        OrderNotFoundError: If the order_id does not exist.
    """
    if not order_id or not isinstance(order_id, str):
        raise OrderNotFoundError(str(order_id))

    clean_id = order_id.strip()
    if clean_id not in MOCK_ORDERS:
        raise OrderNotFoundError(clean_id)

    return dict(MOCK_ORDERS[clean_id])


class OrderTool:
    """Tool wrapper providing get_order_status conforming to Tool protocol."""

    spec: ToolSpec = ToolSpec(
        name="get_order_status",
        description="Retrieve the current fulfillment status, items, and estimated delivery date for an order.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Unique identifier of the order, e.g. 'ord_101', 'ord_102', 'ord_103'.",
                }
            },
            "required": ["order_id"],
        },
    )

    async def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_order_status with provided args."""
        order_id = args.get("order_id", "")
        return get_order_status(order_id)


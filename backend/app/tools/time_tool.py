"""Time tool implementation.

Follows design doc §11:
- get_current_time(timezone) -> {iso, human}
- invalid timezone raises ValueError
"""

from __future__ import annotations

import zoneinfo
from datetime import datetime
from typing import Any, Dict

from backend.app.adapters.protocols import Tool, ToolSpec


def get_current_time(timezone: str) -> Dict[str, str]:
    """Return the current time in the requested IANA timezone as ISO and human-readable formats.

    Raises:
        ValueError: If timezone string is invalid or not found.
    """
    if not timezone or not isinstance(timezone, str):
        raise ValueError(f"Invalid timezone: {timezone}")

    try:
        tz = zoneinfo.ZoneInfo(timezone.strip())
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, KeyError) as e:
        raise ValueError(f"Invalid timezone: {timezone}") from e

    now = datetime.now(tz)
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%Y-%m-%d %I:%M:%S %p %Z"),
    }


class TimeTool:
    """Tool wrapper providing get_current_time conforming to Tool protocol."""

    spec: ToolSpec = ToolSpec(
        name="get_current_time",
        description="Get current date and time for a given IANA timezone (e.g. 'UTC', 'Asia/Tokyo', 'America/New_York').",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone string, e.g. 'UTC', 'America/New_York', 'Asia/Tokyo'.",
                }
            },
            "required": ["timezone"],
        },
    )

    async def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_current_time with provided args."""
        tz = args.get("timezone", "UTC")
        return get_current_time(tz)


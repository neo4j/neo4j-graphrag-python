from typing import Any


class Component:
    """Interface that needs to be implemented
    by all components
    ."""

    async def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

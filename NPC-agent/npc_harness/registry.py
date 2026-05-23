"""Tool abstraction and registry -- standalone reimplementation of nanobot's tool layer.

Architectural concept: each capability is a Tool with a JSON-Schema descriptor
and an async execute(); ToolRegistry holds all tools for a session and can
produce the OpenAI tool-schema list the LLM sees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    @property
    def read_only(self) -> bool:
        return False

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any: ...

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in sorted(self._tools.values(), key=lambda t: t.name)]

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

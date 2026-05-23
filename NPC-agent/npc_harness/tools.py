"""Game tools -- wraps game function schemas as Tool objects.

Each ``GameTool`` carries a ``kind`` -- ``query`` (check_*/search_*, read-only) vs
``action`` (select/equip/sell, side effects) -- mirroring NPC-RL's tool_registry vs
action_registry split. Execution is routed to a ``GameBackend`` so the same tools work
against gold data (eval), the knowledge base (interactive), or a real game server.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Union

try:
    from .tool_builder import build_all_tools
except ImportError:  # pragma: no cover
    build_all_tools = None  # type: ignore

from .backend import GameBackend
from .registry import Tool, ToolRegistry


class GameTool(Tool):
    """One game function, wrapping an OpenAI-format schema and routing to a backend."""

    def __init__(self, function_def: Dict[str, Any], backend: GameBackend, kind: str):
        self._fn = function_def           # {name, description, parameters}
        self._backend = backend
        self._kind = kind                 # "query" | "action"

    @property
    def name(self) -> str:
        return self._fn["name"]

    @property
    def description(self) -> str:
        return self._fn.get("description", "")

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._fn.get("parameters") or {"type": "object", "properties": {}}

    @property
    def read_only(self) -> bool:
        return self._kind == "query"

    async def execute(self, **kwargs: Any) -> Any:
        results = await self._backend.execute(
            [{"name": self.name, "parameters": kwargs, "kind": self._kind}]
        )
        if not results:
            return [{"information": "n/a"}]
        return results[0].get("return", results[0])


def _resolve_registries(tools_spec: Union[str, List[str], None]):
    """(tool_registry, action_registry) dicts for an NpcContext tool spec."""
    from function_calls import tool_map, action_map  # lazy: pulls langchain via game stubs

    if tools_spec is None:
        tools_spec = "function_list_id_0001"
    if isinstance(tools_spec, str):
        if tools_spec not in tool_map and tools_spec not in action_map:
            raise KeyError(f"Unknown tool list '{tools_spec}'. Known: {sorted(tool_map)}")
        return (
            copy.deepcopy(tool_map.get(tools_spec, {"function_registry": {}})),
            copy.deepcopy(action_map.get(tools_spec, {"function_registry": {}})),
        )
    if isinstance(tools_spec, (list, tuple)):
        names = set(tools_spec)
        tool_fr, action_fr = {}, {}
        for reg in tool_map.values():
            for n, fdef in reg["function_registry"].items():
                if n in names and n not in tool_fr:
                    tool_fr[n] = copy.deepcopy(fdef)
        for reg in action_map.values():
            for n, fdef in reg["function_registry"].items():
                if n in names and n not in action_fr:
                    action_fr[n] = copy.deepcopy(fdef)
        return {"function_registry": tool_fr}, {"function_registry": action_fr}
    raise TypeError(f"`tools` must be a str or list, got {type(tools_spec).__name__}")


def build_registry_from_registries(tool_registry: Dict, action_registry: Dict, backend: GameBackend) -> ToolRegistry:
    """Build a ToolRegistry of GameTools from raw game function registries."""
    all_tools = build_all_tools(copy.deepcopy(tool_registry), copy.deepcopy(action_registry))
    action_names = set((action_registry or {}).get("function_registry", {}))
    registry = ToolRegistry()
    for t in all_tools:
        fn = t["function"]
        kind = "action" if fn["name"] in action_names else "query"
        registry.register(GameTool(fn, backend, kind))
    return registry


def build_registry_from_context(ctx, backend: GameBackend) -> ToolRegistry:
    """Build a ToolRegistry for an NpcContext using function_calls registries."""
    tool_reg, action_reg = _resolve_registries(ctx.tools)
    return build_registry_from_registries(tool_reg, action_reg, backend)

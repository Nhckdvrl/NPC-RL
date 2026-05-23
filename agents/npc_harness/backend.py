"""Phase-2 execution backends (the swappable game seam).

The model decides *what* to call; a GameBackend decides *what comes back*. Two impls ship:

  * ``GoldGameBackend``      -- wraps the eval ``function_calls.Executor`` (gold matching +
    call recording). Used by the eval adapter.
  * ``KnowledgeGameBackend`` -- answers calls from the NpcContext's own knowledge base so a
    live NPC works offline (the game tool bodies are empty stubs).

A production deployment replaces these with a backend that talks to real game state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GameBackend(ABC):
    @abstractmethod
    async def execute(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """calls: ``[{name, parameters}]`` -> ``[{name, parameters, return}]``."""
        ...


class GoldGameBackend(GameBackend):
    """Delegates to the sample ``function_calls.Executor`` (records calls for F1)."""

    def __init__(self, executor: Any):
        self.executor = executor

    async def execute(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = [{"name": c["name"], "parameters": c.get("parameters", {})} for c in calls]
        return self.executor.execute(items)  # sync, no I/O


class KnowledgeGameBackend(GameBackend):
    """Answers tool calls from the NPC's knowledge base (interactive mode)."""

    _CHECK_FIELDS = {
        "check_price": "price",
        "check_type": "type",
        "check_attack": "attack",
        "check_level": "level",
        "check_duration": "duration",
        "check_reward": "reward",
    }

    def __init__(self, ctx):
        self.items: List[Dict[str, Any]] = (ctx.knowledge or {}).get("knowledge_info", []) or []

    def _find(self, target: Optional[str]) -> Optional[Dict[str, Any]]:
        if not target:
            return None
        target = str(target).strip().lower()
        for it in self.items:
            if str(it.get("name", "")).strip().lower() == target:
                return it
        return None

    def _check(self, name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        item = self._find(params.get("item_name") or params.get("quest_name"))
        if item is None:
            return [{"information": "n/a"}]
        field = self._CHECK_FIELDS.get(name)
        if field:
            return [{"name": item.get("name"), field: item.get(field, "n/a")}]
        return [item]

    def _search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        qtype = str(params.get("item_type") or params.get("quest_level") or "").lower()
        qname = str(params.get("item_name") or params.get("quest_name") or "").lower()
        qdesc = str(params.get("item_description") or params.get("quest_description") or "").lower()
        # Tokenise the name query so "light sword" can match partial item names/descriptions
        name_words = [w for w in qname.split() if len(w) > 2]
        desc_words = [w for w in qdesc.split() if len(w) > 2]

        results = []
        for it in self.items:
            it_name = str(it.get("name", "")).lower()
            it_type = str(it.get("type", "")).lower()
            it_desc = str(it.get("description", "")).lower()
            it_combined = it_name + " " + it_type + " " + it_desc
            if qtype and qtype not in it_type and qtype not in it_name:
                continue
            # item_name: accept if ALL query words appear anywhere in item's combined text
            if name_words and not all(w in it_combined for w in name_words):
                continue
            if desc_words and not any(w in it_combined for w in desc_words):
                continue
            results.append({"name": it.get("name"), "reason": it.get("description", "")})
        return results or [{"information": "n/a"}]

    async def execute(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for c in calls:
            name = c["name"]
            params = c.get("parameters", {}) or {}
            if name.startswith("search"):
                ret = self._search(params)
            elif name.startswith("check"):
                ret = self._check(name, params)
            else:  # actions: select / equip / sell / start / ...
                ret = [{"status": "success"}]
            out.append({"name": name, "parameters": params, "return": ret})
        return out

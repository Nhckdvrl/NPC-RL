"""Evaluator-compatible adapter.

Implements the same ``generate_functions_and_responses(...)`` contract as
``agents/openai_agent/main_agent.py`` so the existing pipeline can score the harness.
Drives the async two-phase engine on a persistent event loop, and routes Phase-2 execution
through the *provided* gold ``executor`` (wrapped in ``GoldGameBackend``) so tool calls are
recorded for F1 exactly as before.

Enable via ``HARNESS=npc`` in ``agents/user_config.py``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from .backend import GoldGameBackend
from .engine import FALLBACK_REPLY, NpcTurnEngine
from .npc_context import NpcContext
from .providers import OpenAICompatProvider
from .tools import build_registry_from_registries


class NPCHarnessAgent:
    """Two-phase NPC harness packaged as a drop-in for the evaluator."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._provider = None
        self._engine: NpcTurnEngine | None = None

    def _ensure_engine(self, ctx: NpcContext) -> NpcTurnEngine:
        if self._engine is None:
            self._provider = OpenAICompatProvider(
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                api_base=os.getenv("OPENAI_BASE_URL", "http://localhost:8112/v1"),
                default_model=ctx.resolved_model(),
            )
            self._engine = NpcTurnEngine(self._provider)
        return self._engine

    def generate_functions_and_responses(
        self,
        tool_registry: Dict,
        action_registry: Dict,
        worldview: str,
        persona: Dict[str, str],
        role: str,
        knowledge: Dict[str, Any],
        state: Dict[str, str],
        dialogue: List[Dict[str, str]],
        executor: Any,
        exam_id: str = None,
    ) -> Dict[str, Any]:
        ctx = NpcContext(
            name=(persona or {}).get("name", "NPC") if isinstance(persona, dict) else "NPC",
            worldview=worldview or "",
            role=role or "",
            persona=persona or {},
            state=state or {},
            knowledge=knowledge or {},
            model=os.getenv("OPENAI_MODEL", ""),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            toolcall_max_tokens=int(os.getenv("MAX_TOKENS_FUNCTION_CALL", "256")),
            reply_max_tokens=int(os.getenv("MAX_TOKENS_RESPONSE_GENERATION", "80")),
            turn_budget_s=float(os.getenv("NPC_TURN_BUDGET_S", "0")),
        )
        engine = self._ensure_engine(ctx)
        backend = GoldGameBackend(executor)
        registry = build_registry_from_registries(tool_registry, action_registry, backend)

        try:
            result = self._loop.run_until_complete(
                engine.run_turn(ctx, dialogue, registry=registry, backend=backend)
            )
            final = result["final_responses"] or FALLBACK_REPLY
        except Exception as e:
            print(f"NPCHarnessAgent ERROR: {type(e).__name__}: {e}")
            final = FALLBACK_REPLY

        return {"prompts": "", "final_responses": final}

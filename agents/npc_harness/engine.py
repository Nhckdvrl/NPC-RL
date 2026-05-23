"""NpcTurnEngine -- the two-phase NPC turn.

Per player turn (matching the model's training contract):

    Phase 1  toolcall  -> provider.chat(last player msg, tools) => Hermes tool calls
    Phase 2  execute   -> backend.execute(calls)                 => results
    Phase 3  roleplay  -> provider.chat(full history + results)  => short in-character reply

Each turn has an optional wall-clock budget (ctx.turn_budget_s).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from .backend import GameBackend
from .prompt_builder import NpcContextBuilder

FALLBACK_REPLY = "Hmm, it seems I missed part of what you said—could you say that again?"


class NpcTurnEngine:
    def __init__(self, provider, builder: Optional[NpcContextBuilder] = None):
        self.provider = provider
        self.builder = builder or NpcContextBuilder()

    async def run_turn(self, ctx, dialogue: List[Dict[str, Any]], *, registry, backend: GameBackend) -> Dict[str, Any]:
        start = time.monotonic()
        budget = ctx.turn_budget_s if ctx.turn_budget_s and ctx.turn_budget_s > 0 else float("inf")

        def remaining() -> float:
            return budget - (time.monotonic() - start)

        # Phase 1 -- toolcall
        calls: List[Dict[str, Any]] = []
        try:
            tc_msgs = self.builder.build_toolcall_messages(ctx, dialogue)
            resp = await self._chat(
                tc_msgs, ctx, tools=registry.get_definitions(),
                tool_choice="auto", max_tokens=ctx.toolcall_max_tokens, timeout=remaining(),
            )
            calls = self._to_calls(resp)
        except Exception as e:  # timeout or backend error -> proceed tool-free
            print(f"NpcTurnEngine: toolcall phase skipped ({type(e).__name__}: {e})")

        # Phase 2 -- execute
        tool_results: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        if calls:
            try:
                tool_results = await backend.execute(calls)
            except Exception as e:
                print(f"NpcTurnEngine: execute phase failed ({type(e).__name__}: {e})")
            functions = [{"name": c["name"], "parameters": c["parameters"]} for c in calls]

        # Phase 3 -- roleplay
        reply = FALLBACK_REPLY
        try:
            rp_msgs = self.builder.build_roleplay_messages(ctx, dialogue, tool_results)
            resp = await self._chat(
                rp_msgs, ctx, tools=None, max_tokens=ctx.reply_max_tokens, timeout=remaining(),
            )
            reply = (resp.content or "").strip() or FALLBACK_REPLY
        except Exception as e:
            print(f"NpcTurnEngine: roleplay phase failed ({type(e).__name__}: {e})")

        return {"final_responses": reply, "functions": functions, "tool_results": tool_results}

    async def _chat(self, messages, ctx, *, tools, max_tokens, tool_choice=None, timeout=float("inf")):
        coro = self.provider.chat(
            messages=messages,
            tools=tools,
            model=ctx.resolved_model(),
            max_tokens=max_tokens,
            temperature=ctx.temperature,
            tool_choice=tool_choice,
        )
        if timeout is None or timeout == float("inf"):
            return await coro
        if timeout <= 0:
            raise asyncio.TimeoutError("turn budget exhausted")
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    def _to_calls(resp) -> List[Dict[str, Any]]:
        if not getattr(resp, "tool_calls", None):
            return []
        return [
            {"id": tc.id, "name": tc.name, "parameters": tc.arguments or {}}
            for tc in resp.tool_calls
        ]

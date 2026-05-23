"""Dual prompt construction for the two-phase NPC turn.

Builds two distinct message lists per player turn:

  * Phase 1 (toolcall): only the latest player utterance -- no history.
    The Qwen3 chat template injects the tools block; prior assistant turns cause
    the model to stay in roleplay mode and skip tool calls.
  * Phase 3 (roleplay): full conversation history + injected tool results.

Ported and cleaned from ``agents/openai_agent/message_constructor.py``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _history_messages(dialogue: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for turn in dialogue or []:
        role = "assistant" if turn.get("speaker") == "npc" else "user"
        out.append({"role": role, "content": turn.get("text", "")})
    return out


def _kv_block(data: Dict[str, Any], empty: str = "N/A") -> str:
    parts = [f"{k}: {v}" for k, v in (data or {}).items()]
    return "\n".join(parts) if parts else empty


def _item_block(items: List[Dict[str, Any]], empty: str = "N/A") -> str:
    lines = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        name = it.get("name", "Unnamed")
        desc = it.get("description", "")
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    return "\n".join(lines) if lines else empty


def _results_block(tool_results: List[Dict[str, Any]]) -> str:
    """Render executed tool calls + their returns for the roleplay prompt."""
    if not tool_results:
        return ""
    lines = ["# Information you gathered (tool results)"]
    for func in tool_results:
        name = func.get("name", "unknown")
        params = func.get("parameters", {})
        returns = func.get("return", [])
        # n/a returns mean "the action succeeded / nothing to report"
        if isinstance(returns, list) and returns == [{"information": "n/a"}]:
            returns = [{"status": "success"}]
        lines.append(f"- {name}({json.dumps(params, ensure_ascii=False)}) -> "
                     f"{json.dumps(returns, ensure_ascii=False)}")
    return "\n".join(lines)


# The Qwen3 chat template auto-injects the tools block when tools= is passed to the API.
# A custom system message here is concatenated BEFORE that block, which can confuse the
# model and suppress tool calls.  We send no system message in the toolcall phase so the
# template's injected instruction is the only tool-calling context the model sees.


ROLEPLAY_SYSTEM = """\
You are an NPC in a video game, talking with a player. Stay fully in character.

# NPC Settings (act as this character)
{npc_info}

# Knowledge
## General
{general_knowledge}
## Items / Quests
{item_knowledge}

# Worldview
{worldview}

# Current State
{current_state}

# The player you are talking to
{role}

{results}

## Instructions
- Reply in character as the NPC above, using the gathered information and current situation.
- Reference only the provided knowledge, tool results, and state. Do not invent facts.
- Keep the reply short and natural.
Return only the NPC's spoken reply."""


class NpcContextBuilder:
    """Builds Phase-1 (toolcall) and Phase-3 (roleplay) message lists from an NpcContext."""

    def build_toolcall_messages(self, ctx, dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Send only the current player utterance — no history.
        # The toolcall SFT was single-turn; with prior assistant turns the model
        # switches into roleplay mode and stops emitting tool calls.
        last_player = next(
            (t for t in reversed(dialogue or []) if t.get("speaker") == "player"), None
        )
        if not last_player:
            return []
        return [{"role": "user", "content": last_player.get("text", "")}]

    def build_roleplay_messages(
        self,
        ctx,
        dialogue: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        knowledge = ctx.knowledge or {}
        system = ROLEPLAY_SYSTEM.format(
            npc_info=_kv_block(ctx.persona),
            general_knowledge=knowledge.get("general_info") or "N/A",
            item_knowledge=_item_block(knowledge.get("knowledge_info", [])),
            worldview=ctx.worldview or "N/A",
            current_state=_kv_block(ctx.state),
            role=ctx.role or "An adventurer.",
            results=_results_block(tool_results),
        )
        return [{"role": "system", "content": system.strip()}, *_history_messages(dialogue)]

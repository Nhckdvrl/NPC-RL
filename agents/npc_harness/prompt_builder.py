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

from .memory import extract_memory, render_memory


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
        # Action tools often return no payload; query tools should keep n/a visible.
        if (
            isinstance(returns, list)
            and returns == [{"information": "n/a"}]
            and not (name.startswith("check") or name.startswith("search"))
        ):
            returns = [{"status": "success"}]
        lines.append(f"- {name}({json.dumps(params, ensure_ascii=False)}) -> "
                     f"{json.dumps(returns, ensure_ascii=False)}")
    return "\n".join(lines)


def _tool_result_instruction(tool_results: List[Dict[str, Any]]) -> str:
    names = {str(r.get("name", "")) for r in tool_results or []}
    if any(n.startswith("search") for n in names):
        return "The player is asking for options or recommendations. Mention matching item names and short reasons from the current search results. Do not mention prices unless prices are present in the current tool results."
    if "check_price" in names:
        return "The player is asking for price. State the exact current price."
    if "check_attack" in names:
        return "The player is asking about attack or damage. State the exact current attack value."
    if "check_description" in names or "check_basic_info" in names:
        return "The player is asking for details. Summarize the current item description and relevant facts from the current tool result."
    return "Use the current tool results directly."


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

# Conversation Memory
{memory}

# The player you are talking to
{role}

## Instructions
- Reply in character as the NPC above, using the current turn's tool results when present.
- Use conversation memory for direct references such as the player's name or "it / that one / 多少呢".
- If the player previously told you their name and later asks who they are or asks for "the name" in that context, answer with the player's name from memory.
- If the player asks for an exact factual value such as price, attack, type, level, reward, or duration, include the exact value from the tool result.
- Reference only the provided knowledge, tool results, and state. Do not invent facts.
- Reply in the same language as the player's latest message unless the player asks otherwise.
- Keep the reply short and natural.
Return only the NPC's spoken reply."""


CURRENT_TURN_WITH_RESULTS = """\
Latest player message (reply in {language}):
{player_text}

Current turn tool results:
{results}

Current turn answer policy:
{tool_instruction}

Answer the player's latest message using the current tool results above. Current turn tool results override older conversation history."""


CURRENT_TURN = """\
Latest player message (reply in {language}):
{player_text}"""


def _language_hint(text: str) -> str:
    return "Chinese" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "English"


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
        memory = extract_memory(ctx, dialogue[:-1])
        if memory:
            content = (
                "Conversation memory for resolving references only:\n"
                f"{render_memory(memory)}\n\n"
                f"Player says:\n{last_player.get('text', '')}"
            )
        else:
            content = last_player.get("text", "")
        return [{"role": "user", "content": content}]

    def build_roleplay_messages(
        self,
        ctx,
        dialogue: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        knowledge = ctx.knowledge or {}
        memory = extract_memory(ctx, dialogue)
        system = ROLEPLAY_SYSTEM.format(
            npc_info=_kv_block(ctx.persona),
            general_knowledge=knowledge.get("general_info") or "N/A",
            item_knowledge=_item_block(knowledge.get("knowledge_info", [])),
            worldview=ctx.worldview or "N/A",
            current_state=_kv_block(ctx.state),
            memory=render_memory(memory),
            role=ctx.role or "An adventurer.",
        )
        current_player = next(
            (t for t in reversed(dialogue or []) if t.get("speaker") == "player"), None
        )
        prior_dialogue = list(dialogue or [])
        if current_player and prior_dialogue and prior_dialogue[-1] is current_player:
            prior_dialogue = prior_dialogue[:-1]

        messages = [{"role": "system", "content": system.strip()}, *_history_messages(prior_dialogue)]
        player_text = current_player.get("text", "") if current_player else ""
        if tool_results:
            messages.append({
                "role": "user",
                "content": CURRENT_TURN_WITH_RESULTS.format(
                    language=_language_hint(player_text),
                    player_text=player_text,
                    results=_results_block(tool_results),
                    tool_instruction=_tool_result_instruction(tool_results),
                ).strip(),
            })
        elif player_text:
            messages.append({
                "role": "user",
                "content": CURRENT_TURN.format(
                    language=_language_hint(player_text),
                    player_text=player_text,
                ).strip(),
            })
        return messages

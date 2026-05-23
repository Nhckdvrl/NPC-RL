"""Lightweight conversation memory for live NPC sessions."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def extract_memory(ctx, dialogue: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract stable, explicit state from recent dialogue.

    This is not an LLM summarizer. It only records facts that are directly
    stated or grounded in the scenario knowledge, so it is safe to inject into
    both toolcall and roleplay prompts.
    """

    memory: Dict[str, Any] = {}
    player_name = _extract_player_name(dialogue)
    if player_name:
        memory["player_name"] = player_name

    last_item = _extract_last_item(ctx, dialogue)
    if last_item:
        memory["last_item"] = last_item.get("name")

    return memory


def render_memory(memory: Dict[str, Any]) -> str:
    if not memory:
        return "No stable conversation memory yet."
    lines = []
    if memory.get("player_name"):
        lines.append(f"Player name: {memory['player_name']} (use this when the player asks who they are or asks about their name)")
    if memory.get("last_item"):
        lines.append(f"Last focused item: {memory['last_item']}")
    return "\n".join(lines)


def _extract_player_name(dialogue: List[Dict[str, Any]]) -> str | None:
    patterns = [
        re.compile(r"^\s*(?:我叫|我是)\s*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_\-]{0,20})\s*[。.!！]?\s*$"),
        re.compile(r"^\s*(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z0-9_\-]{0,30})\s*[.!]?\s*$", re.I),
    ]
    for turn in reversed(dialogue or []):
        if turn.get("speaker") != "player":
            continue
        text = str(turn.get("text", "")).strip()
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip("。.!?？,， ")
                if name and name.lower() not in {"garrick", "npc", "谁", "誰", "who"}:
                    return name
    return None


def _extract_last_item(ctx, dialogue: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for turn in reversed(dialogue or []):
        text = str(turn.get("text", ""))
        match = match_known_item(ctx, text)
        if match:
            return match
    return None


def match_known_item(ctx, text: str) -> Dict[str, Any] | None:
    lowered = str(text).lower()
    best: tuple[int, Dict[str, Any]] | None = None
    for item in (ctx.knowledge or {}).get("knowledge_info", []) or []:
        names = [item.get("name", ""), *(item.get("aliases") or [])]
        for raw_name in names:
            name = str(raw_name).strip().lower()
            if not name or name not in lowered:
                continue
            score = len(name)
            if best is None or score > best[0]:
                best = (score, item)
    return best[1] if best else None

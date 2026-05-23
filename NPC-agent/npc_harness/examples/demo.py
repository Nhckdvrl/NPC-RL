"""Offline end-to-end demo of the NPC harness -- no GPU / no vLLM needed.

Runs the *real* harness (in-project MessageBus, LLMProvider, ToolRegistry,
SessionManager, and the two-phase NpcTurnEngine / NpcAgentLoop) over a
scripted conversation.  The only thing faked is the model itself:
``ScriptedNpcProvider`` is a real LLMProvider subclass that imitates the
trained model's two-phase behavior with simple rules, so you can watch the
toolcall -> execute -> roleplay pipeline end to end without a GPU.

Run:
    python NPC-agent/npc_harness/examples/demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from npc_harness.bus import InboundMessage, MessageBus
from npc_harness.backend import GameBackend, KnowledgeGameBackend
from npc_harness.engine import NpcTurnEngine
from npc_harness.loop import NpcAgentLoop
from npc_harness.npc_context import load_context
from npc_harness.providers import LLMProvider, LLMResponse, ToolCallRequest
from npc_harness.registry import ToolRegistry
from npc_harness.sessions import SessionManager
from npc_harness.tools import GameTool


# --------------------------------------------------------------------------- model stub

class ScriptedNpcProvider(LLMProvider):
    """Simulates the trained two-phase NPC model with simple rules."""

    def __init__(self, items):
        self.items = items
        self._last_item = None

    async def chat(self, messages, *, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, tool_choice=None, **kwargs) -> LLMResponse:
        text = next((m.get("content") or "" for m in reversed(messages)
                     if m.get("role") == "user"), "")
        low = text.lower()
        if tools:  # Phase 1 -- decide which game functions to call
            return LLMResponse(content=None, tool_calls=self._decide(low), finish_reason="tool_calls")
        # Phase 3 -- in-character reply, grounded in injected tool results
        sys_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        return LLMResponse(content=self._reply(low, sys_prompt), finish_reason="stop")

    def _match_item(self, low: str):
        return next((it for it in self.items if str(it["name"]).lower() in low), None)

    def _decide(self, low: str):
        item = self._match_item(low)
        if item:
            self._last_item = item
        buy_item = item or self._last_item

        def tc(name, args):
            return ToolCallRequest(id=f"call_{name}", name=name, arguments=args)

        if any(k in low for k in ("how much", "price", "cost")) and item:
            return [tc("check_price", {"item_name": item["name"]})]
        if any(k in low for k in ("i'll take", "ill take", "i will take", "buy", "purchase")) and buy_item:
            return [tc("sell", {"item_name": [buy_item["name"]]})]
        if any(k in low for k in ("show", "what", "any", "got", "looking", "need", "have")) \
                and any(t in low for t in ("sword", "axe", "bow", "spear", "whip", "weapon")):
            args = {}
            for t in ("sword", "axe", "bow", "spear", "whip"):
                if t in low:
                    args["item_type"] = "single-handed sword" if t == "sword" else t
                    break
            if "light" in low or "cheap" in low:
                args["item_description"] = "light"
            return [tc("search_item", args or {"item_description": "weapon"})]
        return []

    @staticmethod
    def _reply(low: str, sys_prompt: str) -> str:
        results = sys_prompt.split("# Information you gathered")[-1] if "# Information" in sys_prompt else ""
        names = list(dict.fromkeys(re.findall(r'"name":\s*"([^"]+)"', results)))
        prices = re.findall(r'"price":\s*"([^"]+)"', results)
        if any(k in low for k in ("i'll take", "ill take", "i will take", "buy", "purchase")):
            return "A fine choice. Here you are — may it serve you well out in the marsh."
        if prices and names:
            return f"The {names[0]}? That'll run you {prices[0]}. Honest steel, mind you."
        if names:
            listed = ", ".join(names)
            return f"Aye, I've got the {listed}. Any of 'em catch your eye?"
        if any(k in low for k in ("hello", "hi", "hey", "greet")):
            return "Welcome to my armory, traveler. Lookin' to arm yourself?"
        return "Hmph. Speak plainly — what is it you're after?"


# ------------------------------------------------------------------------- tool wiring

class TracingBackend(GameBackend):
    """Wraps a backend and prints each call/result so the demo shows Phase 2."""

    def __init__(self, inner: GameBackend):
        self.inner = inner

    async def execute(self, calls):
        results = await self.inner.execute(calls)
        for c, r in zip(calls, results):
            params = json.dumps(c.get("parameters", {}), ensure_ascii=False)
            ret = json.dumps(r.get("return"), ensure_ascii=False)
            print(f"    [tool] {c['name']}({params}) -> {ret}")
        return results


def _schema(name, desc, props, required=()):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}


def build_demo_registry(backend: GameBackend) -> ToolRegistry:
    reg = ToolRegistry()
    name_arg = {"item_name": {"type": "string", "description": "weapon name"}}
    reg.register(GameTool(_schema(
        "search_item", "Search weapons by type/price/feature.",
        {"item_type": {"type": "string"}, "item_description": {"type": "string"}}), backend, "query"))
    reg.register(GameTool(_schema(
        "check_price", "Check a weapon's price.", name_arg, ["item_name"]), backend, "query"))
    reg.register(GameTool(_schema(
        "check_basic_info", "Check a weapon's basic info.", name_arg, ["item_name"]), backend, "query"))
    reg.register(GameTool(_schema(
        "sell", "Sell weapon(s) to the player.",
        {"item_name": {"type": "array", "items": {"type": "string"}}}, ["item_name"]), backend, "action"))
    reg.register(GameTool(_schema(
        "equip", "Equip a weapon for the player.", name_arg, ["item_name"]), backend, "action"))
    return reg


# ------------------------------------------------------------------------------- driver

SCRIPT = [
    "Hello there!",
    "What light swords do you have?",
    "How much is the Short Sword?",
    "I'll take it.",
]


async def main() -> None:
    ctx = load_context(os.path.join(os.path.dirname(__file__), "shopkeeper.yaml"))

    bus = MessageBus()
    provider = ScriptedNpcProvider(ctx.knowledge.get("knowledge_info", []))
    backend = TracingBackend(KnowledgeGameBackend(ctx))
    registry = build_demo_registry(backend)
    engine = NpcTurnEngine(provider)

    workspace = Path(tempfile.mkdtemp(prefix="npc_demo_"))
    sessions = SessionManager(workspace)
    loop = NpcAgentLoop(bus, engine, sessions, resolver=lambda _m: (ctx, registry, backend))
    loop_task = asyncio.create_task(loop.run())

    print(f"=== NPC harness demo — talking to {ctx.name} (scripted model, no GPU) ===")
    print(f"    tools available: {registry.tool_names}\n")
    for line in SCRIPT:
        print(f"Player> {line}")
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="p1", chat_id="demo", content=line)
        )
        out = await bus.consume_outbound()
        print(f"{ctx.name}> {out.content}\n")

    loop.stop()
    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)

    session = sessions.get_or_create("cli:demo")
    print(f"[session persisted: {len(session.messages)} messages at {workspace}/sessions]")


if __name__ == "__main__":
    asyncio.run(main())

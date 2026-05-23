"""NpcAgentLoop -- bus-driven driver for the two-phase NPC turn.

Pulls InboundMessages off the bus, runs NpcTurnEngine (toolcall -> execute -> roleplay),
persists history via SessionManager, and pushes OutboundMessages back onto the bus.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Tuple

from .backend import GameBackend
from .bus import InboundMessage, MessageBus, OutboundMessage
from .engine import FALLBACK_REPLY, NpcTurnEngine

# resolve(InboundMessage) -> (ctx, ToolRegistry, GameBackend)
Resolver = Callable[[InboundMessage], Tuple[Any, Any, GameBackend]]


class NpcAgentLoop:
    def __init__(self, bus: MessageBus, engine: NpcTurnEngine, sessions: Any, resolver: Resolver):
        self.bus = bus
        self.engine = engine
        self.sessions = sessions
        self.resolve = resolver
        self._running = False

    @staticmethod
    def _session_to_dialogue(session) -> List[Dict[str, str]]:
        dialogue: List[Dict[str, str]] = []
        for m in session.get_history():
            content = m.get("content")
            if not isinstance(content, str):
                continue
            role = m.get("role")
            if role == "user":
                dialogue.append({"speaker": "player", "text": content})
            elif role == "assistant":
                dialogue.append({"speaker": "npc", "text": content})
        return dialogue

    async def _dispatch(self, msg: InboundMessage) -> None:
        ctx, registry, backend = self.resolve(msg)
        session = self.sessions.get_or_create(msg.session_key)

        dialogue = self._session_to_dialogue(session)
        dialogue.append({"speaker": "player", "text": msg.content})

        result = await self.engine.run_turn(ctx, dialogue, registry=registry, backend=backend)
        reply = result["final_responses"]

        session.add_message("user", msg.content)
        session.add_message("assistant", reply)
        self.sessions.save(session)

        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=reply,
                metadata={
                    "functions": result.get("functions", []),
                    "tool_results": result.get("tool_results", []),
                },
            )
        )

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self._dispatch(msg)
            except Exception as e:
                print(f"NpcAgentLoop: dispatch failed ({type(e).__name__}: {e})")
                await self.bus.publish_outbound(
                    OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=FALLBACK_REPLY)
                )

    def stop(self) -> None:
        self._running = False

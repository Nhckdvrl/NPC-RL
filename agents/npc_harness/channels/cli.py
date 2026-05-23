"""A stdin/stdout debug channel.

Reads a line from stdin, publishes it as an InboundMessage; prints OutboundMessages back.
Real channels (websocket, game-engine adapter, etc.) replace this for production use.
"""

from __future__ import annotations

import asyncio
import sys

from ..bus import InboundMessage, OutboundMessage
from .base import BaseChannel


class CliChannel(BaseChannel):
    name = "cli"
    display_name = "CLI"

    def __init__(self, bus, npc_name: str = "NPC"):
        super().__init__(bus)
        self.npc_name = npc_name

    async def start(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        print(f"== Talking to {self.npc_name}. Type /quit to exit. ==")
        while self._running:
            print("You> ", end="", flush=True)
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:  # EOF
                break
            text = line.strip()
            if text in ("/quit", "/exit"):
                break
            if not text:
                continue
            await self.bus.publish_inbound(
                InboundMessage(channel=self.name, sender_id="player", chat_id="cli", content=text)
            )
        self._running = False

    async def stop(self) -> None:
        self._running = False

    async def send(self, msg: OutboundMessage) -> None:
        print(f"\n{self.npc_name}> {msg.content}\n", flush=True)

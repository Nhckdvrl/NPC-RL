"""Abstract channel interface -- standalone reimplementation of nanobot's channel layer.

Architectural concept: a Channel owns one transport (CLI, WebSocket, …); it
reads from that transport, wraps incoming text as InboundMessages and pushes
them onto the bus, and reads OutboundMessages from the bus to deliver back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..bus import InboundMessage, MessageBus, OutboundMessage  # noqa: F401 (re-export)


class BaseChannel(ABC):
    name: str = "base"

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None: ...

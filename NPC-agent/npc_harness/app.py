"""Wire the harness together and run it over a channel.

Connects MessageBus, OpenAICompatProvider (pointed at the vLLM server), SessionManager,
NpcTurnEngine, NpcAgentLoop, and CliChannel into a runnable interactive NPC session.

    OPENAI_BASE_URL=http://localhost:8112/v1 OPENAI_MODEL=<served-model> \\
    python -m agents.npc_harness --context agents/npc_harness/examples/shopkeeper.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from .backend import KnowledgeGameBackend
from .bus import MessageBus
from .channels.cli import CliChannel
from .engine import NpcTurnEngine
from .loop import NpcAgentLoop
from .npc_context import load_context
from .providers import OpenAICompatProvider
from .sessions import SessionManager
from .tools import build_registry_from_context


def build_provider(ctx) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        api_base=os.getenv("OPENAI_BASE_URL", "http://localhost:8112/v1"),
        default_model=ctx.resolved_model(),
    )


async def _outbound_pump(bus: MessageBus, channel: CliChannel) -> None:
    while True:
        msg = await bus.consume_outbound()
        await channel.send(msg)


async def run_cli(context_path: str, workspace: str | None = None) -> None:
    ctx = load_context(context_path)

    bus = MessageBus()
    provider = build_provider(ctx)
    backend = KnowledgeGameBackend(ctx)
    registry = build_registry_from_context(ctx, backend)
    engine = NpcTurnEngine(provider)

    ws = Path(workspace) if workspace else Path(tempfile.gettempdir()) / "npc_harness"
    sessions = SessionManager(ws)

    loop = NpcAgentLoop(bus, engine, sessions, resolver=lambda _msg: (ctx, registry, backend))
    channel = CliChannel(bus, npc_name=ctx.name)

    pump = asyncio.create_task(_outbound_pump(bus, channel))
    loop_task = asyncio.create_task(loop.run())
    try:
        await channel.start()  # returns on /quit or EOF
    finally:
        loop.stop()
        pump.cancel()
        loop_task.cancel()
        await asyncio.gather(pump, loop_task, return_exceptions=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run an NPC harness agent (CLI debug channel).")
    p.add_argument("--context", required=True, help="Path to an NPC .yaml/.json context file")
    p.add_argument("--workspace", default=None, help="Directory for session storage")
    args = p.parse_args(argv)
    asyncio.run(run_cli(args.context, args.workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

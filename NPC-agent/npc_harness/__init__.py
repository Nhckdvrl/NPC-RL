"""NPC harness for the trained NPC-RL model.

Self-contained agent runtime inspired by nanobot's architecture (decoupled bus, provider,
registry, session, channel) but with zero external framework dependencies.  Replaces the
open ReAct brain with a two-phase ``NpcTurnEngine`` matching the model's training contract:
toolcall -> execute -> roleplay, once per player turn.

Layout:
    bus.py           MessageBus + Inbound/OutboundMessage
    providers.py     LLMProvider (abstract) + OpenAICompatProvider
    registry.py      Tool (abstract) + ToolRegistry
    sessions.py      Session + SessionManager (JSONL persistence)
    channels/        BaseChannel + CliChannel
    npc_context.py   NpcContext + load_context
    prompt_builder.py NpcContextBuilder (dual: toolcall + roleplay prompts)
    backend.py       GameBackend + Gold/Knowledge impls (Phase-2 execution seam)
    tools.py         GameTool builders (query vs action)
    engine.py        NpcTurnEngine (two-phase brain)
    loop.py          NpcAgentLoop (bus-driven)
    app.py           wiring + run (vLLM provider + sessions + loop)
    eval_adapter.py  NPCHarnessAgent (drop-in for the existing evaluator)
"""

from .npc_context import NpcContext, context_from_dict, load_context

__all__ = ["NpcContext", "context_from_dict", "load_context"]

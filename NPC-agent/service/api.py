"""FastAPI wrapper around the two-phase NPC harness.

The model itself is served by vLLM's OpenAI-compatible endpoint.  This service
owns the game-agent contract: session history, tool execution, and the
toolcall -> backend -> roleplay turn pipeline.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.npc_harness.backend import GameBackend, KnowledgeGameBackend
from agents.npc_harness.bus import InboundMessage, MessageBus
from agents.npc_harness.engine import NpcTurnEngine
from agents.npc_harness.loop import NpcAgentLoop
from agents.npc_harness.npc_context import NpcContext, load_context
from agents.npc_harness.providers import LLMProvider, LLMResponse, OpenAICompatProvider, ToolCallRequest
from agents.npc_harness.sessions import SessionManager
from agents.npc_harness.registry import ToolRegistry
from agents.npc_harness.tools import GameTool, build_registry_from_context


DEFAULT_CONTEXT = ROOT / "agents" / "npc_harness" / "examples" / "shopkeeper.yaml"
DEFAULT_WORKSPACE = Path(os.getenv("NPC_SERVICE_WORKSPACE", tempfile.gettempdir())) / "npc_service"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    context_path: str | None = None
    provider: Literal["vllm", "scripted"] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reply_max_tokens: int | None = Field(default=None, ge=16, le=512)
    toolcall_max_tokens: int | None = Field(default=None, ge=16, le=1024)


class ToolTrace(BaseModel):
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class ChatResponse(BaseModel):
    session_id: str
    npc: str
    reply: str
    tools: list[ToolTrace] = Field(default_factory=list)


class ScriptedDemoProvider(LLMProvider):
    """No-GPU provider used by the web demo when vLLM is not running."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self._last_item: dict[str, Any] | None = None

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        text = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        low = text.lower()
        if tools:
            return LLMResponse(content=None, tool_calls=self._decide(low), finish_reason="tool_calls")
        return LLMResponse(content=self._reply(low), finish_reason="stop")

    def _match_item(self, low: str) -> dict[str, Any] | None:
        return next((it for it in self.items if str(it.get("name", "")).lower() in low), None)

    def _decide(self, low: str) -> list[ToolCallRequest]:
        item = self._match_item(low)
        if item:
            self._last_item = item
        buy_item = item or self._last_item

        def tc(name: str, args: dict[str, Any]) -> ToolCallRequest:
            return ToolCallRequest(id=f"call_{name}", name=name, arguments=args)

        if any(k in low for k in ("how much", "price", "cost")) and item:
            return [tc("check_price", {"item_name": item["name"]})]
        if any(k in low for k in ("attack", "hit", "damage")) and item:
            return [tc("check_attack", {"item_name": item["name"]})]
        if any(k in low for k in ("take", "buy", "purchase")) and buy_item:
            return [tc("sell", {"item_name": [buy_item["name"]]})]
        if any(k in low for k in ("show", "what", "any", "got", "looking", "need", "have")):
            args: dict[str, Any] = {}
            for item_type in ("sword", "axe", "bow", "spear", "whip"):
                if item_type in low:
                    args["item_type"] = "single-handed sword" if item_type == "sword" else item_type
                    break
            if "light" in low or "cheap" in low:
                args["item_description"] = "light"
            if args:
                return [tc("search_item", args)]
        return []

    def _reply(self, low: str) -> str:
        if any(k in low for k in ("hello", "hi", "hey")):
            return "Welcome to my armory, traveler. Looking for steel or just shelter from the marsh?"
        if any(k in low for k in ("take", "buy", "purchase")) and self._last_item:
            return f"Done. The {self._last_item['name']} is yours. Keep the edge dry."
        if self._last_item and any(k in low for k in ("price", "cost", "how much")):
            return f"The {self._last_item['name']} costs {self._last_item.get('price', 'a fair bit')}."
        if self._last_item and any(k in low for k in ("attack", "hit", "damage")):
            return f"The {self._last_item['name']} hits at {self._last_item.get('attack', 'unknown')} attack."
        if self._last_item:
            return f"I'd point you at the {self._last_item['name']}. {self._last_item.get('description', '')}"
        return "Speak plainly and I will check the racks for you."


class ServiceState:
    def __init__(self) -> None:
        self.sessions = SessionManager(DEFAULT_WORKSPACE)
        self._lock = asyncio.Lock()

    def load_ctx(self, context_path: str | None, req: ChatRequest) -> NpcContext:
        path = Path(context_path or os.getenv("NPC_CONTEXT", str(DEFAULT_CONTEXT))).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"NPC context not found: {path}")
        ctx = load_context(str(path))
        if req.temperature is not None:
            ctx.temperature = req.temperature
        if req.reply_max_tokens is not None:
            ctx.reply_max_tokens = req.reply_max_tokens
        if req.toolcall_max_tokens is not None:
            ctx.toolcall_max_tokens = req.toolcall_max_tokens
        return ctx

    def build_provider(self, ctx: NpcContext, provider_name: str | None) -> LLMProvider:
        selected = provider_name or os.getenv("NPC_PROVIDER", ctx.provider).lower()
        if selected == "scripted":
            return ScriptedDemoProvider(ctx.knowledge.get("knowledge_info", []) if ctx.knowledge else [])
        return OpenAICompatProvider(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            api_base=os.getenv("OPENAI_BASE_URL", "http://localhost:8112/v1"),
            default_model=ctx.resolved_model(),
        )

    def build_registry(self, ctx: NpcContext, backend: GameBackend) -> ToolRegistry:
        try:
            return build_registry_from_context(ctx, backend)
        except ModuleNotFoundError as exc:
            if exc.name not in {"langchain", "langchain_core"}:
                raise
            return build_lightweight_registry(backend)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        ctx = self.load_ctx(req.context_path, req)
        backend: GameBackend = KnowledgeGameBackend(ctx)
        registry = self.build_registry(ctx, backend)
        provider = self.build_provider(ctx, req.provider)
        engine = NpcTurnEngine(provider)

        session_id = req.session_id or uuid4().hex
        bus = MessageBus()
        loop = NpcAgentLoop(bus, engine, self.sessions, resolver=lambda _msg: (ctx, registry, backend))
        loop_task = asyncio.create_task(loop.run())
        try:
            await bus.publish_inbound(
                InboundMessage(
                    channel="web",
                    sender_id=session_id,
                    chat_id=session_id,
                    content=req.message,
                )
            )
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=max(ctx.turn_budget_s, 0) or 120)
        finally:
            loop.stop()
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)

        reply = clean_spoken_reply(outbound.content)
        if reply != outbound.content:
            session = self.sessions.get_or_create(f"web:{session_id}")
            for msg in reversed(session.messages):
                if msg.get("role") == "assistant" and msg.get("content") == outbound.content:
                    msg["content"] = reply
                    self.sessions.save(session)
                    break

        tool_results = outbound.metadata.get("tool_results") or []
        traces = [
            ToolTrace(
                name=tr.get("name", ""),
                parameters=tr.get("parameters") or {},
                result=tr.get("return"),
            )
            for tr in tool_results
        ]
        return ChatResponse(session_id=session_id, npc=ctx.name, reply=reply, tools=traces)


def clean_spoken_reply(text: str) -> str:
    """Remove leading planning tags sometimes learned from training transcripts."""

    text = text.strip()
    text = re.sub(r"^(?:\[[^\]]{1,140}\]\s*)+", "", text).strip()
    return text or "Could you say that again?"


def _schema(name: str, desc: str, props: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)},
    }


def build_lightweight_registry(backend: GameBackend) -> ToolRegistry:
    """Small OpenAI-tool registry for demos when LangChain is not installed."""

    reg = ToolRegistry()
    item_name = {"item_name": {"type": "string", "description": "Weapon name."}}
    reg.register(GameTool(_schema(
        "search_item",
        "Search weapons by type, name, price, attack level, or description.",
        {
            "item_name": {"type": "string"},
            "item_price": {"type": "string"},
            "item_type": {"type": "string"},
            "item_attack": {"type": "string"},
            "item_description": {"type": "string"},
            "item_price_operator": {"type": "string"},
        },
    ), backend, "query"))
    for name, desc in [
        ("check_basic_info", "Check a weapon's basic information."),
        ("check_price", "Check a weapon's price."),
        ("check_type", "Check a weapon's type."),
        ("check_attack", "Check a weapon's attack value."),
        ("check_description", "Check a weapon's description."),
    ]:
        reg.register(GameTool(_schema(name, desc, item_name, ("item_name",)), backend, "query"))
    reg.register(GameTool(_schema(
        "sell",
        "Sell one or more weapons to the player.",
        {"item_name": {"type": "array", "items": {"type": "string"}}},
        ("item_name",),
    ), backend, "action"))
    reg.register(GameTool(_schema("equip", "Equip a weapon for the player.", item_name, ("item_name",)), backend, "action"))
    return reg


app = FastAPI(
    title="NPC-RL Playable Agent API",
    description="FastAPI service for a vLLM-served post-trained NPC dialogue model.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("NPC_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
state = ServiceState()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_context": os.getenv("NPC_CONTEXT", str(DEFAULT_CONTEXT)),
        "provider": os.getenv("NPC_PROVIDER", "vllm"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:8112/v1"),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    return await state.chat(req)

"""NpcTurnEngine -- the two-phase NPC turn.

Per player turn (matching the model's training contract):

    Phase 1  toolcall  -> provider.chat(last player msg, tools) => Hermes tool calls
    Phase 2  execute   -> backend.execute(calls)                 => results
    Phase 3  roleplay  -> provider.chat(full history + results)  => short in-character reply

Each turn has an optional wall-clock budget (ctx.turn_budget_s).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from .backend import GameBackend
from .memory import extract_memory, match_known_item
from .prompt_builder import NpcContextBuilder

FALLBACK_REPLY = "Hmm, it seems I missed part of what you said—could you say that again?"


class NpcTurnEngine:
    def __init__(self, provider, builder: Optional[NpcContextBuilder] = None):
        self.provider = provider
        self.builder = builder or NpcContextBuilder()

    async def run_turn(self, ctx, dialogue: List[Dict[str, Any]], *, registry, backend: GameBackend) -> Dict[str, Any]:
        start = time.monotonic()
        budget = ctx.turn_budget_s if ctx.turn_budget_s and ctx.turn_budget_s > 0 else float("inf")

        def remaining() -> float:
            return budget - (time.monotonic() - start)

        # Phase 1 -- toolcall
        calls: List[Dict[str, Any]] = []
        try:
            tc_msgs = self.builder.build_toolcall_messages(ctx, dialogue)
            resp = await self._chat(
                tc_msgs, ctx, tools=registry.get_definitions(),
                tool_choice="auto", max_tokens=ctx.toolcall_max_tokens, timeout=remaining(),
            )
            calls = self._to_calls(resp)
        except Exception as e:  # timeout or backend error -> proceed tool-free
            print(f"NpcTurnEngine: toolcall phase skipped ({type(e).__name__}: {e})")

        # Phase 2 -- execute
        tool_results: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        calls = self._repair_calls(ctx, dialogue, calls)
        if calls:
            try:
                tool_results = await backend.execute(calls)
            except Exception as e:
                print(f"NpcTurnEngine: execute phase failed ({type(e).__name__}: {e})")
            functions = [{"name": c["name"], "parameters": c["parameters"]} for c in calls]

        # Phase 3 -- roleplay
        reply = FALLBACK_REPLY
        try:
            rp_msgs = self.builder.build_roleplay_messages(ctx, dialogue, tool_results)
            resp = await self._chat(
                rp_msgs, ctx, tools=None, max_tokens=ctx.reply_max_tokens, timeout=remaining(),
            )
            reply = (resp.content or "").strip() or FALLBACK_REPLY
        except Exception as e:
            print(f"NpcTurnEngine: roleplay phase failed ({type(e).__name__}: {e})")

        reply = self._ground_memory_reply(ctx, dialogue, reply)
        reply = self._ground_factual_reply(dialogue, reply, tool_results)
        return {"final_responses": reply, "functions": functions, "tool_results": tool_results}

    async def _chat(self, messages, ctx, *, tools, max_tokens, tool_choice=None, timeout=float("inf")):
        coro = self.provider.chat(
            messages=messages,
            tools=tools,
            model=ctx.resolved_model(),
            max_tokens=max_tokens,
            temperature=ctx.temperature,
            top_p=getattr(ctx, "top_p", None),
            tool_choice=tool_choice,
        )
        if timeout is None or timeout == float("inf"):
            return await coro
        if timeout <= 0:
            raise asyncio.TimeoutError("turn budget exhausted")
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    def _to_calls(resp) -> List[Dict[str, Any]]:
        if not getattr(resp, "tool_calls", None):
            return []
        return [
            {"id": tc.id, "name": tc.name, "parameters": tc.arguments or {}}
            for tc in resp.tool_calls
        ]

    @staticmethod
    def _repair_calls(ctx, dialogue: List[Dict[str, Any]], calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ground obvious factual queries to the exact game API.

        The post-trained model is still allowed to choose tools, but live game
        agents should defensively normalize high-confidence cases such as a
        bilingual "how much is <item>" query. This keeps the harness faithful to
        game backend semantics when the model emits a nearby but wrong search call.
        """

        latest = next((t for t in reversed(dialogue or []) if t.get("speaker") == "player"), {})
        latest_text = str(latest.get("text", ""))

        lowered = latest_text.lower()
        repair_name = None
        if any(k in lowered for k in ("how much", "price", "cost", "多少钱", "多少", "价格", "價錢", "售价", "售價")):
            repair_name = "check_price"
        elif any(k in lowered for k in ("attack", "damage", "hit", "攻击", "攻擊", "伤害", "傷害")):
            repair_name = "check_attack"
        elif any(k in lowered for k in ("type", "kind", "类型", "類型", "种类", "種類")):
            repair_name = "check_type"
        elif any(k in lowered for k in ("describe", "description", "tell me about", "介绍", "介紹", "说明", "說明")):
            repair_name = "check_description"

        if not repair_name:
            return calls

        matched_item = NpcTurnEngine._resolve_item_for_repair(ctx, dialogue, latest_text)
        if not matched_item:
            return calls

        repaired = {"id": f"repair_{repair_name}", "name": repair_name, "parameters": {"item_name": matched_item["name"]}}
        others = [c for c in calls if not str(c.get("name", "")).startswith(("search", "check"))]
        return [repaired, *others]

    @staticmethod
    def _resolve_item_for_repair(ctx, dialogue: List[Dict[str, Any]], latest_text: str) -> Dict[str, Any] | None:
        current = match_known_item(ctx, latest_text)
        if current:
            return current

        # Context is only for anaphora/ellipsis. If the current turn names an
        # item, the current turn wins; if it says "how much is it/多少呢", use the
        # most recent player-mentioned item. This prevents older entities from
        # overriding a new explicit item.
        lowered = latest_text.lower().strip()
        anaphoric = (
            lowered in {"how much", "how much?", "price?", "cost?", "多少", "多少呢", "多少钱", "多少钱啊"}
            or any(k in lowered for k in ("it", "that one", "它", "这个", "那个", "這個", "那個"))
        )
        if not anaphoric:
            return None

        prior = list(dialogue or [])[:-1]
        for turn in reversed(prior):
            if turn.get("speaker") == "player":
                match = match_known_item(ctx, str(turn.get("text", "")))
                if match:
                    return match
        for turn in reversed(prior):
            if turn.get("speaker") == "npc":
                match = match_known_item(ctx, str(turn.get("text", "")))
                if match:
                    return match
        return None

    @staticmethod
    def _ground_factual_reply(
        dialogue: List[Dict[str, Any]],
        reply: str,
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """Ensure exact check_* tool facts are surfaced in the spoken reply."""

        if not tool_results:
            return reply
        latest = next((t for t in reversed(dialogue or []) if t.get("speaker") == "player"), {})
        chinese = any("\u4e00" <= ch <= "\u9fff" for ch in str(latest.get("text", "")))
        weak = (
            not reply.strip()
            or reply == FALLBACK_REPLY
            or "Could you say that again" in reply
            or reply.rstrip().endswith("...")
            or reply.rstrip().endswith("是")
        )

        for result in tool_results:
            name = result.get("name", "")
            returns = result.get("return") or []
            if not isinstance(returns, list) or not returns or not isinstance(returns[0], dict):
                continue
            item = returns[0]
            item_name = item.get("name") or result.get("parameters", {}).get("item_name") or "That"

            if name == "check_price" and item.get("price"):
                value = str(item["price"])
                if weak or value not in reply:
                    return f"{item_name} costs {value}." if not chinese else f"{item_name}售价{value}。"
            if name == "check_attack" and item.get("attack"):
                value = str(item["attack"])
                if weak or value not in reply:
                    return f"{item_name} has {value} attack." if not chinese else f"{item_name}攻击力是{value}。"
            if name == "check_type" and item.get("type"):
                value = str(item["type"])
                if weak or value not in reply:
                    return f"{item_name} is a {value}." if not chinese else f"{item_name}是{value}。"
            if name in {"check_description", "check_basic_info"} and item.get("description"):
                value = str(item["description"])
                if weak:
                    return value
        return reply

    @staticmethod
    def _ground_memory_reply(ctx, dialogue: List[Dict[str, Any]], reply: str) -> str:
        """Keep replies consistent with explicit conversation memory."""

        memory = extract_memory(ctx, dialogue)
        player_name = memory.get("player_name")

        latest = next((t for t in reversed(dialogue or []) if t.get("speaker") == "player"), {})
        text = str(latest.get("text", "")).lower()
        chinese = any("\u4e00" <= ch <= "\u9fff" for ch in str(latest.get("text", "")))

        asks_player_identity = any(
            key in text
            for key in (
                "我是谁",
                "猜猜我是谁",
                "我叫什么",
                "我的名字",
                "名字呢",
                "who am i",
                "guess who i am",
                "my name",
            )
        )
        asks_knowing = any(key in text for key in ("认识我", "認識我", "do you know me"))
        asks_npc_identity = any(key in text for key in ("你是谁", "你是誰", "who are you"))
        self_intro = any(key in text for key in ("我是", "我叫", "my name is", "i am ", "i'm "))
        greeting = any(key in text for key in ("你好", "hello", "hi", "hey"))
        weak = NpcTurnEngine._is_weak_reply(reply)

        if asks_npc_identity:
            persona = ctx.persona or {}
            name = persona.get("name") or ctx.name
            occupation = persona.get("occupation")
            if chinese:
                return f"我是{name}，{occupation}。" if occupation else f"我是{name}。"
            return f"I'm {name}, {occupation}." if occupation else f"I'm {name}."
        if asks_player_identity and player_name:
            return f"你是{player_name}。" if chinese else f"You're {player_name}."
        if asks_knowing and player_name:
            return (
                f"你告诉我你叫{player_name}，但除此之外我还不了解你。"
                if chinese
                else f"You told me your name is {player_name}, but I do not know much about you yet."
            )
        if self_intro and player_name:
            return f"{player_name}，记住了。" if chinese else f"{player_name}. I'll remember that."
        if greeting and weak:
            return "你好。需要看武器的话就说一声。" if chinese else "Welcome. Tell me what kind of weapon you need."
        return reply

    @staticmethod
    def _is_weak_reply(reply: str) -> bool:
        text = reply.strip()
        return (
            not text
            or text == FALLBACK_REPLY
            or "Could you say that again" in text
            or text in {"嗯？", "嗯。", "Hmph.", "Hmm."}
            or text.rstrip().endswith("...")
        )

"""LLM provider interface + OpenAI-compatible implementation.

Standalone reimplementation of nanobot's provider layer.  Keeps only what the
two-phase NpcTurnEngine actually needs: a single async ``chat()`` call that
returns tool calls or text content.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"


class LLMProvider(ABC):
    """Abstract async chat interface."""

    @abstractmethod
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
    ) -> LLMResponse: ...


class OpenAICompatProvider(LLMProvider):
    """Calls any OpenAI-compatible endpoint (vLLM, SGLang, etc.)."""

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8112/v1",
        default_model: str = "npc-model",
    ) -> None:
        import openai  # lazy: only needed when actually calling the model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
        self._default_model = default_model

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
        kw: dict[str, Any] = dict(
            model=model or self._default_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            kw["top_p"] = kwargs["top_p"]
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice or "auto"

        resp = await self._client.chat.completions.create(**kw)
        choice = resp.choices[0]
        msg = choice.message

        tcs: list[ToolCallRequest] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tcs.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args))

        return LLMResponse(
            content=msg.content,
            tool_calls=tcs,
            finish_reason=choice.finish_reason or "stop",
        )

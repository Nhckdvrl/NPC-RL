"""NPC context: structured per-NPC configuration loaded from a single YAML or JSON file.

Fields: worldview / persona / role / knowledge / state / tools.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class NpcContext:
    # Identity / roleplay grounding
    name: str = "NPC"
    worldview: str = ""
    role: str = ""                       # the *player's* role, for context
    persona: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    knowledge: Dict[str, Any] = field(default_factory=dict)

    # Tools: a `function_list_id_XXXX` key into function_calls.{tool,action}_map,
    # or an explicit list of tool names to expose from those maps.
    tools: Union[str, List[str], None] = "function_list_id_0001"

    # Backend / generation
    provider: str = "vllm"               # pluggable seam; OpenAI-compatible (vLLM) today
    model: str = ""                      # served model name (falls back to $OPENAI_MODEL)
    temperature: float = 0.7
    top_p: float = 0.8
    toolcall_max_tokens: int = 256       # phase-1 budget (Hermes tool calls)
    reply_max_tokens: int = 80           # phase-3 budget (short in-character reply)
    turn_budget_s: float = 0.0           # wall-clock budget per turn; 0 = unlimited

    def resolved_model(self) -> str:
        return self.model or os.getenv("OPENAI_MODEL", "npc-model")


_KNOWN_FIELDS = set(NpcContext.__dataclass_fields__.keys())


def context_from_dict(data: Dict[str, Any]) -> NpcContext:
    """Build an NpcContext from a plain dict, ignoring unknown keys."""
    return NpcContext(**{k: v for k, v in data.items() if k in _KNOWN_FIELDS})


def load_context(path: str) -> NpcContext:
    """Load an NPC context from a .yaml/.yml/.json file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"NPC context at {path} must be a mapping, got {type(data).__name__}")
    return context_from_dict(data)

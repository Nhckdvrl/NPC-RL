"""LLM-as-judge scoring for NPC roleplay turns (DeepSeek / OpenAI-compatible).

``evaluate_model_answer`` scores a candidate NPC reply against the dialogue
context using a CoSER-style rubric, returning a list of floats in [0, 1] (one per
candidate). It is deliberately fail-safe: any error (no endpoint, parse failure,
network) yields a neutral 0.0 so GRPO never crashes and offline unit tests pass.

Configure the judge via environment variables:
    JUDGE_BASE_URL  (default https://api.deepseek.com)
    JUDGE_API_KEY   (default from DEEPSEEK_API_KEY, else "EMPTY")
    JUDGE_MODEL     (default deepseek-chat)
    JUDGE_TIMEOUT   (seconds, default 30)
"""

import hashlib
import json
import os
import re
from typing import Any, List, Optional

# Only call the LLM judge for 1/JUDGE_SAMPLE_RATE of roleplay rollouts.
# Others get JUDGE_SKIP_SCORE (neutral). This cuts API cost by ~87.5% at rate=8.
_JUDGE_SAMPLE_RATE = int(os.getenv("JUDGE_SAMPLE_RATE", "8"))
_JUDGE_SKIP_SCORE = float(os.getenv("JUDGE_SKIP_SCORE", "0.5"))

_RUBRIC = """You are a strict judge for game NPC dialogue. Rate the NPC reply on a 0-100 scale across these dimensions, then give one overall score:
1. Scenario adherence & quest progression
2. Believability & engagement
3. Persona consistency
4. Dialogue flow & coherence
5. Functional relevance (consistency with any tool/action results)

Penalize out-of-character speech, contradictions, repetition, ignoring context, and being overly verbose.
Respond with ONLY a JSON object: {"score": <0-100 integer>}."""


def _client():
    from openai import OpenAI  # imported lazily so the package loads without openai

    base_url = os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com")
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "EMPTY"
    return OpenAI(base_url=base_url, api_key=api_key)


def _context_from_extra(extra_info) -> str:
    """Prefer the dialogue/scene context stored by the data converter in
    extra_info["question"] (the <QUESTION>...</QUESTION> history). This is what a
    CoSER-style judge should grade against, rather than a single gold line."""
    if isinstance(extra_info, dict):
        q = extra_info.get("question") or extra_info.get("conversation_history")
        if q:
            return q if isinstance(q, str) else json.dumps(q, ensure_ascii=False)
    return ""


def _build_context(ground_truth: Any) -> str:
    """ground_truth may be a JSON string ({conversation, evaluation_criteria}),
    a plain gold-response string, or a dict. Render it as judge context."""
    gt = ground_truth
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception:
            return f"Reference / gold response:\n{ground_truth}"
    if isinstance(gt, dict):
        parts = []
        conv = gt.get("conversation") or gt.get("dialogue")
        if conv:
            parts.append("Conversation so far:\n" + json.dumps(conv, ensure_ascii=False))
        if gt.get("evaluation_criteria"):
            parts.append("Evaluation criteria:\n" + str(gt["evaluation_criteria"]))
        if gt.get("gold_response") or gt.get("expected_response"):
            parts.append("Gold response:\n" + str(gt.get("gold_response") or gt.get("expected_response")))
        return "\n\n".join(parts) if parts else json.dumps(gt, ensure_ascii=False)
    return str(gt)


def _parse_score(text: str) -> float:
    if not text:
        return 0.0
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            raw = float(obj.get("score"))
            return max(0.0, min(raw / 100.0, 1.0))
    except Exception:
        pass
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", text)
    if m:
        return max(0.0, min(float(m.group(1)) / 100.0, 1.0))
    return 0.0


def _score_one(solution_str: str, ground_truth: Any, extra_info=None) -> float:
    try:
        context = _context_from_extra(extra_info) or _build_context(ground_truth)
        messages = [
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": f"{context}\n\nNPC reply to evaluate:\n{solution_str}"},
        ]
        resp = _client().chat.completions.create(
            model=os.getenv("JUDGE_MODEL", "deepseek-chat"),
            messages=messages,
            temperature=0.0,
            max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "1024")),
            timeout=float(os.getenv("JUDGE_TIMEOUT", "30")),
        )
        return _parse_score(resp.choices[0].message.content)
    except Exception:
        return 0.0


def _should_judge(solution_str: str) -> bool:
    """Deterministically select 1/JUDGE_SAMPLE_RATE of calls for actual judging.
    Uses a hash of the response so the sampling is reproducible across workers."""
    if _JUDGE_SAMPLE_RATE <= 1:
        return True
    h = int(hashlib.md5(solution_str.encode()[:64]).hexdigest(), 16)
    return (h % _JUDGE_SAMPLE_RATE) == 0


def evaluate_model_answer(
    solution_str,
    ground_truth: Any = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> List[float]:
    """Score one or many candidate replies; always returns list[float] in [0,1].

    Only 1/JUDGE_SAMPLE_RATE responses are actually sent to the LLM judge;
    the rest receive JUDGE_SKIP_SCORE (default 0.5) to reduce API cost.
    """
    if isinstance(solution_str, (list, tuple)):
        golds = ground_truth if isinstance(ground_truth, (list, tuple)) else [ground_truth] * len(solution_str)
        eis = extra_info if isinstance(extra_info, (list, tuple)) else [extra_info] * len(solution_str)
        return [
            _score_one(s, g, e) if _should_judge(s) else _JUDGE_SKIP_SCORE
            for s, g, e in zip(solution_str, golds, eis)
        ]
    if _should_judge(solution_str):
        return [_score_one(solution_str, ground_truth, extra_info)]
    return [_JUDGE_SKIP_SCORE]

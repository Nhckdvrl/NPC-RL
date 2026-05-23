"""NPC-RL reward package for verl GRPO.

Public entry points:
    compute_score(data_source, solution_str, ground_truth, extra_info=None)
    _default_compute_score(...)            # actual scoring logic

Per-source metrics live in ``utils_metric.DATASOURCE_METRICS``. The primary metric
value is mirrored into ``score`` (used by verl) and ``acc``.

Submodules ``toolcall_executor`` (rule F1) and ``llm_evaluate`` (LLM judge)
are referenced via the module object on purpose, so unit tests can patch
``reward_score.toolcall_executor.calculate_score`` /
``reward_score.llm_evaluate.evaluate_model_answer``.
"""

import re
from typing import Any, List, Optional, Union

from . import toolcall_executor, llm_evaluate, utils_metric

__all__ = ["compute_score", "_default_compute_score"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(solution_str: str) -> str:
    """Strip reasoning, return the answer body (inside <answer> if present)."""
    if not isinstance(solution_str, str):
        return ""
    text = _THINK_RE.sub("", solution_str).strip()
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text


def _compute_format_score(solution_str: str, data_source: Optional[str] = None) -> float:
    """1.0 if the generation carries the structure expected for its source."""
    if not isinstance(solution_str, str):
        return 0.0
    if data_source == "npc/toolcall":
        return 1.0 if "<tool_call>" in solution_str else 0.0
    return 1.0


def _score_single(data_source: str, solution_str, ground_truth, extra_info=None) -> dict:
    metrics = utils_metric.metrics_for(data_source)
    # Fixed insertion order so reward_extra_keys list is identical across all sources/workers
    result: dict = {"toolcall_f1": 0.0, "llm": 0.0}

    if data_source == "npc/toolcall":
        result["toolcall_f1"] = float(
            toolcall_executor.calculate_score(solution_str, ground_truth)
        )
    elif data_source == "npc/roleplay":
        scores = llm_evaluate.evaluate_model_answer(solution_str, ground_truth, extra_info)
        result["llm"] = float(scores[0]) if scores else 0.0
    else:
        result["score"] = _compute_format_score(solution_str, data_source)

    primary = metrics[0] if metrics else "score"
    score = float(result.get(primary, result.get("score", 0.0)))
    result["score"] = score
    result["acc"] = score
    return result


def _default_compute_score(
    data_source: Union[str, list],
    solution_str: Union[str, list],
    ground_truth: Union[Any, list],
    extra_info: Optional[Union[dict, list]] = None,
):
    """Score one example (dict) or a batch (list of dicts)."""
    if isinstance(data_source, (list, tuple)):
        n = len(data_source)
        sols = solution_str if isinstance(solution_str, (list, tuple)) else [solution_str] * n
        gts = ground_truth if isinstance(ground_truth, (list, tuple)) else [ground_truth] * n
        eis = extra_info if isinstance(extra_info, (list, tuple)) else [extra_info] * n
        return [_score_single(ds, s, g, e) for ds, s, g, e in zip(data_source, sols, gts, eis)]
    return _score_single(data_source, solution_str, ground_truth, extra_info)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Public verl entry point; thin wrapper over ``_default_compute_score``."""
    return _default_compute_score(data_source, solution_str, ground_truth, extra_info)

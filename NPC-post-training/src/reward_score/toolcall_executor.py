"""Rule-based tool-call scoring for NPC-RL GRPO.

``calculate_score`` parses ``<tool_call>{...}</tool_call>`` blocks out of a model
generation and compares them, by exact (name + parameters) match, against the
gold function list. The returned score is the F1 of exact matches — identical in
spirit to ``eval/evaluation_metrics.evaluate_tool_calls`` (replicated here so the
package has no cross-tree import dependency).
"""

import json
import re
from collections.abc import Hashable
from typing import Any, Dict, List, Tuple, Union

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def make_hashable(value: Any) -> Union[Hashable, Tuple, frozenset]:
    if isinstance(value, (list, tuple)):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, dict):
        return frozenset((k, make_hashable(v)) for k, v in sorted(value.items()))
    if isinstance(value, set):
        return frozenset(make_hashable(v) for v in value)
    return value


def _params_of(func: Dict) -> Dict:
    # Accept both OpenAI/hermes ("arguments") and NPC ("parameters") schemas.
    if "parameters" in func and func["parameters"] is not None:
        return func["parameters"]
    if "arguments" in func and func["arguments"] is not None:
        args = func["arguments"]
        if isinstance(args, str):
            try:
                return json.loads(args)
            except Exception:
                return {}
        return args
    return {}


def func_to_comparable(func: Dict) -> Tuple:
    if not isinstance(func, dict) or "name" not in func:
        return tuple()
    params = _params_of(func)
    if not isinstance(params, dict):
        return tuple()
    try:
        processed = [(k, make_hashable(v)) for k, v in sorted(params.items())]
        return (func["name"], frozenset(processed))
    except Exception:
        return tuple()


def _parse_predicted(solution_str: str) -> List[Dict]:
    funcs: List[Dict] = []
    for block in _TOOL_CALL_RE.findall(solution_str or ""):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict):
            funcs.append(obj)
        elif isinstance(obj, list):
            funcs.extend(o for o in obj if isinstance(o, dict))
    return funcs


def _normalize_gold(gold) -> List[Dict]:
    if gold is None:
        return []
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except Exception:
            return []
    if isinstance(gold, dict):
        gold = [gold]
    if not isinstance(gold, list):
        return []
    return [g for g in gold if isinstance(g, dict)]


def calculate_score(solution_str: str, gold_functions) -> float:
    """Return F1 of exact tool-call matches between prediction and gold."""
    gold = _normalize_gold(gold_functions)
    pred = _parse_predicted(solution_str)

    gold_set = {c for c in (func_to_comparable(g) for g in gold) if c}
    pred_set = {c for c in (func_to_comparable(p) for p in pred) if c}

    # No tool call expected and none produced -> perfect.
    if not gold_set and not pred_set:
        return 1.0

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

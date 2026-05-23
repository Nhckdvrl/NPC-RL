"""
This module contains functions for calculating evaluation metrics.
"""
from collections.abc import Hashable
from typing import Any, Dict, List, Union, Set, Tuple

def make_hashable(value: Any) -> Union[Hashable, Tuple, frozenset]:
    """Recursively converts values to be hashable for inclusion in sets/frozensets."""
    if isinstance(value, (list, tuple)):
        return tuple(make_hashable(v) for v in value)
    elif isinstance(value, dict):
        return frozenset((k, make_hashable(v)) for k, v in sorted(value.items()))
    elif isinstance(value, set):
        return frozenset(make_hashable(v) for v in value)
    return value

def func_to_comparable(func_dict: Dict) -> Tuple[str, frozenset]:
    """Converts a function dictionary to a comparable representation."""
    if not isinstance(func_dict, dict) or 'name' not in func_dict or 'parameters' not in func_dict:
        return tuple()
    try:
        processed_params = []
        for k, v in sorted(func_dict['parameters'].items()):
            processed_params.append((k, make_hashable(v)))
        return (func_dict['name'], frozenset(processed_params))
    except Exception as e:
        print(f"Warning: Could not make parameters hashable for function {func_dict.get('name', 'UNKNOWN_FUNC')}: {e}")
        print(f"Problematic parameters: {func_dict.get('parameters')}")
        return (func_dict.get('name', 'ERROR_FUNC_HASHING'), frozenset([('__error__', str(e))]))

def evaluate_tool_calls(gold_functions: List[Dict], predicted_functions: List[Dict]) -> Dict[str, float]:
    """
    Evaluates tool call accuracy (Precision, Recall, F1).
    
    Args:
        gold_functions: List of ground truth function calls
        predicted_functions: List of predicted function calls
        
    Returns:
        Dictionary containing precision, recall, f1, tp, fp, fn
    """
    if gold_functions is None:
        gold_functions = []
    if predicted_functions is None:
        predicted_functions = []

    gold_func_set = {func_to_comparable(f) for f in gold_functions if func_to_comparable(f)}
    pred_func_set = {func_to_comparable(f) for f in predicted_functions if func_to_comparable(f)}

    tp = len(gold_func_set.intersection(pred_func_set))
    fp = len(pred_func_set - gold_func_set)
    fn = len(gold_func_set - pred_func_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0.0

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate multiple metrics dictionaries by summing up counts and recalculating metrics.
    
    Args:
        metrics_list: List of metrics dictionaries, each containing tp, fp, fn, etc.
        
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}
        
    # Initialize sums
    sums = {k: 0.0 for k in metrics_list[0].keys()}
    
    # Sum up all metrics
    for metrics in metrics_list:
        for k, v in metrics.items():
            if k in sums:
                sums[k] += v
    
    # Recalculate precision, recall, f1 if we have tp, fp, fn
    if 'tp' in sums and 'fp' in sums and 'fn' in sums:
        tp, fp, fn = sums['tp'], sums['fp'], sums['fn']
        sums['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sums['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sums['f1'] = calculate_f1(sums['precision'], sums['recall'])
    
    return sums

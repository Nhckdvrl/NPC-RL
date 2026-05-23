"""
Evaluation package for dialogue systems.

This package provides tools for evaluating different types of dialogue system responses,
including tool calls, direct responses, and game role-playing interactions.
"""

from .evaluation_metrics import evaluate_tool_calls, calculate_f1, aggregate_metrics
from .evaluation_handlers import get_evaluator, get_game_evaluator, EvaluationResult
from .data_io import load_json_file, save_results_to_csv, is_game_format, is_task2_format

__all__ = [
    'evaluate_tool_calls',
    'calculate_f1',
    'aggregate_metrics',
    'get_evaluator',
    'get_game_evaluator',
    'EvaluationResult',
    'load_json_file',
    'save_results_to_csv',
    'is_game_format',
    'is_task2_format'
]
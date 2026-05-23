#!/usr/bin/env python3
"""
Main script for evaluating model responses across different tasks.

This script provides a unified interface for evaluating model responses for:
- Task 1: Tool calls + response quality
- Task 2: Direct response quality
- Game Role-Playing: NPC responses in a game context
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_eval import LLMJudge

# Import local modules
from evaluation_metrics import evaluate_tool_calls, aggregate_metrics
from evaluation_handlers import get_evaluator, get_game_evaluator, EvaluationResult
from data_io import load_json_file, save_results_to_csv, is_game_format, is_task2_format


def detect_task_type(gold_data, model_responses_data):
    """
    Detect the task type based on the data format.
    
    Args:
        gold_data: Loaded gold data
        model_responses_data: Loaded model responses data
        
    Returns:
        Tuple of (task_type, task_type_str)
    """
    if not gold_data:
        raise ValueError("No data found in gold data file")
        
    if is_game_format(gold_data[0]):
        print("Detected Game Role-Playing format")
        return 1, "Game Role-Playing"
    elif is_task2_format(model_responses_data):
        print("Detected Task 2 format (direct string responses)")
        return 2, "Task 2"
    else:
        print("Detected Task 1 format (response objects with functions)")
        return 1, "Task 1"


def initialize_llm_judge() -> LLMJudge:
    """
    Initialize the LLM Judge for evaluation.
    
    Returns:
        Initialized LLMJudge instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        llm_judge = LLMJudge()
        print(f"Successfully initialized LLM Judge with model: {llm_judge.model_name}")
        if 'gpt-4o' not in llm_judge.model_name.lower() and 'gpt-4' not in llm_judge.model_name.lower():
            print(f"Warning: LLM Judge model is '{llm_judge.model_name}'. "
                  f"For best results, 'gpt-4o' or a similar advanced model is recommended for evaluation.")
        return llm_judge
    except Exception as e:
        print(f"Failed to initialize LLMJudge: {e}")
        print("Ensure llm_eval.py and its dependencies are correctly set up, "
              "and API keys/models are configured via environment variables.")
        sys.exit(1)


def process_conversation(
    gold_conv_data: Dict[str, Any],
    model_conv_responses: Dict[str, Any],
    evaluator: Any,
    llm_judge: Optional[LLMJudge] = None
) -> Optional[EvaluationResult]:
    """
    Process a single conversation with the given evaluator.
    
    Args:
        gold_conv_data: Gold conversation data
        model_conv_responses: Model responses for the conversation
        evaluator: Evaluator instance
        llm_judge: Optional LLM Judge instance
        
    Returns:
        EvaluationResult if successful, None otherwise
    """
    conv_id = gold_conv_data.get('data_id', gold_conv_data.get('id', 'unknown'))
    print(f"\n--- Processing Conversation (ID: {conv_id}) ---")
    
    if not isinstance(gold_conv_data, dict):
        print(f"  Warning: Gold conversation data for ID {conv_id} is not a dict. Skipping.")
        return None
    if not isinstance(model_conv_responses, dict):
        print(f"  Warning: Model response data for ID {conv_id} is not a dict. Skipping.")
        return None
    
    # Determine number of turns based on evaluator type
    if hasattr(evaluator, 'is_game_evaluator') and evaluator.is_game_evaluator:
        # For game format, use the dialogue list length
        dialogue = gold_conv_data.get('dialogue', [])
        num_turns = len(dialogue)
    else:
        # For task1/task2, use the number of turn_X keys
        turn_keys = [k for k in model_conv_responses.keys() if k.startswith('turn_')]
        num_turns = len(turn_keys)
    
    print(f"  Found {num_turns} turns to evaluate")
    
    # Process each turn
    results = []
    for turn_idx in range(num_turns):
        turn_key = f'turn_{turn_idx}'
        gold_turn = gold_conv_data.get(turn_key, {})
        model_turn = model_conv_responses.get(turn_key, {})
        
        try:
            result = evaluator.evaluate_turn(
                gold_data=gold_turn,
                model_response=model_turn,
                turn_idx=turn_idx,
                llm_judge=llm_judge
            )
            if result:
                results.append(result)
                score = result.metrics.get('coser_score', result.metrics.get('tool_f1', 'N/A'))
                print(f"  Turn {turn_idx}: Evaluated (Score: {score})")
        except Exception as e:
            print(f"  Error evaluating turn {turn_idx}: {str(e)}")
    
    # Aggregate results for this conversation
    if results:
        return evaluator.aggregate_results(results)
    return None


def main():
    """Main function to run the evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model responses for dialogue tasks.')
    parser.add_argument('--data_file', type=str, required=True, 
                       help='Path to the ground truth data file (e.g., task1_sample.json)')
    parser.add_argument('--response_file', type=str, required=True, 
                       help='Path to the model responses file (e.g., results/task1_responses.json)')
    parser.add_argument('--task_type', type=int, choices=[1, 2], 
                       help='Task type (1 or 2). If not provided, will try to auto-detect.')
    parser.add_argument('--model_name', type=str, 
                       help='Name of the model being evaluated')
    args = parser.parse_args()
    
    # Load data
    try:
        gold_data = load_json_file(args.data_file)
        model_responses_data = load_json_file(args.response_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Detect task type if not specified
    if args.task_type is None:
        try:
            task_type, task_type_str = detect_task_type(gold_data, model_responses_data)
        except Exception as e:
            print(f"Error detecting task type: {e}")
            sys.exit(1)
    else:
        task_type = args.task_type
        task_type_str = f"Task {task_type}"
    
    # Initialize LLM Judge (required for all evaluations)
    llm_judge = initialize_llm_judge()
    
    # Get appropriate evaluator
    if is_game_format(gold_data[0]):
        evaluator = get_game_evaluator()
    else:
        evaluator = get_evaluator(task_type)
    
    # Check data length
    if len(gold_data) != len(model_responses_data):
        print(f"Warning: Number of conversations mismatch between data file ({len(gold_data)}) "
              f"and response file ({len(model_responses_data)}).")
    
    # Process each conversation
    all_results = []
    for i, (gold_conv, model_resp) in enumerate(zip(gold_data, model_responses_data)):
        result = process_conversation(gold_conv, model_resp, evaluator, llm_judge)
        if result:
            all_results.append(result)
    
    # Aggregate overall results
    if all_results:
        overall_result = evaluator.aggregate_results(all_results)
        
        # Prepare metrics for saving
        metrics = {
            'CoSER Score': overall_result.metrics.get('coser_score', 0.0),
            'Turns Evaluated': sum(r.metrics.get('num_turns', 1) for r in all_results if r),
            'Conversations': len(all_results)
        }
        
        # Add tool metrics if available
        if 'tool_tp' in overall_result.metrics:
            metrics.update({
                'Tool Precision': overall_result.metrics.get('tool_precision', 0.0),
                'Tool Recall': overall_result.metrics.get('tool_recall', 0.0),
                'Tool F1': overall_result.metrics.get('tool_f1', 0.0),
                'Tool TP': int(overall_result.metrics.get('tool_tp', 0)),
                'Tool FP': int(overall_result.metrics.get('tool_fp', 0)),
                'Tool FN': int(overall_result.metrics.get('tool_fn', 0))
            })
        
        # Save results
        results_dir = Path("/path/to/npc-rl/results")
        csv_path = save_results_to_csv(
            results_dir=results_dir,
            metrics=metrics,
            task_type=task_type_str,
            model_name=args.model_name or "unknown_model",
            use_rag=os.getenv('USE_RAG', '0') == '1'
        )
        
        print(f"\n--- Evaluation Complete ---")
        print(f"Results saved to: {csv_path}")
        print(f"Task Type: {task_type_str}")
        print(f"Model: {args.model_name or 'unknown_model'}")
        print(f"CoSER Score: {metrics['CoSER Score']:.4f}")
        if 'Tool F1' in metrics:
            print(f"Tool F1: {metrics['Tool F1']:.4f}")
    else:
        print("No valid results were generated from the evaluation.")


if __name__ == '__main__':
    main()

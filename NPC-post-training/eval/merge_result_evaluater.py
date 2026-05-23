#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CPDC Merge Result Evaluator

This script evaluates merge result files against gold standards using the CPDC scoring logic.
It analyzes the scoring patterns and provides detailed statistics on the evaluation results.
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Union

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the scoring functions
from src.reward_score.cpdc_toolcall_executor import calculate_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("merge_result_evaluater")


def load_merge_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load merge results from a JSON file.
    
    Args:
        file_path: Path to the merge results JSON file
        
    Returns:
        List of merge result entries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded merge results from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load merge results from {file_path}: {e}")
        sys.exit(1)


def evaluate_turn(prediction: str, gold_standard: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a single turn's prediction against gold standard.
    
    Args:
        prediction: The model's prediction text
        gold_standard: List of gold standard tool calls
        
    Returns:
        Dictionary with evaluation results
    """
    score = calculate_score(prediction, gold_standard)
    
    # Count tool calls in prediction
    import re
    tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', prediction, re.DOTALL)
    
    return {
        "score": score,
        "num_gold_calls": len(gold_standard),
        "num_pred_calls": len(tool_calls)
    }


def evaluate_merge_results(merge_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate all turns in the merge results.
    
    Args:
        merge_data: List of merge result entries
        
    Returns:
        Dictionary with evaluation statistics
    """
    results = {
        "total_turns": 0,
        "total_score": 0.0,
        "score_distribution": defaultdict(int),
        "score_by_num_gold": defaultdict(list),
        "score_by_num_pred": defaultdict(list),
        "score_by_match_pattern": defaultdict(list),
        "detailed_results": []
    }
    
    for entry in merge_data:
        data_id = entry["data_id"]
        turns = entry["turns"]
        
        for turn_id, turn_data in turns.items():
            # Skip turns without gold functions
            if "gold" not in turn_data or "functions" not in turn_data["gold"]:
                continue
                
            gold_functions = turn_data["gold"]["functions"]
            
            # Skip if no gold functions
            if not gold_functions:
                continue
                
            # Get model prediction
            if "prediction" in turn_data:
                prediction = turn_data["prediction"]
            else:
                logger.warning(f"No prediction found for {data_id} {turn_id}, skipping")
                continue
            
            # Evaluate this turn
            eval_result = evaluate_turn(prediction, gold_functions)
            eval_result["data_id"] = data_id
            eval_result["turn_id"] = turn_id
            
            # Update statistics
            results["total_turns"] += 1
            results["total_score"] += eval_result["score"]
            
            # Round score to 1 decimal place for distribution
            rounded_score = round(eval_result["score"] * 10) / 10
            results["score_distribution"][rounded_score] += 1
            
            # Group by number of gold and prediction calls
            num_gold = eval_result["num_gold_calls"]
            num_pred = eval_result["num_pred_calls"]
            results["score_by_num_gold"][num_gold].append(eval_result["score"])
            results["score_by_num_pred"][num_pred].append(eval_result["score"])
            
            # Create a pattern key to analyze match patterns
            pattern_key = f"gold:{num_gold}_pred:{num_pred}"
            results["score_by_match_pattern"][pattern_key].append(eval_result["score"])
            
            # Store detailed result
            results["detailed_results"].append(eval_result)
    
    # Calculate average score
    if results["total_turns"] > 0:
        results["average_score"] = results["total_score"] / results["total_turns"]
    else:
        results["average_score"] = 0.0
        
    # Convert defaultdicts to regular dicts for JSON serialization
    results["score_distribution"] = dict(results["score_distribution"])
    results["score_by_num_gold"] = {k: {"scores": v, "avg": sum(v)/len(v) if v else 0} 
                                   for k, v in results["score_by_num_gold"].items()}
    results["score_by_num_pred"] = {k: {"scores": v, "avg": sum(v)/len(v) if v else 0} 
                                   for k, v in results["score_by_num_pred"].items()}
    
    # Analyze match patterns more deeply
    for pattern, scores in list(results["score_by_match_pattern"].items()):
        gold, pred = pattern.split('_')
        gold_num = int(gold.split(':')[1])
        pred_num = int(pred.split(':')[1])
        
        # Calculate statistics for this pattern
        avg_score = sum(scores) / len(scores) if scores else 0
        count = len(scores)
        
        # Store pattern analysis
        results["score_by_match_pattern"][pattern] = {
            "count": count,
            "avg_score": avg_score,
            "gold_calls": gold_num,
            "pred_calls": pred_num,
            "theoretical_f1": calculate_theoretical_f1(gold_num, pred_num, gold_num),  # Assuming all gold matched
            "difference": avg_score - calculate_theoretical_f1(gold_num, pred_num, gold_num)
        }
    
    return results


def calculate_theoretical_f1(gold_count: int, pred_count: int, matched_count: int) -> float:
    """
    Calculate the theoretical F1 score based on counts.
    
    Args:
        gold_count: Number of gold standard items
        pred_count: Number of prediction items
        matched_count: Number of matched items
        
    Returns:
        Theoretical F1 score
    """
    if gold_count == 0 and pred_count == 0:
        return 1.0
    if gold_count == 0 or pred_count == 0 or matched_count == 0:
        return 0.0
        
    precision = matched_count / pred_count
    recall = matched_count / gold_count
    
    if precision + recall == 0:
        return 0.0
        
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 2)


def analyze_scoring_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze scoring patterns to identify discrepancies from theoretical F1.
    
    Args:
        results: Evaluation results
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "patterns": [],
        "observations": []
    }
    
    # Look for patterns where actual scores differ from theoretical F1
    for pattern, data in results["score_by_match_pattern"].items():
        if abs(data["difference"]) > 0.01:  # If there's a significant difference
            analysis["patterns"].append({
                "pattern": pattern,
                "actual_avg": data["avg_score"],
                "theoretical_f1": data["theoretical_f1"],
                "difference": data["difference"],
                "count": data["count"]
            })
    
    # Sort patterns by absolute difference
    analysis["patterns"].sort(key=lambda x: abs(x["difference"]), reverse=True)
    
    # Generate observations based on the memory and patterns
    observations = []
    
    # Check for the specific patterns mentioned in the memory
    multi_gold_multi_pred_match = next((p for p in analysis["patterns"] 
                                      if p["pattern"].startswith("gold:2_pred:2") 
                                      and abs(p["actual_avg"] - 0.5) < 0.1), None)
    if multi_gold_multi_pred_match:
        observations.append(
            "对于多个 Gold 标准和多个预测全部匹配的情况，得分为 0.5 而非预期的 1.0"
        )
    
    multi_gold_partial_match = next((p for p in analysis["patterns"] 
                                   if p["pattern"].startswith("gold:2_pred:1") 
                                   and abs(p["actual_avg"] - 0.4) < 0.1), None)
    if multi_gold_partial_match:
        observations.append(
            "对于多个 Gold 标准，预测只匹配部分的情况，得分为 0.4 而非理论上的 2/3 (约 0.67)"
        )
    
    extra_pred_match = next((p for p in analysis["patterns"] 
                           if p["pattern"].startswith("gold:1_pred:2") 
                           and abs(p["actual_avg"] - 0.4) < 0.1), None)
    if extra_pred_match:
        observations.append(
            "对于额外的预测调用，得分为 0.4 而非理论上的 0.8"
        )
    
    # Add general observation
    if analysis["patterns"]:
        observations.append(
            "当前的评分逻辑可能使用了不同于标准 F1 分数的计算方式"
        )
    
    analysis["observations"] = observations
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Evaluate CPDC merge results')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the merge results JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the evaluation results (optional)')
    args = parser.parse_args()
    
    # Load merge results
    merge_data = load_merge_results(args.input)
    
    # Evaluate results
    results = evaluate_merge_results(merge_data)
    
    # Analyze scoring patterns
    analysis = analyze_scoring_patterns(results)
    results["analysis"] = analysis
    
    # Print summary
    print("\n===== CPDC Merge Result Evaluation Summary =====")
    print(f"Input file: {args.input}")
    print(f"Total turns evaluated: {results['total_turns']}")
    print(f"Average score: {results['average_score']:.4f}")
    
    # Print score distribution
    print("\nScore Distribution:")
    for score, count in sorted(results["score_distribution"].items()):
        print(f"  {score:.1f}: {count} turns")
    
    # Print pattern analysis
    print("\nScoring Pattern Analysis:")
    for pattern in analysis["patterns"][:5]:  # Show top 5 patterns
        print(f"  Pattern: {pattern['pattern']}")
        print(f"    Actual avg score: {pattern['actual_avg']:.2f}")
        print(f"    Theoretical F1: {pattern['theoretical_f1']:.2f}")
        print(f"    Difference: {pattern['difference']:.2f}")
        print(f"    Count: {pattern['count']}")
    
    # Print observations
    print("\nObservations:")
    for i, obs in enumerate(analysis["observations"], 1):
        print(f"  {i}. {obs}")
    
    # Save results if output path is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to {args.output}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

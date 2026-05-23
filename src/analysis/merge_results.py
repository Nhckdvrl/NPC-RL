#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge Results Script

This script merges model outputs from task1_responses.json with gold standard answers
from the original data file (e.g., task1_sample.json) to create a comparison file.
"""

import json
import os
import sys
import argparse
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def load_json_data(file_path: str) -> Dict:
    """Load JSON data from the specified file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_results(model_results_path: str, gold_data_path: str, output_path: str) -> None:
    """Merge model results with gold standard answers for comparison."""
    # Load data
    model_results = load_json_data(model_results_path)
    gold_data = load_json_data(gold_data_path)
    
    merged_results = []
    
    # Create a dictionary for quick lookup of model results by data_id
    model_results_dict = {item['data_id']: item for item in model_results}
    
    # Process each conversation in the gold data
    for gold_item in gold_data:
        data_id = gold_item['data_id']
        model_item = model_results_dict.get(data_id, {})
        
        merged_item = {
            'data_id': data_id,
            'total_turn': gold_item.get('total_turn', 0),
            'worldview': gold_item.get('worldview', ''),
            'player': gold_item.get('player', {}),
            'npc': gold_item.get('npc', {}),
            'turns': {}
        }
        
        # Process each turn that has gold functions and responses
        for turn_key, turn_data in gold_item.items():
            if turn_key.startswith("turn_"):
                # Get gold data
                gold_response = turn_data.get("gold_response", "")
                gold_functions = turn_data.get("gold_functions", [])
                dialogue = turn_data.get("dialogue", [])
                
                # Get model data if available
                model_turn_data = model_item.get(turn_key, {})
                model_response = model_turn_data.get("response", "")
                model_functions = model_turn_data.get("functions", [])
                
                # Create merged turn data
                merged_item['turns'][turn_key] = {
                    "dialogue": dialogue,
                    "gold": {
                        "response": gold_response,
                        "functions": gold_functions
                    },
                    "model": {
                        "response": model_response,
                        "functions": model_functions
                    }
                }
        
        merged_results.append(merged_item)
    
    # Save merged results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    
    print(f"Merged results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge model outputs with gold standard answers.')
    parser.add_argument('--model_results', type=str, default='/path/to/npc-rl/results/task1_responses.json',
                        help='Path to the model results JSON file')
    parser.add_argument('--gold_data', type=str, default=os.environ.get('TASK1_DATA_FILE', 'data/task1_sample.json'),
                        help='Path to the gold data JSON file')
    parser.add_argument('--output', type=str, default='/path/to/npc-rl/results/merge_results.json',
                        help='Path to save the merged results')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    model_results_path = os.path.join(base_dir, args.model_results) if not os.path.isabs(args.model_results) else args.model_results
    gold_data_path = os.path.join(base_dir, args.gold_data) if not os.path.isabs(args.gold_data) else args.gold_data
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    merge_results(model_results_path, gold_data_path, output_path)


if __name__ == "__main__":
    main()

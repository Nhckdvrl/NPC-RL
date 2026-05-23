#!/usr/bin/env python3
# agents/tool_classifier/prepare_dataset.py

import json
import os
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import random


def load_json_file(file_path: str) -> List[dict]:
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_turn_data(data: List[dict]) -> List[Tuple[str, List[str]]]:
    """
    Extract (text, [function_names]) pairs from the data.
    
    Args:
        data: List of conversation data items
        
    Returns:
        List of tuples where each tuple is (text, [function_name1, function_name2, ...])
    """
    results = []
    
    for item in data:
        # Get the total number of turns for this conversation
        total_turns = item.get('total_turn', 0)
        
        # Process each turn
        for turn_num in range(total_turns):
            turn_key = f"turn_{turn_num}"
            if turn_key not in item:
                continue
                
            turn_data = item[turn_key]
            
            # Get the last dialogue entry (player's text)
            if 'dialogue' not in turn_data or not turn_data['dialogue']:
                continue
                
            # Get the last player utterance
            player_texts = [d['text'] for d in turn_data['dialogue'] 
                          if d.get('speaker') == 'player']
            
            if not player_texts:
                continue
                
            last_player_text = player_texts[-1].strip()
            
            # Get all function names for this turn
            function_names = []
            if 'gold_functions' in turn_data and turn_data['gold_functions']:
                for func in turn_data['gold_functions']:
                    if 'name' in func:
                        function_names.append(func['name'])
            
            # Add to results if we have at least one function name
            if function_names:
                results.append((last_player_text, function_names))
    
    return results


def save_dataset(data: List[Tuple[str, List[str]]], output_file: str):
    """Save the processed data to a JSON file."""
    # Convert to list of dicts for JSON serialization
    output_data = [{"text": text, "functions": funcs} for text, funcs in data]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(output_data)} samples to {output_file}")


def analyze_dataset(data: List[Tuple[str, List[str]]]):
    """Print statistics about the dataset."""
    if not data:
        print("No data to analyze.")
        return
    
    total_samples = len(data)
    function_counts = defaultdict(int)
    multi_function_samples = 0
    
    for text, funcs in data:
        for func in funcs:
            function_counts[func] += 1
        if len(funcs) > 1:
            multi_function_samples += 1
    
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {total_samples}")
    print(f"Samples with multiple functions: {multi_function_samples} "
          f"({multi_function_samples/total_samples*100:.1f}%)")
    
    print("\nFunction counts:")
    for func, count in sorted(function_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {func}: {count} ({count/total_samples*100:.1f}%)")


def main():
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    output_dir = os.path.join(os.path.dirname(__file__), 'references_data')
    
    train_file = os.path.join(data_dir, 'task1_train.json')
    test_file = os.path.join(data_dir, 'task1_test.json')
    
    output_train = os.path.join(output_dir, 'train_data.json')
    output_test = os.path.join(output_dir, 'test_data.json')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    train_data = load_json_file(train_file)
    train_samples = extract_turn_data(train_data)
    save_dataset(train_samples, output_train)
    analyze_dataset(train_samples)
    
    # Process test data
    print("\nProcessing test data...")
    test_data = load_json_file(test_file)
    test_samples = extract_turn_data(test_data)
    save_dataset(test_samples, output_test)
    analyze_dataset(test_samples)
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tool Usage Analysis Script

This script analyzes the JSON data files from the CPDC dataset to extract insights about:
1. Which tools are most commonly used
2. The typical parameter patterns for each tool
3. Correlations between dialogue content and tool usage

Files analyzed:
- task1_sample.json
- task1_train.json
- task2_sample.json
- task2_train.json
"""

import json
import os
import sys
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import re

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Define paths to data files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
TASK1_SAMPLE = os.path.join(DATA_DIR, 'task1_sample.json')
TASK1_TRAIN = os.path.join(DATA_DIR, 'task1_train.json')
TASK2_SAMPLE = os.path.join(DATA_DIR, 'task2_sample.json')
TASK2_TRAIN = os.path.join(DATA_DIR, 'task2_train.json')

# Output directory for results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from the specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def extract_dialogue_text(dialogue: List[Dict]) -> str:
    """Extract and concatenate all text from dialogue entries."""
    return " ".join([entry.get("text", "") for entry in dialogue])


def analyze_function_usage(data: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    """Analyze function usage patterns in the data.
    
    Returns:
        Tuple containing:
        - function_counts: Counter of function name occurrences
        - parameter_patterns: Dict mapping function names to their parameter patterns
        - dialogue_function_pairs: List of (dialogue, function) pairs
        - function_examples: Dict mapping function names to example usages
    """
    function_counts = Counter()
    parameter_patterns = defaultdict(lambda: defaultdict(int))
    dialogue_function_pairs = []
    function_examples = defaultdict(list)
    
    for conversation in data:
        function_list_id = conversation.get("function_list_id", "unknown")
        
        # Process each turn in the conversation
        for turn_idx in range(20):  # Assuming max 20 turns per conversation
            turn_key = f"turn_{turn_idx}"
            if turn_key not in conversation:
                continue
                
            turn = conversation[turn_key]
            dialogue = turn.get("dialogue", [])
            gold_functions = turn.get("gold_functions", [])
            
            dialogue_text = extract_dialogue_text(dialogue)
            
            # Analyze each function call
            for func in gold_functions:
                func_name = func.get("name", "")
                if not func_name:
                    continue
                    
                # Count function occurrences
                function_counts[func_name] += 1
                
                # Store dialogue-function pairs
                dialogue_function_pairs.append((dialogue_text, func))
                
                # Analyze parameter patterns
                params = func.get("parameters", {})
                for param_name, param_value in params.items():
                    # Skip empty parameters
                    if not param_value and not isinstance(param_value, (int, float, bool)):
                        continue
                        
                    # For string parameters, categorize them
                    if isinstance(param_value, str):
                        if param_value.strip() == "":
                            param_type = "empty_string"
                        elif re.search(r'\d+G', param_value):
                            param_type = "currency"
                        elif param_name.endswith("_operator"):
                            param_type = f"operator:{param_value}"
                        else:
                            param_type = "text"
                    elif isinstance(param_value, (list, dict)):
                        param_type = type(param_value).__name__
                    else:
                        param_type = str(type(param_value).__name__)
                        
                    parameter_patterns[func_name][f"{param_name}:{param_type}"] += 1
                
                # Store example usages (up to 5 per function)
                if len(function_examples[func_name]) < 5:
                    function_examples[func_name].append({
                        "function_list_id": function_list_id,
                        "parameters": params,
                        "return": func.get("return", []),
                        "dialogue_snippet": dialogue_text[:100] + "..." if len(dialogue_text) > 100 else dialogue_text
                    })
    
    return function_counts, parameter_patterns, dialogue_function_pairs, function_examples


def analyze_dialogue_function_correlation(dialogue_function_pairs: List[Tuple[str, Dict]]) -> Dict:
    """Analyze correlations between dialogue content and function calls."""
    correlations = defaultdict(lambda: defaultdict(int))
    
    # Define keyword patterns to look for
    keyword_patterns = {
        "price_query": r"(how much|price|cost|expensive|cheap)",
        "type_query": r"(what type|kind of|category)",
        "attack_query": r"(attack|power|damage|strength)",
        "description_query": r"(tell me about|describe|details|information)",
        "search_request": r"(looking for|search|find|recommend|suggest)",
        "level_query": r"(level|difficulty|rank|grade)",
        "duration_query": r"(how long|duration|time|hours|days)",
        "reward_query": r"(reward|payment|compensation|gold)",
    }
    
    for dialogue_text, func in dialogue_function_pairs:
        func_name = func.get("name", "")
        if not func_name:
            continue
            
        # Check for keyword patterns in dialogue
        for pattern_name, pattern in keyword_patterns.items():
            if re.search(pattern, dialogue_text, re.IGNORECASE):
                correlations[func_name][pattern_name] += 1
    
    return correlations


def generate_report(task_name: str, function_counts: Dict, parameter_patterns: Dict, 
                   correlations: Dict, function_examples: Dict) -> str:
    """Generate a detailed report of the analysis results."""
    report = f"# Tool Usage Analysis Report for {task_name}\n\n"
    
    # Most common functions
    report += "## Most Commonly Used Functions\n\n"
    report += "| Function Name | Count | Percentage |\n"
    report += "|--------------|-------|------------|\n"
    
    total_calls = sum(function_counts.values())
    for func_name, count in function_counts.most_common():
        percentage = (count / total_calls) * 100 if total_calls > 0 else 0
        report += f"| {func_name} | {count} | {percentage:.2f}% |\n"
    
    # Parameter patterns
    report += "\n## Parameter Usage Patterns\n\n"
    
    for func_name, patterns in parameter_patterns.items():
        report += f"### {func_name}\n\n"
        report += "| Parameter | Count | Percentage |\n"
        report += "|-----------|-------|------------|\n"
        
        total_params = sum(patterns.values())
        for param_pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_params) * 100 if total_params > 0 else 0
            report += f"| {param_pattern} | {count} | {percentage:.2f}% |\n"
        
        report += "\n"
    
    # Dialogue-Function Correlations
    report += "\n## Dialogue-Function Correlations\n\n"
    
    for func_name, patterns in correlations.items():
        if not patterns:
            continue
            
        report += f"### {func_name}\n\n"
        report += "| Dialogue Pattern | Count |\n"
        report += "|-----------------|-------|\n"
        
        for pattern_name, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            report += f"| {pattern_name} | {count} |\n"
        
        report += "\n"
    
    # Example usages
    report += "\n## Example Function Usages\n\n"
    
    for func_name, examples in function_examples.items():
        report += f"### {func_name}\n\n"
        
        for i, example in enumerate(examples, 1):
            report += f"#### Example {i} (from {example['function_list_id']})\n\n"
            report += f"**Dialogue Context:** {example['dialogue_snippet']}\n\n"
            report += "**Parameters:**\n```json\n"
            report += json.dumps(example['parameters'], indent=2) + "\n"
            report += "```\n\n"
            report += "**Return Value:**\n```json\n"
            report += json.dumps(example['return'], indent=2) + "\n"
            report += "```\n\n"
    
    return report


def generate_visualizations(task_name: str, function_counts: Dict, parameter_patterns: Dict):
    """Generate visualizations for the analysis results."""
    # Function usage bar chart
    plt.figure(figsize=(12, 6))
    functions = [func for func, _ in function_counts.most_common(10)]
    counts = [count for _, count in function_counts.most_common(10)]
    
    plt.bar(functions, counts)
    plt.title(f'Top 10 Most Used Functions in {task_name}')
    plt.xlabel('Function Name')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"{task_name.replace(' ', '_')}_function_counts.png"))
    plt.close()
    
    # Parameter heatmap for top functions
    for func_name, patterns in parameter_patterns.items():
        if func_name not in [f for f, _ in function_counts.most_common(5)]:
            continue
            
        # Get top 10 parameters
        top_params = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        if not top_params:
            continue
            
        param_names = [param.split(':')[0] for param, _ in top_params]
        param_counts = [count for _, count in top_params]
        
        plt.figure(figsize=(10, 5))
        plt.bar(param_names, param_counts)
        plt.title(f'Top Parameters for {func_name} in {task_name}')
        plt.xlabel('Parameter Name')
        plt.ylabel('Number of Uses')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, f"{task_name.replace(' ', '_')}_{func_name}_params.png"))
        plt.close()


def analyze_file(file_path: str, file_name: str):
    """Analyze a single data file and generate reports."""
    print(f"Analyzing {file_name}...")
    data = load_json_data(file_path)
    
    if not data:
        print(f"No data found in {file_name}")
        return
        
    function_counts, parameter_patterns, dialogue_function_pairs, function_examples = analyze_function_usage(data)
    correlations = analyze_dialogue_function_correlation(dialogue_function_pairs)
    
    # Generate report
    report = generate_report(file_name, function_counts, parameter_patterns, correlations, function_examples)
    report_path = os.path.join(OUTPUT_DIR, f"{file_name.replace('.json', '')}_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to {report_path}")
    
    # Generate visualizations
    generate_visualizations(file_name.replace('.json', ''), function_counts, parameter_patterns)
    
    return {
        "file_name": file_name,
        "function_counts": function_counts,
        "parameter_patterns": parameter_patterns,
        "correlations": correlations,
        "total_conversations": len(data),
        "total_function_calls": sum(function_counts.values())
    }


def generate_comparative_report(results: List[Dict]):
    """Generate a comparative report across all analyzed files."""
    report = "# Comparative Tool Usage Analysis\n\n"
    
    # Basic statistics
    report += "## Dataset Statistics\n\n"
    report += "| Dataset | Conversations | Function Calls | Calls/Conversation |\n"
    report += "|---------|---------------|---------------|-------------------|\n"
    
    for result in results:
        calls_per_conv = result["total_function_calls"] / result["total_conversations"] if result["total_conversations"] > 0 else 0
        report += f"| {result['file_name']} | {result['total_conversations']} | {result['total_function_calls']} | {calls_per_conv:.2f} |\n"
    
    # Function usage comparison
    report += "\n## Function Usage Comparison\n\n"
    
    # Get all unique function names
    all_functions = set()
    for result in results:
        all_functions.update(result["function_counts"].keys())
    
    # Create comparison table
    report += "| Function Name | " + " | ".join([result["file_name"] for result in results]) + " |\n"
    report += "|--------------|" + "-|"*len(results) + "\n"
    
    for func_name in sorted(all_functions):
        report += f"| {func_name} | "
        for result in results:
            count = result["function_counts"].get(func_name, 0)
            total = sum(result["function_counts"].values())
            percentage = (count / total) * 100 if total > 0 else 0
            report += f"{count} ({percentage:.1f}%) | "
        report += "\n"
    
    # Save comparative report
    report_path = os.path.join(OUTPUT_DIR, "comparative_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Comparative report saved to {report_path}")
    
    # Generate comparative visualization
    plt.figure(figsize=(15, 8))
    
    # Get top 10 functions across all datasets
    combined_counts = Counter()
    for result in results:
        for func, count in result["function_counts"].items():
            combined_counts[func] += count
    
    top_functions = [func for func, _ in combined_counts.most_common(10)]
    
    # Create grouped bar chart
    bar_width = 0.8 / len(results)
    index = range(len(top_functions))
    
    for i, result in enumerate(results):
        counts = [result["function_counts"].get(func, 0) for func in top_functions]
        plt.bar([x + i * bar_width for x in index], counts, bar_width, 
                label=result["file_name"])
    
    plt.xlabel('Function Name')
    plt.ylabel('Number of Calls')
    plt.title('Top 10 Functions Across All Datasets')
    plt.xticks([x + bar_width * (len(results) - 1) / 2 for x in index], top_functions, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_function_usage.png"))
    plt.close()


def main():
    """Main function to run the analysis."""
    print("Starting tool usage analysis...")
    
    # Analyze each file
    results = []
    for file_path, file_name in [
        (TASK1_SAMPLE, "task1_sample.json"),
        (TASK1_TRAIN, "task1_train.json"),
        (TASK2_SAMPLE, "task2_sample.json"),
        (TASK2_TRAIN, "task2_train.json")
    ]:
        if os.path.exists(file_path):
            result = analyze_file(file_path, file_name)
            if result:
                results.append(result)
        else:
            print(f"File not found: {file_path}")
    
    # Generate comparative report
    if len(results) > 1:
        generate_comparative_report(results)
    
    print("Analysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

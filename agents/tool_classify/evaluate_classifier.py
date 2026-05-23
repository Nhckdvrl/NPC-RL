import json
import os
from tool_classifier import ToolClassifier # Assumes tool_classifier.py is in the same directory
from typing import List, Dict, Any, Tuple
import sys
import time
import psutil

# Add project root to sys.path to allow importing function_calls
project_root_for_sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root_for_sys_path not in sys.path:
    sys.path.insert(0, project_root_for_sys_path)
from function_calls import tool_map, action_map # Import tool_map and action_map

# Define paths
# Assuming this script is in agents/tool_classify/
BASE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLASSIFY_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "data", "tool_classify")
ORIGINAL_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "data")
DETAILS_OUTPUT_PATH = os.path.join(BASE_PROJECT_PATH, "results", "tool_classify", "details")

# DATASET_NAMES = ["sample", "test", "train"]
DATASET_NAMES = ["sample", "test", "train"]
# TOP_N_VALUES = [5, 10, 15]
TOP_N_VALUES = [5, 10]

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from - {file_path}. Error: {e}", file=sys.stderr)
        return []

def build_tool_list_map() -> Dict[str, List[str]]:
    """
    Builds a map from function_list_id to a list of actual tool names
    using the tool_map from function_calls.
    """
    function_id_to_tool_names = {}
    for func_list_id, outer_dict in tool_map.items():
        if isinstance(outer_dict, dict) and 'function_registry' in outer_dict:
            registry_dict = outer_dict['function_registry']
            if isinstance(registry_dict, dict):
                tool_names = list(registry_dict.keys())
                function_id_to_tool_names[func_list_id] = tool_names
            else:
                print(f"Warning (tool_map): 'function_registry' for {func_list_id} is not a dictionary. Found: {type(registry_dict)}", file=sys.stderr)
                function_id_to_tool_names[func_list_id] = []
        else:
            print(f"Warning (tool_map): Could not find 'function_registry' or item is not a dict for {func_list_id}. Item type: {type(outer_dict)}", file=sys.stderr)
            function_id_to_tool_names[func_list_id] = []
    # print(function_id_to_tool_names["function_list_id_0006"])
    return function_id_to_tool_names

def build_action_list_map() -> Dict[str, List[str]]:
    """
    Builds a map from function_list_id to a list of actual action names
    using the action_map from function_calls.
    """
    function_id_to_action_names = {}
    for func_list_id, outer_dict in action_map.items():
        if isinstance(outer_dict, dict) and 'function_registry' in outer_dict: 
            registry_dict = outer_dict['function_registry']
            if isinstance(registry_dict, dict):
                action_names = list(registry_dict.keys()) 
                function_id_to_action_names[func_list_id] = action_names 
            else:
                print(f"Warning (action_map): 'function_registry' for {func_list_id} is not a dictionary. Found: {type(registry_dict)}", file=sys.stderr)
                function_id_to_action_names[func_list_id] = [] 
        else:
            print(f"Warning (action_map): Could not find 'function_registry' or item is not a dict for {func_list_id}. Item type: {type(outer_dict)}", file=sys.stderr)
            function_id_to_action_names[func_list_id] = [] 
    # print(function_id_to_action_names["function_list_id_0006"])
    return function_id_to_action_names

def build_function_list_map() -> Dict[str, List[str]]:
    """
    Builds a map from function_list_id to a combined list of actual tool and action names.
    """
    tools_map = build_tool_list_map()
    actions_map = build_action_list_map()
    
    combined_map = {}
    all_func_list_ids = set(tools_map.keys()) | set(actions_map.keys())
    
    for func_list_id in all_func_list_ids:
        tool_names = tools_map.get(func_list_id, [])
        action_names = actions_map.get(func_list_id, [])
        # Combine and ensure uniqueness
        combined_names = list(set(tool_names + action_names))
        combined_map[func_list_id] = combined_names
        
    return combined_map

def calculate_recall(predicted_tools: List[str], gold_tools: List[str]) -> float:
    """
    Calculates recall.
    Recall = (Number of correctly recalled gold functions) / (Total number of gold functions)
    """
    if not gold_tools:
        # If there are no gold tools, recall is undefined or 1.0 if no predictions, 0.0 if predictions.
        # For this purpose, if gold_tools is empty, we can't measure recall meaningfully in this context.
        return 0.0 # Or handle as a special case, e.g., by not including it in average

    correctly_recalled_count = 0
    gold_tool_set = set(gold_tools)
    for tool in predicted_tools:
        if tool in gold_tool_set:
            correctly_recalled_count += 1
    
    return correctly_recalled_count / len(gold_tools)

def main():
    # Get the current process for CPU utilization monitoring
    process = psutil.Process(os.getpid())
    # Record initial CPU times for utilization calculation
    initial_cpu_times = process.cpu_times()
    overall_start_time = time.time()
    print("Initializing Tool Classifier...")
    # You can specify a different model if needed, e.g., ToolClassifier(model_name='paraphrase-MiniLM-L3-v2')
    model_name = os.getenv('EMBEDDING_MODEL_NAME', 'all-mpnet-base-v2')
    classifier = ToolClassifier(model_name=model_name)

    print("Building combined function/action list map from function_calls...")
    function_id_to_tool_names_map = build_function_list_map()
    if not function_id_to_tool_names_map:
        print("Error: Could not build function list map. Exiting.", file=sys.stderr)
        return

    for dataset_name in DATASET_NAMES:
        print(f"\n--- Evaluating on {dataset_name} dataset ---")
        eval_file_path = os.path.join(CLASSIFY_DATA_PATH, f"tool_classify_{dataset_name}.json")
        evaluation_data = load_json_file(eval_file_path)

        if not evaluation_data:
            print(f"No data to evaluate for {dataset_name}. Skipping.")
            continue

        dataset_recalls = {n: [] for n in TOP_N_VALUES}
        valid_samples_count = {n: 0 for n in TOP_N_VALUES}
        detailed_results = {n: [] for n in TOP_N_VALUES}

        # Initialize per-dataset accumulators for timing
        total_inference_time_dataset = 0.0
        inference_count_dataset = 0

        for i, entry in enumerate(evaluation_data):
            # if (i + 1) % 200 == 0:
            #     print(f"  Processing entry {i + 1}/{len(evaluation_data)} for {dataset_name}...")

            query = entry.get("user_utterance")
            gold_function_objects = entry.get("gold_functions", [])
            function_list_id = entry.get("function_list_id")

            if not query or not function_list_id:
                # print(f"Skipping entry due to missing query or function_list_id: {entry.get('user_utterance', 'N/A')}", file=sys.stderr)
                continue

            actual_gold_tool_names = [func.get("name") for func in gold_function_objects if func.get("name")]
            
            if not actual_gold_tool_names:
                # If there are no gold standard tools for this utterance, skip for recall calculation.
                # print(f"Skipping entry with no gold functions for recall: {query}")
                continue

            all_available_tools_for_id = function_id_to_tool_names_map.get(function_list_id)
            # print(all_available_tools_for_id)
            if not all_available_tools_for_id:
                # print(f"Warning: No tool list found for function_list_id '{function_list_id}'. Skipping entry: {query}", file=sys.stderr)
                continue
            
            if not isinstance(all_available_tools_for_id, list) or not all(isinstance(tool, str) for tool in all_available_tools_for_id):
                # print(f"Warning: Tool list for function_list_id '{function_list_id}' is invalid. Skipping entry: {query}", file=sys.stderr)
                continue
            
            if not all_available_tools_for_id: # No tools to select from
                # print(f"Warning: Empty tool list for function_list_id '{function_list_id}'. Skipping entry: {query}", file=sys.stderr)
                for n_val in TOP_N_VALUES:
                    # If no tools available, but gold tools exist, recall is 0.
                    recall = calculate_recall([], actual_gold_tool_names)
                    dataset_recalls[n_val].append(recall)
                    valid_samples_count[n_val] += 1
                continue

            inference_start_time = time.time() # Start timing for this entry's processing
            for n_val in TOP_N_VALUES:
                # Get top N predictions
                # The ToolClassifier's recall_top_n handles if n_val > len(all_available_tools_for_id)
                predicted_tuples = classifier.recall_top_n(query, all_available_tools_for_id, top_n=n_val)
                predicted_tool_names = [pt[0] for pt in predicted_tuples]
                
                recall = calculate_recall(predicted_tool_names, actual_gold_tool_names)
                dataset_recalls[n_val].append(recall)
                valid_samples_count[n_val] += 1

                # Store detailed results for this N value
                detailed_results[n_val].append({
                    'entry_index': i,
                    'query': query,
                    'available_tools': all_available_tools_for_id,
                    'gold_tools': actual_gold_tool_names,
                    'predicted_tools': dict(predicted_tuples),
                    'recall_for_entry': recall
                })
            # End of for n_val loop for this entry
            inference_end_time = time.time() # End timing for this entry's processing

            # Accumulate total inference time and count for this dataset
            total_inference_time_dataset += (inference_end_time - inference_start_time)
            inference_count_dataset += 1

        print(f"\nResults for {dataset_name} dataset:")
        for n_val in TOP_N_VALUES:
            if valid_samples_count[n_val] > 0:
                avg_recall = sum(dataset_recalls[n_val]) / valid_samples_count[n_val]
                print(f"  Average Recall @{n_val}: {avg_recall:.4f} (from {valid_samples_count[n_val]} valid samples)")
            else:
                print(f"  Average Recall @{n_val}: N/A (No valid samples to calculate for {dataset_name})")

            # Save detailed results to JSON files
            for n_val, results_list in detailed_results.items():
                output_filename = os.path.join(DETAILS_OUTPUT_PATH, f"{dataset_name}_{n_val}.json")
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(results_list, f, indent=4)
                    print(f"  Detailed results for @{n_val} saved to {output_filename}")
                except IOError as e:
                    print(f"Error writing detailed results to {output_filename}: {e}", file=sys.stderr)
        
        if inference_count_dataset > 0:
            avg_inference_time_per_entry = total_inference_time_dataset / inference_count_dataset
            print(f"  Average processing time per entry for {dataset_name}: {avg_inference_time_per_entry:.4f} seconds")

    overall_end_time = time.time()
    total_script_duration = overall_end_time - overall_start_time
    final_cpu_times = process.cpu_times()

    # Calculate CPU utilization
    # Total CPU time used by the process = (user_time_after - user_time_before) + (system_time_after - system_time_before)
    # Total elapsed wall-clock time for the process = overall_end_time - overall_start_time
    # CPU utilization = (Total CPU time used by the process / number_of_cores) / Total elapsed wall-clock time * 100
    # Simplified: (total_cpu_time_diff / total_script_duration) * 100 (for a single core equivalent usage)
    cpu_time_used = (final_cpu_times.user - initial_cpu_times.user) + (final_cpu_times.system - initial_cpu_times.system)
    
    # psutil.cpu_count(logical=False) gives physical cores, psutil.cpu_count(logical=True) gives logical (threads)
    # Using logical=True for a more common representation of 'available processing units'
    num_cores = psutil.cpu_count(logical=True) 
    cpu_utilization = (cpu_time_used / num_cores / total_script_duration) * 100 if total_script_duration > 0 and num_cores else 0

    print(f"\n--- Overall Script Metrics ---")
    print(f"Total script execution time: {total_script_duration:.2f} seconds")
    print(f"Total CPU time used by script (user + system): {cpu_time_used:.2f} seconds")
    print(f"Approximate average CPU Utilization: {cpu_utilization:.2f}% (across {num_cores} logical cores)")


if __name__ == "__main__":
    main()

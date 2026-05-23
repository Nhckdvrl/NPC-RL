import json
import random
import os

def sample_utterances_from_file():
    """
    Reads triggered utterances from a JSON file, samples a few for each tool,
    and prints them. This output will be used to manually create a
    tool_scopes.md file.
    """
    base_project_dir = "/path/to/npc-rl/src/cpdc-boost"
    json_file_path = os.path.join(base_project_dir, "data-insights/results/tool_call_insights/triggered_utterances.json")
    output_script_dir = os.path.join(base_project_dir, "utterances2prompts")

    # Ensure the directory for this script itself exists, though it should if running from there.
    os.makedirs(output_script_dir, exist_ok=True)

    sampled_utterances_for_analysis = {}

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            triggered_utterances = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        print("Please ensure the file exists and the path is correct.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        print("Please ensure the file is a valid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")
        return None

    print("--- Sampled Utterances for Tool Scope Analysis ---")
    for tool_name, utterances in triggered_utterances.items():
        if not isinstance(utterances, list):
            print(f"\nTool: {tool_name}")
            print(f"  Warning: Expected a list of utterances, but got {type(utterances)}. Skipping.")
            sampled_utterances_for_analysis[tool_name] = {'error': 'Invalid data format'}
            continue

        if not utterances:
            print(f"\nTool: {tool_name}")
            print("  No utterances found for this tool.")
            sampled_utterances_for_analysis[tool_name] = {"samples": [], "message": "No utterances"}
            continue

        if len(utterances) <= 5:
            sample = list(utterances)  # Take all if 5 or fewer, ensure it's a list
        else:
            sample = random.sample(utterances, 5)
        
        print(f"\nTool: {tool_name}")
        for utt in sample:
            print(f"  - {utt}")
        sampled_utterances_for_analysis[tool_name] = {"samples": sample}
    
    print("\n--- End of Sampled Utterances ---")
    # This dictionary is not directly used by this script for file output,
    # but represents the data extracted for potential further programmatic use.
    # For this task, the printed output is key for the next manual step.
    return sampled_utterances_for_analysis

if __name__ == "__main__":
    sample_utterances_from_file()

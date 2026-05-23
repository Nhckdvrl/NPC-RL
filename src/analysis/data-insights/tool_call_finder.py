import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import os
import re

# Basic list of English stop words
STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", 
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", 
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", 
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", 
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", 
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", 
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "tell", "me", "please", "show", 
    "give", "let's", "let", "see", "look", "looking", "want", "like", "would", "could", "may", "might", "about", 
    "what's", "it's", "i'm", "i'll", "i'd", "you're", "you'll", "you'd", "he's", "she's", "we're", "they're", 
    "one", "two", "how", "much", "many", "good", "well", "really", "think", "go", "get", "got", "know", "need",
    "okay", "ok", "yes", "yeah", "yep", "sure", "thanks", "thank", "hello", "hi", "hey", "greetings", "morning", 
    "afternoon", "evening", "bye", "goodbye", "alright", "then", "also", "another", "other", "quest", "quests", 
    "item", "items", "weapon", "weapons", "armor", "armors", "what about", "how about", "can you", "could you", 
    "will you", "do you", "i want to", "i would like to", "i need to", "i am", "a", "b", "c"
])

class ToolCallFinder:
    """
    Analyzes conversation data to find user utterances that trigger specific tool calls.
    """

    def __init__(self, data_files: List[str]):
        """
        Initializes the ToolCallFinder.

        Args:
            data_files: A list of paths to JSON data files containing conversations.
                        Each file should be a list of conversations.
                        Each conversation should have a "turns" key, which is a list of turns.
                        Each turn should have "speaker" ("USER" or "ASSISTANT") and "utterance".
                        ASSISTANT turns can have "tool_calls" (a list of tool call dicts, each with "name" and "parameters").
        """
        self.data_files = data_files
        self.tool_utterances = defaultdict(list) # List will now store dicts: {'utterance': str, 'parameters': dict}

    def _load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads data from a single JSON file."""
        exists = os.path.exists(file_path)
        print(f"  Inside _load_data: Checking existence of '{file_path}': {exists}") # Diagnostic print
        if not exists:
            print(f"  Inside _load_data: Checking read access for '{file_path}': {os.access(file_path, os.R_OK)}") # Diagnostic for read access
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"Warning: Data in {file_path} is not a list as expected. Skipping.")
                    return []
                return data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path} during _load_data. Skipping.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: JSON decoding error in {file_path}: {e}. Skipping.")
            return []
        except Exception as e:
            print(f"Error: An unexpected error occurred while loading {file_path}: {type(e).__name__} - {e}. Skipping.")
            return []

    def process_data(self) -> None:
        """
        Processes all specified data files to find tool call triggers.
        Results are stored in self.tool_utterances.
        """
        print("\nProcessing data files...")
        if not self.data_files:
            print("No data files specified for processing.")
            return

        for file_path in self.data_files:
            print(f"Processing {file_path} (repr: {repr(file_path)})...") # Added repr() for the path
            # Ensure file_path is a string and not None or other types
            if not isinstance(file_path, str):
                print(f"Warning: Invalid file path type ({type(file_path)}): {file_path}. Skipping.")
                continue

            game_scenarios = self._load_data(file_path)
            if not game_scenarios: # Skip if file loading failed or returned empty
                continue

            for scenario_idx, scenario in enumerate(game_scenarios):
                if not isinstance(scenario, dict):
                    print(f"Warning: Scenario {scenario_idx} in {file_path} is not a dict. Skipping.")
                    continue

                # Find all turn_X keys and sort them numerically
                turn_keys = []
                for key in scenario.keys():
                    if key.startswith("turn_") and key.split('_')[-1].isdigit():
                        turn_keys.append(key)
                
                # Sort keys like turn_0, turn_1, ..., turn_10, etc.
                turn_keys.sort(key=lambda x: int(x.split('_')[-1]))

                for turn_key in turn_keys:
                    turn_data = scenario.get(turn_key)
                    if not isinstance(turn_data, dict):
                        print(f"Warning: Content of {turn_key} in scenario {scenario_idx}, file {file_path} is not a dict. Skipping.")
                        continue

                    dialogue_history = turn_data.get("dialogue")
                    gold_functions = turn_data.get("gold_functions")

                    last_player_utterance = None
                    if isinstance(dialogue_history, list):
                        for dialogue_entry in dialogue_history:
                            if isinstance(dialogue_entry, dict) and \
                               dialogue_entry.get("speaker") == "player" and \
                               isinstance(dialogue_entry.get("text"), str):
                                last_player_utterance = dialogue_entry.get("text")
                    
                    if last_player_utterance and isinstance(gold_functions, list):
                        for tc_idx, tool_call in enumerate(gold_functions):
                            if not isinstance(tool_call, dict):
                                print(f"Warning: Tool call {tc_idx} in {turn_key}, scenario {scenario_idx}, file {file_path} is not a dict. Skipping.")
                                continue
                            tool_name = tool_call.get("name")
                            tool_parameters = tool_call.get("parameters", {}) # Get parameters, default to empty dict

                            if isinstance(tool_name, str):
                                tool_name = tool_name.strip() # Ensure consistency
                                self.tool_utterances[tool_name].append({
                                    "utterance": last_player_utterance,
                                    "parameters": tool_parameters
                                })
                            else:
                                print(f"Warning: Tool call name in {turn_key}, scenario {scenario_idx}, file {file_path} is not a string or missing. Tool call: {tool_call}")

    def get_triggered_utterances(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the aggregated utterances and their parameters that triggered tool calls.
        Processes data if not already done.

        Returns:
            A dictionary where keys are tool names and values are lists of dictionaries,
            each dictionary containing 'utterance' and 'parameters'.
        """
        if not self.tool_utterances:
            self.process_data()
        return dict(self.tool_utterances)

    def print_results(self, limit_per_tool: Optional[int] = None) -> None:
        """
        Prints the results to the console.

        Args:
            limit_per_tool: Optional. If set, limits the number of unique utterance-parameter pairs printed per tool.
        """
        results = self.get_triggered_utterances()
        if not results:
            print("No tool call triggers found or data processed.")
            return

        for tool_name, utterance_objects in results.items():
            # Create a string representation for uniqueness check if needed, or just print all
            # For simplicity, we'll print all occurrences up to a limit if specified, 
            # rather than trying to define "unique" based on utterance + params string.
            print(f"\nTool: {tool_name} (Triggered {len(utterance_objects)} times)")
            
            count = 0
            for utt_obj in utterance_objects:
                if limit_per_tool is not None and count >= limit_per_tool:
                    print(f"  ... and {len(utterance_objects) - count} more triggers.")
                    break
                utterance = utt_obj.get("utterance", "N/A")
                parameters = utt_obj.get("parameters", {})
                print(f"  - Utterance: \"{utterance}\"")
                print(f"    Parameters: {json.dumps(parameters)}")
                count += 1

    def generate_keyword_lists(self, top_n: int = 10) -> Dict[str, List[Tuple[str, int]]]:
        """
        Generates keyword lists for each tool based on utterance frequency.

        Args:
            top_n: The number of top keywords to return for each tool.

        Returns:
            A dictionary where keys are tool names and values are lists of (keyword, frequency) tuples.
        """
        if not self.tool_utterances:
            self.process_data() # Ensure data is processed
        
        keyword_lists = defaultdict(list)
        
        for tool_name, utterance_objects in self.tool_utterances.items():
            all_words = []
            for utt_obj in utterance_objects:
                utt_text = utt_obj.get("utterance", "")
                if not utt_text: # Skip if utterance text is empty
                    continue
                # Normalize: lowercase and remove punctuation (keep alphanumeric and spaces)
                normalized_utt = re.sub(r'[^a-z0-9\s]', '', utt_text.lower())
                words = normalized_utt.split()
                # Filter stop words and short words (e.g., less than 2 chars, often remnants)
                filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
                all_words.extend(filtered_words)
            
            if all_words:
                word_counts = Counter(all_words)
                top_keywords = word_counts.most_common(top_n)
                keyword_lists[tool_name] = top_keywords
        
        return dict(keyword_lists)

    def print_keyword_lists(self, top_n: int = 10) -> None:
        """
        Prints the generated keyword lists to the console.

        Args:
            top_n: The number of top keywords to display for each tool.
        """
        keyword_lists = self.generate_keyword_lists(top_n=top_n)

        if not keyword_lists:
            print("No keywords generated. Ensure data has been processed and contains tool calls.")
            return

        print("\n--- Keyword Lists (Top " + str(top_n) + ") ---")
        for tool_name, keywords in keyword_lists.items():
            print(f"\nTool: {tool_name}")
            if not keywords:
                print("  No significant keywords found after filtering.")
                continue
            for keyword, freq in keywords:
                print(f"  - {keyword} (count: {freq})")

    def save_triggered_utterances_to_json(self, output_path: str) -> None:
        """
        Saves the aggregated triggered utterances to a JSON file.

        Args:
            output_path: The full path to the output JSON file.
        """
        if not self.tool_utterances:
            self.process_data() # Ensure data is processed
        
        # For JSON serialization, convert defaultdict to dict and sets of utterances to lists
        output_data = { 
            tool: [
                {
                    "utterance": utt_obj.get("utterance", ""),
                    "parameters": utt_obj.get("parameters", {})
                } 
                for utt_obj in utts 
            ] 
            for tool, utts in self.tool_utterances.items() 
        }

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"\nSuccessfully saved triggered utterances to: {output_path}")
        except IOError as e:
            print(f"Error saving triggered utterances to {output_path}: {e}")

    def save_keyword_lists_to_json(self, output_path: str, top_n: int = 10) -> None:
        """
        Generates and saves the keyword lists to a JSON file.

        Args:
            output_path: The full path to the output JSON file.
            top_n: The number of top keywords to generate and save for each tool.
        """
        keyword_lists = self.generate_keyword_lists(top_n=top_n)

        if not keyword_lists:
            print("No keywords generated to save.")
            return

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(keyword_lists, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved keyword lists to: {output_path}")
        except IOError as e:
            print(f"Error saving keyword lists to {output_path}: {e}")

if __name__ == "__main__":
    print("ToolCallFinder Example Usage")
    print("=" * 30)
    print("This script helps find user utterances that likely triggered specific tool calls.")
    print("It processes JSON files containing conversation data (game scenarios with turns).")
    print("For each tool call found, it extracts the preceding player utterance.")
    print("Finally, it generates and prints keyword lists for each tool based on these utterances.")

    # Define the project root relative to this script's location
    # Assuming the script is in src/cpdc-boost/data-insights/
    # and data is in data/
    script_path = os.path.abspath(__file__)
    # project_root should be /path/to/npc-rl
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
    print(f"\nCalculated project root: {project_root}")

    # Diagnostic: List contents of the data directory from within Python
    data_dir_path = os.path.join(project_root, "data")
    print(f"Attempting to list contents of data directory: {data_dir_path}")
    try:
        data_dir_contents = os.listdir(data_dir_path)
        print(f"  Contents of '{data_dir_path}': {data_dir_contents}")
    except Exception as e:
        print(f"  Error listing contents of '{data_dir_path}': {type(e).__name__} - {e}")

    # List of example data files relative to the project root
    example_files_relative_to_project_root = [
        os.path.join("data", "task1_train.json"),
        os.path.join("data", "task2_train.json"),
        os.path.join("data", "task1_sample.json"), 
        os.path.join("data", "task2_sample.json"), 
    ]

    print("\nConstructing list of all specified example data files relative to project root:", project_root)
    all_specified_files_to_attempt = []
    for rel_path in example_files_relative_to_project_root:
        abs_f_path = os.path.join(project_root, rel_path)
        all_specified_files_to_attempt.append(abs_f_path)
        print(f"  Will attempt to process: {abs_f_path}")

    if not all_specified_files_to_attempt:
        print("\nNo files were specified in example_files_relative_to_project_root.")
        print("Please ensure example_files_relative_to_project_root is populated or manually create a ToolCallFinder instance.")
    else:
        print(f"\nAttempting to process all specified data files: {all_specified_files_to_attempt}")
        finder = ToolCallFinder(data_files=all_specified_files_to_attempt)
        
        # The ToolCallFinder's _load_data method will handle non-existent files by printing a warning and returning None.
        # Subsequent processing steps will then skip these.

        print("\n--- Results (showing up to 10 unique utterances per tool) ---")
        finder.print_results(limit_per_tool=10)
        
        # Example of getting all triggers for a specific tool:
        # all_triggers = finder.get_triggered_utterances()
        # specific_tool = 'search_item' # Change as needed
        # if specific_tool in all_triggers:
        #     print(f"\nAll (including duplicate) triggers for '{specific_tool}':")
        #     for utt in all_triggers[specific_tool]:
        #         print(f"  - \"{utt}\"")

        # Generate and print keyword lists
        finder.print_keyword_lists(top_n=10)

        # Define output directory and save results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "results", "tool_call_insights")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True) # Redundant if using os.makedirs in save methods, but good for clarity

        utterances_output_path = os.path.join(output_dir, "triggered_utterances.json")
        finder.save_triggered_utterances_to_json(utterances_output_path)

        keywords_output_path = os.path.join(output_dir, "keyword_lists.json")
        finder.save_keyword_lists_to_json(keywords_output_path, top_n=10)

    print("\n" + "=" * 30)
    print("End of Example Usage")

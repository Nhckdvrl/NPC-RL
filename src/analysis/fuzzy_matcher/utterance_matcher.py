import json
import os
import random
from fuzzywuzzy import fuzz
from collections import defaultdict

# Path to the triggered utterances JSON file
# This assumes the script is run from the project root or the path is adjusted accordingly.
# For robustness, let's calculate it from this script's location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to cpdc-boost, then to data-insights, then results, etc.
# SCRIPT_DIR = fuzzy_matcher
# PARENT_OF_SCRIPT_DIR = cpdc-boost
# GRANDPARENT_OF_SCRIPT_DIR = src
# GREAT_GRANDPARENT_OF_SCRIPT_DIR = npc-rl (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
UTTERANCES_FILE_PATH = os.path.join(
    PROJECT_ROOT, 
    "src", 
    "cpdc-boost", 
    "data-insights", 
    "results", 
    "tool_call_insights", 
    "triggered_utterances.json"
)

def load_utterances(file_path: str) -> dict:
    """Loads utterances from the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Utterances file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return {}

def convert_to_knowledge_base(all_utterances_by_tool: dict) -> list:
    """Converts the loaded utterances dict to a flat list of (utterance, tool_name) tuples."""
    knowledge_base = []
    if not all_utterances_by_tool:
        return knowledge_base
    for tool_name, utterances in all_utterances_by_tool.items():
        for utt in utterances:
            knowledge_base.append((utt, tool_name))
    return knowledge_base

def find_similar_examples(query: str, knowledge_base: list, similarity_threshold: int = 55, top_n: int = 3) -> list:
    """
    Finds the top_n most similar example utterances from the knowledge base for a given query.
    Args:
        query: The new user utterance.
        knowledge_base: A list of (utterance, tool_name) tuples.
        similarity_threshold: Minimum similarity score to consider a match.
        top_n: The maximum number of similar examples to return.
    Returns:
        A list of tuples, where each tuple is (example_utterance, tool_name, similarity_score),
        sorted by similarity_score in descending order.
    """
    matches = []
    if not knowledge_base:
        return matches

    for known_utterance, tool_name in knowledge_base:
        score = fuzz.token_sort_ratio(query.lower(), known_utterance.lower())
        if score >= similarity_threshold:
            matches.append((known_utterance, tool_name, score))
    
    # Sort matches by score in descending order
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches[:top_n]

if __name__ == "__main__":
    print(f"Loading knowledge base from: {UTTERANCES_FILE_PATH}")
    all_utterances_dict = load_utterances(UTTERANCES_FILE_PATH)

    if not all_utterances_dict:
        print("No utterances loaded to form knowledge base. Exiting.")
    else:
        knowledge_base_list = convert_to_knowledge_base(all_utterances_dict)
        print(f"Knowledge base loaded with {len(knowledge_base_list)} utterances.")

        if not knowledge_base_list:
            print("Knowledge base is empty. Cannot provide examples.")
        else:
            print("\n--- RAG Prompt Help System Example ---")
            
            example_queries = [
                "I want to find a quest that is not too difficult",
                "How much does that cool sword cost?",
                "Tell me more about the rat extermination quest",
                "I'm ready to start my adventure!"
            ]

            for user_query in example_queries:
                print(f"\nUser Query: '{user_query}'")
                similar_examples = find_similar_examples(user_query, knowledge_base_list, similarity_threshold=55, top_n=3)
                
                if similar_examples:
                    print("Found similar examples:")
                    for utt, tool, score in similar_examples:
                        print(f"  - Example: '{utt}' (Tool: {tool}, Score: {score}%)")
                else:
                    print("  No sufficiently similar examples found in the knowledge base.")

import os
import json
import sys
from fuzzywuzzy import fuzz # type: ignore 

# Global variable to store the loaded knowledge base
_knowledge_base_list = []
_rag_initialized = False

def _get_project_root():
    """Gets the project root directory.
    Assumes this module is in agents/rag/rag_module.py,
    so the project root is three levels up from the file's directory.
    """
    # Path of this file: .../project_root/agents/rag/rag_module.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _load_utterances_from_file(file_path: str) -> dict:
    """Loads utterances from a JSON file."""
    if not os.path.exists(file_path):
        print(f"RAG_MODULE WARNING: Utterances file not found at {file_path}", file=sys.stderr)
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"RAG_MODULE ERROR: Failed to load or parse utterances from {file_path}: {e}", file=sys.stderr)
        return {}

def _convert_utterances_to_kb(all_utterances_dict: dict) -> list:
    """Converts a dictionary of utterances to a flat list for the knowledge base.
    Assumes each value in all_utterances_dict is a list of objects,
    where each object has an 'utterance' key and a 'parameters' key.
    Example: {'tool_name': [{'utterance': 'text', 'parameters': {'p1': 'v1'}}]}
    """
    knowledge_base = []
    for tool_name, utterance_objects in all_utterances_dict.items():
        if isinstance(utterance_objects, list):
            for utterance_obj in utterance_objects:
                if isinstance(utterance_obj, dict):
                    text = utterance_obj.get("utterance", "")
                    params = utterance_obj.get("parameters", {})
                    if text: # Only add if there's an utterance text
                        knowledge_base.append((text, tool_name, params))
                else:
                    # Fallback for old format (list of strings), though parameters will be empty
                    if isinstance(utterance_obj, str):
                        knowledge_base.append((utterance_obj, tool_name, {}))
        # Optional: handle cases where the structure isn't as expected, or log a warning
    return knowledge_base

def initialize_rag_system():
    """Initializes the RAG system by loading utterances and preparing the knowledge base."""
    global _knowledge_base_list, _rag_initialized
    if _rag_initialized:
        return

    project_root = _get_project_root()
    utterances_file_path = os.path.join(
        project_root,
        "src",
        "cpdc-boost",
        "data-insights",
        "results",
        "tool_call_insights",
        "triggered_utterances.json"
    )

    all_utterances_dict = _load_utterances_from_file(utterances_file_path)
    _knowledge_base_list = _convert_utterances_to_kb(all_utterances_dict)
    _rag_initialized = True

    if _knowledge_base_list:
        print(f"RAG_MODULE INFO: RAG Knowledge base loaded with {len(_knowledge_base_list)} utterances.", file=sys.stdout)
    else:
        print("RAG_MODULE WARNING: RAG Knowledge base is empty after loading.", file=sys.stderr)

def _find_similar_examples_for_query(query: str, top_n: int = 2, similarity_threshold: int = 55) -> list:
    """Finds similar examples from the knowledge base for a given query."""
    if not _rag_initialized or not _knowledge_base_list:
        return []

    scores = []
    for utterance_text, tool_name, params in _knowledge_base_list:
        score = fuzz.token_sort_ratio(query.lower(), utterance_text.lower())
        if score >= similarity_threshold:
            scores.append((score, utterance_text, tool_name, params))

    scores.sort(key=lambda x: x[0], reverse=True)
    # Return list of (utterance_text, tool_name, parameters, score)
    return [(utt_text, tool, p, scr) for scr, utt_text, tool, p in scores[:top_n]]

def get_formatted_rag_examples(user_query: str, top_n: int = 2, similarity_threshold: int = 55) -> str:
    """Retrieves similar examples for a user query and formats them into a string."""
    if not user_query or not _rag_initialized: # Ensure RAG is initialized
        return ""
        
    examples = _find_similar_examples_for_query(user_query, top_n=top_n, similarity_threshold=similarity_threshold)
    
    if not examples:
        return ""

    example_lines = ["## Relevant Examples from Past Interactions:"] 
    for utt, tool, params, score in examples:
        params_str = json.dumps(params) if params else "{}"
        example_lines.append(f"### User query\n{utt}")
        example_lines.append(f"### Function name\n{tool}")
        example_lines.append(f"### Function parameter\n{params_str}")
        example_lines.append("---") 
    
    if example_lines and example_lines[-1] == "---":
        example_lines.pop()

    return "\n".join(example_lines) + "\n\n"

if __name__ == "__main__":
    print("--- Running RAG Module Test ---")
    initialize_rag_system()

    test_queries = [
        "I want to buy a sword",
        "show me a cheap shield",
        "what quests are available?",
        "how much is the broadsword?",
        "sell my dagger"
    ]

    if not _rag_initialized or not _knowledge_base_list:
        print("RAG system not initialized or knowledge base empty. Cannot run tests.")
    else:
        print(f"Knowledge base size: {len(_knowledge_base_list)}")
        # Example of how parameters would look if present in _knowledge_base_list
        # for utt, tool, params in _knowledge_base_list[:2]:
        #     print(f"KB item: Utterance='{utt}', Tool='{tool}', Params='{params}'")

        for query in test_queries:
            print(f"\n--- Test RAG Output for query: '{query}' ---")
            examples_text = get_formatted_rag_examples(query, top_n=3, similarity_threshold=50)
            if examples_text:
                print(examples_text)
            else:
                print(f"--- No RAG examples found for query: '{query}' ---")
    print("--- RAG Module Test Finished ---")
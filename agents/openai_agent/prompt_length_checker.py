import os
import pathlib

# Determine the project's 'agents' directory
# __file__ is .../agents/openai_agent/prompt_length_checker.py
# .parent is .../agents/openai_agent/
# .parent.parent is .../agents/
project_agents_dir = pathlib.Path(__file__).resolve().parent.parent
tiktoken_cache_dir = project_agents_dir / ".tiktoken_cache"

# Create the cache directory if it doesn't exist
tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)

# Set the TIKTOKEN_CACHE_DIR environment variable BEFORE importing tiktoken
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

import tiktoken
import json
import copy
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from . import config
# It's a good practice to initialize the tokenizer once and reuse it.
# 'cl100k_base' is the encoding for models like gpt-4, gpt-3.5-turbo, etc.
try:
    encoding = tiktoken.encoding_for_model(config.get_openai_model())
except KeyError:
    encoding = tiktoken.get_encoding("gpt-4o-mini")
from .config import MAX_TOKENS_FUNCTION_CALL_COMPETITION


def load_keyword_mapping() -> Dict[str, List[str]]:
    """Load the keyword mapping from the JSON file.
    
    Returns:
        Dict[str, List[str]]: A dictionary mapping tool names to lists of keywords.
    """
    keyword_file = Path(__file__).parent / "keyword_mapping.json"
    try:
        with open(keyword_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load keyword mapping: {e}")
        return {}

def calculate_tool_relevance(tool_name: str, user_message: str, keyword_mapping: Dict[str, List[str]]) -> int:
    """Calculate the relevance score of a tool based on keyword matching.
    
    Args:
        tool_name (str): The name of the tool.
        user_message (str): The user's message.
        keyword_mapping (Dict[str, List[str]]): The keyword mapping.
        
    Returns:
        int: A relevance score (higher is more relevant).
    """
    if tool_name not in keyword_mapping:
        return 0
    
    score = 0
    user_message = user_message.lower()
    for keyword in keyword_mapping[tool_name]:
        if keyword.lower() in user_message:
            score += 1
    return score

def reorder_tools_by_relevance(tools: List[Dict[str, Any]], user_message: str, keyword_mapping: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Reorder tools based on their relevance to the user's message.
    
    Args:
        tools (List[Dict[str, Any]]): The list of tools.
        user_message (str): The user's message.
        keyword_mapping (Dict[str, List[str]]): The keyword mapping.
        
    Returns:
        List[Dict[str, Any]]: The reordered list of tools.
    """
    if not user_message or not keyword_mapping:
        return tools
    
    # Calculate relevance scores for each tool
    tool_scores = []
    for tool in tools:
        tool_name = tool.get("function", {}).get("name", "")
        score = calculate_tool_relevance(tool_name, user_message, keyword_mapping)
        tool_scores.append((tool, score))
    
    # Sort tools by relevance score (descending)
    tool_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return tools in order of relevance
    return [tool for tool, _ in tool_scores]

def trim_tool_descriptions(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_attempts: int = 3
) -> List[Dict[str, Any]]:
    """
    Trims the descriptions of tools if the prompt token count exceeds the maximum allowed.
    First tries to reorder tools based on relevance to the user's message, then removes
    the least relevant tools, and finally trims descriptions if needed.

    Args:
        messages (List[Dict[str, Any]]): The messages to be sent to the model.
        tools (List[Dict[str, Any]]): The list of tools available.
        max_attempts (int): Maximum number of tools to remove before falling back to description trimming.

    Returns:
        List[Dict[str, Any]]: The list of tools with potentially trimmed descriptions.
    """
    if not tools:
        logging.warning("No tools provided, nothing to optimize")
        return tools
        
    # Make a deep copy to avoid modifying the original
    processed_tools = copy.deepcopy(tools)
    original_tool_count = len(processed_tools)
    original_tool_names = [t.get("function", {}).get("name", f"tool_{i}") for i, t in enumerate(processed_tools)]
    
    # Get the last user message if available
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    # Load keyword mapping
    keyword_mapping = load_keyword_mapping()
    
    # Initial token count check
    token_count = get_prompt_token_count(messages=messages, tool_list=processed_tools)
    if token_count <= MAX_TOKENS_FUNCTION_CALL_COMPETITION:
        logging.warning(f"Initial token count {token_count} is within limit, no optimization needed")
        return processed_tools
    
    logging.warning(
        f"Token count {token_count} exceeds limit {MAX_TOKENS_FUNCTION_CALL_COMPETITION}. "
        f"Starting optimization of {original_tool_count} tools..."
    )
    logging.warning(f"User Query: {user_message}")
    logging.warning(f"Original tools: {', '.join(original_tool_names) if original_tool_names else 'No named tools found'}")
    
    # Store original descriptions for later comparison using tool names as keys
    original_descriptions = {
        tool.get("function", {}).get("name", f"tool_{i}"): tool.get("function", {}).get("description", "")
        for i, tool in enumerate(processed_tools)
    }
    
    # First, reorder tools by relevance
    if user_message and keyword_mapping:
        processed_tools = reorder_tools_by_relevance(processed_tools, user_message, keyword_mapping)
        Reordering_tool_names = [t.get("function", {}).get("name", f"tool_{i}") for i, t in enumerate(processed_tools)]
        logging.warning(f"Reordering tools: {Reordering_tool_names}")
        token_count = get_prompt_token_count(messages=messages, tool_list=processed_tools)
        if token_count <= MAX_TOKENS_FUNCTION_CALL_COMPETITION:
            logging.warning("✓ Tool reordering brought token count under limit")
            return processed_tools
    
    # Track removed tools
    removed_tools = []
    
    # Try removing the least relevant tools (up to max_attempts)
    for attempt in range(min(max_attempts, len(processed_tools) - 1)):
        if not processed_tools:
            break
            
        # Get the tool to be removed
        removed_tool = processed_tools[-1]
        tool_name = removed_tool.get("function", {}).get("name", f"tool_{len(processed_tools)-1}")
        
        # Remove the last tool (least relevant after reordering)
        processed_tools = processed_tools[:-1]
        removed_tools.append(tool_name)
        
        # Calculate new token count
        token_count = get_prompt_token_count(messages=messages, tool_list=processed_tools)
        
        logging.warning(f"Removed tool '{tool_name}'. New token count: {token_count}")
        
        if token_count <= MAX_TOKENS_FUNCTION_CALL_COMPETITION:
            logging.warning(
                f"✓ Successfully reduced token count by removing {len(removed_tools)} tools. "
                f"Removed tools: {', '.join(removed_tools)}"
            )
            return processed_tools
    
    # If we still exceed the limit, trim descriptions
    if not processed_tools:
        logging.warning("No tools remaining after removal attempts")
        return processed_tools
        
    logging.warning("Tool removal was not enough, falling back to trimming descriptions...")
    
    # Calculate total description length before trimming
    total_original_length = sum(
        len(tool.get("function", {}).get("description", "")) 
        for tool in processed_tools
    )
    
    if total_original_length == 0:
        logging.warning("No descriptions available to trim")
        return processed_tools
    
    # Continue trimming until the token count is below the threshold
    iteration = 0
    while token_count > MAX_TOKENS_FUNCTION_CALL_COMPETITION:
        iteration += 1
        chars_trimmed_this_round = 0
        
        # Trim 10% of the description of each tool
        for tool in processed_tools:
            function = tool.get("function", {})
            description = function.get("description", "")
            if description:
                # Keep at least 10 chars or 90% of original, whichever is larger
                min_keep_length = max(10, int(len(description) * 0.9))
                # Calculate how many characters to trim (10% of current length)
                chars_to_trim = int(len(description) * 0.1)
                # Ensure we don't trim too much (keep at least min_keep_length)
                new_length = max(min_keep_length, len(description) - chars_to_trim)
                chars_trimmed = len(description) - new_length
                chars_trimmed_this_round += chars_trimmed
                function["description"] = description[:new_length]
        
        new_token_count = get_prompt_token_count(messages=messages, tool_list=processed_tools)
        
        if new_token_count == token_count:  # No more progress can be made
            logging.warning("✗ Cannot reduce token count further by trimming descriptions")
            break
            
        logging.warning(
            f"Trim iteration {iteration}: Removed {chars_trimmed_this_round} chars, "
            f"tokens: {token_count} -> {new_token_count}"
        )
        token_count = new_token_count
    
    # Generate final report
    logging.warning("\n=== Tool Optimization Summary ===")
    logging.warning(f"Original tools: {original_tool_count}")
    logging.warning(f"Tools after optimization: {len(processed_tools)}")
    
    if removed_tools:
        logging.warning(f"✂️ Removed {len(removed_tools)} tools: {', '.join(removed_tools)}")
    else:
        logging.warning("✓ No tools were removed")
    
    # Calculate total trimmed characters
    total_trimmed = 0
    for tool in processed_tools:
        tool_name = tool.get("function", {}).get("name", "unknown_tool")
        original_desc = original_descriptions.get(tool_name, "")
        if not original_desc:
            continue
            
        new_desc = tool.get("function", {}).get("description", "")
        if len(original_desc) > len(new_desc):
            chars_trimmed = len(original_desc) - len(new_desc)
            total_trimmed += chars_trimmed
            # The remaining characters is simply the length of the new description
            remaining = len(new_desc)
            logging.warning(
                f"✂️ Trimmed {chars_trimmed} chars from '{tool_name}' "
                f"({remaining}/{len(original_desc)} chars remaining)"
            )
    
    if total_trimmed > 0:
        logging.warning(f"✂️ Total characters trimmed: {total_trimmed}")
    else:
        logging.warning("✓ No description trimming was needed")
    
    logging.warning(f"Final token count: {token_count} (limit: {MAX_TOKENS_FUNCTION_CALL_COMPETITION})")
    logging.warning("================================\n")
    
    return processed_tools


def optimize_dialogue(npc_info: str, general_knowledge: str, item_knowledge: str, worldview: str, current_state: str, role: str, action_str: str, level: int) -> str:
    """
    Optimizes the dialogue prompt by progressively trimming content based on a reduction level,
    prioritizing cuts based on data-driven importance scores.

    Args:
        npc_info (str): NPC information.
        general_knowledge (str): General knowledge.
        item_knowledge (str): Item knowledge.
        worldview (str): Worldview information.
        current_state (str): Current state.
        role (str): NPC Role information.
        action_str (str): Action string.
        level (int): The level of reduction to apply (1 is least aggressive).

    Returns:
        str: The optimized system prompt string.
    """
    # Initialize modified versions of the inputs
    modified_npc_info = npc_info
    modified_general_knowledge = general_knowledge
    modified_item_knowledge = item_knowledge
    modified_worldview = worldview
    modified_current_state = current_state
    modified_role = role

    # Importance Ranking: state < role < worldview < general_knowledge < item_knowledge < npc_info
    
    # Level 1: Trim Current State (Least Important)
    if level >= 1:
        modified_current_state = "State info has been trimmed to save space."

    # Level 2: Trim Role
    if level >= 2:
        lines = role.split('\n')
        new_lines = [l for l in lines if l.startswith('name:') or l.startswith('occupation:')]
        modified_role = '\n'.join(new_lines) if new_lines else "Role info has been trimmed."

    # Level 3: Trim Worldview
    if level >= 3:
        modified_worldview = "The world is a typical fantasy RPG setting."

    # Level 4: Trim General Knowledge
    if level >= 4:
        sections = general_knowledge.split('### ')
        if len(sections) > 3:
            modified_general_knowledge = '### '.join(sections[:3])
        else:
            modified_general_knowledge = modified_general_knowledge[:int(len(modified_general_knowledge) * 0.5)]

    # Level 5: Trim Item Knowledge
    if level >= 5:
        lines = modified_item_knowledge.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith("description:"):
                desc = line.split(":", 1)[1]
                new_lines.append(f"description: {desc[:int(len(desc) * 0.8)].rstrip()}")
            else:
                new_lines.append(line)
        modified_item_knowledge = '\n'.join(new_lines)

    # Level 6: Trim NPC Info (Lightly)
    if level >= 6:
        lines = npc_info.split('\n')
        new_lines = [l for l in lines if not any(l.startswith(p) for p in ['hobbies:', 'daily routines:', 'background:'])]
        modified_npc_info = '\n'.join(new_lines)

    # Level 7: More Aggressive Trimming
    if level >= 7:
        modified_general_knowledge = "Basic RPG guild and quest system knowledge applies."
        modified_worldview = ""
        lines = modified_item_knowledge.split('\n')
        new_lines = [l for l in lines if l.startswith('name:')]
        modified_item_knowledge = '\n'.join(new_lines)

    # Level 8: Drastic Trimming (Core Persona Only)
    if level >= 8:
        lines = modified_npc_info.split('\n')
        new_lines = [l for l in lines if any(l.startswith(p) for p in ['name:', 'personality traits:', 'appearance:', 'role:'])]
        modified_npc_info = '\n'.join(new_lines)
        modified_item_knowledge = ""
        modified_general_knowledge = ""

    system_prompt_template = ( 
        "Now you play as an NPC in a video game, who will engage with a player in dialogues.\n"
        "Use the following settings and knowledge to create your response.\n" 
        "# Your Settings\n"
        "{npc_info}\n"
        "# Knowledge\n"
        "There are two parts of knowledge. The first part is the general knowledge for this video game and the second part is a list of specific items:\n"
        "## General Knowledge\n"
        "{general_knowledge}\n"
        "## Knowledge of items\n"
        "{item_knowledge}\n"
        "# Worldview: It describes the setting of the game world.\n"
        "{worldview}\n"
        "Current State:\n{current_state}\n"
        "{action_str}\n"
        "{role}\n"
        "Control your answer within 70 words."
    )

    final_system_prompt = system_prompt_template.format(
        npc_info=modified_npc_info,
        general_knowledge=modified_general_knowledge,
        item_knowledge=modified_item_knowledge,
        worldview=modified_worldview,
        current_state=modified_current_state,
        role=modified_role,
        action_str=action_str
    )
    
    return final_system_prompt.strip()


def get_prompt_token_count(messages: List[Dict], tool_list: Optional[List[Dict]] = None) -> int:
    """
    Calculates the approximate number of tokens for a given list of messages and tools.

    This is a simplified implementation. OpenAI's actual token counting can be more complex,
    with extra tokens per message and per name/function call. This function provides a
    good estimate for prompt length management.

    Args:
        messages (List[Dict]): A list of message objects, following the OpenAI API format.
        tool_list (Optional[List[Dict]], optional): An optional list of tool definitions.

    Returns:
        int: The estimated total number of tokens for the prompt.
    """
    total_tokens = 0

    # Calculate tokens for messages
    if messages:
        for message in messages:
            # A common approximation is 4 tokens per message (for role, content markers, etc.)
            total_tokens += 4
            for key, value in message.items():
                if value is None:
                    continue
                # Convert non-string values (like tool_calls list) to a JSON string
                str_value = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
                total_tokens += len(encoding.encode(str_value))
                if key == "name": # Add a small overhead for the 'name' field
                    total_tokens += 1

    # Calculate tokens for tools
    if tool_list:
        # Add a rough approximation for the function definition wrapper
        total_tokens += 16
        for tool in tool_list:
            tool_str = json.dumps(tool, ensure_ascii=False)
            total_tokens += len(encoding.encode(tool_str))

    return total_tokens

# Example Usage:
if __name__ == '__main__':
    example_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather like in Boston?"}
    ]

    example_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Example 1: Messages only
    length1 = get_prompt_token_count(messages=example_messages)
    print(f"Estimated token count for messages only: {length1}")

    # Example 2: Messages and tools
    length2 = get_prompt_token_count(messages=example_messages, tool_list=example_tools)
    print(f"Estimated token count for messages and tools: {length2}")

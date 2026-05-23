import sys
import os
from typing import List, Dict, Any, Tuple
import json
# Attempt to import RAG and tool_builder modules with relative paths
try:
    from ..rag.rag_module import get_formatted_rag_examples
except ImportError:
    # Fallback for different execution contexts if needed
    from agents.rag.rag_module import get_formatted_rag_examples

try:
    from ..tool_builder import build_all_tools
except ImportError:
    # Fallback for different execution contexts if needed
    from agents.tool_builder import build_all_tools

# Import configurations
from . import config
import logging

# --- Centralized Function Usage Strategies ---
FUNCTION_USAGE_STRATEGIES = {
    "check_description_vs_check_basic_info":"""
- Use **`check_description`** when the user is asking about:
  - Purpose, effectiveness, tactics, or contextual usage
  - Subjective assessments like “what makes it special?”, “how does it perform in battle?”, “is it good for rainy weather?”
- Use **`check_basic_info`** when the user is asking about:
  - Basic factual attributes (>=2)
  - Wants to compare multiple items or quests
Tip:
If the user's question involves **subjective use-case** or **comparison of combat utility**, prefer `check_description`.  
If the user asks for **specific fields** or **general info** about items or quests, prefer `check_basic_info`.""",
    "select_vs_select_request_confirm": """- Use `select_request_confirm` if the user shows **interest without clear confirmation** (e.g., “I think I’ll go for it”).
- Use `select` if the user **explicitly confirms** the quest (e.g., “Sign me up for [quest name]”) and already confirmed the quest on the previous turn.
- If only one quest is clearly implied, `select` may still be used.""",
    "select": """### `select`:
- Use `select` if the user **explicitly confirms** the quest (e.g., “Sign me up for [quest name]”).""",
    "check_description": """### `check_description`:
Use when the user asks about:
- Purpose, usage, effectiveness, or context  
(e.g., “what makes it special?”, “good for rainy weather?”, “how do others use it?”)""",
    "check_level": """### `check_level`:
Use when the user asks about quest difficulty or challenge  
(e.g., “is it difficult?”, “will it test my skills?”)  
**Do not** use if asking about monster strength or tactics — use `check_description` instead.""",
    "check_basic_info": """### `check_basic_info`:
Example Toolcalls parameters:
{{
    "item_description": "something light, but effective for defense and offense.|Something versatile|something with more reach|something that can be swung more easily",
    "item_name": "Short Sword|Long Sword|Light-Weight Durable Knife",
    "item_name_operator": "other than",
    "item_price": "1000G",
    "item_price_operator": "or less",
    "item_type": "Sword"
}},
{{
    "item_description": "a weapon for an upcoming battle"
}}
""",
    "search_quest": """### search_quest:
    Example Toolcalls parameters:
{{
    "quest_level": "B|C",
    "quest_level_operator": "other than"
}},
{{
    "quest_duration": "a week",
    "quest_duration_operator": "or less",
    "quest_description": "Something with a satisfying experience.",
    "quest_level": "challenging"
}}
    """
}

# --- Refactored System Prompt Template ---
system_prompt_template = (
    """You are an assistant in estimating function names and arguments given some dialogues in a video game world.
Your goal is to select the appropriate function(s) and fill in parameters based on the user’s input.

{add_info}

## Instructions:
    Fill only the clearly mentioned or strongly implied parameters. When calling functions such as search or check, you must fill in the parameters based on all the items/tasks mentioned in the context.
Use only the information explicitly stated or strongly implied in the original content, and do not add any parameters that are not mentioned.

## Note: You may call 0 - 3 functions.  
- Do **not** add parameters that are not mentioned or implied.
- Do **not** guess or hallucinate any values.
- Do **not** preemptively call functions for items or quests the player has not yet asked to explore.
- **Only call functions for the specific item or quest that the player is currently requesting details for.**
    - If the player says “Start with the first one,” you must only call the function for the first quest. 
    - Do not call functions for the rest until the player explicitly asks.
- If the player expresses a **subjective or situational need** (e.g. “something reliable for night missions”), use `search_item` and fill `item_description` with relevant free-form text.
- You may call multiple functions **only if** the current turn of conversation requires it.

Calling extra functions outside the current request will be considered an error.
{FUNCTION_USAGE_GUIDELINES}
    {RAG_EXAMPLES_SECTION}\n
    {MENTIONED_ITEMS_SECTION}"""
)

def create_messages_for_function_calling(tool_functions: List[Dict], action_functions: List[Dict], dialogue: List[Dict], worldview: str, persona: Dict[str, str], role: str, knowledge: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Creates the tools and messages payload for the function calling API request.
    """
    all_tools = build_all_tools(tool_functions, action_functions)

    # Dynamically build function usage guidelines based on available tools
    guideline_texts = set() # Use a set to avoid duplicate guidelines (like for select/select_request_confirm)
    for tool in all_tools:
        tool_name = tool.get('function', {}).get('name')
        
    tool_name_list = [tool.get('function', {}).get('name') for tool in all_tools]
    if "select" and "select_request_confirm" in tool_name_list:
        tool_name_list.remove("select")
        tool_name_list.remove("select_request_confirm")
        guideline_texts.add(FUNCTION_USAGE_STRATEGIES["select_vs_select_request_confirm"])
    if "check_description" and "check_basic_info" in tool_name_list:
        tool_name_list.remove("check_description")
        tool_name_list.remove("check_basic_info")
        guideline_texts.add(FUNCTION_USAGE_STRATEGIES["check_description_vs_check_basic_info"])
    for tool_name in tool_name_list:
        if tool_name in FUNCTION_USAGE_STRATEGIES:
            guideline_texts.add(FUNCTION_USAGE_STRATEGIES[tool_name])
    # Join the unique guidelines into a single string
    function_guidelines_str = ""
    if guideline_texts:
        function_guidelines_str = "## Function Usage Guidelines:\n" + "\n".join(sorted(list(guideline_texts)))

    # --- Start building the prompt content ---

    # 1. Inject dynamic guidelines into the main template
    prompt_with_guidelines = system_prompt_template.replace(
        "{FUNCTION_USAGE_GUIDELINES}",
        function_guidelines_str
    )

    # 2. Handle RAG examples
    user_query = ""
    if dialogue and isinstance(dialogue, list):
        last_turn = dialogue[-1]
        if last_turn.get("dialogue") and isinstance(last_turn["dialogue"], list):
            for turn_event in reversed(last_turn["dialogue"]):
                if turn_event.get("speaker") == "Player" and turn_event.get("utterance"):
                    user_query = turn_event["utterance"]
                    break
        elif last_turn.get("text"):
            user_query = last_turn['text'].strip()

    similar_examples_formatted_text = ""
    if user_query and config.USE_RAG:
        try:
            similar_examples_formatted_text = get_formatted_rag_examples(
                user_query,
                top_n=config.RAG_TOP_N,
                similarity_threshold=config.RAG_SIMILARITY_THRESHOLD
            )
        except Exception as e:
            print(f"MessageConstructor WARNING: Error getting RAG examples: {e}", file=sys.stderr)
            similar_examples_formatted_text = ""
    
    prompt_with_rag_injected = prompt_with_guidelines.replace(
        "{RAG_EXAMPLES_SECTION}",
        similar_examples_formatted_text
    )

    # 3. Handle mentioned items
    mentioned_items_text = ""
    if dialogue and isinstance(dialogue, list) and dialogue[-1].get("target_item") and isinstance(dialogue[-1]["target_item"], list):
        target_items = dialogue[-1]["target_item"]
        if target_items:
            item_names = [info["name"] for info in target_items if isinstance(info, dict) and "name" in info]
            if item_names:
                mentioned_items_text = "## You should call functions for those mentioned items:\n" + "\n- ".join(item_names) + "\n"

    prompt_with_mentioned_items = prompt_with_rag_injected.replace(
        "{MENTIONED_ITEMS_SECTION}",
        mentioned_items_text
    )

    # 4. Handle additional info (worldview, persona, etc.)
    add_info_text = ""
    # if worldview and persona and role and knowledge:
    #     add_info_text = f"Worldview: {worldview}\nPersona: {persona}\nRole: {role}\nKnowledge: {knowledge}\nIf the player is asking for information that are related to those items, you should call some functions to get the correct information to answer the player.\n"
    final_system_prompt_content = prompt_with_mentioned_items.replace(
        "{add_info}",
        add_info_text
    )

    # --- Construct final messages ---
    history_list = []
    for item in dialogue:
        speaker_role = "user" # Default to user
        if item.get("speaker") == "npc": # Check if 'speaker' key exists
            speaker_role = "assistant"
        # Ensure 'text' key exists, provide default if not
        text_content = item.get("text", "[no utterance]") 
        history_list.append({"role": speaker_role, "content": text_content})

    messages = [
        {"role": "system", "content": final_system_prompt_content.strip()}
    ]
    messages.extend(history_list)

    if config.get_openai_api_key() != "123":
        from .prompt_length_checker import trim_tool_descriptions, get_prompt_token_count
        
        # Get the last user message for better tool relevance assessment
        last_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            ""
        )
        
        # Log initial tool count and names for debugging
        tool_names = [t.get("function", {}).get("name", "unnamed") for t in all_tools]
        logging.info(f"Processing {len(all_tools)} tools: {', '.join(tool_names)}")
        
        try:
            # Initial token count check
            initial_token_count = get_prompt_token_count(messages, all_tools)
            print(f"toolcall prompt token count: {initial_token_count}")
            
            # Only proceed with optimization if we're over the limit
            if initial_token_count > config.MAX_TOKENS_FUNCTION_CALL_COMPETITION:
                # Optimize tools using the enhanced trim_tool_descriptions
                all_tools = trim_tool_descriptions(
                    messages=messages,
                    tools=all_tools,
                    max_attempts=4  # Try removing up to 3 least relevant tools first
                )
                
                # Log optimization results
                final_token_count = get_prompt_token_count(messages, all_tools)
                remaining_tools = len(all_tools)
                
                # Log if we're still over the limit after optimization
                if final_token_count > config.MAX_TOKENS_FUNCTION_CALL_COMPETITION:
                    print(
                        f"Warning: Token count ({final_token_count}) still exceeds limit "
                        f"({config.MAX_TOKENS_FUNCTION_CALL_COMPETITION}) after optimization"
                    )
        except Exception as e:
            pass
            # In case of error, continue with the original tools

    return all_tools, messages


def create_messages_for_dialogue_generation(worldview: str, persona: Dict[str, str], role: str, 
                                          knowledge: Dict[str, Any], state: Dict[str, str], 
                                          dialogue: List[Dict[str, str]], function_results: List[Dict], action=None) -> List[Dict]:
    """
    Creates the messages payload for the dialogue generation API request.
    """
    system_prompt_template = """
Now you play as an NPC in a video game, who will engage with a player in dialogues.
Use the following settings and knowledge to create your response.

# NPC Settings
{npc_info}
{role}

# Knowledge
There are two parts of knowledge. The first part is the general knowledge for this video game and the second part is a list of specific items:

## General Knowledge
{general_knowledge}

## Knowledge of items
{item_knowledge}

# Worldview: It describes the setting of the game world.
{worldview}

# Current State
{current_state}

# Action
{action_str}

## Instructions
- Respond in character as the NPC described above.
- Reference only the provided knowledge and current situation.
- Stay consistent with the NPC’s personality and the game world.
- Focus only on the given context—avoid information not in the prompt.
Return only the NPC's dialogue.

Control your answer within 50 tokens.
"""
    system_prompt_template3=( 
        "Now you play as an NPC in a video game, who will engage with a player in dialogues.\n"
        "Use the following settings and knowledge to create your response.\n" 
        "# NPC Settings (You should act as this character):\n"
        "{npc_info}\n"
        "\n"
        "# Knowledge\n"
        "There are two parts of knowledge. The first part is the general knowledge for this video game, the second part is a list of specific items in the video game:\n"
        "## General Knowledge\n"
        "{general_knowledge}\n"
        "## Knowledge of items\n"
        "{item_knowledge}\n"
        "# Worldview: It describes the setting of the world in the video game.\n"
        "{worldview}\n"
        "Current State:\n{current_state}\n"
        "This is the player's person who you will talk to:\n"
        "{role}\n"
        "{action_str}"
        "Control your answer within 50 tokens."
    )
    system_prompt_template2 = """You are an NPC in a video game. Engage in realistic, in-character dialogue with a player according to the settings below.  
# NPC Character  
{npc_info}  
# Game Information  
## General Knowledge  
{general_knowledge}  
## Quests and Items  
{item_knowledge}  
# World Setting  
{worldview}  
# Current Situation  
{current_state}  
"{action_str}"
## Instructions  
- Respond in character as the NPC described above.  
- Reference only the provided knowledge and current situation.  
- Keep each response under 70 words.  
- If the player selects a quest, always confirm their choice before proceeding.  
- Stay consistent with the NPC’s personality and the game world.  
- Focus only on the given context—avoid information not in the prompt.  
Return only the NPC's dialogue. """
    action_str = ""
    if action or function_results:
        if action:
            action_str += "The action description:\n" + action + "\n"
        if function_results:
            # 处理函数执行结果，替换 ["information": "n/a"] 为 success
            processed_results = []
            for func_result in function_results:
                processed_func = func_result.copy()
                if "return" in processed_func and isinstance(processed_func["return"], list):
                    for i, ret_item in enumerate(processed_func["return"]):
                        if isinstance(ret_item, dict) and len(ret_item) == 1 and "information" in ret_item and ret_item["information"] == "n/a":
                            processed_func["return"][i] = {"status": "success"}
                processed_results.append(processed_func)
            
            action_str += "The action you have done and received:\n"
            for func in processed_results:
                name = func.get("name", "unknown")
                parameters = func.get("parameters", {})
                returns = func.get("return", [])

                action_str += f"name: {name}\n"
                action_str += f"parameters: {json.dumps(parameters, indent=2)}\n"

                # 判断 return 是否为 [{'status': 'success'}]
                if isinstance(returns, list) and len(returns) == 1 and returns[0].get("status") == "success" and len(returns[0]) == 1:
                    action_str += '"status": "success"\n\n'
                else:
                    action_str += f"return: {json.dumps(returns, indent=2)}\n\n"

    # Prepare NPC Info
    npc_info_parts = []
    for key, value in persona.items():
        npc_info_parts.append(f"{key}: {value}")
    npc_info_str = "\n".join(npc_info_parts) if npc_info_parts else "N/A"

    # Prepare General Knowledge
    general_knowledge_str = knowledge.get("general_info", "N/A")

    # Prepare Item Knowledge
    item_knowledge_lines = []
    if isinstance(knowledge.get("knowledge_info"), list):
        for item in knowledge["knowledge_info"]:
            if isinstance(item, dict):
                name = item.get("name", "Unnamed")
                desc = item.get("description", "")
                item_knowledge_lines.append(f"name: {name}")
                item_knowledge_lines.append(f"description: {desc}")
            
    item_knowledge_str = "\n".join(filter(None, item_knowledge_lines)) 
    if not item_knowledge_str.strip():
        item_knowledge_str = "No item knowledge to display."

    # Prepare Worldview
    worldview_str = worldview if worldview else "N/A"

    # Prepare Current State
    current_state_parts = []
    for key, value in state.items():
        current_state_parts.append(f"{key}: {value}")
    current_state_str = "\n".join(current_state_parts) if current_state_parts else "N/A"

    # Prepare Player Info
    role_str = role if role else "N/A" 

    # Format the final system prompt
    # final_system_prompt = system_prompt_template.format(
    #     npc_info=npc_info_str,
    #     general_knowledge=general_knowledge_str,
    #     item_knowledge=item_knowledge_str,
    #     worldview=worldview_str,
    #     current_state=current_state_str,
    #     role=role_str,
    #     action_str=action_str
    # )
    final_system_prompt = system_prompt_template3.format(
        npc_info=npc_info_str,
        general_knowledge=general_knowledge_str,
        item_knowledge=item_knowledge_str,
        worldview=worldview_str,
        current_state=current_state_str,
        role=role_str,
        action_str=action_str
    )
    # final_system_prompt = system_prompt_template2.format(
    # npc_info=npc_info_str,
    # general_knowledge=general_knowledge_str,
    # worldview=worldview_str,
    # current_state=current_state_str,
    # item_knowledge=item_knowledge_str,
    # action_str=action_str
# )
    history_list = []
    for item in dialogue:
        speaker_role = "user" # Default to user
        if item.get("speaker") == "npc": # Check if 'speaker' key exists
            speaker_role = "assistant"
        # Ensure 'text' key exists, provide default if not
        text_content = item.get("text", "[no utterance]") 
        history_list.append({"role": speaker_role, "content": text_content})

    messages = [{"role": "system", "content": final_system_prompt.strip()}]
    messages.extend(history_list)

    # messages = [{"role": "system", "content": final_system_prompt.strip() + "\nHistory Conversations: " + json.dumps(history_list)}]

    if config.get_openai_api_key() != "123":
            from .prompt_length_checker import get_prompt_token_count, optimize_dialogue
            prompt_length = get_prompt_token_count(messages)
            print(f"roleplay prompt token count: {prompt_length}")
            
            # 如果token数量超过config.MAX_TOKENS_FUNCTION_CALL_COMPETITION，循环优化系统提示
            level = 1
            while prompt_length > config.MAX_TOKENS_FUNCTION_CALL_COMPETITION and level <= 8: # Add a max level to prevent infinite loops
                print(f"Token count {prompt_length} exceeds {config.MAX_TOKENS_FUNCTION_CALL_COMPETITION}, optimizing dialogue components at level {level}...")
                optimized_prompt = optimize_dialogue(
                    npc_info=npc_info_str,
                    general_knowledge=general_knowledge_str,
                    item_knowledge=item_knowledge_str,
                    worldview=worldview_str,
                    current_state=current_state_str,
                    role=role_str,
                    action_str=action_str,
                    level=level
                )
                messages[0]['content'] = optimized_prompt
                # 重新计算优化后的token数量
                prompt_length = get_prompt_token_count(messages)
                print(f"Optimized roleplay prompt token count: {prompt_length}")
                level += 1
        
    # print(messages[0])
    return messages
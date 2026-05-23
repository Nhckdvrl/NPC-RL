# /path/to/npc-rl/eval/coser_eval.py
import json

critic_prompts = {
    "self-play-deduct-template": """You are an expert game critic specializing in evaluating AI-driven NPC and player dialogue quality in role-playing games. 
Your task is to holistically evaluate the model's response based on the provided game context and ALL the quality dimensions listed below.

1. Read and understand the provided materials about the game {game_title}:
   * World overview and current game state.
   * Profiles of the Player ({player_name}) and NPC ({npc_name}).
   * The dialogue history leading up to the current turn.
   * The gold standard response for the current turn as a reference.

2. Evaluate the model's generated response for the current turn ({turn_number}) against ALL the following dimensions and their criteria:
{all_dimension_criteria_text}
   {additional_instructions}

## Game Context for Current Turn

### World Overview
{world_overview}

### Current Turn State & Knowledge
{current_turn_context}

### Character Profiles
{character_profiles}

### Dialogue History (leading up to current turn {turn_number})
{dialogue_history_text}

### Player's Query for Current Turn ({turn_number})
{player_query_for_current_turn}

### Gold Standard NPC Response for Current Turn ({turn_number})
Expected NPC Response: {gold_standard_npc_response_text}

### Model's Response to Evaluate (for turn: {turn_number})
{model_response_text}

## Scoring Guidelines

1. Identify all instances of flaws in the model's response for the current turn, considering all provided dimensions.
2. For each flaw identified, specify which dimension it belongs to.
3. For each flaw, determine its level of severity from 1 to 5, where 1 indicates minor, 3 indicates moderate, and 5 indicates severe.

## Output Requirements

Provide your evaluation in JSON format. The JSON should contain:
1. A key "all_flaws" which is a list of all identified flaws. Each flaw object in the list must include:
   - "dimension": "<Name of the dimension the flaw belongs to>"
   - "instance": "<Specific quote or observation about the flaw in the model's response>"
   - "type": "<Type of flaw as defined in the criteria for that dimension>"
   - "severity": <A number from 1 (minor) to 5 (severe)>
2. A key "overall_summary_comment" with a brief overall comment on the model's performance across all dimensions.

Example Output:
```json
{{
    "all_flaws": [
        {{
            "dimension": "Persona Consistency (NPC and Player)",
            "instance": "The NPC suddenly acted very timid, which is unlike their brave warrior persona.",
            "type": "Out of Character (Behavior/Speech)",
            "severity": 4
        }},
        {{
            "dimension": "Dialogue Flow & Coherence",
            "instance": "The NPC's response about the weather was irrelevant to my question about the quest.",
            "type": "Non-Sequitur / Irrelevant Response",
            "severity": 3
        }}
        // ... more flaws if any
    ],
    "overall_summary_comment": "The NPC showed some inconsistencies in persona and dialogue flow, but was otherwise adequate."
}}
```
""",

    "dimension_details": {
        "Scenario Adherence & Quest Progression": {
            "dimension_brief": "How well the NPC's response adheres to the current game scenario, objectives, and helps progress the quest or task.",
            "dimension_criteria": """### Scenario Adherence & Quest Progression
   - Type: Off-Quest or Task Irrelevant
     * Response deviates significantly from the current quest, task, or established scenario.
     * Introduces elements external to the current context without justification.
   - Type: Plot Inconsistency
     * Contradicts established plot points, world lore, or current game state.
     * Fails to acknowledge or correctly use critical information provided in the scenario.
   - Type: Lack of Progression
     * Response does not move the dialogue or task forward meaningfully.
     * Stalls or circles without offering new information or actions relevant to progression.
   - Type: Misinterpretation of Objectives
     * Demonstrates a misunderstanding of the player's goals or the objectives of the current scenario/quest."""
        },
        "NPC/Player Believability & Engagement": {
            "dimension_brief": "How believable, natural, and engaging the NPC's (or player model's) response is, fostering an immersive experience.",
            "dimension_criteria": """### NPC/Player Believability & Engagement
   - Type: Unnatural Dialogue
     * Dialogue feels stilted, robotic, overly formal/informal for the context, or uses awkward phrasing.
     * Response sounds like a generic AI assistant rather than a character in a game world.
   - Type: Poor Engagement
     * Response is bland, uninspired, or fails to elicit interest or emotional response from the player.
     * Fails to ask relevant follow-up questions or react appropriately to player's input to maintain conversation flow.
   - Type: Lack of Initiative
     * Overly passive, only responding to direct questions without contributing to the interaction proactively (when appropriate for the character).
   - Type: Emotional Inconsistency / Flatness
     * Displays emotions that are inappropriate for the situation or character's persona.
     * Lacks appropriate emotional depth or nuance, appearing emotionally flat."""
        },
        "Persona Consistency (NPC and Player)": {
            "dimension_brief": "How consistently the NPC (or player model) maintains its defined personality, role, knowledge, and background.",
            "dimension_criteria": """### Persona Consistency (NPC and Player)
   (Applies to both Player: {player_name} and NPC: {npc_name})
   - Type: Out of Character (Behavior/Speech)
     * Actions, decisions, or speech patterns violate the character's established personality, traits, or typical behavior.
     * Expresses opinions or values that are contradictory to their defined persona.
   - Type: Knowledge Inconsistency
     * Character reveals knowledge they shouldn't possess (metagaming, future knowledge) or lacks knowledge they should have.
     * Forgets significant past interactions or established facts about themselves or the world.
   - Type: Role Inconsistency
     * Acts outside the bounds of their defined role in the game (e.g., a commoner displaying noble etiquette they shouldn't know).
   - Type: Inconsistent Motivations/Goals
     * Actions or statements contradict the character's established motivations or goals."""
        },
        "Dialogue Flow & Coherence": {
            "dimension_brief": "The logical flow, relevance, and coherence of the NPC's response in the context of the ongoing conversation.",
            "dimension_criteria": """### Dialogue Flow & Coherence
   - Type: Non-Sequitur / Irrelevant Response
     * Response is unrelated to the player's query or the preceding dialogue turn.
     * Abruptly changes topic without logical connection.
   - Type: Repetitive
     * Unnecessarily repeats information already stated by themselves or the player.
     * Gets stuck in a loop, offering similar responses multiple times.
   - Type: Contradictory Statements
     * Makes statements that contradict what they or another character said previously in the same conversation.
   - Type: Poor Understanding / Misinterpretation
     * Response indicates a misunderstanding of the player's question or statement.
   - Type: Overly Verbose or Too Brief
     * Response is excessively long and rambling, losing focus.
     * Response is too short or curt, failing to adequately address the player's input when more detail is expected."""
        },
        "Functional Relevance (Tool Use & Actions)": {
            "dimension_brief": "How appropriately and correctly the NPC utilizes any available tools, game functions, or performs actions relevant to the situation.",
            "dimension_criteria": """### Functional Relevance (Tool Use & Actions) - If applicable
   - Type: Incorrect Function/Tool Use
     * Attempts to use a game function, tool, or API call that is not available or suitable for the situation.
     * Uses a function/tool with incorrect parameters or in a way that doesn't make sense.
   - Type: Missed Functional Opportunity
     * Fails to use a relevant and available function, tool, or perform an action that would be logical and beneficial in the current context.
   - Type: Hallucinated Function/Action
     * Mentions or attempts to perform actions or use tools/functions that do not exist in the game's defined capabilities.
   - Type: Action-Dialogue Mismatch
     * The described action (e.g., in <action> tags or narration) does not match the spoken dialogue or intent."""
        }
        # Removed "Overall Game Interaction Quality" as it's now covered by holistic eval and overall_summary_comment
    }
}

def format_dialogue_history(history: list) -> str:
    """Formats dialogue history into a readable string."""
    if not history:
        return "No prior dialogue in this conversation."
    formatted = []
    for i, entry in enumerate(history):
        speaker = entry.get('speaker', 'Unknown')
        text = entry.get('text', '')
        # Ensure speaker and text are strings
        speaker_str = str(speaker) if speaker is not None else "Unknown"
        text_str = str(text) if text is not None else ""
        formatted.append(f"{speaker_str}: {text_str}")
    return '\n'.join(formatted) if formatted else "No previous dialogue history."

def evaluate_game_roleplay_turn(
    turn_number: int,
    worldview_details: str,
    current_turn_state_knowledge: str, # Combined state and knowledge for the turn
    player_profile: dict,
    npc_profile: dict,
    full_dialogue_history: list, # List of dicts: {'speaker': 'Player/NPC name', 'text': 'dialogue text'} - history BEFORE player's current query
    player_query_for_current_turn: str, # Player's utterance for the current turn
    model_response_text: str, # Model's (NPC) response to the player's query
    gold_standard_npc_response_text: str, # Gold standard for NPC's response
    gold_standard_npc_functions: list, # Gold standard for NPC's functions - kept for data consistency, but not used in this prompt
    dimension_to_evaluate: str, # This will be effectively ignored as we evaluate all dimensions
    llm_judge_instance, # Instance of LLMJudge from llm_eval.py
    game_title: str = "This Game",
    additional_instructions_text: str = ""
) -> dict:
    """Evaluates a single turn of a game role-play conversation using an LLM judge across all defined dimensions, applying GCA scoring."""
    
    formatted_history_str = format_dialogue_history(full_dialogue_history)
    
    debug_info = {
        'input_parameters': {
            'turn_number': turn_number,
            'worldview_details': worldview_details,
            'current_turn_state_knowledge': current_turn_state_knowledge,
            'player_profile': player_profile,
            'npc_profile': npc_profile,
            'full_dialogue_history_raw': full_dialogue_history,
            'formatted_dialogue_history': formatted_history_str,
            'player_query': player_query_for_current_turn,
            'model_response': model_response_text,
            'gold_response': gold_standard_npc_response_text,
            'dimension_to_evaluate_param': dimension_to_evaluate, # Log original param
            'game_title': game_title,
            'additional_instructions': additional_instructions_text
        },
        'evaluation_steps': [],
        'formatted_prompt': '',
        'raw_llm_response': '',
        'parsed_llm_output': None
    }

    all_dimension_criteria_text_parts = []
    defined_dimensions_for_scoring = list(critic_prompts["dimension_details"].keys())

    for dim_name, details in critic_prompts["dimension_details"].items():
        all_dimension_criteria_text_parts.append(f"Dimension: {dim_name}\nBrief: {details['dimension_brief']}\nCriteria:\n{details['dimension_criteria']}\n---")
    all_dimension_criteria_text = "\n\n".join(all_dimension_criteria_text_parts)

    prompt_template = critic_prompts["self-play-deduct-template"]
    
    format_args = {
        "game_title": game_title,
        "player_name": player_profile.get('name', 'Player'),
        "npc_name": npc_profile.get('name', 'NPC'),
        "all_dimension_criteria_text": all_dimension_criteria_text,
        "additional_instructions": additional_instructions_text,
        "world_overview": worldview_details if worldview_details else "No specific world overview provided.",
        "current_turn_context": current_turn_state_knowledge if current_turn_state_knowledge else "No specific turn context provided.",
        "character_profiles": f"""Player ({player_profile.get('name', 'Player')}): {player_profile.get('persona', 'No persona details')}. Inventory: {player_profile.get('inventory', 'Unknown')}. Goals: {player_profile.get('goals', 'Unknown')}
NPC ({npc_profile.get('name', 'NPC')}): {npc_profile.get('persona', 'No persona details')}. Role: {npc_profile.get('role', 'Unknown')}. Attitude: {npc_profile.get('attitude', 'Unknown')}""",
        "dialogue_history_text": formatted_history_str,
        "player_query_for_current_turn": player_query_for_current_turn if player_query_for_current_turn else "N/A",
        "model_response_text": model_response_text if model_response_text else "[No response generated]",
        "gold_standard_npc_response_text": gold_standard_npc_response_text if gold_standard_npc_response_text else "N/A",
        "turn_number": turn_number
    }

    try:
        prompt = prompt_template.format(**format_args)
    except KeyError as e:
        error_msg = f"KeyError formatting prompt: {e}. Available keys in format_args: {list(format_args.keys())}"
        debug_info['evaluation_steps'].append(error_msg)
        print(f"ERROR: {error_msg}")
        # Return error structure
        return {
            'overall_score': 0.0,
            'dimension_scores': {dim: 0.0 for dim in defined_dimensions_for_scoring},
            'reasoning': error_msg,
            'raw_llm_output': '',
            'parsed_llm_output': None,
            'all_flaws': [],
            'debug_info': debug_info
        }
        
    debug_info['formatted_prompt'] = prompt
    
    raw_llm_response_str = ""
    try:
        # Ensure llm_judge_instance and its predict method are correctly called
        if hasattr(llm_judge_instance, 'predict') and callable(llm_judge_instance.predict):
            raw_llm_response_full = llm_judge_instance.predict(prompt)
            # Standardize extraction of JSON block
            if "```json" in raw_llm_response_full:
                raw_llm_response_str = raw_llm_response_full.split("```json")[-1].split("```")[0].strip()
            elif raw_llm_response_full.strip().startswith("{") and raw_llm_response_full.strip().endswith("}"):
                 raw_llm_response_str = raw_llm_response_full.strip() # Assume it's a direct JSON object
            else:
                raw_llm_response_str = raw_llm_response_full # Fallback if no markdown
        else:
            raise ValueError("llm_judge_instance does not have a callable 'predict' method.")
    except Exception as e:
        error_msg = f"Error calling LLM judge: {e}"
        debug_info['evaluation_steps'].append(error_msg)
        print(f"ERROR: {error_msg}")
        return {
            'overall_score': 0.0,
            'dimension_scores': {dim: 0.0 for dim in defined_dimensions_for_scoring},
            'reasoning': error_msg,
            'raw_llm_output': str(e), # Store the exception as raw output
            'parsed_llm_output': None,
            'all_flaws': [],
            'debug_info': debug_info
        }

    debug_info['raw_llm_response'] = raw_llm_response_str
    
    overall_gca_score = 0.0
    dimension_scores = {dim: 100.0 for dim in defined_dimensions_for_scoring} # Default to 100
    reasoning = "Evaluation successful. See flaws list for details."
    parsed_llm_output = None
    all_identified_flaws = []
    total_severity_all_flaws = 0
    dimension_specific_severities = {dim: 0 for dim in defined_dimensions_for_scoring}
    actor_rounds_for_turn = 1.0 # For single turn evaluation context

    # print(f"✈✈✈✈ PROMPT:\n{prompt}\n🌼🌼🌼🌼 RAW LLM RESPONSE:\n{raw_llm_response_str}")

    try:
        parsed_llm_output = json.loads(raw_llm_response_str)
        debug_info['parsed_llm_output'] = parsed_llm_output

        # Check if essential keys are present in the parsed JSON
        if "all_flaws" in parsed_llm_output and "overall_summary_comment" in parsed_llm_output:
            flaws_from_llm = parsed_llm_output.get("all_flaws", [])
            all_identified_flaws.extend(flaws_from_llm)

            if not all_identified_flaws:  # No flaws found by LLM
                overall_gca_score = 100.0
                # dimension_scores remain 100.0 as initialized
                reasoning = parsed_llm_output.get("overall_summary_comment", "No flaws identified, excellent response.")
            else:  # Flaws were found
                for flaw_item in all_identified_flaws:
                    severity = flaw_item.get("severity")
                    flaw_dimension = flaw_item.get("dimension")

                    if not isinstance(severity, (int, float)) or not (1 <= severity <= 5):
                        severity = 1  # Default severity for malformed entries
                        debug_info['evaluation_steps'].append(f"Warning: Invalid severity '{flaw_item.get('severity')}' for flaw: {flaw_item.get('instance')}. Defaulting to 1.")
                    
                    total_severity_all_flaws += severity
                    if flaw_dimension in dimension_specific_severities:
                        dimension_specific_severities[flaw_dimension] += severity
                    else:
                        debug_info['evaluation_steps'].append(f"Warning: Flaw reported for unknown dimension '{flaw_dimension}'. Severity added to total but not to specific dimension scores.")
                
                # Calculate overall GCA score based on total severity of all flaws
                overall_gca_score = max(0.0, min(100.0 - (total_severity_all_flaws - 0.3 * actor_rounds_for_turn) * 5.0, 100.0))

                # Calculate GCA score for each dimension
                for dim_name_key in defined_dimensions_for_scoring:
                    dim_total_severity = dimension_specific_severities.get(dim_name_key, 0.0)
                    dimension_scores[dim_name_key] = max(0.0, min(100.0 - (dim_total_severity - 0.3 * actor_rounds_for_turn) * 5.0, 100.0))
                
                reasoning = parsed_llm_output.get("overall_summary_comment", "Evaluation based on identified flaws.")
            
            overall_score = overall_gca_score  # Assign the GCA score to the returnable overall_score

        else:  # Essential keys ('all_flaws' or 'overall_summary_comment') are missing from parsed JSON
            reasoning = f"LLM response JSON was missing 'all_flaws' or 'overall_summary_comment'. Raw: {raw_llm_response_str[:200]}"
            for dim_name_key in defined_dimensions_for_scoring:
                dimension_scores[dim_name_key] = 0.0
            overall_score = 0.0  # Overall score is 0
            all_identified_flaws.clear()  # No valid flaws to report
            debug_info['error'] = "LLM response missing essential keys ('all_flaws' or 'overall_summary_comment')"

    except json.JSONDecodeError as je:
        reasoning = f"Failed to parse LLM JSON response: {str(je)}. Raw: {raw_llm_response_str[:200]}"
        # Initialize scores for all defined dimensions in case of early exit
        for dim_name_key in defined_dimensions_for_scoring:
            dimension_scores[dim_name_key] = 0.0
        if dimension_to_evaluate not in defined_dimensions_for_scoring and dimension_to_evaluate:
             dimension_scores[dimension_to_evaluate] = 0.0 # for the case where a specific (possibly invalid) dimension was requested
        overall_score = 0.0
        all_flaws_from_llm = []
        debug_info['error'] = f"JSON Decode Error: {str(je)}"
        debug_info['parsed_llm_output'] = None # Ensure it's cleared

    except Exception as e:
        reasoning = f"Error during LLM response processing: {str(e)}. Raw: {raw_llm_response_str[:200]}"
        # Initialize scores for all defined dimensions
        for dim_name_key in defined_dimensions_for_scoring:
            dimension_scores[dim_name_key] = 0.0
        if dimension_to_evaluate not in defined_dimensions_for_scoring and dimension_to_evaluate:
             dimension_scores[dimension_to_evaluate] = 0.0
        overall_score = 0.0 # Assign a minimal score for general processing errors
        all_flaws_from_llm = []
        debug_info['error'] = f"Processing Error: {str(e)}"
        debug_info['parsed_llm_output'] = None # Ensure it's cleared
    
    # Add final debug info
    debug_info['final_dimension_scores'] = dimension_scores
    debug_info['final_overall_score'] = overall_score
    
    # Print debug information
    print("\n=== DEBUG: Evaluation Details ===")
    print(f"Turn {turn_number} - {dimension_to_evaluate}")
    print(f"Player Query: {player_query_for_current_turn}")
    print(f"Model Response: {model_response_text}")
    print(f"Gold Response: {gold_standard_npc_response_text}")
    print("\nDimension Scores (0-100 GCA Scale):")
    for dim, score in dimension_scores.items():
        print(f"- {dim}: {score:.2f}")
    print(f"Overall GCA Score (0-100): {overall_score:.2f}")
    print("=" * 40 + "\n")
    
    return {
        "overall_score": overall_score,  # This is the GCA overall score
        "dimension_scores": dimension_scores,  # GCA dimension scores
        "reasoning": reasoning,
        "all_flaws": all_identified_flaws,  # List of flaw objects
        "raw_llm_response_str": raw_llm_response_str,  # Always the raw string from LLM
        "parsed_llm_output": parsed_llm_output,  # Python dict if JSON parsing succeeded, else None
        "debug_info": debug_info
    }

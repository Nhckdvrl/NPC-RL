import json
import os
from typing import List, Dict, Any, Optional

from . import config

def log_interaction(
    exam_id: Optional[str],
    function_gen_messages: List[Dict],
    generated_functions: List[Dict],
    function_execution_results: List[Dict],
    dialogue_gen_messages: List[Dict],
    final_response_content: Optional[str],
    raw_tool_calls_from_dialogue_response: Optional[List[Dict]],
    fn_gen_usage_info: Optional[Dict],
    dialogue_gen_usage_info: Optional[Dict]
) -> None:
    """
    Logs the details of an interaction to a JSON file if DEBUG_MODE is enabled.
    """
    if not config.DEBUG_MODE:
        return

    try:
        with open(config.INTERACTION_LOG_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Prepare the response data structure for logging
    # This mimics the structure from the original agent's logging
    response_data_for_log = {
        "choices": [{
            "message": {
                "content": final_response_content,
                "tool_calls": raw_tool_calls_from_dialogue_response if raw_tool_calls_from_dialogue_response else []
            }
        }]
    }
    
    # Calculate combined totals for usage and cost
    total_usage_info = None
    if fn_gen_usage_info or dialogue_gen_usage_info:
        total_prompt = (fn_gen_usage_info.get("prompt_tokens", 0) if fn_gen_usage_info else 0) + \
                       (dialogue_gen_usage_info.get("prompt_tokens", 0) if dialogue_gen_usage_info else 0)
        total_completion = (fn_gen_usage_info.get("completion_tokens", 0) if fn_gen_usage_info else 0) + \
                           (dialogue_gen_usage_info.get("completion_tokens", 0) if dialogue_gen_usage_info else 0)
        total_tokens_combined = (fn_gen_usage_info.get("total_tokens", 0) if fn_gen_usage_info else 0) + \
                              (dialogue_gen_usage_info.get("total_tokens", 0) if dialogue_gen_usage_info else 0)
        total_cost_estimate = (fn_gen_usage_info.get("cost_usd_estimate", 0.0) if fn_gen_usage_info else 0.0) + \
                              (dialogue_gen_usage_info.get("cost_usd_estimate", 0.0) if dialogue_gen_usage_info else 0.0)
        
        if total_tokens_combined > 0:
            total_usage_info = {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens_combined,
                "cost_usd_estimate": total_cost_estimate
            }

    entry = {
        "exam_id": exam_id if exam_id else "Unknown",
        "function_gen_messages": function_gen_messages,
        "generated_functions": generated_functions,
        "function_execution_results": function_execution_results,
        "dialogue_gen_messages": dialogue_gen_messages,
        "response_for_log": response_data_for_log, # Renamed to avoid confusion with direct API response object
        "usage_info": {
            "function_generation": fn_gen_usage_info,
            "dialogue_generation": dialogue_gen_usage_info,
            "overall_total": total_usage_info
        }
    }
    
    data.append(entry)

    try:
        with open(config.INTERACTION_LOG_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"InteractionLogger ERROR: Failed to write to log file {config.INTERACTION_LOG_FILE}: {e}")


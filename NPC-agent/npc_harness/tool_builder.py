#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tool Builder Module

This module provides functions to build tool definitions from function registries
for use with OpenAI function calling API.
"""

from typing import Dict, List, Any


def build_all_tools(tool_functions: Dict, action_functions: Dict) -> List[Dict]:
    """
    Build tool definitions from both tool functions and action functions.
    
    Parameters
    ----------
    tool_functions : Dict
        Dictionary containing tool function registry with functions already converted to OpenAI format
    action_functions : Dict
        Dictionary containing action function registry with functions already converted to OpenAI format
        
    Returns
    -------
    List[Dict]
        A list of all tool definitions in OpenAI function calling format
    """
    all_tools = []
    
    # Add tool functions
    for name, func_def in tool_functions['function_registry'].items():
        if func_def.get("description"):
            func_def["description"] = func_def.get("description").replace("item_name_operator : str\n    Specified weapon characteristics (e.g. beginner-friendly, light, etc.). Uses the characteristics of the weapon mentioned in the conversation.", "item_name_operator : str\n    Exclusion modifier used with the weapon name specified by item_name.Uses 'other than' as the modifier.")
        tool_def = {
            'type': 'function',
            'function': func_def
        }
        all_tools.append(tool_def)
    
    # Add action functions
    for name, func_def in action_functions['function_registry'].items():
        tool_def = {
            'type': 'function',
            'function': func_def
        }
        all_tools.append(tool_def)

    for tool in all_tools:
        desc = tool['function']['description']
        tool['function']['description'] = desc.split("Returns:")[0].strip()
        if tool['function']['name'] == 'search_item':
            tool['function']['description'] = (
                "Search for weapons based on player-specified criteria such as name, price, type, attack power, or special features "
                "(e.g., lightweight, beginner-friendly). Returns matching weapon names with reasons.\n\n"
                "### Parameters:\n"
                "- item_name: weapon names (e.g., 'Avis Wind', 'Short Sword')\n"
                "- item_price: numeric price (e.g., '500G')\n"
                "- item_type: one of ['axe', 'blunt weapon', 'bow', 'sword', 'double-handed sword', 'single-handed sword', 'spear', 'whip']\n"
                "- item_attack: attack value (e.g., '80')\n"
                "- item_description: free-form traits or features (e.g., 'something light')\n\n"
                "### *_operator fields:\n"
                "Include *_operator **only** if the user's input implies a comparison or exclusion:\n"
                "- Use 'other than' for exclusions (e.g., 'not A or B' ⇒ item_name='A|B', item_name_operator='other than')\n"
                "- For comparisons (e.g., 'more than 1000G'), allowed values are:\n"
                "  'no limit', 'or more', 'or less', 'more than', 'less than', 'highest', 'high', 'average', 'low', 'lowest', 'other than', 'n/a' (for no comparison)"
            )
        if tool['function']['name'] == 'search_quest':
            tool['function']['description'] = (
                "Search for quests based on player-specified criteria such as name, level, duration, reward, or special features "
                "(e.g., for magic users, investigation-type). Returns matching quest names with reasons.\n\n"
                "### Parameters:\n"
                "- quest_name: quest names (e.g., 'Collecting Medical Herbs')\n"
                "- quest_level: difficulty level (e.g., 'A', 'B', 'C')\n"
                "- quest_duration: duration (e.g., '2 hours', '3 days')\n"
                "- quest_reward: numeric reward (e.g., '10G')\n"
                "- quest_description: free-form attributes or special requirements\n\n"
                "### *_operator fields:\n"
                "Include *_operator **only** if the user's input implies a comparison or exclusion:\n"
                "- Use 'other than' for exclusions (e.g., 'not A or B' ⇒ quest_name='A|B', quest_name_operator='other than')\n"
                "- For comparisons (e.g., 'easier than B'), allowed values are:\n"
                "  'or more', 'or less', 'more than', 'less than', 'or above', 'or below',\n"
                "  'highest', 'high', 'average', 'low', 'lowest',\n"
                "  'about', 'long', 'short', 'shortest', 'longest',\n"
                "  'easy', 'difficult', 'most difficult', 'other than', 'n/a' (for no comparison)"
            )

        #     # 清理旧字段 这样也不行
        #     tool['function']['parameters'].pop("required", None)


        # if tool['function']['name'] == 'sell':
        #     tool['function']['parameters']['properties']['item_name']={
        #             "type": "array"
        #           }
        #     tool['function']['description'] = "Sell the specified weapon (e.g. Avis Wind, Short Sword, etc.).\n\nParameters:\n----------\nitem_name: List[str]\n    Specified weapon names (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.\n\nReturns:\n-------\nNone"

    return all_tools

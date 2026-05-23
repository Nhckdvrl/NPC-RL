#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for function_call_langchain.executor

This script allows you to test different tools and functions from the function_call_langchain module
and observe the system's return values.
"""

import sys
import os
import json
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import required modules
from function_calls import tool_map, action_map, Executor


def print_function_registry(registry):
    """Print the available functions in a registry."""
    print("Available functions:")
    for func_name, func_info in registry['function_registry'].items():
        print(f"  - {func_name}: {func_info['description'][:60]}...")
        print(f"    Parameters: {list(func_info['args'].keys())}")
        print()


def test_function_call(function_list_id, function_name, parameters, gold_return=None):
    """Test a function call and print the results."""
    # Get the appropriate registries
    tool_registry = tool_map.get(function_list_id, {})
    action_registry = action_map.get(function_list_id, {})
    
    # Prepare gold function for the executor
    gold_functions = []
    if gold_return is not None:
        gold_functions = [
            {
                "name": function_name,
                "parameters": parameters,
                "return": gold_return
            }
        ]
    
    # Create the executor
    executor = Executor(tool_registry, action_registry, gold_functions)
    
    # Prepare the function call
    function_call = [
        {
            "name": function_name,
            "parameters": parameters
        }
    ]
    
    # Execute the function call
    result = executor.execute(function_call)
    
    # Print the result
    print("\nFunction Call Result:")
    pprint(result)
    print("\nFunction Call Stats:")
    pprint(executor.function_call_stats)
    
    return result


def main():
    """Main function to run the test script."""
    print("\n===== Function Call Langchain Executor Test =====\n")
    
    # Display available function list IDs
    print("Available Function List IDs:")
    for function_list_id in tool_map.keys():
        print(f"  - {function_list_id}")
    print()
    
    while True:
        # Get the function list ID from the user
        function_list_id = input("Enter function list ID (e.g., function_list_id_0001) or 'q' to quit: ")
        if function_list_id.lower() == 'q':
            break
        
        if function_list_id not in tool_map:
            print(f"Error: Function list ID '{function_list_id}' not found.")
            continue
        
        # Display available functions in the selected registry
        print("\n----- Tool Functions -----")
        print_function_registry(tool_map[function_list_id])
        print("\n----- Action Functions -----")
        print_function_registry(action_map[function_list_id])
        
        # Get the function name from the user
        function_name = input("\nEnter function name to test or 'b' to go back: ")
        if function_name.lower() == 'b':
            continue
        
        # Check if the function exists in either registry
        tool_functions = tool_map[function_list_id]['function_registry']
        action_functions = action_map[function_list_id]['function_registry']
        
        if function_name not in tool_functions and function_name not in action_functions:
            print(f"Error: Function '{function_name}' not found in either registry.")
            continue
        
        # Get the function parameters from the user
        print("\nEnter function parameters (JSON format):")
        print("Example: {\"item_name\": \"Avis Wind\"}")
        parameters_json = input("> ")
        
        try:
            parameters = json.loads(parameters_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            continue
        
        # Ask if the user wants to provide a gold return value
        use_gold = input("\nDo you want to provide a gold return value? (y/n): ").lower() == 'y'
        gold_return = None
        
        if use_gold:
            print("\nEnter gold return value (JSON format):")
            print("Example: [{\"information\": \"This is a test\"}]")
            gold_return_json = input("> ")
            
            try:
                gold_return = json.loads(gold_return_json)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format. Using default return value.")
        
        # Test the function call
        test_function_call(function_list_id, function_name, parameters, gold_return)
        
        # Ask if the user wants to continue testing
        continue_testing = input("\nDo you want to test another function? (y/n): ").lower() == 'y'
        if not continue_testing:
            break
    
    print("\nThank you for using the Function Call Langchain Executor Test!")


if __name__ == "__main__":
    main()

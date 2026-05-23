#!/usr/bin/env python3
import json
import os
import sys

def check_json_file(file_path):
    """
    Checks a JSON file for validity and structure.
    Specifically validates the format expected for SFT training data.
    """
    print(f"Checking JSON file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return False
    
    # Try to load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print(f"✓ JSON is valid and could be parsed successfully")
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON format: {e}")
                return False
    except Exception as e:
        print(f"Error: Could not open file: {e}")
        return False
    
    # Check if data is a list
    if not isinstance(data, list):
        print(f"Error: Expected a JSON array at the top level, got {type(data).__name__}")
        return False
    
    print(f"✓ Top-level structure is a list with {len(data)} items")
    
    # Check each item in the list
    valid_items = 0
    invalid_items = 0
    role_pattern_issues = 0
    missing_fields = 0
    
    for i, item in enumerate(data):
        item_valid = True
        
        # Check if item is a dictionary
        if not isinstance(item, dict):
            print(f"  - Item {i}: Error: Expected a dictionary, got {type(item).__name__}")
            invalid_items += 1
            continue
        
        # Check for required fields
        if 'conversations' not in item:
            print(f"  - Item {i}: Error: Missing 'conversations' field")
            missing_fields += 1
            item_valid = False
        
        if 'data_source' not in item:
            print(f"  - Item {i}: Error: Missing 'data_source' field")
            missing_fields += 1
            item_valid = False
        
        # Check conversations structure if it exists
        if 'conversations' in item and isinstance(item['conversations'], list):
            conversations = item['conversations']
            
            # Check if conversations is empty
            if not conversations:
                print(f"  - Item {i}: Error: 'conversations' list is empty")
                invalid_items += 1
                continue
            
            # Check role pattern (system, human, gpt, human, gpt, ...)
            role_pattern_valid = True
            
            # First message should be system
            if not conversations or conversations[0].get('from') != 'system':
                print(f"  - Item {i}: Error: First message is not from 'system'")
                role_pattern_valid = False
            
            # Check alternating pattern after system
            if len(conversations) < 3:
                print(f"  - Item {i}: Error: Conversation too short (< 3 messages)")
                role_pattern_valid = False
            
            # Check if length is odd (should end with gpt)
            if len(conversations) % 2 == 0:
                print(f"  - Item {i}: Error: Even number of messages ({len(conversations)}), should end with 'gpt'")
                role_pattern_valid = False
            
            # Check alternating pattern
            for j in range(1, len(conversations)):
                expected_role = 'human' if j % 2 == 1 else 'gpt'
                actual_role = conversations[j].get('from')
                
                if actual_role != expected_role:
                    print(f"  - Item {i}: Error: At position {j}, expected '{expected_role}', got '{actual_role}'")
                    role_pattern_valid = False
                    break
            
            # Check that each message has 'from' and 'value' fields
            for j, msg in enumerate(conversations):
                if not isinstance(msg, dict):
                    print(f"  - Item {i}: Error: Message {j} is not a dictionary")
                    item_valid = False
                    continue
                    
                if 'from' not in msg:
                    print(f"  - Item {i}: Error: Message {j} missing 'from' field")
                    item_valid = False
                
                if 'value' not in msg:
                    print(f"  - Item {i}: Error: Message {j} missing 'value' field")
                    item_valid = False
                
                # Check that values are not empty
                if 'value' in msg and not msg['value'].strip():
                    print(f"  - Item {i}: Warning: Message {j} has empty 'value'")
            
            if not role_pattern_valid:
                role_pattern_issues += 1
                item_valid = False
        
        if item_valid:
            valid_items += 1
        else:
            invalid_items += 1
    
    # Print summary
    print("\nSummary:")
    print(f"Total items: {len(data)}")
    print(f"Valid items: {valid_items}")
    print(f"Invalid items: {invalid_items}")
    print(f"Items with role pattern issues: {role_pattern_issues}")
    print(f"Items with missing fields: {missing_fields}")
    
    # Calculate percentages
    if len(data) > 0:
        valid_percent = (valid_items / len(data)) * 100
        print(f"Valid percentage: {valid_percent:.2f}%")
    
    return valid_items > 0

if __name__ == "__main__":
    # Default file path
    file_path = "/path/to/npc-rl/data/sft/task1/stage1_sync_by_lian-3m.json"
    
    # Allow custom file path from command line
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    check_json_file(file_path)

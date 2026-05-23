#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict, Counter
import shutil

def fix_conversation_structures(file_path):
    """
    Fixes conversation structures in a JSON file to ensure they follow the system-human-gpt pattern.
    1. Renames 'assistant' to 'gpt'
    2. Ensures conversations end with 'gpt' message
    3. Ensures proper alternating pattern
    """
    print(f"Processing file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return False
    
    # Load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False
    
    if not isinstance(data, list):
        print(f"Error: Expected a list at the top level, got {type(data).__name__}")
        return False
    
    print(f"Total items in the dataset before processing: {len(data)}")
    
    # Initialize counters
    stats = {
        "total_before": len(data),
        "renamed_assistant_to_gpt": 0,
        "removed_trailing_human": 0,
        "fixed_role_patterns": 0,
        "skipped_invalid": 0,
        "total_after": 0
    }
    
    fixed_data = []
    
    # Process each item
    for i, item in enumerate(data):
        if 'conversations' not in item or not isinstance(item['conversations'], list):
            print(f"Skipping item {i}: Missing or invalid 'conversations' field")
            stats["skipped_invalid"] += 1
            continue
        
        conversations = item['conversations']
        if not conversations:
            print(f"Skipping item {i}: Empty conversations list")
            stats["skipped_invalid"] += 1
            continue
        
        # Fix the conversation structure
        fixed_conversations = []
        
        # Step 1: Rename 'assistant' to 'gpt'
        for msg in conversations:
            if msg.get('from') == 'assistant':
                msg['from'] = 'gpt'
                stats["renamed_assistant_to_gpt"] += 1
            fixed_conversations.append(msg)
        
        # Step 2: Ensure the conversation starts with 'system'
        if not fixed_conversations or fixed_conversations[0].get('from') != 'system':
            print(f"Skipping item {i}: First message is not 'system'")
            stats["skipped_invalid"] += 1
            continue
        
        # Step 3: Ensure proper alternating pattern after system
        valid_conversations = [fixed_conversations[0]]  # Start with system message
        
        current_role = 'human'  # After system, we expect human
        for j in range(1, len(fixed_conversations)):
            msg = fixed_conversations[j]
            if msg.get('from') == current_role:
                valid_conversations.append(msg)
                # Toggle expected role
                current_role = 'gpt' if current_role == 'human' else 'human'
                stats["fixed_role_patterns"] += 1
        
        # Step 4: Ensure conversation ends with 'gpt'
        if valid_conversations and valid_conversations[-1].get('from') == 'human':
            valid_conversations.pop()  # Remove trailing human message
            stats["removed_trailing_human"] += 1
        
        # Step 5: Ensure we have at least system-human-gpt
        if len(valid_conversations) < 3:
            print(f"Skipping item {i}: Too short after fixing (< 3 messages)")
            stats["skipped_invalid"] += 1
            continue
        
        # Add the fixed item to our new dataset
        item['conversations'] = valid_conversations
        fixed_data.append(item)
    
    stats["total_after"] = len(fixed_data)
    
    # Create a backup of the original file
    backup_path = file_path + '.backup'
    print(f"Creating backup of original file at: {backup_path}")
    shutil.copy2(file_path, backup_path)
    
    # Write the fixed data back to the original file
    print(f"Writing fixed data back to: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n===== Processing Statistics =====")
    print(f"Total items before processing: {stats['total_before']}")
    print(f"Total items after processing: {stats['total_after']}")
    print(f"Items skipped due to invalid structure: {stats['skipped_invalid']}")
    print(f"'assistant' renamed to 'gpt': {stats['renamed_assistant_to_gpt']}")
    print(f"Trailing 'human' messages removed: {stats['removed_trailing_human']}")
    print(f"Role patterns fixed: {stats['fixed_role_patterns']}")
    
    return True

def process_all_coser_files():
    """
    Process all coser dataset files
    """
    base_path = "data/raw/coser"
    files = [
        "data-00000-of-00005.json",
        "data-00001-of-00005.json",
        "data-00002-of-00005.json",
        "data-00003-of-00005.json",
        "data-00004-of-00005.json"
    ]
    
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        print(f"\n{'='*50}")
        print(f"Processing file: {file_name}")
        print(f"{'='*50}")
        success = fix_conversation_structures(file_path)
        if success:
            print(f"Successfully processed: {file_name}")
        else:
            print(f"Failed to process: {file_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process a specific file if provided
        file_path = sys.argv[1]
        fix_conversation_structures(file_path)
    else:
        # Process all coser files
        process_all_coser_files()

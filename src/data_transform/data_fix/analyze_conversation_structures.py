#!/usr/bin/env python3
import json
import os
from collections import defaultdict, Counter

def analyze_conversation_structures(file_path):
    """
    Analyzes the conversation structures in a JSON file to identify patterns and anomalies.
    """
    print(f"Analyzing conversation structures in: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return
    
    # Load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    if not isinstance(data, list):
        print(f"Error: Expected a list at the top level, got {type(data).__name__}")
        return
    
    print(f"Total items in the dataset: {len(data)}")
    
    # Initialize counters and trackers
    conversation_lengths = Counter()
    role_patterns = Counter()
    role_sequences = Counter()
    first_role_counts = Counter()
    last_role_counts = Counter()
    empty_value_counts = defaultdict(int)
    anomalies = []
    
    # Analyze each item
    for i, item in enumerate(data):
        if 'conversations' not in item:
            anomalies.append(f"Item {i}: Missing 'conversations' field")
            continue
        
        conversations = item['conversations']
        if not isinstance(conversations, list):
            anomalies.append(f"Item {i}: 'conversations' is not a list")
            continue
        
        # Count conversation length
        conv_length = len(conversations)
        conversation_lengths[conv_length] += 1
        
        # Check for empty conversations
        if conv_length == 0:
            anomalies.append(f"Item {i}: Empty conversations list")
            continue
        
        # Analyze role pattern
        roles = [msg.get('from', 'unknown') for msg in conversations]
        role_pattern = '-'.join(roles)
        role_patterns[role_pattern] += 1
        
        # Track first and last roles
        if roles:
            first_role_counts[roles[0]] += 1
            last_role_counts[roles[-1]] += 1
        
        # Check for alternating pattern
        expected_pattern = True
        if len(roles) >= 1:
            if roles[0] not in ['system', 'human']:
                expected_pattern = False
            
            # Check alternating pattern after first message
            for j in range(1, len(roles)):
                if j % 2 == 1 and roles[j] != 'human':  # Odd positions should be 'human'
                    if j == 1 and roles[0] == 'human':  # Unless first was already 'human'
                        if roles[j] != 'gpt':
                            expected_pattern = False
                    else:
                        expected_pattern = False
                elif j % 2 == 0 and roles[j] != 'gpt':  # Even positions should be 'gpt'
                    expected_pattern = False
        
        if not expected_pattern:
            anomalies.append(f"Item {i}: Unexpected role pattern: {role_pattern}")
        
        # Check for empty values
        for j, msg in enumerate(conversations):
            if 'value' not in msg:
                anomalies.append(f"Item {i}, Message {j}: Missing 'value' field")
            elif not msg['value'].strip():
                role = msg.get('from', 'unknown')
                empty_value_counts[role] += 1
        
        # Track role sequences (first 3 roles, first 5 roles, etc.)
        if len(roles) >= 3:
            role_sequences['-'.join(roles[:3])] += 1
        if len(roles) >= 5:
            role_sequences['-'.join(roles[:5])] += 1
    
    # Print analysis results
    print("\n===== Conversation Structure Analysis =====")
    
    print("\n--- Conversation Lengths ---")
    for length, count in sorted(conversation_lengths.items()):
        print(f"Length {length}: {count} conversations ({count/len(data)*100:.2f}%)")
    
    print("\n--- First Message Role ---")
    for role, count in first_role_counts.most_common():
        print(f"Role '{role}': {count} conversations ({count/len(data)*100:.2f}%)")
    
    print("\n--- Last Message Role ---")
    for role, count in last_role_counts.most_common():
        print(f"Role '{role}': {count} conversations ({count/len(data)*100:.2f}%)")
    
    print("\n--- Most Common Role Patterns ---")
    for pattern, count in role_patterns.most_common(10):
        print(f"Pattern '{pattern}': {count} conversations ({count/len(data)*100:.2f}%)")
    
    print("\n--- Most Common Role Sequences (first few roles) ---")
    for sequence, count in role_sequences.most_common(10):
        print(f"Sequence '{sequence}': {count} conversations ({count/len(data)*100:.2f}%)")
    
    print("\n--- Empty Values by Role ---")
    for role, count in empty_value_counts.items():
        print(f"Role '{role}' has {count} empty values")
    
    print("\n--- Anomalies ---")
    if anomalies:
        print(f"Found {len(anomalies)} anomalies:")
        for i, anomaly in enumerate(anomalies[:20]):  # Show first 20 anomalies
            print(f"  {i+1}. {anomaly}")
        if len(anomalies) > 20:
            print(f"  ... and {len(anomalies) - 20} more anomalies")
    else:
        print("No anomalies found.")
    
    # Detailed analysis of specific patterns
    print("\n--- Detailed Pattern Analysis ---")
    
    # Check for system-human-gpt pattern
    system_human_gpt_pattern = sum(count for pattern, count in role_patterns.items() 
                                if pattern.startswith("system-human-gpt"))
    print(f"system-human-gpt pattern (standard): {system_human_gpt_pattern} conversations ({system_human_gpt_pattern/len(data)*100:.2f}%)")
    
    # Check for human-gpt pattern
    human_gpt_pattern = sum(count for pattern, count in role_patterns.items() 
                          if pattern.startswith("human-gpt"))
    print(f"human-gpt pattern (no system): {human_gpt_pattern} conversations ({human_gpt_pattern/len(data)*100:.2f}%)")
    
    # Check for system-human pattern (incomplete)
    system_human_pattern = sum(count for pattern, count in role_patterns.items() 
                             if pattern == "system-human")
    print(f"system-human pattern (incomplete): {system_human_pattern} conversations ({system_human_pattern/len(data)*100:.2f}%)")
    
    # Check for other patterns
    other_patterns = len(data) - system_human_gpt_pattern - human_gpt_pattern - system_human_pattern
    print(f"Other patterns: {other_patterns} conversations ({other_patterns/len(data)*100:.2f}%)")
    
    # Sample a few examples of each major pattern
    print("\n--- Sample Conversations by Pattern ---")
    
    pattern_samples = {
        "system-human-gpt": [],
        "human-gpt": [],
        "other": []
    }
    
    for i, item in enumerate(data):
        if len(pattern_samples["system-human-gpt"]) >= 3 and len(pattern_samples["human-gpt"]) >= 3 and len(pattern_samples["other"]) >= 3:
            break
            
        if 'conversations' in item and isinstance(item['conversations'], list):
            conversations = item['conversations']
            if len(conversations) >= 3:
                roles = [msg.get('from', 'unknown') for msg in conversations]
                role_pattern = '-'.join(roles)
                
                if role_pattern.startswith("system-human-gpt") and len(pattern_samples["system-human-gpt"]) < 3:
                    pattern_samples["system-human-gpt"].append((i, role_pattern))
                elif role_pattern.startswith("human-gpt") and len(pattern_samples["human-gpt"]) < 3:
                    pattern_samples["human-gpt"].append((i, role_pattern))
                elif len(pattern_samples["other"]) < 3:
                    pattern_samples["other"].append((i, role_pattern))
    
    for pattern_type, samples in pattern_samples.items():
        print(f"\n{pattern_type} pattern samples:")
        for i, (item_idx, pattern) in enumerate(samples):
            print(f"  Sample {i+1}: Item {item_idx}, Pattern: {pattern}")

if __name__ == "__main__":
    file_path = "data/raw/coser/data-00000-of-00005.json"
    # file_path = "/path/to/npc-rl/data/sft/task1/stage1_sync_by_lian-3m.json"
    analyze_conversation_structures(file_path)

import json
import os

# Global counters for validation failures
empty_conversation_count = 0
too_short_conversation_count = 0
even_length_conversation_count = 0
role_mismatch_count = 0
valid_conversation_count = 0
user_assistant_pattern_count = 0

def validate_conversation_roles(conversation_list, context_info, filename_for_log):
    """
    Validates if a conversation list follows the 'system', then 'human', 'gpt' alternating pattern.
    The conversation must start with 'system' and end with 'gpt'.
    Args:
        conversation_list: The list of messages, each a dict with "from" and "value".
        context_info: A string describing the source (e.g., "line 123", "item 5 from full JSON").
        filename_for_log: The name of the file being processed (for logging purposes).
    Returns:
        True if valid, False otherwise.
    """
    global empty_conversation_count, too_short_conversation_count, even_length_conversation_count
    global role_mismatch_count, valid_conversation_count, user_assistant_pattern_count
    
    if not conversation_list:
        # print(f"Warning: Empty conversation provided for validation. Context: {context_info} in {filename_for_log}.")
        empty_conversation_count += 1
        return False

    # A valid sequence after system is (human, gpt, human, gpt, ... gpt)
    # This means the total length must be odd and at least 3 (system, human, gpt).
    
    if len(conversation_list) < 2:
        too_short_conversation_count += 1
        return False
        
    if conversation_list[0].get("from") == "user" and conversation_list[1].get("from") == "assistant":
        user_assistant_pattern_count += 1
        return True
        
    if len(conversation_list) % 2 == 0:
        # This implies it ends with 'human' or is structured like (system, human) after system.
        # The user's original code already attempts to remove a final 'human' message.
        # If it's still even, it's an invalid structure for 'system, ..., gpt'.
        # print(f"Error: Conversation (Context: {context_info} in {filename_for_log}) has an even number of messages "
        #       f"({len(conversation_list)}), which is not expected for a 'system, ..., gpt' sequence. Full: {conversation_list}")
        even_length_conversation_count += 1
        return False

    # 2. Check alternation for messages after 'system'
    # Expected: human (idx 1), gpt (idx 2), human (idx 3), ...
    for i in range(1, len(conversation_list)):
        current_role = conversation_list[i].get("from")
        expected_role = "human" if i % 2 == 1 else "gpt"
        if current_role != expected_role:
            # print(f"Error: Role mismatch in conversation (Context: {context_info} in {filename_for_log}). "
            #       f"At index {i}, expected '{expected_role}', found '{current_role}'. Full: {conversation_list}")
            role_mismatch_count += 1
            return False
            
    # If all checks pass, the conversation is valid.
    # The last message will be 'gpt' due to odd length and correct alternation.
    valid_conversation_count += 1
    return True


def merge_and_tag_jsonl_files():
    """
    Reads multiple JSONL files, adds a 'data_source' field based on the filename,
    and writes the combined result to a single JSON file.
    """
    # Reset global counters before processing
    global empty_conversation_count, too_short_conversation_count, even_length_conversation_count
    global role_mismatch_count, valid_conversation_count, user_assistant_pattern_count
    
    empty_conversation_count = 0
    too_short_conversation_count = 0
    even_length_conversation_count = 0
    role_mismatch_count = 0
    valid_conversation_count = 0
    user_assistant_pattern_count = 0
    
    # Track additional statistics
    total_conversations_processed = 0
    total_conversations_skipped = 0
    skipped_by_reason = {
        "empty_after_processing": 0,
        "too_short_after_trimming": 0,
        "validation_failed": 0,
        "json_decode_error": 0
    }
    
    # base_data_path = "/path/to/npc-rl/data/chat_format/gpt-4o_2024-05-13"
    base_data_path = "/path/to/npc-rl/data/3m"
    output_file_path = "/path/to/npc-rl/data/sft/task1/stage1_sync_by_lian-3m.json"

    # input_files = [
    #     "direct-gen-full-withIntent-withGameInfo.json",
    #     "direct-gen-full-withIntent.json",
    #     "direct-gen-full.json",
    #     "GivenContextGenDialogue-AllFromTrain-N.json",
    #     "GivenContextGenDialogue-AllFromTrain-withIntent.json",
    #     "GivenContextGenDialogue-AllFromTrain.json"
    # ]
    input_files = [
        "direct-gen-full-flipRoles.json",
        "direct-gen-full.json",
        "situational_conversation_1turn_persona.json",
        "situational_conversation_1turn.json"
    ]
    combined_data = []

    for filename in input_files:
        file_path = os.path.join(base_data_path, filename)
        data_source_name = os.path.basename(filename)
        
        print(f"Processing {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # First, try to load the entire file as a single JSON entity
                try:
                    full_data = json.load(f)
                    if isinstance(full_data, list):
                        # The file is a JSON array
                        for item in full_data:
                            if isinstance(item, dict):
                                item['data_source'] = data_source_name
                            else:
                                print(f"Warning: Item in list is not a dictionary in {filename}")
                        combined_data.extend(full_data)
                    elif isinstance(full_data, dict):
                        # The file is a single JSON object
                        full_data['data_source'] = data_source_name
                        combined_data.append(full_data)
                except json.JSONDecodeError:
                    # If loading the whole file fails, it might be JSONL. Reset seek and read line-by-line.
                    f.seek(0)
                    print(f"Could not parse as single JSON, trying as JSONL for {filename}...")
                    line_num = 0 # Initialize line counter
                    for line in f:
                        line_num += 1 # Increment line counter
                        total_conversations_processed += 1
                        
                        if line.strip():
                            try:
                                # As per user feedback, each line is the content for the 'conversations' key.
                                conversations_content = json.loads(line)
                                
                                # 将每条 message 转换为符合 LLaMA Factory 格式
                                transformed_conversation = []
                                for message in conversations_content:
                                    role = message.get("role")
                                    content = message.get("content")
                                    if role == "system":
                                        from_role = "system"
                                    elif role == "user":
                                        from_role = "human"
                                    elif role == "assistant":
                                        from_role = "gpt"
                                    else:
                                        print(f"Warning: Unknown role '{role}' in {filename} line {line_num}, skipping this message.")
                                        continue

                                    if content.strip() == "":
                                        continue
                                    transformed_conversation.append({
                                        "from": from_role,
                                        "value": content
                                    })
                                if transformed_conversation and transformed_conversation[-1].get("from") == "human":
                                    transformed_conversation = transformed_conversation[:-1]
                                if len(transformed_conversation) < 2:
                                    print(f"Warning: Conversation too short after trimming, skipping.")
                                    skipped_by_reason["too_short_after_trimming"] += 1
                                    total_conversations_skipped += 1
                                    continue

                                # Validate the transformed conversation
                                if not transformed_conversation:
                                    print(f"Warning: Conversation in {filename} at line {line_num} is empty after processing. Skipping.")
                                    skipped_by_reason["empty_after_processing"] += 1
                                    total_conversations_skipped += 1
                                elif validate_conversation_roles(transformed_conversation, f"line {line_num}", filename):
                                    # 构造最终结构
                                    new_record = {
                                        "conversations": transformed_conversation,
                                        "data_source": data_source_name
                                    }   
                                    combined_data.append(new_record)
                                else:
                                    # Validation failed
                                    skipped_by_reason["validation_failed"] += 1
                                    total_conversations_skipped += 1
                                    # print(f"Info: Skipping record from {filename} at line {line_num} due to role validation failure.")
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON from line in {filename} at line {line_num}: {line.strip()}")
                                skipped_by_reason["json_decode_error"] += 1
                                total_conversations_skipped += 1
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {file_path}")

    print(f"\nWriting combined data to {output_file_path}...")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print("\n===== Processing Statistics =====")
    print(f"Total conversations processed: {total_conversations_processed}")
    print(f"Total conversations included in output: {len(combined_data)}")
    print(f"Total conversations skipped: {total_conversations_skipped}")
    
    print("\n--- Skipped by reason ---")
    for reason, count in skipped_by_reason.items():
        print(f"{reason}: {count}")
    
    print("\n--- Validation Statistics ---")
    print(f"Empty conversations: {empty_conversation_count}")
    print(f"Too short conversations: {too_short_conversation_count}")
    print(f"Even length conversations: {even_length_conversation_count}")
    print(f"Role mismatch conversations: {role_mismatch_count}")
    print(f"Valid conversations (system-human-gpt pattern): {valid_conversation_count}")
    print(f"User-assistant pattern conversations: {user_assistant_pattern_count}")
    
    print("\nProcessing complete.")
    print(f"Total items in output file: {len(combined_data)}")


if __name__ == "__main__":
    merge_and_tag_jsonl_files()

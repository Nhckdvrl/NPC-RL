import json
import argparse
import os

def convert_log_entry_to_stage0(log_entry):
    conversations = []
    # Process messages from stage_0
    if 'stage_0' in log_entry and 'messages' in log_entry['stage_0']:
        for msg in log_entry['stage_0']['messages']:
            if msg['role'] == 'system':
                conversations.append({'from': 'system', 'value': msg['content'].strip()})
            elif msg['role'] == 'user':
                conversations.append({'from': 'human', 'value': msg['content']})
            elif msg['role'] == 'assistant': # Should ideally not happen before gold_toolcall
                conversations.append({'from': 'gpt', 'value': msg['content']})
    
    # Add gold_toolcall as assistant's response
    if 'gold_toolcall' in log_entry and log_entry['gold_toolcall']:
        toolcalls = []
        for toolcall in log_entry['gold_toolcall']:
            # Create a new tool call object with only name and arguments
            toolcall_obj = {
                "name": toolcall.get("name", ""),
                "arguments": toolcall.get("parameters", {})
            }
            toolcalls.append(toolcall_obj)
        
        function_value = json.dumps(toolcalls[0] if len(toolcalls) == 1 else toolcalls, ensure_ascii=False)
        conversations.append({'from': 'function_call', 'value': function_value})
    else:
        # If no gold_toolcall, use an empty string
        conversations.append({'from': 'gpt', 'value': 'No need toolcall'}) 

    tools_definition = []
    if 'stage_0' in log_entry and 'tools' in log_entry['stage_0']:
        for tool in log_entry['stage_0']['tools']:
            tools_definition.append(tool)

    return {
        'conversations': conversations,
        'tools': json.dumps(tools_definition)
    }

def convert_log_entry_to_stage1(log_entry):
    conversations = []
    # Process messages from stage_1
    if 'stage_1' in log_entry and 'messages' in log_entry['stage_1']:
        for msg in log_entry['stage_1']['messages']:
            if msg['role'] == 'system':
                conversations.append({'from': 'system', 'value': msg['content'].strip()})
            elif msg['role'] == 'user':
                conversations.append({'from': 'human', 'value': msg['content']})
            elif msg['role'] == 'assistant':
                if msg.get('tool_calls'):
                    conversations.append({'from': 'function_call', 'value': json.dumps(msg['tool_calls'])})
                elif msg.get('content'): # Regular assistant message
                    conversations.append({'from': 'gpt', 'value': msg['content']})
                # If assistant message is empty or only has tool_calls, it's handled or implicitly skipped if no content
            # elif msg['role'] == 'tool':
            #     # Ensure content is a string, as it's often pre-stringified JSON
            #     content_val = msg['content']
            #     if not isinstance(content_val, str):
            #         content_val = json.dumps(content_val)
            #     conversations.append({'from': 'observation', 'value': content_val})

    # Add gold_response as assistant's final response
    if 'gold_response' in log_entry and log_entry['gold_response'] is not None:
        conversations.append({'from': 'gpt', 'value': log_entry['gold_response']})

    return {
        'conversations': conversations
    }

def main():
    parser = argparse.ArgumentParser(description='Convert LLM interaction logs to training data format.')
    parser.add_argument('--input_file', '-i', type=str, 
                      help='Path to the input log file')
    args = parser.parse_args()
# /path/to/npc-rl/logs/task1_logs_1751014533.6055717.json
# /path/to/npc-rl/logs/task1_logs_1751013479.9913943.json
    input_filepath = args.input_file
    # output_directory = os.path.dirname(input_filepath)
    output_directory = "/path/to/npc-rl/data/sft/task1"
    output_stage0_filepath = os.path.join(output_directory, 'stage_0_test.json')
    output_stage1_filepath = os.path.join(output_directory, 'stage_1_test.json')

    try:
        with open(input_filepath, 'r') as f:
            logs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filepath}")
        return

    stage0_data = []
    stage1_data = []

    # Logs are a dictionary with turn numbers as keys
    for turn_key in sorted(logs.keys(), key=int): # Sort by turn number
        log_entry = logs[turn_key]
        stage0_data.append(convert_log_entry_to_stage0(log_entry))
        stage1_data.append(convert_log_entry_to_stage1(log_entry))

    with open(output_stage0_filepath, 'w') as f:
        json.dump(stage0_data, f, indent=2)
    print(f"Successfully converted and saved stage_0 data to {output_stage0_filepath}")

    with open(output_stage1_filepath, 'w') as f:
        json.dump(stage1_data, f, indent=2)
    print(f"Successfully converted and saved stage_1 data to {output_stage1_filepath}")

if __name__ == '__main__':
    main()

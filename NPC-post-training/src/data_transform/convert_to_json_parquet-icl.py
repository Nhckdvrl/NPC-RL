#!/usr/bin/env python3
"""
Script to convert NPC-RL dataset files into structured JSON and Parquet formats.
This script processes each file individually, adds metadata, calculates token lengths,
and saves in both formats. Stage0 files are treated as task1/toolcall and stage1 files
are treated as task2/roleplay.
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


# Simple tokenizer for estimating token count
def estimate_token_count(text: str) -> int:
    """
    Estimate token count using a simple word-based approach.
    This is a basic approximation and not as accurate as a proper tokenizer.
    """
    # Split by whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return len(tokens)

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSON file and return its contents.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                return [data]
            return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

def create_entry(item: Dict[str, Any], data_source: str, task_id: str, 
                source_file: str, source_index: int) -> Dict[str, Any]:
    """
    Create a structured entry from a conversation item.
    
    Args:
        item: The conversation item from the source data
        data_source: The data source identifier ('npc/toolcall' or 'npc/roleplay')
        task_id: The task identifier ('task1' or 'task2')
        source_file: The source file name
        source_index: The index of the item in the source file
        
    Returns:
        A dictionary with the structured entry
    """
    # Extract conversations
    conversations = item.get('conversations', [])
    # 获取最后一个turn作为ground truth（assistant的回复）
    ground_truth = ""
    if conversations and len(conversations) > 0:
        last_turn = conversations[-1]
        if last_turn.get('from') in ['gpt', 'assistant', 'function_call']:
            ground_truth = last_turn.get('value', '')
    # 处理对话，将角色映射为正确的格式
    prompt_messages = []
    for i, conv in enumerate(conversations[:-1]):  # 排除最后一个turn
        role = conv.get('from', '')
        content = conv.get('value', '')
        
        # 将角色名称映射为标准格式
        if role == 'human':
            role = 'user'
        elif role in ['gpt', 'assistant']:
            role = 'assistant'
        elif role == 'system':
            role = 'system'
            content = content.replace("Control your answer within 200 tokens.", "Control your answer within 50 tokens.")
            content += "\nThe ground truth is: " + ground_truth
        # 添加到消息列表
        prompt_messages.append({"role": role, "content": content})
    prompt_str = "<QUESTION>"
    if task_id == "task1":
        prompt_str += "\n".join([f"<{msg['role']}>{msg['content']}</{msg['role']}>" for msg in prompt_messages]) + "\nWhat tool assistant should call?"
    else:
        prompt_str += "\n".join([f"<{msg['role']}>{msg['content']}</{msg['role']}>" for msg in prompt_messages]) + "\nWhat assistant should response?"
    prompt_str += "</QUESTION>"

        
    
    # 提取唯一标识符或生成一个
    entry_id = f"{task_id}_{source_file.replace('.json', '')}_{source_index}"
    
    # 计算token长度（使用所有消息，包括最后一个turn）
    all_text = " ".join([msg.get('content', '') for msg in prompt_messages]) + " " + ground_truth
    token_length = estimate_token_count(all_text)
    
    # 创建结构化条目
    entry = {
        "id": entry_id,
        "data_source": data_source,
        "prompt": prompt_messages,  # 不包含最后一个turn的消息列表
        "ability": "tool_use" if "npc/toolcall" in data_source else "roleplay",
        "reward_model": {
            "ground_truth": ground_truth,  # 最后一个turn作为ground truth
            "style": "rule"
        },
        "extra_info": {
            "id": entry_id,
            "task_name": task_id,
            "question": prompt_str,
            "source_file": source_file,
            "source_index": source_index,
            "token_length": token_length
        }
    }
    
    # 如果有工具信息，添加到entry中
    if 'tools' in item:
        entry["tools"] = json.loads(item["tools"])
    
    return entry

def process_file(file_path: str, output_dir: str) -> None:
    """
    Process a single JSON file and save as JSON and Parquet.
    
    Args:
        file_path: Path to the input JSON file
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file name and determine task_id and data_source based on file name
    file_name = os.path.basename(file_path)
    
    # Determine task_id and data_source based on whether it's stage0 or stage1
    if "stage_0" in file_name or "stage0" in file_name:
        task_id = "task1"
        data_source = "npc/toolcall/icl-rl"
    elif "stage_1" in file_name or "stage1" in file_name:
        task_id = "task2"
        data_source = "npc/roleplay"
    elif "gpt-4o-toolcall-sft" in file_path:
        task_id = "task1"
        data_source = "npc/toolcall/gpt-4o-synthesis"
    else:
        print(f"Cannot determine task type for {file_name}, skipping...")
        return
    
    print(f"Processing {file_name} as {task_id} ({data_source})...")
    
    # Read the JSON file
    items = read_json_file(file_path)
    
    # Process each item
    entries = []
    for idx, item in enumerate(items):
        entry = create_entry(
            item=item,
            data_source=data_source,
            task_id=task_id,
            source_file=file_name,
            source_index=idx
        )
        entries.append(entry)
    
    # Create output file name based on input file name
    output_base = os.path.splitext(file_name)[0]
    
    # Save as JSON
    json_output_path = os.path.join(output_dir, f"{output_base}.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    
    print(f"Saved JSON output to {json_output_path}")
    
    # Convert to DataFrame and save as Parquet
    df = pd.DataFrame(entries)
    parquet_output_path = os.path.join(output_dir, f"{output_base}.parquet")
    df.to_parquet(parquet_output_path, engine='pyarrow', index=False)
    
    print(f"Saved Parquet output to {parquet_output_path}")
    print(f"Processed {len(entries)} entries from {file_name}")

def main():
    parser = argparse.ArgumentParser(description='Convert NPC-RL data to JSON and Parquet formats')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='List of JSON files to process')
    parser.add_argument('--output_dir', type=str, default='/path/to/npc-rl/data/verl/icl-rl',
                        help='Directory to save processed JSON and Parquet files')
    
    args = parser.parse_args()
    
    # Process each file individually
    for file_path in args.files:
        if os.path.exists(file_path):
            process_file(file_path, args.output_dir)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
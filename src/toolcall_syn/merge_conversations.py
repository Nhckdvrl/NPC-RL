#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并task1_train.json和stage_0.json的脚本
根据task1_train.json中的total_turn值，从stage_0.json中提取相应数量的对话
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

def load_json_file(file_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_data(task1_train_path: str, stage_0_path: str, output_path: str) -> None:
    """
    合并task1_train.json和stage_0.json的数据
    
    Args:
        task1_train_path: task1_train.json的路径
        stage_0_path: stage_0.json的路径
        output_path: 输出文件路径
    """
    # 加载数据
    task1_train_data = load_json_file(task1_train_path)
    stage_0_data = load_json_file(stage_0_path)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化结果列表和用于跟踪stage_0数据使用情况的索引
    result_data = []
    stage_0_index = 0
    
    # 遍历task1_train.json中的每个条目
    for task_item in tqdm(task1_train_data, desc="处理数据"):
        # 获取total_turn值
        total_turn = task_item.get("total_turn", 0)
        
        if total_turn <= 0:
            print(f"警告: 发现无效的total_turn值: {total_turn}，跳过此条目")
            continue
        
        # 确保有足够的stage_0数据可用
        if stage_0_index + total_turn > len(stage_0_data):
            print(f"警告: stage_0数据不足，需要{total_turn}条，但只剩{len(stage_0_data) - stage_0_index}条")
            break
        
        # 从stage_0.json中提取指定数量的对话
        conversations_data = stage_0_data[stage_0_index:stage_0_index + total_turn]
        stage_0_index += total_turn
        
        # 创建新的数据条目，包含task1_train.json的键和stage_0.json的对话
        new_item = {
            "worldview": task_item.get("worldview", ""),
            "total_turn": total_turn,
            "player": task_item.get("player", ""),
            "knowledge": task_item.get("knowledge", {"knowledge_info": "", "general_info": ""}),
            "conversations": []
        }
        
        # 添加所有对话
        for conv_item in conversations_data:
            new_item["conversations"].append({
                "conversations": conv_item.get("conversations", []),
                "tools": conv_item.get("tools", "[]")
            })
        
        result_data.append(new_item)
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成! 共处理了{len(result_data)}个条目")
    print(f"结果已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="合并task1_train.json和stage_0.json的数据")
    parser.add_argument("--task1_train", type=str, default="/path/to/npc-rl/data/task1_train.json",
                        help="task1_train.json的路径")
    parser.add_argument("--stage_0", type=str, default="/path/to/npc-rl/data/sft/task1/stage_0.json",
                        help="stage_0.json的路径")
    parser.add_argument("--output", type=str, default="/path/to/npc-rl/data/gpt-4o-toolcall-sft/raw_stage0_train.json",
                        help="输出文件路径")
    
    args = parser.parse_args()
    
    merge_data(args.task1_train, args.stage_0, args.output)

if __name__ == "__main__":
    main()

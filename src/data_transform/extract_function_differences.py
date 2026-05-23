#!/usr/bin/env python3
"""
提取对话历史和gold与model工具调用不一致的样本
输出格式化JSON以便分析
"""

import json
import sys
import copy
import os
from collections import defaultdict
import argparse

def normalize_function(func):
    """标准化函数调用，移除return字段"""
    func_copy = copy.deepcopy(func)
    if "return" in func_copy:
        del func_copy["return"]
    return func_copy

def compare_functions(gold_funcs, model_funcs):
    """比较两组函数调用是否相同"""
    if len(gold_funcs) != len(model_funcs):
        return False
    
    # 标准化两边的函数
    normalized_gold = [normalize_function(f) for f in gold_funcs]
    normalized_model = [normalize_function(f) for f in model_funcs]
    
    # 检查每个函数是否都能在另一边找到匹配
    for g_func in normalized_gold:
        found = False
        for m_func in normalized_model:
            if g_func["name"] == m_func["name"] and g_func.get("parameters", {}) == m_func.get("parameters", {}):
                found = True
                break
        if not found:
            return False
    
    return True

def extract_differences(input_file, output_file):
    """提取对话历史和gold与model工具调用不一致的样本"""
    print(f"Processing {input_file}...")
    
    # 统计信息
    total_turns = 0
    different_functions = 0
    differences = []
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 遍历每个样本
            for sample_idx, sample in enumerate(data):
                if "turns" not in sample:
                    continue
                
                # 遍历每个回合
                for turn_key, turn in sample["turns"].items():
                    total_turns += 1
                    
                    # 获取gold和model的函数调用
                    gold_functions = turn.get("gold", {}).get("functions", [])
                    model_functions = turn.get("model", {}).get("functions", [])
                    
                    # 比较函数调用
                    if not compare_functions(gold_functions, model_functions):
                        different_functions += 1
                        
                        # 提取对话历史
                        dialogue_history = turn.get("dialogue", [])
                        
                        # 创建差异记录
                        difference = {
                            "sample_idx": sample_idx,
                            "turn": turn_key,
                            "dialogue": dialogue_history,
                            "gold": {
                                "response": turn.get("gold", {}).get("response", ""),
                                "functions": gold_functions
                            },
                            "model": {
                                "response": turn.get("model", {}).get("response", ""),
                                "functions": model_functions
                            }
                        }
                        
                        differences.append(difference)
                        
                        # 输出进度
                        if len(differences) % 10 == 0:
                            print(f"Found {len(differences)} differences so far...")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return
    
    # 输出统计信息
    print(f"\nTotal turns: {total_turns}")
    print(f"Turns with different functions: {different_functions} ({different_functions/total_turns*100:.2f}%)")
    
    # 保存差异到文件
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(differences, f, indent=2, ensure_ascii=False)
    print(f"Detailed differences saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract dialogue history and function call differences')
    parser.add_argument('input_file', help='Path to the merge_result JSON file')
    parser.add_argument('--output', '-o', default='function_differences_with_dialogue.json', 
                        help='Output file path (default: function_differences_with_dialogue.json)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    extract_differences(args.input_file, args.output)

if __name__ == "__main__":
    main()

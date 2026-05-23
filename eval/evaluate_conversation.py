#!/usr/bin/env python3
"""
Evaluate conversation quality for the given dataset.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from llm_eval import LLMJudge
from coser_eval import evaluate_game_roleplay_turn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ConversationEvaluator:
    """Evaluator for conversation quality."""
    
    def __init__(self, llm_judge: Optional[LLMJudge] = None):
        """Initialize with an optional LLM judge."""
        self.llm_judge = llm_judge

    def process_turn_data(self, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process turn data and extract necessary information for evaluation.
        
        Args:
            turn_data: Dictionary containing turn data with dialogue, gold, and model responses
        
        Returns:
            Dictionary containing processed turn information
        """
        dialogue_history = []
        player_query = ""
        gold_response = turn_data.get('gold', {}).get('response', '')
        model_response = turn_data.get('model', {}).get('response', '')
        gold_functions = turn_data.get('gold', {}).get('functions', [])
        model_functions = turn_data.get('model', {}).get('functions', [])
        
        # Extract dialogue history and player query
        for entry in turn_data.get('dialogue', []):
            if entry['speaker'] == 'player':
                player_query = entry['text']
                dialogue_history.append({
                    'speaker': 'Player',
                    'text': entry['text']
                })
            elif entry['speaker'] == 'npc':
                dialogue_history.append({
                    'speaker': 'NPC',
                    'text': entry['text']
                })
        
        return {
            'player_query': player_query,
            'gold_response': gold_response,
            'model_response': model_response,
            'gold_functions': gold_functions,
            'model_functions': model_functions,
            'dialogue_history': dialogue_history
        }

    def evaluate_turn(self, turn_data: Dict[str, Any], conversation: Dict[str, Any], turn_idx: int) -> Dict[str, Any]:
        """评估单个对话轮次
        Args:
            turn_data: 当前轮次的数据，包含对话、gold和model的响应
            conversation: 整个对话的上下文数据
            turn_idx: 当前轮次的索引
            
        Returns:
            包含评估结果的字典
        """
        # 处理轮次数据
        processed = self.process_turn_data(turn_data)
        
        result = {
            'turn_metrics': {
                'has_functions': False,
                'tool_metrics': None,
                'roleplay_score': None
            }
        }
        
        # 1. 评估工具调用
        gold_functions = processed['gold_functions']
        model_functions = processed['model_functions']
        
        if gold_functions or model_functions:
            result['turn_metrics']['has_functions'] = True
            result['turn_metrics']['tool_metrics'] = self._evaluate_tool_calls(
                gold_functions, model_functions
            )
        
        # 2. 评估角色扮演质量
        try:
            # 构建角色扮演评估所需的参数
            roleplay_result = evaluate_game_roleplay_turn(
                turn_number=turn_idx + 1,
                worldview_details=conversation.get('worldview', ''),
                current_turn_state_knowledge=conversation.get('state_knowledge', ''),
                player_profile=conversation.get('player', {}).get('persona', {}),
                npc_profile=conversation.get('npc', {}).get('persona', {}),
                full_dialogue_history=processed['dialogue_history'][:-1],  # 排除当前轮次的玩家输入
                player_query_for_current_turn=processed['player_query'],
                model_response_text=processed['model_response'],
                gold_standard_npc_response_text=processed['gold_response'],
                gold_standard_npc_functions=gold_functions,
                dimension_to_evaluate="all",
                llm_judge_instance=self.llm_judge,
                game_title=conversation.get('worldview', 'Dialogue Evaluation').split('.')[0]
            )
            
            # 处理成功的角色扮演评估结果
            result['turn_metrics']['roleplay_score'] = {
                'overall_score': roleplay_result.get('overall_score', 0.0),
                'dimension_scores': roleplay_result.get('dimension_scores', {}),
                'reasoning': roleplay_result.get('reasoning', ''),
                'flaws': roleplay_result.get('all_flaws', [])
            }
            
        except Exception as e:
            print(f"Error in roleplay evaluation: {e}")
            result['turn_metrics']['roleplay_score'] = {
                'overall_score': 0.0,
                'dimension_scores': {},
                'error': str(e),
                'flaws': []
            }
        
        return result

    def _evaluate_tool_calls(self, gold_functions: List[Dict], model_functions: List[Dict]) -> Dict[str, Any]:
        """评估工具调用的准确率
        
        Args:
            gold_functions: 标准答案中的工具调用列表
            model_functions: 模型预测的工具调用列表
            
        Returns:
            包含评估指标的字典，包括精确率、召回率、F1等
        """
        # 确保输入是列表
        gold_functions = gold_functions or []
        model_functions = model_functions or []
        
        # 初始化默认结果
        result = {
            'tool_name_precision': 0.0,
            'tool_name_recall': 0.0,
            'tool_name_f1': 0.0,
            'gold_tool_count': len(gold_functions),
            'predicted_tool_count': len(model_functions),
            'correct_tool_count': 0,
            'evaluated': False
        }
        
        # 如果两者都为空，直接返回默认结果
        if not gold_functions and not model_functions:
            return result
            
        try:
            # 1. 工具名称匹配
            gold_names = {str(f.get('name', '')).strip().lower() for f in gold_functions if f.get('name')}
            model_names = {str(f.get('name', '')).strip().lower() for f in model_functions if f.get('name')}
            
            # 计算名称匹配的精确率、召回率和F1
            correct_names = gold_names.intersection(model_names)
            precision = len(correct_names) / len(model_names) if model_names else 0.0
            recall = len(correct_names) / len(gold_names) if gold_names else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 更新结果
            result.update({
                'tool_name_precision': round(precision, 4),
                'tool_name_recall': round(recall, 4),
                'tool_name_f1': round(f1, 4),
                'correct_tool_count': len(correct_names),
                'evaluated': True
            })
            
            # 2. 参数匹配（可选）
            # 这里可以添加更详细的参数匹配逻辑
            
        except Exception as e:
            print(f"Error in tool call evaluation: {e}")
            result['error'] = str(e)
        
        return result
    def evaluate_roleplay_quality(self, conversation: Dict[str, Any], turn_data: Dict[str, Any], turn_idx: int) -> Dict[str, Any]:
        """评估角色扮演质量。"""
        try:
            # 从对话数据中提取必要的信息
            player_query = turn_data.get('player_query', '')
            model_response = turn_data.get('model', {}).get('content', '')
            gold_response = turn_data.get('gold', {}).get('content', '')
            
            # 构建对话历史
            dialogue_history = []
            for i in range(turn_idx):
                turn_key = f"turn_{i}"
                if turn_key in conversation.get('turns', {}):
                    turn = conversation['turns'][turn_key]
                    if 'player_query' in turn:
                        dialogue_history.append({
                            'speaker': 'Player',
                            'text': turn['player_query']
                        })
                    if 'gold' in turn and 'content' in turn['gold']:
                        dialogue_history.append({
                            'speaker': 'NPC',
                            'text': turn['gold']['content']
                        })
            
            # 调用coser_eval进行评估
            roleplay_result = evaluate_game_roleplay_turn(
                turn_number=turn_idx + 1,
                worldview_details=conversation.get('worldview', ''),
                current_turn_state_knowledge=conversation.get('state_knowledge', ''),
                player_profile=conversation.get('player_profile', {}),
                npc_profile=conversation.get('npc_profile', {}),
                full_dialogue_history=dialogue_history,
                player_query_for_current_turn=player_query,
                model_response_text=model_response,
                gold_standard_npc_response_text=gold_response,
                gold_standard_npc_functions=turn_data.get('gold', {}).get('function_call', []),
                dimension_to_evaluate="all",
                llm_judge_instance=self.llm_judge,
                game_title=conversation.get('scenario', 'Dialogue Evaluation')
            )
            
            return {
                'success': True,
                'overall_score': roleplay_result.get('overall_score', 0.0),
                'dimension_scores': roleplay_result.get('dimension_scores', {}),
                'reasoning': roleplay_result.get('reasoning', ''),
                'flaws': roleplay_result.get('all_flaws', [])
            }
        except Exception as e:
            print(f"Error in roleplay evaluation: {e}")
            return {
                'success': False,
                'error': str(e),
                'overall_score': 0.0,
                'dimension_scores': {},
                'reasoning': f'评估出错: {str(e)}',
                'flaws': []
            }
    def evaluate_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """评估整个对话
        
        Args:
            conversation: 包含对话数据的字典
            
        Returns:
            包含评估结果的字典
        """
        result = {
            'conversation_id': conversation.get('data_id', 'unknown'),
            'total_turns': conversation.get('total_turn', 0),
            'turns': [],
            'tool_metrics': {
                'avg_tool_f1': 0.0,
                'tool_precision': 0.0,
                'tool_recall': 0.0
            },
            'roleplay_metrics': {
                'avg_roleplay_score': 0.0
            }
        }
        
        turns = conversation.get('turns', {})
        if not turns:
            return result
            
        tool_f1_scores = []
        tool_precisions = []
        tool_recalls = []
        roleplay_scores = []
        
        for turn_idx, (turn_key, turn_data) in enumerate(turns.items()):
            try:
                # 评估当前轮次
                turn_result = self.evaluate_turn(
                    turn_data=turn_data,
                    conversation=conversation,
                    turn_idx=turn_idx
                )
                
                # 添加轮次元数据
                turn_result['turn_id'] = turn_key
                turn_result['turn_idx'] = turn_idx
                
                # 处理工具调用指标
                tool_metrics = turn_result.get('turn_metrics', {}).get('tool_metrics')
                if tool_metrics and isinstance(tool_metrics, dict):
                    if 'tool_name_f1' in tool_metrics:
                        tool_f1_scores.append(tool_metrics['tool_name_f1'])
                    if 'tool_name_precision' in tool_metrics:
                        tool_precisions.append(tool_metrics['tool_name_precision'])
                    if 'tool_name_recall' in tool_metrics:
                        tool_recalls.append(tool_metrics['tool_name_recall'])
                
                # 处理角色扮演分数
                roleplay_score = turn_result.get('turn_metrics', {}).get('roleplay_score', {}).get('overall_score')
                if roleplay_score is not None:
                    roleplay_scores.append(roleplay_score)
                
                result['turns'].append(turn_result)
                
            except Exception as e:
                print(f"Error evaluating turn {turn_key}: {e}")
                continue
        
        # 计算平均指标
        if tool_f1_scores:
            result['tool_metrics']['avg_tool_f1'] = sum(tool_f1_scores) / len(tool_f1_scores)
            
        if tool_precisions:
            result['tool_metrics']['tool_precision'] = sum(tool_precisions) / len(tool_precisions)
            
        if tool_recalls:
            result['tool_metrics']['tool_recall'] = sum(tool_recalls) / len(tool_recalls)
        
        if roleplay_scores:
            result['roleplay_metrics']['avg_roleplay_score'] = sum(roleplay_scores) / len(roleplay_scores)
        
        return result
        

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load conversation data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        return [data]
    return data

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save evaluation results to a JSON file."""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate conversation quality.')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to the input JSON file with conversations')
    parser.add_argument('--output_file', type=str, default='results/evaluation_results.json',
                       help='Path to save the evaluation results')
    
    args = parser.parse_args()
    
    # Initialize LLM judge if needed
    llm_judge = None
    try:
        llm_judge = LLMJudge()
        print("LLM judge initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize LLM judge: {e}")
        print("Falling back to rule-based evaluation.")
        raise Exception
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    conversations = load_data(args.data_file)
    print(f"Loaded {len(conversations)} conversations.")
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(llm_judge=llm_judge)
    
    # Evaluate each conversation
    results = []
    for conv in conversations:
        result = evaluator.evaluate_conversation(conv)
        results.append(result)
        
        # Print progress
        conv_id = result['conversation_id']
        avg_f1 = result.get('avg_function_f1', 0)
        avg_len_ratio = result.get('avg_length_ratio', 0)
        print(f"Evaluated {conv_id}: F1={avg_f1:.2f}, Length Ratio={avg_len_ratio:.2f}")
    
    # Save results
    save_results(results, args.output_file)
    print(f"\nEvaluation complete. Results saved to {args.output_file}")
    
    # Print summary
    if results:
        avg_f1 = sum(r.get('avg_function_f1', 0) for r in results) / len(results)
        avg_len_ratio = sum(r.get('avg_length_ratio', 0) for r in results) / len(results)
        
        print("\nSummary:")
        print(f"  Average Function F1: {avg_f1:.4f}")
        print(f"  Average Length Ratio: {avg_len_ratio:.4f}")

if __name__ == "__main__":
    main()

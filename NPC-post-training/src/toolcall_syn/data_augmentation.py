#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据增广脚本：使用GPT-4o生成工具调用数据
适用于新的数据格式，包含worldview、player、knowledge等元数据
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm

from llm_client import LLMJudge, LLMJudgeConfig

class ToolCallAugmentor:
    """工具调用数据增广器"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        初始化增广器
        
        Args:
            model_name: 使用的模型名称，默认为gpt-4o
        """
        config = LLMJudgeConfig(
            temperature=0.7,  # 使用较高的温度以增加多样性
            max_tokens=1024,
            model_name=model_name
        )
        self.llm = LLMJudge(config)
        
    def create_tool_augmentation_prompt(self, sample: Dict[str, Any], conv_index: int, tool_name: str) -> str:
        """
        创建针对特定工具的数据增广提示
        
        Args:
            sample: 包含worldview、knowledge等元数据的样本
            conv_index: 当前处理的对话索引
            tool_name: 当前要生成的工具名称
        
        Returns:
            str: 提示文本
        """
        # 获取元数据
        worldview = sample.get("worldview", "")
        player = sample.get("player", "")
        knowledge = sample.get("knowledge", {})
        knowledge_info = knowledge.get("knowledge_info", "")
        general_info = knowledge.get("general_info", "")
        
        # 获取当前对话
        conversation_data = sample["conversations"][conv_index]
        conversations = conversation_data.get("conversations", [])
        tools_json = conversation_data.get("tools", "[]")
        
        # 提取除最后一个human-response对之外的所有对话
        context_conversations = conversations[:-2] if len(conversations) > 2 else conversations
        
        # 解析工具列表
        tools = json.loads(tools_json)
        
        # 找到目标工具的描述
        target_tool = None
        for tool in tools:
            if tool["type"] == "function" and tool["function"]["name"] == tool_name:
                target_tool = tool
                break
        
        if not target_tool:
            return "SKIP_TOOL"  # 工具不存在
        
        # 构建提示
        prompt = f"""你是一个专门生成工具调用数据的助手。请基于以下背景信息、对话历史和工具描述，生成一个合理的用户请求和相应的工具调用。

## 背景信息:
世界观: {worldview}
玩家: {player}
知识信息: {knowledge_info}
通用信息: {general_info}

## 对话历史:
"""
        
        # 添加对话历史
        for msg in context_conversations:
            role = msg["from"]
            content = msg["value"]
            prompt += f"{role}: {content}\n\n"
        
        # 添加目标工具描述
        tool_desc = target_tool["function"]["description"]
        tool_params = target_tool["function"]["parameters"]
        
        prompt += f"""
## 目标工具:
名称: {tool_name}
描述: {tool_desc}
参数: {json.dumps(tool_params, indent=2)}

## 任务:
1. 生成一个合理的用户请求，该请求应该自然地引导到使用 {tool_name} 工具
2. 生成一个适当的工具调用JSON响应

如果你认为在当前对话上下文中，用户不太可能需要使用 {tool_name} 工具，请回复 "SKIP_TOOL"。

请使用以下格式使用英语输出:
<user>用户请求内容</user>
<assistant>{{"name": "{tool_name}", "arguments": {{参数}}}}</assistant>

确保JSON格式正确，只包含在对话中明确提到或强烈暗示的参数。不要包含空值、默认值或不确定的值。
"""
        # print(prompt)
        return prompt
    
    def parse_augmentation_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析增广响应
        
        Args:
            response: GPT-4o的响应
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (用户请求, 工具调用JSON)，如果跳过则都为None
        """
        if response.strip() == "SKIP_TOOL":
            return None, None
        
        # 提取用户请求
        user_match = re.search(r'<user>(.*?)</user>', response, re.DOTALL)
        if not user_match:
            return None, None
        user_request = user_match.group(1).strip()
        
        # 提取工具调用
        assistant_match = re.search(r'<assistant>(.*?)</assistant>', response, re.DOTALL)
        if not assistant_match:
            return None, None
        tool_call = assistant_match.group(1).strip()
        
        return user_request, tool_call
    
    def augment_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        增广单个样本
        
        Args:
            sample: 包含worldview、knowledge等元数据的样本
            
        Returns:
            List[Dict[str, Any]]: 增广后的样本列表，格式为[{conversations, tools}]
        """
        augmented_samples = []
        
        # 遍历样本中的所有对话
        for conv_index, conversation_data in enumerate(sample.get("conversations", [])):
            tools_json = conversation_data.get("tools", "[]")
            
            # 解析工具列表
            tools = json.loads(tools_json)
            
            # 为每个工具生成一个增广样本
            for tool in tools:
                if tool["type"] != "function":
                    continue
                    
                tool_name = tool["function"]["name"]
                
                # 创建提示
                prompt = self.create_tool_augmentation_prompt(sample, conv_index, tool_name)
                
                if prompt == "SKIP_TOOL":
                    continue
                
                # 调用LLM
                response = self.llm.predict(prompt)
                
                # 解析响应
                user_request, tool_call = self.parse_augmentation_response(response)
                
                if user_request and tool_call:
                    # 获取原始对话
                    original_conversations = conversation_data.get("conversations", [])
                    
                    # 创建新的增广样本 (使用原始格式 {conversations, tools})
                    new_sample = {
                        "conversations": [],
                        "tools": tools_json
                    }
                    
                    # 复制除最后一个human-response对之外的所有对话
                    for msg in original_conversations[:-2]:
                        new_sample["conversations"].append(msg.copy())
                    
                    # 添加新的对话对
                    new_sample["conversations"].append({
                        "from": "human",
                        "value": user_request
                    })
                    new_sample["conversations"].append({
                        "from": "function_call",
                        "value": tool_call
                    })
                    
                    augmented_samples.append(new_sample)
        
        return augmented_samples
    
    def augment_dataset(self, input_file: str, output_file: str, max_samples: int = None) -> None:
        """
        增广整个数据集
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            max_samples: 最大处理样本数，None表示处理所有
        """
        # 创建输出目录
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 限制样本数
        if max_samples is not None:
            data = data[:max_samples]
        
        # 增广数据
        augmented_data = []
        for sample in tqdm(data, desc="Augmenting samples"):
            augmented_samples = self.augment_sample(sample)
            augmented_data.extend(augmented_samples)
            
            # 每10个样本保存一次，避免数据丢失
            if len(augmented_data) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(augmented_data, f, ensure_ascii=False, indent=2)
                print(f"已处理 {len(augmented_data)} 个样本，中间结果已保存")
        
        # 保存最终结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
        print(f"增广完成！原始样本数: {len(data)}, 增广后样本数: {len(augmented_data)}")
        print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="工具调用数据增广")
    parser.add_argument("--input", type=str, default="/path/to/npc-rl/data/gpt-4o-toolcall-sft/raw_stage0_train.json",
                        help="输入文件路径")
    parser.add_argument("--output", type=str, default="/path/to/npc-rl/data/gpt-4o-toolcall-sft/toolcall-v1.json",
                        help="输出文件路径")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="使用的模型名称")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数，None表示处理所有")
    
    args = parser.parse_args()
    
    augmentor = ToolCallAugmentor(model_name=args.model)
    augmentor.augment_dataset(args.input, args.output, args.max_samples)


if __name__ == "__main__":
    main()

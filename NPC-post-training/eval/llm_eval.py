from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
import re
import json
import time
import os
import pandas as pd
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta
@dataclass
class LLMJudgeConfig:
    """LLM Judge的配置类"""
    api_key: str = "123"
    base_url: str = "http://0.0.0.0:8112/v1"
    temperature: float = 0.3
    max_tokens: int = 1024
    model_name: Optional[str] = None

class LLMCache:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, cache_file: str = "results/cache_llm.csv"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.cache_file = Path(cache_file)
                cls._instance.cache = {}
                cls._instance.last_save_time = time.time()
                cls._instance.save_interval = 600  # 10 minutes in seconds
                cls._instance._load_cache()
            return cls._instance
    
    def _load_cache(self) -> None:
        """Load cache from file if it exists"""
        try:
            if self.cache_file.exists():
                df = pd.read_csv(self.cache_file)
                self.cache = dict(zip(df['prompt'], df['response']))
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
    
    def _save_cache(self) -> None:
        """Save cache to file"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame([{"prompt": k, "response": v} for k, v in self.cache.items()])
            df.to_csv(self.cache_file, index=False)
            self.last_save_time = time.time()
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def get(self, prompt: str) -> Tuple[bool, str]:
        """Get response from cache if exists"""
        return prompt in self.cache, self.cache.get(prompt, "")
    
    def set(self, prompt: str, response: str) -> None:
        """Add or update cache entry"""
        self.cache[prompt] = response
        # Check if we need to save to disk
        if time.time() - self.last_save_time >= self.save_interval:
            self._save_cache()
    
    def save_if_needed(self) -> None:
        """Save cache to disk if enough time has passed"""
        if time.time() - self.last_save_time >= self.save_interval:
            self._save_cache()

class LLMJudge:
    """使用LLM来评估模型输出质量的评分器"""
    
    def __init__(self, config: Optional[LLMJudgeConfig] = None):
        """初始化LLM Judge
        
        Args:
            config: LLMJudgeConfig实例，如果为None则使用默认配置
        """
        self.config = config or LLMJudgeConfig()
        if os.getenv("LLMEVAL_API_TYPE", "openai") == "openai":
            from openai import OpenAI
            api_key =os.getenv("LLMEVAL_API_KEY", self.config.api_key)
            base_url =os.getenv("LLMEVAL_API_BASE", self.config.base_url)
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.model_name = os.getenv("LLMEVAL_API_MODEL", self._get_model_name())
        elif os.getenv("LLMEVAL_API_TYPE", "openai") == "azure_key":
            from openai import AzureOpenAI
            endpoint = os.getenv("LLMEVAL_API_BASE")
            subscription_key = os.getenv("LLMEVAL_API_KEY")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=subscription_key,
                api_version="2025-01-01-preview",
            )

            self.model_name = os.getenv("LLMEVAL_API_MODEL", self._get_model_name())
        else:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import AzureOpenAI
            api_base = os.getenv('OPENAI_API_BASE')
            api_type = os.getenv('OPENAI_API_TYPE')
            api_version = os.getenv('OPENAI_API_VERSION')
            credential = AzureCliCredential()
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAI(
                azure_endpoint=api_base,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )
            self.model_name = os.getenv("LLMEVAL_API_MODEL")
        self.cache = LLMCache()
        self.cost_init()

    def cost_init(self):
        """
        初始化成本信息
        """
        self.INPUT_COST_PER_1000_TOKENS = {
            "gpt-4o": 0.0025,
            "gpt-4o-mini": 0.00015,
            "o4-mini": 0.00055,
            "o3-mini": 0.00110,
            "o1": 0.015,
            "o1-mini": 0.00110,
            "gpt-3.5-turbo": 0.0005,
            "gpt-35-turbo": 0.0005,
            }

        self.OUTPUT_COST_PER_1000_TOKENS = {
            "gpt-4o": 0.01,
            "gpt-4o-mini": 0.0006,
            "o4-mini": 0.0022,
            "o3-mini": 0.0044,
            "o1": 0.06,
            "o1-mini": 0.0044,
            "gpt-3.5-turbo": 0.0015,
            "gpt-35-turbo": 0.0015,
        }
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        
    def _get_model_name(self) -> str:
        """获取可用的模型名称"""
        if self.config.model_name:
            return self.config.model_name
            
        try:
            models = self.client.models.list()
            for model in models:
                if model.id:
                    return model.id
        except Exception as e:
            print(f"Warning: Failed to get model list: {e}")
        
        return "default_model"  # 如果无法获取模型列表，返回默认值

    @staticmethod
    def _extract_xml_content(text: str, tag: str) -> tuple[bool, str]:
        """从XML标签中提取内容"""
        if not text or not tag:
            return False, ""
        
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            if content:  # 检查内容是否为空
                return True, content
        
        return False, ""

    def predict(self, prompt: str) -> str:
        """
        单个预测处理，带缓存功能和重试机制
        
        Args:
            prompt: 输入提示
            
        Returns:
            str: 生成的文本或失败时返回0
        """
        # 检查缓存
        in_cache, cached_response = self.cache.get(prompt)
        if in_cache:
            return cached_response
            
        # 缓存未命中，调用 API
        max_retries = 3  # 最大重试次数
        retry_delay = 10  # 重试间隔（秒）
        
        for retry_count in range(max_retries):
            try:
                conversation = [{"role": "user", "content": prompt}]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                # 提取生成的文本并缓存
                generated_text = response.choices[0].message.content.strip()
                
                # 更新成本
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self._update_cost(input_tokens, output_tokens)
                
                # 保存到缓存
                self.cache.set(prompt, generated_text)
                self.save_cost()
                return generated_text
                
            except Exception as e:
                error_msg = f"调用预测 API 失败 (尝试 {retry_count + 1}/{max_retries}): {str(e)}"
                print(error_msg)
                
                # 如果不是最后一次重试，等待后继续
                if retry_count < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print("所有重试均失败，返回0")
                    return ""  # 所有重试均失败时返回 0
    
    def batch_predict(self, prompts: List[str]) -> List[str]:
        """
        批量预测处理，带缓存功能和重试机制

        Args:
            prompts: 输入提示列表

        Returns:
            List[str]: 生成的文本列表，失败项返回""
        """
        if not prompts:
            return []
            
        results = []
        to_process = []
        to_process_indices = []  # 记录未缓存项的原始索引
        
        # 首先检查缓存
        for i, prompt in enumerate(prompts):
            in_cache, cached_response = self.cache.get(prompt)
            if in_cache:
                results.append(cached_response)
            else:
                to_process.append(prompt)
                to_process_indices.append(i)
        
        # 如果所有项都已缓存，直接返回缓存结果
        if not to_process:
            return results
            
        # 预先填充结果列表，未处理的位置先用None占位
        while len(results) < len(prompts):
            results.append(None)
        
        # 批量处理未缓存的提示
        batch_size = 5  # 较小的批量以避免超时，可以根据需要调整
        for i in range(0, len(to_process), batch_size):
            batch_prompts = to_process[i:i+batch_size]
            batch_indices = to_process_indices[i:i+batch_size]
            
            # 为当前批次定义重试机制
            max_retries = 3  # 最大重试次数
            retry_delay = 10  # 重试间隔（秒）
            
            for retry_count in range(max_retries):
                try:
                    # 将提示转换为对话格式
                    conversations = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
                    
                    # 批量处理当前批次
                    batch_results = []
                    for j, conversation in enumerate(conversations):
                        prompt = batch_prompts[j]
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=conversation,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens
                        )
                        
                        # 提取生成的文本并缓存
                        generated_text = response.choices[0].message.content.strip()
                        
                        # 更新成本
                        input_tokens = response.usage.prompt_tokens
                        output_tokens = response.usage.completion_tokens
                        self._update_cost(input_tokens, output_tokens)
                        
                        # 保存到缓存并添加到结果
                        self.cache.set(prompt, generated_text)
                        batch_results.append(generated_text)
                    
                    # 将批处理结果放回对应位置
                    for j, result_text in enumerate(batch_results):
                        results[batch_indices[j]] = result_text
                    
                    # 成功处理，退出重试循环
                    break
                    
                except Exception as e:
                    error_msg = f"批量处理出错 (尝试 {retry_count + 1}/{max_retries}): {str(e)}"
                    print(error_msg)
                    
                    # 如果不是最后一次重试，等待后继续
                    if retry_count < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        import time
                        time.sleep(retry_delay)
                    else:
                        print("所有重试均失败，对当前批次返回0")
                        # 所有重试均失败时，当前批次的所有项返回 ""
                        for idx in batch_indices:
                            results[idx] = ""
        
        # 保存成本
        for prompt in prompts:
            in_cache, cached_response = self.cache.get(prompt)
            if in_cache and cached_response in results:
                final_results.append(cached_response)
            elif new_idx < len(to_process):
                final_results.append(results[new_idx])
                new_idx += 1
        self.save_cost()
        return final_results


    def _update_cost(self, input_tokens, output_tokens):
        """
        更新累计的 token 数量和成本。
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        cost = (input_tokens / 1000) * self.INPUT_COST_PER_1000_TOKENS.get(self.model_name, 0) + (output_tokens / 1000) * self.OUTPUT_COST_PER_1000_TOKENS.get(self.model_name, 0)
        self.total_cost += cost


    def save_cost(self):
        """
        保存当前累计的成本信息到 cost.jsonl，并清零累计值。
        """
        cost_data = {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost
        }
        
        # 追加保存到 cost.jsonl
        with open("results/cost.jsonl", "a") as f:
            json.dump(cost_data, f)
            f.write("\n")
        
        # 输出成本信息并清零
        print(f"Saved cost: {cost_data}")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        return cost_data
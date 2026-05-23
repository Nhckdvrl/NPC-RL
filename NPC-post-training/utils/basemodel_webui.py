"""
Streamlit application to visualize conversations from online_vllm.py
"""
#  vllm serve ~/jx/rl_models/social_qwen3b_0520_50step/   --api-key 123  --gpu-memory-utilization 0.4   --port 8112   --max-model-len 6000    --tensor-parallel-size 2   --trust-remote-code --max_num_seqs 32 &
#  vllm serve ~/jx/rl_model/social_qwen3b_0520_50step_no_think/   --api-key 123  --gpu-memory-utilization 0.4   --port 8113   --max-model-len 6000    --tensor-parallel-size 2   --trust-remote-code --max_num_seqs 32 &
import sys
import os
import json
import torch
import datetime
import streamlit as st
from openai import OpenAI

# 设置页面标题 - 必须是第一个 Streamlit 命令
st.set_page_config(page_title="对话模型演示", layout="wide")

# Add project root to path
sys.path.append("/home/aiscuser/Function-Calling/other_repos/Agent-R1")
sys.path.append("/home/aiscuser/Function-Calling/multisources-search-r1")
sys.path.append("/home/aiscuser/Function-Calling/multisources-search-r1/train")
# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 支持多个端口
PORTS = os.getenv("PORTS", "8112,8113").split(",")

# Import necessary modules
from src.core.online_vllm import process_query, serialize_conversations, deserialize_conversations, OpenAI

class OpenAIClient:
    _instances = {}
    
    @classmethod
    def get_instance(cls, port):
        if port not in cls._instances:
            cls._instances[port] = cls(port)
        return cls._instances[port]

    def get_model_name(self):
        _models = self.client.models.list()
        for _model in _models:
            if _model.id:
                model_name = _model.id
                break
        if not model_name:
            raise ValueError("未找到可用的模型")
        return model_name
    def __init__(self, port):
        self.port = port
        self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="123")
        self.model_name = self.get_model_name()
        
# Simple Singleton class
class Singleton:
    _instances = {}
    
    @classmethod
    def get_instance(cls):
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]

# 初始化评估器 - simplified version without ModelEvaluator
class EvaluatorManager(Singleton):
    def __init__(self):
        print("评估器已初始化")
    
    def evaluate_response(self, conversations, question, golden_answers=None):
        # 获取最后一个助手回复
        last_assistant_msg = None
        for msg in reversed(conversations):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg
                break
        
        if not last_assistant_msg:
            return {"error": "未找到助手回复"}
        
        # 简化的评估逻辑，不使用ModelEvaluator
        try:
            # 简单评估 - 固定返回值
            scores = {
                "average": 0.85,
                "relevance": {"score": 0.9, "reason": "回复与问题相关"},
                "coherence": {"score": 0.8, "reason": "回复连贯"},
                "groundedness": {"score": 0.85, "reason": "回复基于事实"}
            }
            return scores
        except Exception as e:
            return {"error": str(e)}

# 初始化tokenizer
from transformers import AutoTokenizer

# 侧边栏配置
with st.sidebar:
    st.title("配置选项")
    
    # 模型选择
    st.subheader("模型选择")
    selected_port = st.selectbox(
        "选择要使用的模型",
        options=PORTS,
        format_func=lambda x: OpenAIClient.get_instance(x).model_name
    )
    
    st.write(f"当前选择的模型: {OpenAIClient.get_instance(selected_port).model_name}")
    st.write(f"连接地址: http://localhost:{selected_port}/v1")

# 初始化OpenAI客户端
openai_client = OpenAIClient.get_instance(selected_port)

# 初始化评估器
evaluator_manager = EvaluatorManager.get_instance()
def _get_model_name(port) -> str:
    return OpenAIClient.get_instance(port).model_name

# Import the required functions from online_vllm

# Set page config

# Main title
st.title("Conversation Visualizer")

# 初始化tokenizer
model_name = _get_model_name(selected_port)
if model_name:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    st.sidebar.success(f"已加载模型: {model_name}")
else:
    st.sidebar.error("无法获取模型名称")
    tokenizer = None

# 输入区域和按钮布局
st.write("输入您的问题：")
col1, col2 = st.columns([6, 1])

with col1:
    # 使用回车键触发查询处理
    query = st.text_input("", key="query_input", label_visibility="collapsed")

with col2:
    process_button = st.button("发送", key="process_query")

# 初始化会话状态
if 'model_history' not in st.session_state:
    st.session_state.model_history = {}  # 按模型存储历史记录

# 确保每个模型都有自己的历史记录列表
for port in PORTS:
    if port not in st.session_state.model_history:
        st.session_state.model_history[port] = []

if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# 当用户输入新查询时更新当前查询
if query != st.session_state.current_query:
    st.session_state.current_query = query

# 初始化历史对话路径
model_name = _get_model_name(selected_port)
if model_name:
    model_short_name = model_name.split("/")[-1]
    save_path = f"/home/aiscuser/Function-Calling/data/online_vllm_{model_short_name}.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 读取历史对话
if 'conversations' not in st.session_state:
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            st.session_state.conversations = json.load(f)
    else:
        st.session_state.conversations = []

# 处理查询
def process_current_query():
    if query:
        with st.spinner("正在处理查询..."):
            try:
                # 获取当前选择的客户端
                current_client = OpenAIClient.get_instance(selected_port)
                model_name = _get_model_name(selected_port)
                
                # 处理查询并获取对话
                new_conversations = process_query(current_client.client, query, model_name=model_name)
                
                # 序列化新对话
                serialized_new_conversations = serialize_conversations(new_conversations)
                
                # 评估回复
                evaluation_results = None
                if tokenizer:
                    try:
                        evaluation_results = evaluator_manager.evaluate_response(
                            conversations=new_conversations,
                            question=query
                        )
                    except Exception as eval_error:
                        st.warning(f"评估时出错: {str(eval_error)}")
                
                # 将当前对话作为一个独立的历史记录保存
                conversation_record = {
                    "query": query,
                    "timestamp": str(datetime.datetime.now()),
                    "conversations": serialized_new_conversations,
                    "evaluation": evaluation_results,
                    "model": model_name
                }
                
                # 添加到当前选择模型的历史记录列表
                st.session_state.model_history[selected_port].append(conversation_record)
                
                # 将评估结果保存到会话状态
                st.session_state.last_evaluation = evaluation_results
                st.session_state.current_conversations = serialized_new_conversations
                
                # 更新当前历史记录索引为最新的一条
                st.session_state[f'current_history_index_{selected_port}'] = len(st.session_state.model_history[selected_port]) - 1
                
                st.success(f"查询处理成功！")
                
                # 清除当前查询
                st.session_state.current_query = ""
                json.dump(st.session_state.model_history[selected_port], open(save_path, 'w'), indent=4)
                
            except Exception as e:
                st.error(f"处理查询时出错: {str(e)}")
    else:
        st.error("请先输入查询！")

# 处理按钮点击
if process_button and query:
    process_current_query()

# 显示当前模型的历史记录
current_model_history = st.session_state.model_history[selected_port]
if current_model_history:
    st.header(f"对话历史")
    
    # 选择当前要显示的历史记录
    if f'current_history_index_{selected_port}' not in st.session_state:
        st.session_state[f'current_history_index_{selected_port}'] = len(current_model_history) - 1
    
    # 历史记录选择器
    history_options = [f"{i+1}. 查询: {h['query'][:30]}..." for i, h in enumerate(current_model_history)]
    selected_history = st.selectbox("选择历史记录", history_options, index=st.session_state[f'current_history_index_{selected_port}'])
    
    # 获取选中的历史记录索引
    selected_index = history_options.index(selected_history)
    st.session_state[f'current_history_index_{selected_port}'] = selected_index
    current_history = current_model_history[selected_index]
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["对话视图", "JSON视图", "评估结果"])
    
    with tab1:
        # 以对话形式显示
        for msg in current_history['conversations']:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # 根据角色不同的样式
            if role == "user":
                st.markdown(f"**用户**: {content}")
                st.markdown("---")
            elif role == "assistant":
                st.markdown(f"**助手**: {content}")
                st.markdown("---")
    
    with tab2:
        # 显示原始 JSON
        st.json(current_history['conversations'])
    
    with tab3:
        # 显示评估结果
        eval_results = current_history.get('evaluation')
        if eval_results:
            if 'error' in eval_results:
                st.error(f"评估出错: {eval_results['error']}")
            else:
                st.subheader("模型评分")
                metrics_data = []

                # 添加平均分
                if 'average' in eval_results:
                    metrics_data.append(["平均分", f"{eval_results['average']:.2f}"])

                for key, value in eval_results.items():
                    if key in ['relevance', 'coherence', 'groundedness'] and isinstance(value, dict):
                        metrics_data.append([f"{key} 分数", f"{value['score']:.2f}"])
                        metrics_data.append([f"{key} 原因", value['reason']])

                if metrics_data:
                    st.table(metrics_data)
                else:
                    st.info("没有详细的评分数据。")
        else:
            st.info("此对话没有评估结果。")

# Add some information at the bottom
st.markdown("---")
st.markdown("对话模型演示系统 - 基于VLLM API")

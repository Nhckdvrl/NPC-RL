import sys
import time
import openai
import json
from typing import List, Dict, Any, Tuple, Optional
import os

# Configuration and utility modules
from . import config
from . import message_constructor
from . import llm_invoker
from . import interaction_logger
from .logger import llm_logger

# Attempt to import RAG initialization with relative paths
try:
    from ..rag.rag_module import initialize_rag_system
except ImportError:
    # Fallback for different execution contexts if needed
    from agents.rag.rag_module import initialize_rag_system


def expend_description(generated_functions: List[Dict[str, Any]], tools_for_llm: List[Dict[str, Any]]) -> str:
    """
    将生成的函数调用与工具描述信息结合，生成更详细的描述字符串。
    
    Args:
        generated_functions: 生成的函数调用列表
        tools_for_llm: 提供给LLM的工具列表，包含描述信息
        
    Returns:
        str: 包含函数名称、参数和描述的格式化字符串
    """
    if not generated_functions:
        return None
        
    # 创建工具名称到描述的映射
    tool_descriptions = {}
    for tool in tools_for_llm:
        if tool.get("type") == "function" and "function" in tool:
            function_info = tool["function"]
            tool_descriptions[function_info.get("name")] = function_info.get("description", "")
    
    result = []
    function_names = []
    for func in generated_functions:
        function_name = func.get("name", "")
        function_names.append(function_name)
    
    for function_name in list(set(function_names)):
        description = tool_descriptions.get(function_name, "").split("Parameters:")[0]
        if len(description) > 200:
            description = description[:200] + "..."
        result.append(f"Function: {function_name}\nDescription: {description}")
    
    return "\n".join(result)

class OpenAIAgent(object):
    def __init__(self):
        """Initialize the refactored OpenAI agent."""
        # Initialize RAG system (if enabled and available)
        if config.USE_RAG: # Check if RAG is globally enabled
            try:
                initialize_rag_system()
                if config.DEBUG_MODE:
                    print("OpenAIAgent INFO: RAG system initialized successfully.", file=sys.stdout)
            except Exception as e:
                print(f"OpenAIAgent WARNING: Failed to initialize RAG system: {e}", file=sys.stderr)

    def generate_functions_and_responses(
        self,
        tool_registry: List[Dict],
        action_registry: List[Dict],
        worldview: str,
        persona: Dict[str, str],
        role: str,
        knowledge: Dict[str, Any],
        state: Dict[str, str],
        dialogue: List[Dict[str, str]],
        executor: Any, # Executor object with an 'execute' method
        exam_id: str = None
    ) -> Dict[str, Any]:
        """
        Orchestrates the process of function calling and response generation.
        """
        start_time = time.time()
        # Create new log entry
        log_index = llm_logger.current_index
        if os.getenv("IS_LOCAL", "0")=="1":
            TOTAL_TIMEOUT_LIMIT = 9999  # seconds
        else:
            TOTAL_TIMEOUT_LIMIT = 6.9  # seconds
        GENERIC_TIMEOUT_RESPONSE = "Hmm, it seems I missed part of what you said—could you repeat that for me?"

        final_response_content = GENERIC_TIMEOUT_RESPONSE
        generated_functions = []
        function_execution_results = []
        fn_call_messages = [] # Initialize for logging in case of early timeout
        dialogue_gen_messages = [] # Initialize for logging
        fn_gen_usage_info = None
        dialogue_gen_usage_info = None
        raw_tool_calls_dialogue = None

        try:
            # 1. Create messages for function calling
            tools_for_llm, fn_call_messages = message_constructor.create_messages_for_function_calling(
                tool_functions=tool_registry, 
                action_functions=action_registry, 
                dialogue=dialogue,
                worldview=worldview,
                persona=persona,
                role=role,
                knowledge=knowledge,
            )

            # 2. Invoke LLM for function calling
            time_elapsed = time.time() - start_time
            remaining_time_fn_llm = TOTAL_TIMEOUT_LIMIT - time_elapsed
            if remaining_time_fn_llm <= 0:
                print(f"OpenAIAgent TIMEOUT: Exceeded {TOTAL_TIMEOUT_LIMIT}s before LLM function call.", file=sys.stderr)
                raise openai.APITimeoutError("Timeout before LLM function call")
            if dialogue[-1]["target_item"] and dialogue[-1]["target_item"][0]["name"]:
                target_item = dialogue[-1]["target_item"][0]["name"]
            else:
                target_item = None    
            item_names = [
                'avis wind', 'battle axe', 'bloody saint cross', 'demonic sword', 'dragon cutter',
                'heart breaker', 'hell chain', "hunter's bow", 'lance', 'light-weight durable knife',
                'long bow', 'long sword', 'mace', 'man gauche', 'morning star', 'rope',
                'rose battle whip', 'short sword', 'trident', 'two-handed sword', 'war hammer',
                'zephyr', 'zwei hander'
            ]
            item_types = [
                'sword', 'bow', 'axe', 'blunt weapon', 'single-handed sword', 'single-handed sword',
                'axe', 'whip', 'bow', 'spear', 'single-handed sword', 'bow', 'single-handed sword',
                'blunt weapon', 'single-handed sword', 'blunt weapon', 'whip', 'whip',
                'single-handed sword', 'spear', 'double-handed sword', 'axe', 'spear', 'double-handed sword'
            ] # 'sword'很重要

            # 添加动态扩展的值
            if isinstance(knowledge.get("knowledge_info"), list):
                for item in knowledge["knowledge_info"]:
                    if isinstance(item, dict):
                        item_name = item.get("name", "ERROR").lower()
                        item_type = item.get("type", "ERROR").lower()
                        item_names.append(item_name)
                        item_types.append(item_type)

            # 去重后转换为列表
            item_names = list(set(item_names))
            item_types = list(set(item_types))

            result = llm_invoker.invoke_function_calling_llm(
                messages=fn_call_messages, 
                tools=tools_for_llm,
                request_timeout_seconds=remaining_time_fn_llm,
                log_index=log_index,
                target_item=target_item,
                item_names=item_names,
                item_types=item_types,
            )
            
            # Handle the case when result is None
            if result is None:
                generated_functions, fn_gen_usage_info = [], None
            else:
                generated_functions, fn_gen_usage_info = result
                
            generated_functions = generated_functions if generated_functions is not None else []
            # 3. Execute generated functions
            # Executor is assumed to be fast and not part of the LLM timeout budget here.
            # If executor can also be slow, its timing needs to be managed.
            if config.USE_GOLD_FUNCTIONS_FOR_ROLEPLAY: # 本地运行时，使用gold_functions
                function_execution_results = executor.execute(executor.gold_functions)
                if executor.gold_functions:
                    generated_functions_with_tool_description = expend_description(executor.gold_functions, tools_for_llm)
                else:
                    generated_functions_with_tool_description = None
            else:
                function_execution_results = executor.execute(generated_functions)
                if generated_functions:
                    generated_functions_with_tool_description = expend_description(generated_functions, tools_for_llm)
                else:
                    generated_functions_with_tool_description = None
            
            # 4. Create messages for dialogue generation
            time_elapsed = time.time() - start_time
            remaining_time_dialogue_msg = TOTAL_TIMEOUT_LIMIT - time_elapsed
            if remaining_time_dialogue_msg <= 0:
                print(f"OpenAIAgent TIMEOUT: Exceeded {TOTAL_TIMEOUT_LIMIT}s before dialogue message creation.", file=sys.stderr)
                raise openai.APITimeoutError("Timeout before dialogue message creation")

            dialogue_gen_messages = message_constructor.create_messages_for_dialogue_generation(
                worldview=worldview, 
                persona=persona, 
                role=role, 
                knowledge=knowledge, 
                state=state, 
                dialogue=dialogue, 
                function_results=function_execution_results,
                action=generated_functions_with_tool_description
            )

            # 5. Invoke LLM for dialogue generation
            time_elapsed = time.time() - start_time
            remaining_time_dialogue_llm = TOTAL_TIMEOUT_LIMIT - time_elapsed
            if remaining_time_dialogue_llm <= 0:
                print(f"OpenAIAgent TIMEOUT: Exceeded {TOTAL_TIMEOUT_LIMIT}s before LLM dialogue generation.", file=sys.stderr)
                raise openai.APITimeoutError("Timeout before LLM dialogue generation")

            final_response_content, dialogue_gen_usage_info, raw_tool_calls_dialogue = llm_invoker.invoke_dialogue_generation_llm(
                messages=dialogue_gen_messages,
                request_timeout_seconds=remaining_time_dialogue_llm,
                log_index=log_index
            )
            
            # Log the response
            llm_logger.log_response(log_index, final_response_content, 1)
            # If LLM call was successful and didn't timeout, use its response
            if final_response_content is not None:
                final_response_content = final_response_content
            # If it returned None (e.g. API error but not timeout), keep generic or previous response

        except openai.APITimeoutError as e:
            print(f"OpenAIAgent TIMEOUT: Operation exceeded {TOTAL_TIMEOUT_LIMIT}s. Details: {e}", file=sys.stderr)
            final_response_content = GENERIC_TIMEOUT_RESPONSE
            # Potentially clear generated_functions if timeout occurred during their generation or execution
            # For simplicity, we keep whatever was generated before timeout for logging.
        except Exception as e:
            print(f"OpenAIAgent ERROR: An unexpected error occurred: {e}", file=sys.stderr)
            final_response_content = GENERIC_TIMEOUT_RESPONSE # Fallback for other errors too
            # Clear generated_functions on other critical errors as well
            generated_functions = [] 

        # 6. Log interaction details if in debug mode
        if config.DEBUG_MODE:
            interaction_logger.log_interaction(
                exam_id=exam_id,
                function_gen_messages=fn_call_messages,
                generated_functions=generated_functions,
                function_execution_results=function_execution_results,
                dialogue_gen_messages=dialogue_gen_messages,
                final_response_content=final_response_content,
                raw_tool_calls_from_dialogue_response=raw_tool_calls_dialogue,
                fn_gen_usage_info=fn_gen_usage_info,
                dialogue_gen_usage_info=dialogue_gen_usage_info
            )
        return {
            'prompts': 'Placeholder', 
            "final_responses": final_response_content,
            # "functions": generated_functions # Keep this commented as per user's last state
        }

# Example of how this agent might be instantiated and used (for testing/illustration)
# if __name__ == '__main__':
#     # This is a placeholder for actual registries, executor, and inputs
#     mock_tool_registry = [] 
#     mock_action_registry = []
#     mock_worldview = "A fantasy world."
#     mock_persona = {"name": "Guard", "mood": "stern"}
#     mock_role = "Guard of the city gates."
#     mock_knowledge = {"general_info": "The city is currently peaceful.", "knowledge_info": []}
#     mock_state = {}
#     mock_dialogue = [{"speaker": "Player", "utterance": "Hello there!", "text": "Hello there!"}]
    
#     class MockExecutor:
#         def execute(self, functions_to_call):
#             print(f"Executor called with: {functions_to_call}")
#             return [{ "name": func['name'], "parameters": func['parameters'], "return": [{"status": "executed"}]} for func in functions_to_call]

#     agent = OpenAIAgent()
#     result = agent.generate_functions_and_responses(
#         tool_registry=mock_tool_registry,
#         action_registry=mock_action_registry,
#         worldview=mock_worldview,
#         persona=mock_persona,
#         role=mock_role,
#         knowledge=mock_knowledge,
#         state=mock_state,
#         dialogue=mock_dialogue,
#         executor=MockExecutor(),
#         exam_id="test_001"
#     )
#     print(f"Final Result: {result}")

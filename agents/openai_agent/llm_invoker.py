import json
import os
import openai
from typing import List, Dict, Any, Tuple, Optional

from .logger import llm_logger

from . import config

# Global variable to hold the singleton OpenAI client instance
_openai_client = None
_function_schema = None

def _load_function_schema():
    """Loads the function schema from the JSON file."""
    global _function_schema
    if _function_schema is None:
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'function_schema.json')
            with open(schema_path, 'r', encoding='utf-8') as f:
                _function_schema = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"LLMInvoker ERROR: Could not load or parse function_schema.json: {e}")
            _function_schema = {}
    return _function_schema

def _check_and_correct_args_types(func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Checks and corrects argument types and keys, including special handling for known inconsistencies."""
    schema = _load_function_schema()
    corrected_args = args.copy()

    # --- Start of specific patches for known inconsistencies ---

    # Patch for 'sell' function: correct key 'item_names' to 'item_name' and handle nested dicts
    if func_name == 'sell' and 'item_names' in corrected_args:
        item_list = corrected_args.pop('item_names')  # Use pop to get value and remove old key
        if item_list and isinstance(item_list, list) and len(item_list) > 0 and isinstance(item_list[0], dict):
            new_item_list = [item.get('item_name') for item in item_list if isinstance(item, dict) and item.get('item_name')]
            if new_item_list:
                corrected_args['item_name'] = new_item_list
        else:
            corrected_args['item_name'] = item_list

    # Patch for operator arguments: map symbols to natural language and filter invalid values
    if isinstance(corrected_args, dict):
        # First, filter out empty/None/n/a values
        corrected_args = {
            k: v for k, v in corrected_args.items()
            if v is not None and str(v).strip() != "" and str(v).strip().lower() != "n/a"
        }

        # Then, process operators
        operator_mapping = {
            ">": "more than", "<": "less than", "<=": "or less",
            "less": "less than", "more": "more than", ">=": "or more",
            "no limit": "no limit", "below": "less than", "above": "more than",
            "=": None  # Used for identification and subsequent deletion
        }
        for k in list(corrected_args.keys()):
            if "_operator" in k:
                v = corrected_args.get(k)
                if v == "=":
                    corrected_args.pop(k)
                elif v in operator_mapping:
                    corrected_args[k] = operator_mapping[v]
                
                # Remove any operator that is not in the allowed list after mapping
                allowed_operators = ["more than", "less than", "or less", "or more", "about", "no limit", "highest", "high", "average", "low", "lowest", "other than"]
                if k in corrected_args and corrected_args[k] not in allowed_operators:
                    corrected_args.pop(k)

    # --- End of specific patches ---

    # --- Generic Type Correction based on schema ---
    if not schema or func_name not in schema:
        return corrected_args

    func_params = schema.get(func_name, {})

    for param_name, param_info in func_params.items():
        if param_name in corrected_args:
            expected_type_str = param_info.get("type")
            current_value = corrected_args[param_name]

            # 如果期望类型是 List[str]，但当前不是 list，则包装为 list
            if expected_type_str == "List[str]" and not isinstance(current_value, list):
                if isinstance(current_value, str):
                    corrected_args[param_name] = [current_value]
                elif isinstance(current_value, dict):
                    # 展开 dict 的 value 转为 list[str]
                    corrected_args[param_name] = [str(v) for v in current_value.values()]
                else:
                    # 其他类型统一转为单元素 list[str]
                    corrected_args[param_name] = [str(current_value)]

            # 如果期望类型是 str，但当前不是 str
            elif expected_type_str == "str" and not isinstance(current_value, str):
                if isinstance(current_value, dict):
                    # 展开 dict 的 value 后拼成空格分隔的字符串
                    corrected_args[param_name] = " ".join(str(v) for v in current_value.values())
                elif isinstance(current_value, list):
                    # list 中的每一项转 str 后拼接
                    corrected_args[param_name] = " ".join(str(item) for item in current_value)
                else:
                    # 对于 int、float 等直接转换
                    corrected_args[param_name] = str(current_value)

            # 可以添加其他类型的转换，例如 int、float 等
            elif expected_type_str == "int" and not isinstance(current_value, int):
                try:
                    corrected_args[param_name] = int(current_value)
                except (ValueError, TypeError):
                    pass  # 或记录警告日志

            elif expected_type_str == "float" and not isinstance(current_value, float):
                try:
                    corrected_args[param_name] = float(current_value)
                except (ValueError, TypeError):
                    pass
    
    # Trick: Remove operator keys if their corresponding base keys don't exist
    keys_to_remove = []
    for key in corrected_args.keys():
        if key.endswith("_operator"):
            base_key = key[:-9]  # Remove "_operator" suffix
            if base_key not in corrected_args:
                keys_to_remove.append(key)
    
    # Remove the identified keys
    for key in keys_to_remove:
        corrected_args.pop(key)

    return corrected_args

def _infer_missing_operators_from_query(user_query: str, func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infers missing operators by looking for them in the user query near the value.
    e.g., query="...20G or more", args={"quest_reward": "20G"} -> adds {"quest_reward_operator": "or more"}
    """
    if func_name not in ["search_quest", "search_item"] or not user_query:
        return args

    operators = sorted([
        "more than", "less than", "or less", "or more", "about", "no limit",
        "highest", "high", "average", "low", "lowest", "other than",
        "or above", "or below", "longest", "long", "short", "shortest",
        "easiest", "most difficult", "difficult"
    ], key=len, reverse=True)

    corrected_args = args.copy()
    query_lower = user_query.lower()

    for key in list(corrected_args.keys()):
        if key.endswith("_operator"):
            continue

        operator_key = f"{key}_operator"
        if operator_key in corrected_args:
            continue

        value = corrected_args.get(key)
        if not isinstance(value, str) or not value:
            continue

        value_lower = value.lower()
        
        # Simple check to see if the value is in the query
        if value_lower not in query_lower:
            continue

        # Search for operator patterns near the value in the query
        # We create a small window around the value's position to look for operators
        try:
            start_index = query_lower.index(value_lower)
            # Look in a window of text around the value for an operator
            # Window size can be tuned, e.g., 20 chars before and after
            window_start = max(0, start_index - 20)
            window_end = min(len(query_lower), start_index + len(value_lower) + 20)
            search_window = query_lower[window_start:window_end]

            for op in operators:
                # Check for patterns like "op value" or "value op"
                if f"{op} {value_lower}" in search_window or f"{value_lower} {op}" in search_window:
                    corrected_args[operator_key] = op
                    break
                # Check for patterns without space, e.g., "over15" (if value is "15")
                if f"{op}{value_lower}" in search_window or f"{value_lower}{op}" in search_window:
                    corrected_args[operator_key] = op
                    break
        except ValueError:
            # Value not found in query, continue to next key
            continue
    
    return corrected_args

def _handle_operator_splitting(func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Splits combined operator and value strings into separate arguments.
    e.g., {"quest_duration": "about 10 days"} -> {"quest_duration": "10 days", "quest_duration_operator": "about"}
    """
    if func_name not in ["search_quest", "search_item"]:
        return args

    # A list of operators that might be combined with values.
    # Sorted by length descending to match longer operators first (e.g., "more than" before "more").
    operators = sorted([
        "more than", "less than", "or less", "or more", "about", "no limit", 
        "highest", "high", "average", "low", "lowest", "other than",
        "or above", "or below", "longest", "long", "short", "shortest", 
        "easiest", "most difficult", "difficult"
    ], key=len, reverse=True)

    corrected_args = args.copy()
    
    # Iterate over a copy of keys as the dictionary might be modified
    for key in list(corrected_args.keys()):
        # We only care about base keys, not keys that are already operators
        if key.endswith("_operator"):
            continue

        value = corrected_args.get(key)
        if not isinstance(value, str):
            continue

        # Check for each operator
        for op in operators:
            # Check if value starts with operator
            if value.lower().startswith(op + ' '):
                # Extract value and update args
                new_value = value[len(op) + 1:].strip()
                operator_key = f"{key}_operator"
                
                # Update only if the operator key is not already set
                if operator_key not in corrected_args:
                    corrected_args[key] = new_value
                    corrected_args[operator_key] = op
                    break # Move to the next key
            
            # Check if value ends with operator
            elif value.lower().endswith(' ' + op):
                # Extract value and update args
                new_value = value[:-len(op) - 1].strip()
                operator_key = f"{key}_operator"

                # Update only if the operator key is not already set
                if operator_key not in corrected_args:
                    corrected_args[key] = new_value
                    corrected_args[operator_key] = op
                    break # Move to the next key

    return corrected_args

def _get_client():
    """Initializes and returns the singleton OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        if os.getenv("IS_LOCAL", "false") in ["1", "true", "yes"] and config.get_openai_api_key() != "123":
            from openai import AzureOpenAI as OpenAI
            _openai_client = OpenAI(
                azure_endpoint=config.get_openai_base_url(),
                api_key=config.get_openai_api_key(),
                api_version="2025-01-01-preview",
            )
        else:
            from openai import OpenAI
            _openai_client = OpenAI(
                api_key=config.get_openai_api_key(),
                base_url=config.get_openai_base_url()
            )
    return _openai_client


def _calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates the estimated cost based on token counts."""
    # Cost per 1M tokens (example, replace with actuals if different or model-specific)
    # GPT-4o-mini pricing: $0.15 / 1M input tokens, $0.60 / 1M output tokens (as of May 2024)
    cost_input_per_million = 0.15 
    cost_output_per_million = 0.60
    
    cost = (prompt_tokens / 1_000_000 * cost_input_per_million) + \
           (completion_tokens / 1_000_000 * cost_output_per_million)
    return cost

def invoke_function_calling_llm(messages: List[Dict], tools: List[Dict], request_timeout_seconds: float, log_index: int, target_item: str=None, item_names: List[str]=None, item_types: List[str]=None) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Invokes the LLM for function calling, parses tool calls, and returns them along with usage info.
    Returns an empty list for functions if no tools are provided, IS_TASK2 is true, or an error occurs.
    """
    all_functions = []
    usage_info = None

    if not tools or config.get_if_is_task2():
        return all_functions, usage_info

    try:
        # Log stage 0 data
        llm_logger.log_stage_0(log_index, messages, tools, [], [])
        if config.get_openai_api_key() == "123":
            response = _get_client().chat.completions.create(
                model=config.get_openai_model(),
                messages=messages,
                tools=tools,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                max_tokens=config.MAX_TOKENS_FUNCTION_CALL,
                stream=False,
                tool_choice="auto",
                timeout=request_timeout_seconds,
                extra_body={
                    "repetition_penalty": 1.05,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
        else:
            response = _get_client().chat.completions.create(
                model=config.get_openai_model(),
                messages=messages,
                tools=tools,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                max_tokens=config.MAX_TOKENS_FUNCTION_CALL,
                stream=False,
                tool_choice="auto",
                timeout=request_timeout_seconds,
            )
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            cost = _calculate_cost(prompt_tokens, completion_tokens)
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd_estimate": cost
            }
            # Log stage 1 data
            llm_logger.log_stage_1(log_index, messages, [])
        # print(response.choices)
        all_query = ""
        for msg in messages:
            if msg.get("role") == "user":
                all_query += "\n" * 20 + msg["content"]
        if response.choices and response.choices[0].message.tool_calls:
            checked_basic_info_items = set()
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.type != "function":
                    continue
                try:
                    # trick 1: 去除无效的参数
                    try:
                        args = json.loads(tool_call.function.arguments)

                        # trick 0: 修正类型和值
                        name = tool_call.function.name
                        args = _check_and_correct_args_types(name, args)

                        # trick 4: 分割操作符和值 (e.g., "about 10 days")
                        args = _handle_operator_splitting(name, args)

                        # trick 5: 从用户查询中推断缺失的操作符
                        args = _infer_missing_operators_from_query(all_query, name, args)

                        if isinstance(args, str):
                            # 如果解析后仍然是字符串，尝试再次解析
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                # 如果仍然无法解析，则作为普通字符串处理
                                args = {"value": args}


                    except json.JSONDecodeError as e:
                        args = {"raw_arguments": tool_call.function.arguments}

                    # trick 2: 去除冗余的check函数调用
                    name = tool_call.function.name
                    # 如果是 check_basic_info，记录 item_name 或 quest_name
                    if name == "check_basic_info":
                        if "quest_name" in args:
                            checked_basic_info_items.add(args["quest_name"])
                        if "item_name" in args:
                            checked_basic_info_items.add(args["item_name"])
                    
                    func_call = {
                        'name': name,
                        'parameters': args
                    }
                    all_functions.append(func_call)
                        # Log the function call response

                except json.JSONDecodeError as e:
                    print(f"LLMInvoker ERROR: Failed to parse function arguments for {name}: {tool_call.function.arguments}. Error: {e}")
                    continue # Skip this tool call
            try:
                # 处理冗余的check函数调用
                for func_call in all_functions.copy():
                    redundant_checks = {"check_price", "check_type", "check_attack", "check_level", "check_duration", "check_reward", "check_description"}
                    name = func_call.get("name")
                    params = func_call.get("parameters", {})
                    
                    # 检查是否是冗余函数
                    if name in redundant_checks:
                        # 检查item_name (物品相关)
                        item_name = params.get("item_name")
                        if item_name and item_name in checked_basic_info_items:
                            all_functions.remove(func_call)
                            continue
                            
                        # 检查quest_name (任务相关)
                        quest_name = params.get("quest_name")
                        if quest_name and quest_name in checked_basic_info_items:
                            all_functions.remove(func_call)
                            continue
                
                # 合并多个check函数为一个check_basic_info
                # 按照item_name和quest_name分组收集所有check函数
                item_checks = {}
                quest_checks = {}
                
                for func_call in all_functions.copy():
                    name = func_call.get("name")
                    params = func_call.get("parameters", {})
                    
                    if name in {"check_price", "check_type", "check_attack", "check_level", "check_duration", "check_reward", "check_description"}:
                        # 收集物品相关的check
                        item_name = params.get("item_name")
                        if item_name:
                            if item_name not in item_checks:
                                item_checks[item_name] = []
                            item_checks[item_name].append(func_call)
                        
                        # 收集任务相关的check
                        quest_name = params.get("quest_name")
                        if quest_name:
                            if quest_name not in quest_checks:
                                quest_checks[quest_name] = []
                            quest_checks[quest_name].append(func_call)
                
                # 如果同一个物品/任务有多个check函数，合并为一个check_basic_info
                for item_name, checks in item_checks.items():
                    if len(checks) >= 2:  # 至少有2个check函数才需要合并
                        print(f"Consolidating {len(checks)} check functions for item '{item_name}' into check_basic_info")
                        # 从all_functions中移除这些check函数
                        for check in checks:
                            if check in all_functions:
                                all_functions.remove(check)
                        
                        # 添加一个新的check_basic_info函数
                        all_functions.append({
                            'name': 'check_basic_info',
                            'parameters': {'item_name': item_name}
                        })
                
                for quest_name, checks in quest_checks.items():
                    if len(checks) >= 2:  # 至少有2个check函数才需要合并
                        print(f"Consolidating {len(checks)} check functions for quest '{quest_name}' into check_basic_info")
                        # 从all_functions中移除这些check函数
                        for check in checks:
                            if check in all_functions:
                                all_functions.remove(check)
                        
                        # 添加一个新的check_basic_info函数
                        all_functions.append({
                            'name': 'check_basic_info',
                            'parameters': {'quest_name': quest_name}
                        })
                            
                # 处理冗余的select函数调用（当select和start都存在且参数相同时）
                select_functions = [f for f in all_functions if f.get('name') == 'select']
                start_functions = [f for f in all_functions if f.get('name') == 'start']
                
                if select_functions and start_functions:
                    for select_func in select_functions.copy():
                        select_params = select_func.get('parameters', {})
                        select_quest = select_params.get('quest_name')
                        
                        if select_quest:
                            # 检查是否有相同quest_name的start函数
                            for start_func in start_functions:
                                start_params = start_func.get('parameters', {})
                                if select_quest == start_params.get('quest_name'):
                                    # 找到相同参数的start函数，移除select函数
                                    if select_func in all_functions:
                                        all_functions.remove(select_func)
                                    break
                # 获取所有 sell 函数
                sell_functions = [f for f in all_functions if f.get('name') == 'sell']
                if sell_functions:
                    # 收集所有 equip 的 item_name（小写）
                    equipped_items = {
                        f.get('parameters', {}).get('item_name', '').lower()
                        for f in all_functions
                        if f.get('name') == 'equip' and f.get('parameters', {}).get('item_name')
                    }

                    # 移除任何包含已装备物品的 sell 函数
                    def is_valid_sell(f):
                        if f.get('name') != 'sell':
                            return True
                        item_names = f.get('parameters', {}).get('item_name')
                        if isinstance(item_names, list):
                            return all(name.lower() not in equipped_items for name in item_names if isinstance(name, str))
                        elif isinstance(item_names, str):
                            return item_names.lower() not in equipped_items
                        else:
                            return True  # unexpected type, keep for safety

                    original_count = len(all_functions)
                    all_functions = [f for f in all_functions if is_valid_sell(f)]
                    removed_count = original_count - len(all_functions)
                # 修正target_item：确保所有函数调用中与物品相关的参数都使用正确的target_item
                if target_item:
                    print(f"Applying target_item correction: {target_item}")
                    for f in all_functions:
                        # 检查是否有parameters
                        if not f.get("parameters"):
                            continue
                            
                        # 检查是否是与物品相关的函数
                        name = f.get("name")
                        if name in {"check_basic_info", "check_description", "check_price", "check_type", "check_attack", 
                                  "equip", "sell"}:
                            # 修正item_name参数
                            if "item_name" in f["parameters"]:
                                original_item = f["parameters"]["item_name"]
                                f["parameters"]["item_name"] = target_item
                                print(f"  Corrected: {name} item_name from '{original_item}' to '{target_item}'")
                                
                        # # 特殊处理search_item函数，可能有多个物品名称
                        # elif name == "search_item":
                        #     if "item_name" in f["parameters"] and f["parameters"]["item_name"]:
                        #         # 保留search_item中的item_name，因为它可能是搜索条件而不是目标物品
                for f in all_functions:
                    if f["name"] == "search_item":
                        if f["parameters"].get("item_name"):
                            item_name_list = [name.strip().lower() for name in f["parameters"]["item_name"].split("|")]
                            valid_names = [name for name in item_name_list if name in item_names]
                            if valid_names:
                                f["parameters"]["item_name"] = "|".join(valid_names)
                            else:
                                if "item_name_operator" in f["parameters"]:
                                    f["parameters"].pop("item_name_operator")
                                f["parameters"].pop("item_name")

                        if f["parameters"].get("item_type"):
                            item_type_value = f["parameters"]["item_type"].strip().lower()
                            if item_type_value not in item_types:
                                f["parameters"].pop("item_type")

            except Exception as e:
                print(f"LLMInvoker ERROR: Failed to remove sell functions for equipped items: {e}")
        llm_logger.log_tool(log_index, all_functions)
        llm_logger.log_response(log_index, response.choices[0].message.content, 0)
    # except openai.APITimeoutError: # Explicitly re-raise APITimeoutError
    #     raise
    except Exception as e:
        print(f"LLMInvoker ERROR: OpenAI API call for function generation failed: {e}")
        # In case of an API error, return empty lists for tool calls and None for usage
        return [], None
    # print("all_functions: ", all_functions)
    return all_functions, usage_info
def invoke_dialogue_generation_llm(messages: List[Dict], request_timeout_seconds: float, log_index: int) -> Tuple[str, Optional[Dict], Optional[List[Dict]]]:
    """
    Invokes the LLM for dialogue generation and returns the content and usage info.
    Also returns the raw tool_calls from the response for logging purposes.
    """
    response_content = "I'm so happy to help you. You can give me more details so I can better serve you."
        
    usage_info = None
    raw_tool_calls_for_log = [] # For logging the full tool_call structure if present

    try:
        # Log stage 1 initial data
        llm_logger.log_stage_1(log_index, messages, [])
        if config.get_openai_api_key() == "123":
            response = _get_client().chat.completions.create(
                model=config.get_openai_model(),
                messages=messages,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                max_tokens=config.MAX_TOKENS_RESPONSE_GENERATION,
                stream=False,
                timeout=request_timeout_seconds,
                extra_body={
                    "repetition_penalty": 1.05,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
        else:
            response = _get_client().chat.completions.create(
                model=config.get_openai_model(),
                messages=messages,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                max_tokens=config.MAX_TOKENS_RESPONSE_GENERATION,
                stream=False,
                timeout=request_timeout_seconds,
            )

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            cost = _calculate_cost(prompt_tokens, completion_tokens)
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd_estimate": cost
            }
        
        if response.choices and response.choices[0].message:
            response_content = response.choices[0].message.content
            # Capture raw tool_calls if any (though not expected for pure dialogue generation)
            if response.choices[0].message.tool_calls:
                for tc in response.choices[0].message.tool_calls:
                    raw_tool_calls_for_log.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })

    # except openai.APITimeoutError: # Explicitly re-raise APITimeoutError
    #     raise
    except Exception as e:
        print(f"LLMInvoker ERROR: OpenAI API call for dialogue generation failed: {e}")
        # Return a generic roleplay message and None for usage info in case of an API error
        fallback_response = "I'm so happy to help you. You can give me more details so I can better serve you."
        print(f"LLMInvoker FALLBACK: Returning generic response due to API error: {fallback_response}")
        return fallback_response, None, []  # Always return a tuple of three values
        
    llm_logger.log_response(log_index, response_content, 1)
    # print("Response: ", response_content)
    return response_content, usage_info, raw_tool_calls_for_log

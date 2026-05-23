import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# 使用绝对导入路径
from eval.tool_pattern.modules.pattern_extractor import PatternExtractor
from eval.tool_pattern.modules.context_manager import ContextManager
from eval.tool_pattern.modules.function_builder import FunctionBuilder
from eval.tool_pattern.modules.matcher_helper import MatcherHelper

class ToolPatternMatcher:

    def __init__(self, patterns_file: Optional[str] = None):
        """
        初始化工具模式匹配器
        
        Args:
            patterns_file: 可选的模式文件路径，包含预先分析的模式数据
        """
        # 初始化组件
        self.pattern_extractor = PatternExtractor()
        self.context_manager = ContextManager()
        self.function_builder = FunctionBuilder(self.pattern_extractor)
        # 注意：将function_builder传递给matcher_helper
        self.matcher_helper = MatcherHelper(function_builder=self.function_builder)
        
        # 加载预定义模式（如果提供）
        self.patterns = {}
        if patterns_file:
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    self.patterns = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load patterns file: {e}")
    
    def match_functions(self, query: str, target_items: List[Dict] = None) -> List[Dict]:
        """
        根据查询和目标物品匹配可能的一个或多个函数调用
        
        Args:
            query: 用户查询文本
            target_items: 目标物品列表，每个物品是包含name的字典
            
        Returns:
            包含name和parameters的函数调用字典列表
        """
        # 预处理查询
        query = query.strip()
        
        # 处理空查询
        if not query:
            return []
        
        # 检测查询意图
        intents = self.matcher_helper._detect_query_intent(query)
        
        # 提取物品/任务名称
        item_name = self.pattern_extractor.extract_item_name(query)
        quest_name = self.pattern_extractor.extract_quest_name(query)
        
        # 如果有目标物品，使用目标物品名称
        if target_items and len(target_items) > 0 and not item_name:
            item_name = target_items[0].get('name', '')
        
        # 检查是否是比较查询（涉及多个物品）
        if self.pattern_extractor.is_comparison_query(query):
            all_items = self.pattern_extractor.extract_all_item_names(query)
            all_quests = self.pattern_extractor.extract_all_quest_names(query)
            
            if len(all_items) > 1:
                return self.matcher_helper._handle_multi_item_comparison(all_items, intents)
            elif len(all_quests) > 1:
                return self.matcher_helper._handle_multi_quest_comparison(all_quests, intents)
        
        # 处理确认语句 - 优先级最高
        if intents['is_confirmation']:
            return [self.function_builder.build_confirm_call("")]
        
        # 处理开始/接受任务意图
        if intents['is_start_query']:
            if quest_name:
                return [self.function_builder.build_start_call(quest_name)]
            # 如果没有指定任务名称，但有开始意图，尝试从上下文获取任务名称
            context_quest = self.context_manager.get_last_mentioned_quest()
            if context_quest:
                return [self.function_builder.build_start_call(context_quest)]
            return [] # Was: self.function_builder.build_search_quest_call(query)
        
        # 处理选择意图
        if intents['is_select_intent']:
            if item_name:
                return [self.function_builder.build_select_call(item_name)]
            if quest_name:
                return [self.function_builder.build_select_call(quest_name)]
            # 如果没有指定物品或任务名称，但有选择意图，尝试从上下文获取
            context_item = self.context_manager.get_last_mentioned_item()
            if context_item:
                return [self.function_builder.build_select_call(context_item)]
            context_quest = self.context_manager.get_last_mentioned_quest()
            if context_quest:
                return [self.function_builder.build_select_call(context_quest)]
            return [self.function_builder.build_select_request_confirm_call()]
        
        # 处理销售意图
        if intents['is_sell_intent']:
            if item_name:
                return [self.function_builder.build_sell_call(item_name)]
            # 如果没有指定物品名称，但有销售意图，尝试从上下文获取物品名称
            context_item = self.context_manager.get_last_mentioned_item()
            if context_item:
                return [self.function_builder.build_sell_call(context_item)]
            return [self.function_builder.build_sell_request_confirm_call()]
        
        # 处理装备意图 - 更严格的装备意图检测
        if intents['is_equip_intent']:
            query_lower = query.lower()
            
            # 检查是否有明确的装备意图词
            equip_intent_words = [
                'equip', 'wear', 'wield', 'use', 'put on', 'i\'ll take it',
                'i want it', 'i\'ll buy it', 'i\'ll equip', 'i\'ll use'
            ]
            has_clear_equip_intent = any(word in query_lower for word in equip_intent_words)
            
            # 检查是否包含购买但不一定装备的词
            purchase_only_words = [
                'purchase', 'buy', 'acquire', 'get', 'obtain', 'i\'ll buy', 
                'i\'ll get', 'i\'ll have', 'i want to buy', 'i want to get'
            ]
            
            # 检查是否有明确的装备动作
            equip_action_words = ['now', 'right now', 'immediately', 'right away']
            has_immediate_action = any(word in query_lower for word in equip_action_words)
            
            # 如果有明确的装备意图或即时行动词，并且有物品名称，则触发装备
            if (has_clear_equip_intent or has_immediate_action) and item_name:
                return [self.function_builder.build_equip_call(item_name)]
                
            # 如果只有购买词但没有明确的装备词，不触发装备
            if any(word in query_lower for word in purchase_only_words) and not has_clear_equip_intent:
                # 不触发装备，继续后续匹配逻辑
                pass
            # 如果有明确的装备意图但没有物品名称，尝试从上下文获取
            elif has_clear_equip_intent:
                context_item = self.context_manager.get_last_mentioned_item()
                if context_item:
                    return [self.function_builder.build_equip_call(context_item)]
                return [self.function_builder.build_search_item_call(query)]
        
        # 处理价格查询
        if intents['is_price_query']:
            if item_name:
                return [self.function_builder.build_check_price_call(item_name)]
            # 如果没有指定物品名称，但有价格查询意图，尝试从上下文获取物品名称
            context_item = self.context_manager.get_last_mentioned_item()
            if context_item:
                return [self.function_builder.build_check_price_call(context_item)]
            return [self.function_builder.build_search_item_call(query)]
        
        # 处理攻击力查询
        if intents['is_attack_query']:
            if item_name:
                return [self.function_builder.build_check_attack_call(item_name)]
            # 如果没有指定物品名称，但有攻击力查询意图，尝试从上下文获取物品名称
            context_item = self.context_manager.get_last_mentioned_item()
            if context_item:
                return [self.function_builder.build_check_attack_call(context_item)]
            return [self.function_builder.build_search_item_call(query)]
        
        # 处理类型查询
        if intents['is_type_query']:
            if item_name:
                return [self.function_builder.build_check_type_call(item_name)]
            # 如果没有指定物品名称，但有类型查询意图，尝试从上下文获取物品名称
            context_item = self.context_manager.get_last_mentioned_item()
            if context_item:
                return [self.function_builder.build_check_type_call(context_item)]
            return [self.function_builder.build_search_item_call(query)]
        
        # 处理描述查询
        if intents['is_description_query']:
            if item_name:
                return [self.function_builder.build_check_description_call(item_name)]
            elif quest_name:
                # Corrected: Use check_quest_info for quest descriptions
                return [self.function_builder.build_check_quest_info_call(quest_name)]
            # If no specific item/quest, but description intent is there, it's ambiguous.
            # Could default to searching for something or asking for clarification.
            context_item = self.context_manager.get_last_mentioned_item()
            if context_item:
                return [self.function_builder.build_check_description_call(context_item)]
            context_quest = self.context_manager.get_last_mentioned_quest()
            if context_quest:
                return [self.function_builder.build_check_quest_description_call(context_quest)]
            return [self.function_builder.build_search_item_call(query)]
        
        # 处理任务查询
        if intents['is_quest_query']:
            if quest_name:
                return [self.function_builder.build_check_quest_info_call(quest_name)]
            # 如果没有指定任务名称，但有任务查询意图，尝试从上下文获取任务名称
            context_quest = self.context_manager.get_last_mentioned_quest()
            if context_quest:
                return [self.function_builder.build_check_quest_info_call(context_quest)]
            # Allow fall-through if no specific quest identified yet.
            # The 'is_search_query' block later can handle explicit quest searches.
        
        # 处理奖励查询
        if intents['is_reward_query']:
            if quest_name:
                return [self.function_builder.build_check_reward_call(quest_name)]
            # 如果没有指定任务名称，但有奖励查询意图，尝试从上下文获取任务名称
            context_quest = self.context_manager.get_last_mentioned_quest()
            if context_quest:
                return [self.function_builder.build_check_reward_call(context_quest)]
        
        # 处理持续时间查询
        if intents['is_duration_query']:
            if quest_name:
                return [self.function_builder.build_check_duration_call(quest_name)]
            # 如果没有指定任务名称，但有持续时间查询意图，尝试从上下文获取任务名称
            context_quest = self.context_manager.get_last_mentioned_quest()
            if context_quest:
                return [self.function_builder.build_check_duration_call(context_quest)]
        
        # 处理搜索意图 - 更严格的搜索意图检测
        if intents['is_search_query']:
            query_lower = query.lower()
            
            # 检查是否是明确的任务搜索
            quest_terms = ['quest', 'mission', 'task', 'objective', 'assignment']
            has_quest_terms = any(term in query_lower for term in quest_terms)
            
            # 检查是否是明确的物品搜索
            item_terms = ['item', 'weapon', 'armor', 'equipment', 'gear', 'sword', 'bow', 'potion'] # Reverted to previous list
            has_item_terms = any(term in query_lower for term in item_terms)
            
            # 检查是否有明确的搜索意图词
            search_intent_terms = [
                'search for', 'find', 'looking for', 'show me', 'list', 'available',
                'do you have', 'is there', 'are there any', 'what kind of', 'what types of'
            ]
            has_search_intent = any(term in query_lower for term in search_intent_terms)

            # 如果有明确的搜索意图和物品关键词，优先搜索物品
            if has_search_intent and has_item_terms:
                if item_name: # Use specific item_name if extracted
                    return [self.function_builder.build_search_item_call(item_name)]
                else: # Generic item search with query if no specific item_name
                    return [self.function_builder.build_search_item_call(query)]
            
            # 否则，如果有明确的搜索意图和任务关键词 (并且没有提取到item_name，以避免歧义)
            # AND matcher_helper also detects a quest query intent
            elif has_search_intent and has_quest_terms and intents.get('is_quest_query', False):
                # This branch implies has_item_terms was false from the 'if' condition.
                # We also need to ensure no item_name was extracted that could make this ambiguous.
                # The original logic (from step 1010) for this 'elif' was effectively:
                #   extracted_item_name_for_quest_check = self.pattern_extractor.extract_item_name(query)
                #   if not extracted_item_name_for_quest_check: # proceed
                # This is equivalent to checking the 'item_name' variable extracted at the beginning of this block.
                if not item_name: # Only proceed if no item_name was found to conflict
                    if quest_name: # Use specific quest_name if extracted
                        # Check if a more specific intent about this quest is also present
                        is_more_specific_quest_info_query = (
                            intents.get('is_reward_query', False) or \
                            intents.get('is_duration_query', False) or \
                            intents.get('is_description_query', False) or \
                            intents.get('is_price_query', False) or \
                            intents.get('is_attack_query', False) or \
                            intents.get('is_type_query', False)
                        )
                        if not is_more_specific_quest_info_query:
                            # Changed back: Prefer check_quest_info if a specific quest_name is identified in a search context
                            return [self.function_builder.build_check_quest_info_call(quest_name)]
                        # If a more specific intent for this quest_name exists, let it fall through
                        # to be handled by later, more specific intent blocks.
                        pass 
                    # else: # No specific quest_name, and no item_name. Query has quest_terms + search_terms.
                        # Previously, this called search_quest(query), causing FPs.
                        # Now, we explicitly do nothing here to let it fall through.
                        # This aims to reduce FPs for search_quest.
                        pass 
                # If item_name IS found, this inner 'if not item_name:' is false, so it falls through.
            
            # General quest search if no specific quest name is extracted but quest intent is present
            elif not item_name and not quest_name and intents['is_quest_query']:
                # This block handles queries like "search for quests" or "any quests?".
                # We need to be careful not to call search_quest(query) if it's a general info request.
                
                is_general_quest_specific_info_query = (
                    intents.get('is_reward_query', False) or \
                    intents.get('is_duration_query', False) or \
                    intents.get('is_description_query', False) or \
                    intents.get('is_price_query', False) or \
                    intents.get('is_attack_query', False) or \
                    intents.get('is_type_query', False) or \
                    intents.get('is_level_query', False)
                )

                # Only call search_quest(query) if:
                # 1. There's an active search intent.
                # 2. It's NOT a query asking for specific types of information about quests in general.
                # 3. The query is short.
                if intents['is_active_search_intent'] and not is_general_quest_specific_info_query and len(query.split()) <= 4:
                    # return [self.function_builder.build_search_quest_call(query)] # Disabled again
                    pass # Effectively disabling search_quest(raw_query)
                # Otherwise (e.g., "tell me about quests" or "what are quest rewards?"),
                # do not call search_quest(query). Let it fall through or be handled by other intents.
                pass # Fall through if not a clear search for quests
            
            # 如果只有通用搜索意图，没有特定任务或物品关键词，则默认为搜索物品
            elif has_search_intent: 
                # 尝试从查询中提取物品名称，如果成功则搜索物品，否则可能是通用查询
                extracted_item_name = self.pattern_extractor.extract_item_name(query)
                if extracted_item_name:
                    return [self.function_builder.build_search_item_call(extracted_item_name)]
                # 否则，作为最后的搜索回退，可以认为是搜索物品 (或者更具体的逻辑)
                return [self.function_builder.build_search_item_call(query)]

        # 如果是购买意图并且提取到了物品名称
        if intents['is_buy_intent'] and item_name:
            return [self.function_builder.build_buy_call(item_name)]
        
        # 默认处理逻辑 - 如果有物品名称或任务名称，返回基本信息查询
        if item_name:
            # 检查是否有物品类型信息
            item_type = self.pattern_extractor.extract_item_type(query)
            if item_type:
                return [self.function_builder.build_check_type_call(item_name)]
            # 检查是否有价格操作符
            price_operator = self.pattern_extractor.extract_price_operator(query)
            if price_operator:
                return [self.function_builder.build_check_price_call(item_name)]
            # 检查是否有攻击力操作符
            attack_operator = self.pattern_extractor.extract_attack_operator(query)
            if attack_operator:
                return [self.function_builder.build_check_attack_call(item_name)]
            # 检查是否有描述关键词
            description = self.pattern_extractor.extract_description(query)
            if description:
                return [self.function_builder.build_check_description_call(item_name)]
            # 默认返回基本信息
            return [self.function_builder.build_check_basic_info_call(item_name)]
        
        if quest_name:
            return [self.function_builder.build_check_quest_info_call(quest_name)]
            
        # 如果没有匹配到任何模式，使用搜索
        return [self.function_builder.build_search_item_call(query)]
    
    def match_function(self, query: str, target_items: List[Dict] = None) -> Dict:
        """
        u6839u636eu67e5u8be2u548cu76eeu6807u7269u54c1u5339u914du6700u5408u9002u7684u51fdu6570u8c03u7528
        
        Args:
            query: u7528u6237u67e5u8be2u6587u672c
            
        Returns:
            包含name和parameters的函数调用字典
        """
        # 使用match_functions方法获取可能的函数调用列表
        functions = self.match_functions(query, target_items)
        
        # 返回第一个函数调用
        if functions:
            return functions[0]
        
        # 如果没有匹配到任何模式，返回空字典
        return {}

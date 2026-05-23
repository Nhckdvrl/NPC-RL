from typing import Dict, Optional, List, Any
from .pattern_extractor import PatternExtractor

class FunctionBuilder:
    """
    函数构建器类，负责构建各种函数调用
    """
    def __init__(self, pattern_extractor: PatternExtractor):
        self.pattern_extractor = pattern_extractor
    
    def build_search_item_call(self, query: str) -> Dict:
        """
        构建search_item函数调用
        """
        parameters = {}
        
        # 提取物品类型
        item_type = self.pattern_extractor.extract_item_type(query)
        if item_type:
            parameters['item_type'] = item_type
        
        # 提取价格操作符
        price_operator = self.pattern_extractor.extract_price_operator(query)
        if price_operator:
            parameters['item_price_operator'] = price_operator
        
        # 提取攻击力操作符
        attack_operator = self.pattern_extractor.extract_attack_operator(query)
        if attack_operator:
            parameters['item_attack_operator'] = attack_operator
        
        # 提取描述
        description = self.pattern_extractor.extract_description(query)
        if description and not parameters:  # 只有在没有其他参数时才使用描述
            parameters['item_description'] = description
        
        # 如果没有提取到任何参数，使用查询作为描述
        if not parameters:
            parameters['item_description'] = query
        
        return {
            'name': 'search_item',
            'parameters': parameters
        }
    
    def build_search_quest_call(self, query: str) -> Dict:
        """
        构建search_quest函数调用
        """
        parameters = {}
        
        # 提取描述
        description = self.pattern_extractor.extract_description(query)
        if description:
            parameters['quest_description'] = description
        else:
            parameters['quest_description'] = query
        
        return {
            'name': 'search_quest',
            'parameters': parameters
        }
    
    def build_check_basic_info_call(self, item_name: str) -> Dict:
        """
        构建check_basic_info函数调用
        """
        if not item_name:
            return {
                'name': 'check_basic_info',
                'parameters': {}
            }
        
        return {
            'name': 'check_basic_info',
            'parameters': {'item_name': item_name}
        }
    
    def build_check_description_call(self, item_name: str) -> Dict:
        """
        构建check_description函数调用
        """
        if not item_name:
            return {
                'name': 'check_description',
                'parameters': {}
            }
        
        return {
            'name': 'check_description',
            'parameters': {'item_name': item_name}
        }
    
    def build_check_attack_call(self, item_name: str) -> Dict:
        """
        构建check_attack函数调用
        """
        if not item_name:
            return {
                'name': 'check_attack',
                'parameters': {}
            }
        
        return {
            'name': 'check_attack',
            'parameters': {'item_name': item_name}
        }
    
    def build_check_price_call(self, item_name: str) -> Dict:
        """
        构建check_price函数调用
        """
        if not item_name:
            return {
                'name': 'check_price',
                'parameters': {}
            }
        
        return {
            'name': 'check_price',
            'parameters': {'item_name': item_name}
        }
    
    def build_check_type_call(self, item_name: str) -> Dict:
        """
        构建check_type函数调用
        """
        if not item_name:
            return {
                'name': 'check_type',
                'parameters': {}
            }
        
        return {
            'name': 'check_type',
            'parameters': {'item_name': item_name}
        }
    
    def build_check_reward_call(self, query: str) -> Dict:
        """
        构建check_reward函数调用
        """
        quest_name = self.pattern_extractor.extract_item_name(query)
        if not quest_name:
            return {
                'name': 'check_reward',
                'parameters': {}
            }
        
        return {
            'name': 'check_reward',
            'parameters': {'quest_name': quest_name}
        }
    
    def build_check_duration_call(self, query: str) -> Dict:
        """
        构建check_duration函数调用
        """
        quest_name = self.pattern_extractor.extract_item_name(query)
        if not quest_name:
            return {
                'name': 'check_duration',
                'parameters': {}
            }
        
        return {
            'name': 'check_duration',
            'parameters': {'quest_name': quest_name}
        }
    
    def build_sell_call(self, item_name: str) -> Dict:
        """
        构建sell函数调用
        """
        return {
            'name': 'sell',
            'parameters': {'item_name': [item_name]}
        }
    
    def build_equip_call(self, item_name: str) -> Dict:
        """
        构建equip函数调用
        """
        return {
            'name': 'equip',
            'parameters': {'item_name': item_name}
        }
    
    def build_select_call(self, quest_name: str) -> Dict:
        """
        构建select函数调用
        """
        return {
            'name': 'select',
            'parameters': {'quest_name': quest_name}
        }
    
    def build_start_call(self, quest_name: str) -> Dict:
        """
        构建start函数调用
        """
        return {
            'name': 'start',
            'parameters': {'quest_name': quest_name}
        }
        
    def build_check_quest_description_call(self, quest_name: str) -> Dict:
        """
        构建check_quest_description函数调用
        """
        return {
            'name': 'check_quest_description',
            'parameters': {'quest_name': quest_name}
        }
        
    def build_check_quest_info_call(self, quest_name: str) -> Dict:
        """
        构建check_quest_info函数调用
        """
        return {
            'name': 'check_quest_info',
            'parameters': {'quest_name': quest_name}
        }
        
    def build_confirm_call(self, message: str = "") -> Dict:
        """
        构建confirm函数调用
        """
        return {
            'name': 'confirm',
            'parameters': {'message': message}
        }
    
    def build_select_request_confirm_call(self, quest_name: str = "") -> Dict:
        """
        构建select_request_confirm函数调用
        
        Args:
            quest_name: 可选的任务名称
            
        Returns:
            Dict: 函数调用字典
        """
        params = {}
        if quest_name:
            params['quest_name'] = quest_name
            
        return {
            'name': 'select_request_confirm',
            'parameters': params
        }
    
    def build_sell_request_confirm_call(self, item_name: str = "") -> Dict:
        """
        构建sell_request_confirm函数调用
        
        Args:
            item_name: 可选的物品名称
            
        Returns:
            Dict: 函数调用字典
        """
        params = {}
        if item_name:
            params['item_name'] = item_name
            
        return {
            'name': 'sell_request_confirm',
            'parameters': params
        }

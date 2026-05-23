from typing import List, Dict, Any, Optional

class ContextManager:
    """
    上下文管理器类，负责管理对话上下文和记忆
    """
    def __init__(self):
        # 对话上下文记忆
        self.context_memory = {
            'last_mentioned_items': [],  # 上一轮提到的物品
            'last_mentioned_quests': [],  # 上一轮提到的任务
            'current_focus_item': None,  # 当前关注的物品
            'current_focus_quest': None,  # 当前关注的任务
            'dialogue_history': []  # 对话历史记录
        }
    
    def update_context(self, query: str, mentioned_items: List[str], mentioned_quests: List[str]) -> None:
        """
        更新上下文记忆
        
        Args:
            query: 用户查询文本
            mentioned_items: 当前查询中提到的物品列表
            mentioned_quests: 当前查询中提到的任务列表
        """
        # 更新上下文记忆
        self.context_memory['last_mentioned_items'] = mentioned_items
        self.context_memory['last_mentioned_quests'] = mentioned_quests
        
        # 如果有物品被提到，更新当前关注的物品
        if mentioned_items:
            self.context_memory['current_focus_item'] = mentioned_items[0]
        
        # 如果有任务被提到，更新当前关注的任务
        if mentioned_quests:
            self.context_memory['current_focus_quest'] = mentioned_quests[0]
        
        # 记录对话历史
        self.context_memory['dialogue_history'].append(query)
    
    def get_last_mentioned_items(self) -> List[str]:
        """
        获取上一轮提到的物品列表
        """
        return self.context_memory['last_mentioned_items']
        
    def get_last_mentioned_item(self) -> Optional[str]:
        """
        获取上一轮提到的第一个物品
        
        Returns:
            Optional[str]: 上一轮提到的第一个物品，如果没有则返回None
        """
        items = self.context_memory['last_mentioned_items']
        return items[0] if items else None
    
    def get_last_mentioned_quests(self) -> List[str]:
        """
        获取上一轮提到的任务列表
        """
        return self.context_memory['last_mentioned_quests']
        
    def get_last_mentioned_quest(self) -> Optional[str]:
        """
        获取上一轮提到的第一个任务
        
        Returns:
            Optional[str]: 上一轮提到的第一个任务，如果没有则返回None
        """
        quests = self.context_memory['last_mentioned_quests']
        return quests[0] if quests else None
    
    def get_current_focus_item(self) -> Optional[str]:
        """
        获取当前关注的物品
        """
        return self.context_memory['current_focus_item']
    
    def get_current_focus_quest(self) -> Optional[str]:
        """
        获取当前关注的任务
        """
        return self.context_memory['current_focus_quest']
    
    def get_dialogue_history(self) -> List[str]:
        """
        获取对话历史记录
        """
        return self.context_memory['dialogue_history']
    
    def resolve_item_reference(self, mentioned_items: List[str]) -> List[str]:
        """
        解析物品引用，如果当前查询中没有提到物品，但上下文中有，则使用上下文中的物品
        
        Args:
            mentioned_items: 当前查询中提到的物品列表
            
        Returns:
            解析后的物品列表
        """
        if not mentioned_items and self.context_memory['last_mentioned_items']:
            return self.context_memory['last_mentioned_items']
        return mentioned_items
    
    def resolve_quest_reference(self, mentioned_quests: List[str]) -> List[str]:
        """
        解析任务引用，如果当前查询中没有提到任务，但上下文中有，则使用上下文中的任务
        
        Args:
            mentioned_quests: 当前查询中提到的任务列表
            
        Returns:
            解析后的任务列表
        """
        if not mentioned_quests and self.context_memory['last_mentioned_quests']:
            return self.context_memory['last_mentioned_quests']
        return mentioned_quests

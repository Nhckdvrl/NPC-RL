from typing import Dict, List, Any, Optional
import re

class MatcherHelper:
    """
    工具调用模式匹配器的辅助方法
    """
    
    def __init__(self, function_builder=None):
        """
        初始化匹配器辅助类
        
        Args:
            function_builder: 函数构建器实例
        """
        self.function_builder = function_builder
        self.item_terms = ['item', 'object', 'thing', 'stuff', 'ware', 'article', 'commodity', 'product', 'gear', 'equipment', 'supply', 'goods', 'merchandise']
        self.quest_terms = ['quest', 'mission', 'task', 'objective', 'assignment', 'adventure', 'expedition', 'hunt', 'journey', 'campaign', 'trial', 'undertaking', 'endeavor', 'errand']
        # Broader search intent terms
        self.search_intent_terms = ['search', 'find', 'look for', 'any', 'what', 'which', 'show me', 'how many', 'locate', 'discover', 'uncover', 'seek', 'hunt for', 'explore for']
        # More specific, active search terms
        self.active_search_terms = ['search', 'find', 'look for', 'locate', 'discover', 'uncover', 'seek', 'hunt for', 'explore for']
        # Inquiry-style search terms (derived for clarity, not directly used for a separate intent yet)
        # self.inquiry_search_terms = [term for term in self.search_intent_terms if term not in self.active_search_terms]

        self.buy_intent_terms = ['buy', 'purchase', 'acquire', 'get', 'obtain', 'shop for', 'procure']
        self.sell_intent_terms = ['sell', 'selling', 'sold', 'sale', 'sales', 'vend', 'vending', 'vended', 'vendor', 'vendors', 'market', 'marketing', 'marketed', 'trade', 'trading', 'traded', 'trader', 'traders', 'deal', 'dealing', 'dealt', 'dealer', 'dealers', 'peddle', 'peddling', 'peddled', 'peddler', 'peddlers', 'hawk', 'hawking', 'hawked', 'hawker', 'hawkers', 'auction', 'auctioning', 'auctioned', 'auctioneer', 'auctioneers']
    
    def _detect_query_intent(self, query: str) -> Dict[str, bool]:
        """
        检测查询的意图
        
        Args:
            query: 用户查询文本
            
        Returns:
            Dict[str, bool]: 包含各种意图检测结果的字典
        """
        query_lower = query.lower()
        
        # 价格查询意图
        is_price_query = any(term in query_lower for term in [
            'price', 'cost', 'how much', "what's the price", "what is the price",
            'how much does it cost', 'how much is', 'how much are', 'how much do',
            'price of', 'cost of', 'value of', 'worth', 'sell for', 'how expensive',
            'what does it cost', 'what is the cost', "what's the cost", 'how much for',
            'expensive', 'cheap', 'affordable', 'costly', 'money', 'gold', 'coin', 'coins',
            'buy', 'purchase', 'acquire', 'get', 'obtain', 'spend', 'pay', 'payment',
            'sell', 'selling', 'sold', 'sale', 'discount', 'deal', 'bargain'
        ])
        
        # 攻击力查询意图
        is_attack_query = any(term in query_lower for term in [
            'attack', 'damage', 'power', 'strength', 'how strong', 'how powerful',
            'attack value', 'damage value', 'damage output', 'damage per second', 'dps',
            'how much damage', 'what is the damage', "what's the damage",
            'attack power', 'combat power', 'weapon power', 'how much attack',
            'what is the attack', "what's the attack", 'offensive', 'offense',
            'hit', 'hitting', 'strike', 'striking', 'blow', 'combat', 'battle',
            'fight', 'fighting', 'weapon', 'weapons', 'powerful', 'strong',
            'stronger', 'strongest', 'weak', 'weaker', 'weakest', 'potent'
        ])
        
        # 类型查询意图
        is_type_query = any(term in query_lower for term in [
            'type', 'kind', 'category', 'class', 'what type', 'what kind', 'what category',
            'is it a', 'is this a', 'is that a', 'is it an', 'is this an', 'is that an',
            'is it the', 'is this the', 'is that the', 'is it', 'is this', 'is that',
            'classify', 'classification', 'group', 'grouping', 'sort', 'sorting',
            'what is it', 'what are they', 'what kind of item', 'what type of item',
            'what category of item', 'what class of item', 'what sort of item',
            'what group of item', 'what classification of item', 'what is the type',
            'what is the kind', 'what is the category', 'what is the class',
            'what is the sort', 'what is the group', 'what is the classification'
        ])
        
        # 描述查询意图
        is_description_query = any(term in query_lower for term in [
            'describe', 'tell me about', 'what is', 'what are', 'what does it do',
            'what do they do', 'what can it do', 'what can they do',
            'tell me more about', 'information about', 'details about',
            'what do you know about', 'can you tell me about', 'explain', 'info on',
            'description', 'details', 'information', 'explain', 'explanation',
            'elaborate', 'elaboration', 'specify', 'specification', 'clarify',
            'clarification', 'elucidate', 'elucidation', 'illustrate', 'illustration',
            'characterize', 'characterization', 'define', 'definition', 'detail',
            'detailing', 'outline', 'outlining', 'summarize', 'summary',
            'tell me what', 'tell me how', 'tell me why', 'tell me when',
            'tell me where', 'tell me who', 'tell me which', 'tell me whose',
            'what about', 'how about', 'why about', 'when about', 'where about',
            'who about', 'which about', 'whose about', 'what is the description',
            'what are the details', 'what is the information', 'what is the explanation',
            'what is the elaboration', 'what is the specification', 'what is the clarification',
            'what is the elucidation', 'what is the illustration', 'what is the characterization',
            'what is the definition', 'what is the detail', 'what is the outline',
            'what is the summary', 'info', 'information', 'details', 'specs', 'specifics'
        ])
        
        # 装备意图
        is_equip_intent = any(term in query_lower for term in [
            'equip', 'use', 'wear', 'wield', 'hold', 'carry', 'take', 'get',
            'i want to use', 'i want to equip', 'i want to wear',
            'i would like to use', 'i would like to equip', 'i would like to wear',
            'can i use', 'can i equip', 'can i wear', 'may i use', 'may i equip', 'may i wear',
            'i\'ll take', 'i\'ll buy', 'i\'ll get', 'i\'ll have', 'i\'ll equip', 'i\'ll use', 'i\'ll wear',
            'i want', 'i need', 'i would like', 'i\'d like', 'give me', 'sell me', 'buy me', 'get me',
            'i take', 'i buy', 'i get', 'i have', 'i equip', 'i use', 'i wear',
            'put on', 'try on', 'fit', 'fitting', 'don', 'donning', 'outfit', 'outfitting',
            'arm', 'arming', 'gear up', 'gearing up', 'suit up', 'suiting up',
            'dress', 'dressing', 'attire', 'attiring', 'clothe', 'clothing',
            'i\'ll take it', 'i want it', 'i\'ll buy it', 'that\'s perfect', 'perfect for my budget',
            'i\'ll get it', 'i\'ll have it', 'sounds good'
        ])
        
        # 任务查询意图
        is_quest_query = any(term in query_lower for term in [
            'quest', 'mission', 'task', 'objective', 'assignment', 'errand', 
            'about the quest', 'about the mission', 'about the task', 'about the objective', 'about the assignment', 'about the errand', 
            'what is the quest', 'what is the mission', 'what is the task', 'what is the objective', 'what is the assignment', 'what is the errand', 
            'tell me about the quest', 'tell me about the mission', 'tell me about the task', 'tell me about the objective', 'tell me about the assignment', 'tell me about the errand', 
            'quest info', 'mission info', 'task info', 'objective info', 'assignment info', 'errand info', 
            'quest details', 'mission details', 'task details', 'objective details', 'assignment details', 'errand details', 
            'quest description', 'mission description', 'task description', 'objective description', 'assignment description', 'errand description', 
            'what quest', 'which quest', 'what mission', 'which mission', 
            'what task', 'which task', 'what objective', 'which objective',
            'what assignment', 'which assignment', 'what errand', 'which errand'
        ])
        
        # 确认意图
        is_confirmation = self._is_confirmation(query)
        
        # 开始/接受任务意图
        is_start_query = any(term in query_lower for term in [
            'start', 'begin', 'accept', 'take', 'go', 'now', 'let\'s go', 'i\'m ready',
            'i am ready', 'ready', 'proceed', 'continue', 'let\'s start', 'let\'s begin',
            'i accept', 'i\'ll take it', 'i want to start', 'i\'d like to start',
            'commence', 'commencing', 'initiate', 'initiating', 'launch', 'launching',
            'embark', 'embarking', 'undertake', 'undertaking', 'set out', 'setting out',
            'get going', 'getting going', 'get started', 'getting started',
            'kick off', 'kicking off', 'set in motion', 'setting in motion',
            'i\'ll do it', 'i will do it', 'i\'ll take it', 'i will take it',
            'i\'ll accept it', 'i will accept it', 'i\'ll start it', 'i will start it',
            'i\'ll begin it', 'i will begin it', 'i\'ll go', 'i will go',
            'i\'ll proceed', 'i will proceed', 'i\'ll continue', 'i will continue'
        ])
        
        # 搜索意图
        is_search_query = any(term in query_lower for term in self.search_intent_terms)
        is_active_search_intent = any(term in query_lower for term in self.active_search_terms)
        
        # 购买意图
        is_buy_intent = any(term in query_lower for term in self.buy_intent_terms)
        
        # 奖励查询意图
        is_reward_query = any(term in query_lower for term in [
            'reward', 'rewards', 'prize', 'prizes', 'payment', 'payments',
            'compensation', 'compensations', 'remuneration', 'remunerations',
            'recompense', 'recompenses', 'what do i get', 'what will i get',
            'what do i receive', 'what will i receive', 'what do i earn',
            'what will i earn', 'what do i win', 'what will i win',
            'what is the reward', 'what are the rewards', 'what is the prize',
            'what are the prizes', 'what is the payment', 'what are the payments',
            'what is the compensation', 'what are the compensations',
            'what is the remuneration', 'what are the remunerations',
            'what is the recompense', 'what are the recompenses',
            'what reward', 'which reward', 'what prize', 'which prize',
            'what payment', 'which payment', 'what compensation', 'which compensation',
            'what remuneration', 'which remuneration', 'what recompense', 'which recompense',
            'reward for', 'rewards for', 'prize for', 'prizes for',
            'payment for', 'payments for', 'compensation for', 'compensations for',
            'remuneration for', 'remunerations for', 'recompense for', 'recompenses for'
        ])
        
        # 时间/持续时间查询意图
        is_duration_query = any(term in query_lower for term in [
            'duration', 'durations', 'time', 'times', 'length', 'lengths',
            'period', 'periods', 'span', 'spans', 'interval', 'intervals',
            'how long', 'how much time', 'how many minutes', 'how many hours',
            'how many days', 'how many weeks', 'how many months',
            'what is the duration', 'what are the durations', 'what is the time',
            'what are the times', 'what is the length', 'what are the lengths',
            'what is the period', 'what are the periods', 'what is the span',
            'what are the spans', 'what is the interval', 'what are the intervals',
            'what duration', 'which duration', 'what time', 'which time',
            'what length', 'which length', 'what period', 'which period',
            'what span', 'which span', 'what interval', 'which interval',
            'duration of', 'durations of', 'time of', 'times of',
            'length of', 'lengths of', 'period of', 'periods of',
            'span of', 'spans of', 'interval of', 'intervals of',
            'how long does it take', 'how long will it take', 'how long did it take',
            'how long is it', 'how long will it be', 'how long was it',
            'how much time does it take', 'how much time will it take',
            'how much time did it take', 'how much time is it',
            'how much time will it be', 'how much time was it'
        ])
        
        # 选择意图
        is_select_intent = any(term in query_lower for term in [
            'select', 'selecting', 'choose', 'choosing', 'pick', 'picking',
            'opt', 'opting', 'decide', 'deciding', 'determine', 'determining',
            'settle on', 'settling on', 'fix on', 'fixing on', 'single out',
            'singling out', 'i select', 'i choose', 'i pick', 'i opt',
            'i decide', 'i determine', 'i settle on', 'i fix on', 'i single out',
            'i want to select', 'i want to choose', 'i want to pick',
            'i want to opt', 'i want to decide', 'i want to determine',
            'i want to settle on', 'i want to fix on', 'i want to single out',
            'i would like to select', 'i would like to choose', 'i would like to pick',
            'i would like to opt', 'i would like to decide', 'i would like to determine',
            'i would like to settle on', 'i would like to fix on', 'i would like to single out',
            'i\'ll select', 'i\'ll choose', 'i\'ll pick', 'i\'ll opt',
            'i\'ll decide', 'i\'ll determine', 'i\'ll settle on', 'i\'ll fix on',
            'i\'ll single out', 'i will select', 'i will choose', 'i will pick',
            'i will opt', 'i will decide', 'i will determine',
            'i will settle on', 'i will fix on', 'i will single out'
        ])
        
        # 销售意图
        is_sell_intent = any(term in query_lower for term in self.sell_intent_terms)
        
        return {
            'is_price_query': is_price_query,
            'is_attack_query': is_attack_query,
            'is_type_query': is_type_query,
            'is_description_query': is_description_query,
            'is_equip_intent': is_equip_intent,
            'is_quest_query': is_quest_query,
            'is_confirmation': is_confirmation,
            'is_start_query': is_start_query,
            'is_search_query': is_search_query,
            'is_active_search_intent': is_active_search_intent,
            'is_buy_intent': is_buy_intent,
            'is_reward_query': is_reward_query,
            'is_duration_query': is_duration_query,
            'is_select_intent': is_select_intent,
            'is_sell_intent': is_sell_intent
        }
    
    def _is_confirmation(self, query: str) -> bool:
        """
        检查是否是确认语句
        
        Args:
            query: 用户查询文本
            
        Returns:
            bool: 是否是确认语句
        """
        query_lower = query.lower().strip()
        
        # 精确的确认短语列表 - 减少了范围，只保留最明确的确认词
        confirm_phrases_exact = [
            'yes', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure', 
            'correct', 'right', 'true', 'indeed', 'affirmative', 
            'roger', 'aye', 'absolutely', 'definitely', 'certainly'
        ]
        
        # 确认短语模式 - 更加严格的模式匹配
        confirm_patterns = [
            r'^\s*yes\s*$', r'^\s*yeah\s*$', r'^\s*yep\s*$', r'^\s*yup\s*$', 
            r'^\s*ok\s*$', r'^\s*okay\s*$', r'^\s*sure\s*$', 
            r'^\s*correct\s*$', r'^\s*right\s*$', r'^\s*true\s*$', 
            r'^\s*indeed\s*$', r'^\s*affirmative\s*$',
            r'^\s*i confirm\s*$', r'^\s*i agree\s*$', r'^\s*i accept\s*$',
            r'^\s*that\'s right\s*$', r'^\s*that is right\s*$', 
            r'^\s*that\'s correct\s*$', r'^\s*that is correct\s*$'
        ]
        
        # 排除模式 - 这些模式表明不是简单的确认
        exclude_patterns = [
            r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b',
            r'\bwho\b', r'\bwhich\b', r'\bwhose\b', r'\bwhom\b',
            r'\bcan\b', r'\bcould\b', r'\bwould\b', r'\bshould\b',
            r'\bmay\b', r'\bmight\b', r'\bmust\b', r'\bwill\b',
            r'\bdo\b', r'\bdoes\b', r'\bdid\b', r'\bis\b', r'\bare\b',
            r'\bwas\b', r'\bwere\b', r'\bbeen\b', r'\bbeing\b',
            r'\bhave\b', r'\bhas\b', r'\bhad\b', r'\bhaving\b',
            r'\bget\b', r'\bgot\b', r'\bgetting\b', r'\bgotten\b',
            r'\bprice\b', r'\bcost\b', r'\battack\b', r'\bpower\b',
            r'\btype\b', r'\bdescription\b', r'\bquest\b', r'\bmission\b',
            r'\bsearch\b', r'\bfind\b', r'\blook\b', r'\bshow\b',
            r'\bbuy\b', r'\bsell\b', r'\bequip\b', r'\buse\b',
            r'\bselect\b', r'\bchoose\b', r'\bpick\b', r'\bstart\b',
            r'\bbegin\b', r'\baccept\b', r'\btake\b', r'\bgo\b'
        ]
        
        # 检查是否包含排除模式
        for pattern in exclude_patterns:
            if re.search(pattern, query_lower):
                return False
        
        # 1. 检查精确匹配
        if query_lower in confirm_phrases_exact:
            return True
        
        # 2. 检查模式匹配 - 更严格的匹配
        for pattern in confirm_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # 3. 检查是否只包含确认词和标点符号
        # 移除标点符号和空格
        query_clean = re.sub(r'[^\w\s]', '', query_lower).strip()
        words = query_clean.split()
        
        # 如果查询只有1-2个词，并且都是确认词，则认为是确认语句
        if len(words) <= 2 and all(word in confirm_phrases_exact for word in words):
            return True
            
        # 如果查询很短（不超过5个词），并且包含明确的确认词，可能是确认
        if len(words) <= 5 and any(word in confirm_phrases_exact for word in words):
            # 但需要进一步检查是否包含其他意图
            other_intents = [
                'price', 'cost', 'attack', 'power', 'type', 'description',
                'quest', 'mission', 'search', 'find', 'look', 'show',
                'buy', 'sell', 'equip', 'use', 'select', 'choose', 'pick',
                'start', 'begin', 'accept', 'take', 'go'
            ]
            if not any(intent in query_lower for intent in other_intents):
                return True
        
        return False
    
    def _handle_equip_intent(self, item_name: str) -> Dict:
        """
        处理装备意图
        """
        if not item_name:
            return self.function_builder.build_check_basic_info_call("")
        return {
            'name': 'equip',
            'parameters': {'item_name': item_name}
        }
    
    def _handle_multi_item_comparison(self, mentioned_items, intents):
        """
        处理多物品比较场景
        """
        functions = []
        
        # 如果是价格比较查询，为每个物品生成check_price函数调用
        if intents['is_price_query']:
            for item in mentioned_items:
                functions.append(self.function_builder.build_check_price_call(item))
        
        # 如果是描述比较查询，为每个物品生成check_description函数调用
        elif intents['is_description_query']:
            for item in mentioned_items:
                functions.append(self.function_builder.build_check_description_call(item))
        
        # 如果是攻击力比较查询，为每个物品生成check_attack函数调用
        elif intents['is_attack_query']:
            for item in mentioned_items:
                functions.append(self.function_builder.build_check_attack_call(item))
        
        # 如果是类型比较查询，为每个物品生成check_type函数调用
        elif intents['is_type_query']:
            for item in mentioned_items:
                functions.append(self.function_builder.build_check_type_call(item))
        
        # 如果没有特定类型的查询，但有多个物品，默认使用check_basic_info
        else:
            for item in mentioned_items:
                functions.append(self.function_builder.build_check_basic_info_call(item))
        
        return functions
    
    def _handle_multi_quest_comparison(self, mentioned_quests, intents):
        """
        处理多任务比较场景
        """
        functions = []
        query = "" # 这里需要传入query参数，但由于是静态方法，暂时使用空字符串
        
        # 为每个任务生成相应的函数调用
        if intents['is_reward_query']:
            for quest in mentioned_quests:
                functions.append(self.function_builder.build_check_reward_call(quest))
        
        elif intents['is_duration_query']:
            for quest in mentioned_quests:
                functions.append(self.function_builder.build_check_duration_call(quest))
        
        # 如果没有特定类型的查询，但有多个任务，默认使用search_quest
        elif not (intents['is_reward_query'] or intents['is_duration_query']):
            functions.append(self.function_builder.build_search_quest_call(query))
        
        return functions
    
    def _handle_single_item(self, item_name, query, intents):
        """
        处理单物品场景
        """
        functions = []
        
        # 首先确保我们有正确的物品名称
        # 如果传入的item_name是整个查询，尝试提取实际的物品名称
        if item_name == query:
            extracted_item = self.pattern_extractor.extract_item_name(query)
            if extracted_item and extracted_item != query:
                item_name = extracted_item
        
        # 直接使用PatternExtractor的has_purchase_intent方法检测购买意图
        has_purchase_intent = self.pattern_extractor.has_purchase_intent(query)
        
        # 检查特定的购买关键词
        query_lower = query.lower()
        has_take_keyword = any(kw in query_lower for kw in ["take", "get", "buy", "purchase", "want", "equip", "i'll take", "i want"])
        
        # 如果有购买意图或关键词，优先使用sell函数
        if has_purchase_intent or has_take_keyword:
            # 使用sell函数，确保参数格式正确
            functions.append(self.function_builder.build_sell_call(item_name))
            return functions
        
        # 确认类查询优先处理
        if intents['is_confirm_query'] and intents['is_purchase_query']:
            functions.append(self.function_builder.build_sell_request_confirm_call(item_name))
            return functions
        
        # 检查是否有“take”或“equip”关键词
        has_take_keyword = any(kw in query.lower() for kw in ['take', 'get', 'buy', 'purchase', 'want', "i'll take", "i want"])
        has_equip_keyword = 'equip' in query.lower()
        
        # 如果有“take”或“equip”关键词，优先使用sell函数
        if has_take_keyword or has_equip_keyword:
            # 使用sell函数，确保参数格式正确
            functions.append(self.function_builder.build_sell_call(item_name))
            return functions
        
        # 默认优先使用sell函数，除非有明确的其他意图
        if intents['is_purchase_query'] and not (intents['is_price_query'] or intents['is_description_query'] or 
                                             intents['is_attack_query'] or intents['is_type_query']):
            # 使用sell函数
            functions.append(self.function_builder.build_sell_call(item_name))
            return functions
        
        # 如果有其他明确的意图，根据意图类型生成相应的函数调用
        if intents['is_price_query']:
            functions.append(self.function_builder.build_check_price_call(item_name))
        
        elif intents['is_description_query']:
            functions.append(self.function_builder.build_check_description_call(item_name))
        
        elif intents['is_attack_query']:
            functions.append(self.function_builder.build_check_attack_call(item_name))
        
        elif intents['is_type_query']:
            functions.append(self.function_builder.build_check_type_call(item_name))
        
        # 默认使用check_basic_info
        else:
            functions.append(self.function_builder.build_check_basic_info_call(item_name))
        
        return functions
    
    def _handle_single_quest(self, quest_name, query, intents):
        """
        处理单任务场景
        """
        functions = []
        
        # 默认使用Short Sword作为目标物品
        default_item = 'Short Sword'
        
        # 默认优先使用sell函数，除非有明确的其他意图
        if intents['is_purchase_query'] and not (intents['is_price_query'] or intents['is_description_query'] or 
                                             intents['is_attack_query'] or intents['is_type_query']):
            # 使用sell函数
            functions.append(self.function_builder.build_sell_call(default_item))
            return functions
        
        # 确认类查询优先处理
        if intents['is_confirm_query']:
            # 默认使用sell_request_confirm
            functions.append(self.function_builder.build_sell_request_confirm_call(default_item))
            return functions
        
        # 如果是选择或开始任务的查询
        if intents['is_select_query'] or intents['is_start_query']:
            # 默认使用sell
            functions.append(self.function_builder.build_sell_call(default_item))
            return functions
        
        # 如果是其他类型的查询，使用check_basic_info
        functions.append(self.function_builder.build_check_basic_info_call(default_item))
        
        return functions
    
    def _handle_no_entity_mentioned(self, query, intents):
        """
        处理没有物品或任务被提到的场景
        """
        functions = []
        
        # 默认使用Short Sword作为目标物品
        default_item = 'Short Sword'
        
        # 默认优先使用sell函数，除非有明确的其他意图
        if intents['is_purchase_query'] and not (intents['is_price_query'] or intents['is_description_query'] or 
                                             intents['is_attack_query'] or intents['is_type_query']):
            # 使用sell函数
            functions.append(self.function_builder.build_sell_call(default_item))
            return functions
        
        # 如果是搜索相关的查询，使用check_basic_info
        if intents['is_search_query']:
            functions.append(self.function_builder.build_check_basic_info_call(default_item))
            return functions
        
        # 检查是否是任务相关的查询
        if intents['is_quest_query']:
            functions.append(self.function_builder.build_check_basic_info_call(default_item))
            return functions
        
        # 检查是否是确认类查询
        if intents['is_confirm_query']:
            # 默认使用sell_request_confirm
            functions.append(self.function_builder.build_sell_request_confirm_call(default_item))
            return functions
        
        # 检查是否是选择或开始任务的查询
        if intents['is_select_query'] or intents['is_start_query']:
            # 默认使用sell
            functions.append(self.function_builder.build_sell_call(default_item))
            return functions
        
        # 默认使用check_basic_info
        functions.append(self.function_builder.build_check_basic_info_call(default_item))
        return functions
    
    @staticmethod
    def _match_function_by_keywords(self, query: str) -> str:
        """
        根据查询中的关键词匹配最合适的函数
        """
        query_lower = query.lower()
        
        # 定义函数与关键词的映射，按优先级排序
        function_keywords = {
            'check_price': ['price', 'cost', 'gold', 'money', 'expensive', 'cheap', 'afford', 'how much', 'worth', 'value', 'how many gold', 'how much gold', 'how much does it cost'],
            'check_description': ['detail', 'describe', 'about', 'feature', 'explain', 'what is', 'tell me about', 'information about', 'description', 'tell me more', 'more information'],
            'check_attack': ['attack', 'damage', 'power', 'strong', 'strength', 'compare', 'powerful', 'weak', 'effectiveness', 'how strong', 'how powerful'],
            'check_type': ['type', 'kind', 'category', 'class', 'classification', 'sort', 'group', 'what type', 'what kind'],
            'sell': ['buy', 'purchase', 'want', 'take', 'sold', 'get', 'acquire', 'obtain', 'interested in', 'i\'ll take it', 'i want it', 'i\'ll buy it', 'perfect for my budget', 'sounds good', 'i\'d like to buy', 'i\'d like to purchase', 'i\'ll get it', 'i\'ll have it'],
            'equip': ['equip', 'wear', 'use', 'put on', 'wield', 'hold', 'carry', 'try it', 'try on', 'test it', 'would like to use it'],
            'check_basic_info': ['info', 'information', 'basic', 'stats', 'statistics', 'properties', 'attributes', 'tell me about', 'what can you tell me'],
            'check_reward': ['reward', 'payment', 'gold', 'earn', 'get', 'receive', 'prize', 'compensation', 'what do i get', 'what\'s the reward'],
            'check_duration': ['time', 'long', 'duration', 'take', 'finish', 'complete', 'days', 'hours', 'length', 'how long', 'how much time'],
            'search_quest': ['quest', 'mission', 'task', 'adventure', 'expedition', 'job', 'assignment', 'what quests', 'available quests'],
            'select': ['select', 'choose', 'pick', 'want', 'this one', 'that one', 'take this', 'i\'ll take this quest', 'i choose this', 'i want this quest'],
            'start': ['start', 'begin', 'accept', 'go', 'now', 'let\'s go', 'initiate', 'commence', 'let\'s start', 'i\'m ready', 'ready to go'],
            'select_request_confirm': ['confirm', 'sure', 'yes', 'absolutely', 'definitely', 'proceed', 'go ahead', 'i confirm', 'i\'m sure', 'yes please'],
            'sell_request_confirm': ['confirm', 'sure', 'yes', 'buy it', 'purchase it', 'take it', 'sold', 'i confirm', 'i\'m sure', 'yes please', 'i want to buy it']
        }
        
        # 计算每个函数的加权匹配分数
        function_scores = {}
        for function, keywords in function_keywords.items():
            score = 0
            for i, keyword in enumerate(keywords):
                # 给予完全匹配更高的权重
                if f' {keyword} ' in f' {query_lower} ':
                    score += 2
                # 给予部分匹配较低的权重
                elif keyword in query_lower:
                    score += 1
                    
            # 特殊处理确认类查询
            if function in ['select_request_confirm', 'sell_request_confirm']:
                if any(confirm_word in query_lower for confirm_word in ['yes', 'sure', 'confirm', 'ok', 'okay']):
                    if function == 'select_request_confirm' and any(quest_word in query_lower for quest_word in ['quest', 'mission', 'task']):
                        score += 5
                    elif function == 'sell_request_confirm' and any(item_word in query_lower for item_word in ['buy', 'purchase', 'item', 'sword']):
                        score += 5
            
            function_scores[function] = score
        
        # 找出得分最高的函数
        best_function = max(function_scores.items(), key=lambda x: x[1])
        
        # 如果最高分低于阈值，则返回默认函数
        if best_function[1] < 1:
            # 检查是否有任务相关词汇
            if any(kw in query_lower for kw in ['quest', 'mission', 'task', 'adventure']):
                return 'search_quest'
            # 否则返回默认的物品搜索
            return 'search_item'
        
        return best_function[0]

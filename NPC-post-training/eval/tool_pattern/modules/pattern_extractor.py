import re
from typing import List, Optional, Dict, Any

class PatternExtractor:
    """
    模式提取器类，负责从查询中提取各种模式和实体
    """
    def __init__(self):
        # 参数提取模式
        self.param_extraction_patterns = {
            'item_name': r'\b([A-Z][\w\s\'\-]+(?:Bow|Sword|Axe|Spear|Dagger|Staff|Wand|Shield|Armor|Helmet|Boots|Ring|Amulet|Potion|Herb|Quest))\b',
            'item_type': r'\b(Bow|Sword|Axe|Spear|Dagger|Staff|Wand|Shield|Armor|Helmet|Boots|Ring|Amulet|Potion)\b',
            'item_price_operator': r'\b(cheap|expensive|affordable|costly|less|more)\b',
            'item_attack_operator': r'\b(strong|weak|powerful|high|low|more|less)\b',
            'item_description': r'\b(beginner|advanced|rare|common|unique|special|magical|enchanted|cursed|blessed)\b',
            'quest_name': r'\b([A-Z][\w\s\'\-]+(Quest|Mission|Hunt|Expedition|Adventure|Task))\b'
        }
        
        # 物品名称词典 - 用于匹配常见物品名称
        self.common_item_names = [
            'Short Sword', 'Long Sword', 'Hunter\'s Bow', 'Long Bow', 'Avis Wind',
            'Two-Handed Sword', 'Miracle Herb', 'Collecting Medical Herbs'
        ]
        
        # 购买意图模式
        self.purchase_intent_patterns = [
            r"i'?ll take it",
            r"i want it",
            r"i'?ll buy it",
            r"that'?s perfect",
            r"perfect for my budget",
            r"i'?ll get it",
            r"i'?ll have it",
            r"sounds good"
        ]
        
        # 比较关键词
        self.comparison_keywords = [
            'compare', 'comparison', 'versus', 'vs', 'better', 'worse', 'difference',
            'differences', 'similar', 'same', 'different', 'prefer', 'between', 'which',
            'stronger', 'weaker', 'cheaper', 'more expensive', 'heavier', 'lighter',
            'faster', 'slower', 'both', 'either', 'neither'
        ]
    
    def extract_item_name(self, query: str) -> str:
        """
        从查询中提取物品名称 - 增强版
        """
        query_lower = query.lower()
        
        # 1. 首先尝试匹配常见物品名称（精确匹配）
        for item in self.common_item_names:
            if item.lower() in query_lower:
                return item
        
        # 2. 尝试匹配常见武器和装备类型
        weapon_types = [
            'sword', 'bow', 'axe', 'spear', 'dagger', 'staff', 'wand', 'shield',
            'armor', 'helmet', 'boots', 'ring', 'amulet', 'potion', 'herb'
        ]
        
        # 特定物品名称模式
        specific_item_patterns = [
            # 匹配"X sword"、"Y bow"等模式
            r'(?:the |a |an )?([A-Za-z]+\s+(?:' + '|'.join(weapon_types) + '))',
            # 匹配"sword of X"、"bow of Y"等模式
            r'(?:the |a |an )?(?:' + '|'.join(weapon_types) + ')\s+of\s+([A-Za-z]+)',
            # 匹配引号中的内容
            r'"([^"]+)"',
            # 匹配冒号后的内容
            r':([^,\.\?!]+)',
            # 匹配"about/for/of/check X"模式
            r'(?:about|for|of|check|using|with|equip|buy|sell|price of|attack of|type of|description of)\s+(?:the |a |an )?([A-Za-z\s]+)'
        ]
        
        for pattern in specific_item_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                item_name = match.group(1).strip()
                # 过滤掉太短或太通用的名称
                if len(item_name) > 2 and item_name not in ['it', 'this', 'that', 'the', 'a', 'an']:
                    # 检查是否是已知物品类型
                    if any(weapon_type in item_name for weapon_type in weapon_types):
                        return item_name.title()  # 转换为标题格式
        
        # 3. 尝试从上下文中提取物品名称
        context_patterns = [
            # 匹配"I want the X"、"I need the Y"等模式
            r'(?:i want|i need|i would like|i\'d like|give me|sell me|show me)\s+(?:the |a |an )?([A-Za-z\s]+)',
            # 匹配"X is what I want"、"Y is what I need"等模式
            r'([A-Za-z\s]+)\s+is what (?:i want|i need|i\'m looking for)'
        ]
        
        for pattern in context_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                item_name = match.group(1).strip()
                # 过滤掉太短或太通用的名称
                if len(item_name) > 2 and item_name not in ['it', 'this', 'that', 'the', 'a', 'an']:
                    return item_name.title()  # 转换为标题格式
        
        # 4. 尝试提取可能的物品名称（更宽松的匹配）
        # 查找句子中的名词短语
        words = query_lower.split()
        for i, word in enumerate(words):
            if word in weapon_types and i > 0:
                # 返回形容词+武器类型
                return (words[i-1] + ' ' + word).title()
            elif i < len(words) - 1 and words[i+1] in weapon_types:
                # 返回形容词+武器类型
                return (word + ' ' + words[i+1]).title()
        
        # 5. 如果还是没找到，尝试提取任何可能的物品名称
        for word in words:
            if word in weapon_types:
                return word.title()
        
        return ""
    
    def extract_quest_name(self, query: str) -> str:
        """
        从查询中提取任务名称 - 增强版
        """
        query_lower = query.lower()
        
        # 1. 首先尝试匹配常见任务名称（精确匹配）
        common_quests = [
            'Collecting Medical Herbs',
            'Defeat the Goblins',
            'Rescue the Villagers',
            'Find the Lost Treasure',
            'Escort the Merchant',
            'Dragon Slayer Quest',
            'Lost Treasure Quest',
            'Main Quest'
        ]
        
        for quest in common_quests:
            if quest.lower() in query_lower:
                return quest
        
        # 2. 尝试匹配常见任务类型和关键词
        quest_types = [
            'quest', 'mission', 'task', 'objective', 'assignment',
            'adventure', 'expedition', 'hunt', 'journey', 'campaign'
        ]
        
        specific_info_keywords = ['reward', 'prize', 'loot', 'payment', 'compensation', 'bounty', 
                              'duration', 'description', 'details', 'price', 'cost', 'value', 
                              'attack', 'damage', 'power']

        extracted_quest_names = []

        # 特定任务名称模式
        specific_quest_patterns = [
            # 匹配"X quest"、"Y mission"等模式
            r'(?:the |a |an )?([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:' + '|'.join(quest_types) + ')',
            # 匹配"quest of X"、"mission of Y"等模式
            r'(?:the |a |an )?(?:' + '|'.join(quest_types) + ')\s+of\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})',
            # 匹配引号中的内容
            r'"([^"]+)"',
            # 匹配冒号后的内容
            r':([^,\.\?!]+)',
            # 匹配"about/for/of/check/start/complete X"模式 (后续检查是否含任务关键词)
            r'(?:about|for|of|check|start|begin|accept|take|complete|finish|do)\s+(?:the |a |an )?([A-Za-z]+(?:\s+[A-Za-z]+){0,4})',
            # 匹配"tell me about the X"模式 (后续检查是否含任务关键词)
            r'tell\s+(?:me|us)\s+about\s+(?:the |a |an )?([A-Za-z]+(?:\s+[A-Za-z]+){0,4})',
            # 匹配"what\'s the X about"模式 (后续检查是否含任务关键词)
            r'what(?:\'s|\s+is)\s+(?:the |a |an )?([A-Za-z]+(?:\s+[A-Za-z]+){0,4})\s+about'
        ]
        
        for pattern_idx, pattern in enumerate(specific_quest_patterns):
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                quest_name = match.group(1).strip()
                
                is_potential_quest = False
                if pattern_idx < 2: # Patterns 0 and 1 inherently include quest_types and are less general
                    is_potential_quest = True
                else: # For patterns 2-6 (quoted text, after colon, "about X", etc.)
                    # Assume it's a potential quest initially, unless it contains specific info keywords.
                    if not any(sik.lower() in quest_name.lower() for sik in specific_info_keywords):
                        is_potential_quest = True

                if is_potential_quest:
                    if len(quest_name) > 2 and quest_name not in ['it', 'this', 'that', 'the', 'a', 'an']:
                        if pattern_idx < 2: # For patterns 0 and 1
                            extracted_quest_names.append(quest_name.title())
                        else: # For patterns 2-6 (already filtered for specific_info_keywords)
                              # Now, ensure it's either multi-word or explicitly contains a quest type.
                            if any(qt.lower() in quest_name.lower() for qt in quest_types) or len(quest_name.split()) >= 2:
                                extracted_quest_names.append(quest_name.title())
    
        if extracted_quest_names:
            return max(extracted_quest_names, key=len) # Return the longest valid match from specific patterns

        # 3. 尝试从上下文中提取任务名称
        context_patterns = [
            r'(?:i want to|i need to|i would like to|i\'d like to)\s+(?:do|complete|finish|start|begin|accept|take)\s+(?:the |a |an )?([A-Za-z\s]+)',
            r'([A-Za-z\s]+)\s+is\s+(?:the|a|an)\s+(?:' + '|'.join(quest_types) + ')\s+(?:i want|i need|i\'m looking for)'
        ]
        
        for pattern in context_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                quest_name = match.group(1).strip()
                if len(quest_name) > 2 and quest_name not in ['it', 'this', 'that', 'the', 'a', 'an']:
                    if any(qt.lower() in quest_name.lower() for qt in quest_types) or len(quest_name.split()) >= 2:
                         extracted_quest_names.append(quest_name.title())
        
        if extracted_quest_names:
            return max(extracted_quest_names, key=len) # Return the longest valid match including context patterns

        return None # If no quest name is found after all checks
    
    def extract_all_item_names(self, query: str) -> List[str]:
        """
        从查询中提取所有物品名称
        """
        # 使用正则表达式提取物品名称
        pattern = self.param_extraction_patterns['item_name']
        matches = re.finditer(pattern, query)
        items = []
        for match in matches:
            item_name = match.group(1)
            if item_name not in items:
                items.append(item_name)
        
        # 检查常见物品名称
        for item_name in self.common_item_names:
            if item_name.lower() in query.lower() and item_name not in items:
                items.append(item_name)
        
        # 检查特定模式，如"this sword"、"that bow"等
        this_that_patterns = [
            r'\b(this|that|the)\s+(sword|bow|axe|spear|dagger|staff|wand|shield|armor|helmet|gloves|boots|ring|amulet|potion|herb|quest)\b',
            r'\b(it|one|weapon|item)\b'  # 代词指代
        ]
        
        for pattern in this_that_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                if len(items) == 0 and len(self.common_item_names) > 0:
                    # 如果没有提取到物品名称，但有指代词，使用第一个常见物品
                    items.append(self.common_item_names[0])
                    break
        
        return items
    
    def extract_all_quest_names(self, query: str) -> List[str]:
        """
        从查询中提取所有任务名称
        """
        # 首先检查特定的任务名称
        specific_quests = [
            'Collecting Medical Herbs',
            'Defeat the Goblins',
            'Rescue the Villagers',
            'Find the Lost Treasure',
            'Escort the Merchant'
        ]
        
        quests = []
        query_lower = query.lower()
        
        # 检查特定任务名称
        for quest in specific_quests:
            if quest.lower() in query_lower and quest not in quests:
                quests.append(quest)
        
        # 如果没有找到特定任务，使用正则表达式
        if not quests:
            # 使用正则表达式提取任务名称
            pattern = self.param_extraction_patterns['quest_name']
            matches = re.finditer(pattern, query)
            for match in matches:
                quest_name = match.group(1)
                if quest_name not in quests:
                    quests.append(quest_name)
            
            # 检查常见任务名称
            for item_name in self.common_item_names:
                if 'Quest' in item_name or 'Herb' in item_name or 'Collect' in item_name:
                    if item_name.lower() in query_lower and item_name not in quests:
                        quests.append(item_name)
        
        # 检查特定模式，如"this quest"、"that mission"等
        this_that_patterns = [
            r'\b(this|that|the)\s+(quest|mission|task|adventure|expedition|hunt)\b',
            r'\b(it|one|job|assignment)\b'  # 代词指代
        ]
        
        for pattern in this_that_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                if len(quests) == 0 and 'Collecting Medical Herbs' in self.common_item_names:
                    # 如果没有提取到任务名称，但有指代词，使用默认任务
                    quests.append('Collecting Medical Herbs')
                    break
        
        return quests
    
    def is_comparison_query(self, query: str) -> bool:
        """
        检查是否是比较类查询
        """
        return any(kw in query.lower() for kw in self.comparison_keywords)
    
    def extract_item_name(self, query: str) -> str:
        """
        从查询中提取物品名称
        
        Args:
            query: 用户查询文本
            
        Returns:
            str: 提取到的物品名称，如果没有找到则返回空字符串
        """
        # 定义常见物品类型和名称模式
        common_items = [
            'sword', 'bow', 'shield', 'potion', 'armor', 'helmet', 'gloves',
            'boots', 'ring', 'amulet', 'weapon', 'staff', 'wand', 'dagger',
            'axe', 'mace', 'spear', 'hammer', 'crossbow', 'arrow', 'quiver',
            'longsword', 'shortsword', 'greatsword', 'battleaxe', 'warhammer',
            'short bow', 'longbow', 'crossbow', 'rapier', 'scimitar', 'halberd'
        ]
        
        # 定义常见前缀和后缀
        prefixes = ['the', 'a', 'an', 'my', 'your', 'this', 'that', 'these', 'those', 'some', 'any']
        suffixes = ['please', 'thanks', 'thank you', 'ok', 'okay', 'hey', 'hi', 'hello', 'for me']
        
        # 转换为小写并移除标点符号
        query_lower = query.lower()
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)
        
        # 查找特定模式
        patterns = [
            # 价格查询模式
            r'(?:price|cost|value|worth|how much is|what\'?s the price of|what is the price of|how much does it cost|how much do|how much are|how much is|price of|cost of|value of|worth of|sell for|how expensive|what does it cost|what is the cost|what\'?s the cost|how much for)\s+(?:the|a|an)?\s*([\w\s]+?)(?:\?|$|\s+please|\s+thanks|\s+thank you)',
            # 攻击力查询模式
            r'(?:attack|damage|power|strength|how strong|how powerful|attack value|damage value|damage output|damage per second|dps|how much damage|what is the damage|what\'?s the damage|attack power|combat power|weapon power|how much attack|what is the attack|what\'?s the attack)\s+(?:of|for|on)?\s*(?:the|a|an)?\s*([\w\s]+?)(?:\?|$|\s+please|\s+thanks|\s+thank you)',
            # 类型查询模式
            r'(?:type|kind|category|class|what type|what kind|what category|is it a|is this a|is that a|is it an|is this an|is that an|is it the|is this the|is that the|is it|is this|is that|what is|what are|what\'?s|what is the|what are the|what does the|what do the|what can the|what is a|what are a|what does a|what do a|what can a|what is an|what are an|what does an|what do an|what can an|what\'?s the|what\'?s a|what\'?s an|what\'?s|what is|what are|what does|what do|what can)\s+(?:the|a|an)?\s*([\w\s]+?)(?:\?|$|\s+please|\s+thanks|\s+thank you)',
            # 装备/使用模式
            r'(?:equip|use|wear|wield|hold|carry|take|get|i want to use|i want to equip|i want to wear|i would like to use|i would like to equip|i would like to wear|can i use|can i equip|can i wear|may i use|may i equip|may i wear|i\'?ll take|i\'?ll buy|i\'?ll get|i\'?ll have|i\'?ll equip|i\'?ll use|i\'?ll wear|i want|i need|i would like|i\'d like|give me|sell me|buy me|get me|i take|i buy|i get|i have|i equip|i use|i wear|i want the|i need the|i would like the|i\'d like the|give me the|sell me the|buy me the|get me the|i take the|i buy the|i get the|i have the|i equip the|i use the|i wear the|i want a|i need a|i would like a|i\'d like a|give me a|sell me a|buy me a|get me a|i take a|i buy a|i get a|i have a|i equip a|i use a|i wear a|i want an|i need an|i would like an|i\'d like an|give me an|sell me an|buy me an|get me an|i take an|i buy an|i get an|i have an|i equip an|i use an|i wear an)\s+(?:the|a|an)?\s*([\w\s]+?)(?:\?|$|\s+please|\s+thanks|\s+thank you|\s+for me)',
            # 描述/信息查询模式
            r'(?:describe|tell me about|what is|what are|what does it do|what do they do|what can it do|what can they do|tell me more about|information about|details about|what do you know about|can you tell me about|explain|info on|what is the|what are the|what does the|what do the|what can the|what is a|what are a|what does a|what do a|what can a|what is an|what are an|what does an|what do an|what can an|what\'?s the|what\'?s a|what\'?s an|what\'?s|what is|what are|what does|what do|what can|what\'?s the|what\'?s a|what\'?s an|what\'?s|what is the|what are the|what does the|what do the|what can the|what is a|what are a|what does a|what do a|what can a|what is an|what are an|what does an|what do an|what can an|what\'?s the|what\'?s a|what\'?s an|what\'?s|what is|what are|what does|what do|what can)\s+(?:the|a|an)?\s*([\w\s]+?)(?:\?|$|\s+please|\s+thanks|\s+thank you)'
        ]
        
        # 尝试匹配模式
        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # 获取第一个非空的分组
                for group in match.groups():
                    if group and group.strip() not in prefixes:
                        item_name = group.strip()
                        # 移除常见后缀
                        for suffix in suffixes:
                            if item_name.endswith(' ' + suffix):
                                item_name = item_name[:-len(suffix)].strip()
                        # 检查是否包含常见物品类型
                        if any(item in item_name for item in common_items):
                            return item_name
                        # 如果查询很短，直接返回
                        if len(item_name.split()) <= 3:
                            return item_name
        
        # 如果没有匹配到模式，尝试提取名词短语
        words = query_clean.split()
        words = [w for w in words if w not in prefixes + suffixes]
        
        # 查找包含常见物品类型的短语
        for i, word in enumerate(words):
            if word in common_items:
                # 获取前面的形容词（最多2个）
                start = max(0, i - 2)
                item_phrase = ' '.join(words[start:i+1])
                return item_phrase
        
        # 如果还是没找到，返回第一个名词短语（最多3个词）
        if words:
            return ' '.join(words[:3])
        
        # 如果所有尝试都失败，返回空字符串
        return ""
    
    def extract_item_type(self, query: str) -> Optional[str]:
        """
        从查询中提取物品类型
        
        Args:
            query: 用户查询文本
            
        Returns:
            Optional[str]: 提取到的物品类型，如果没有找到则返回None
        """
        query_lower = query.lower()
        
        # 定义物品类型及其关键词，按优先级排序
        item_types = [
            # 武器类
            ('sword', ['sword', 'blade', 'rapier', 'saber', 'cutlass', 'scimitar', 'claymore', 'katana']),
            ('bow', ['bow', 'longbow', 'shortbow', 'crossbow', 'recurve bow', 'composite bow', 'hunting bow']),
            ('dagger', ['dagger', 'knife', 'dirk', 'stiletto', 'kris', 'kukri']),
            ('axe', ['axe', 'hatchet', 'battleaxe', 'handaxe', 'great axe', 'war axe']),
            ('mace', ['mace', 'club', 'morning star', 'flail', 'morgenstern']),
            ('spear', ['spear', 'javelin', 'pike', 'halberd', 'glaive', 'trident', 'partisan']),
            ('hammer', ['hammer', 'mallet', 'maul', 'warhammer', 'sledge', 'gavel']),
            ('staff', ['staff', 'staffs', 'stave', 'quarterstaff', 'wizard staff', 'archmage staff']),
            ('wand', ['wand', 'rod', 'scepter', 'baton', 'conduit']),
            ('throwing', ['throwing knife', 'throwing axe', 'shuriken', 'dart']),
            
            # 防具类
            ('shield', ['shield', 'buckler', 'tower shield', 'kite shield', 'heater shield']),
            ('armor', ['armor', 'armour', 'plate', 'mail', 'chainmail', 'splint mail', 'scale mail']),
            ('helmet', ['helmet', 'helm', 'hat', 'headgear', 'headpiece', 'coif', 'great helm', 'bascinet']),
            ('gloves', ['gloves', 'gauntlets', 'mittens', 'handwraps', 'bracers']),
            ('boots', ['boots', 'shoes', 'footwear', 'sandals', 'sabatons', 'greaves']),
            ('cloak', ['cloak', 'cape', 'mantle', 'robe']),
            
            # 饰品
            ('ring', ['ring', 'band', 'signet', 'circlet']),
            ('amulet', ['amulet', 'necklace', 'pendant', 'talisman', 'locket', 'medallion']),
            ('belt', ['belt', 'girdle', 'sash']),
            
            # 消耗品
            ('potion', ['potion', 'elixir', 'philter', 'tonic', 'draught']),
            ('scroll', ['scroll', 'parchment', 'tome', 'book', 'grimoire', 'codex', 'manual']),
            ('food', ['food', 'rations', 'meal', 'provisions', 'rations', 'trail rations']),
            ('drink', ['potion', 'flask', 'bottle', 'vial', 'waterskin', 'canteen']),
            
            # 弹药
            ('arrow', ['arrow', 'bolt', 'quarrel', 'dart', 'projectile']),
            ('quiver', ['quiver', 'case', 'holster', 'sheath']),
            
            # 工具
            ('key', ['key', 'lockpick', 'skeleton key', 'thieves tools']),
            ('torch', ['torch', 'lantern', 'lamp', 'candle', 'brazier']),
            ('instrument', ['lute', 'harp', 'flute', 'drum', 'instrument', 'horn', 'lyre']),
            ('tool', ['tool', 'kit', 'set', 'equipment', 'gear', 'apparatus']),
            
            # 材料
            ('material', ['ore', 'ingot', 'bar', 'cloth', 'leather', 'hide', 'fur', 'scale']),
            ('gem', ['gem', 'crystal', 'jewel', 'diamond', 'ruby', 'emerald', 'sapphire']),
            ('herb', ['herb', 'plant', 'root', 'leaf', 'flower', 'mushroom', 'fungus']),
            
            # 容器
            ('container', ['bag', 'pouch', 'backpack', 'sack', 'chest', 'coffer', 'crate']),
            
            # 任务物品
            ('quest', ['quest item', 'artifact', 'relic', 'heirloom', 'token', 'proof', 'evidence'])
        ]
        
        # 1. 首先检查是否有明确的类型关键词
        for item_type, keywords in item_types:
            for keyword in keywords:
                # 使用正则表达式确保匹配完整单词
                if re.search(rf'\b{re.escape(keyword)}\b', query_lower):
                    return item_type
        
        # 2. 如果没有找到明确的类型，尝试从物品名称中推断
        item_name = self.extract_item_name(query)
        if item_name:
            item_name_lower = item_name.lower()
            for item_type, keywords in item_types:
                if any(re.search(rf'\b{re.escape(keyword)}\b', item_name_lower) for keyword in keywords):
                    return item_type
        
        # 3. 检查通用类别关键词
        weapon_terms = ['weapon', 'sword', 'axe', 'mace', 'dagger', 'bow', 'crossbow', 'spear', 'hammer', 'staff', 'wand']
        armor_terms = ['armor', 'armour', 'shield', 'helmet', 'gauntlet', 'glove', 'boot', 'shoe', 'cloak', 'cape']
        
        if any(term in query_lower for term in weapon_terms):
            return 'weapon'
        elif any(term in query_lower for term in armor_terms):
            return 'armor'
        
        # 4. 检查是否有通用物品关键词
        if any(term in query_lower for term in ['item', 'object', 'thing']):
            return 'item'
        
        # 5. 如果还是没找到，返回None
        return None
    
    def extract_price_operator(self, query: str) -> Optional[str]:
        """
        从查询中提取价格操作符
        """
        pattern = self.param_extraction_patterns['item_price_operator']
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        return None
    
    def extract_attack_operator(self, query: str) -> Optional[str]:
        """
        从查询中提取攻击力操作符
        """
        pattern = self.param_extraction_patterns['item_attack_operator']
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        return None
    
    def extract_description(self, query: str) -> Optional[str]:
        """
        从查询中提取描述关键词
        """
        pattern = self.param_extraction_patterns['item_description']
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        return None
    
    def has_purchase_intent(self, query: str) -> bool:
        """
        检查查询中是否有购买或装备意图的强表达
        """
        query_lower = query.lower().strip()
        
        # 1. 检查是否匹配购买意图模式
        for pattern in self.purchase_intent_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # 2. 扩展的购买/装备动词和短语
        purchase_verbs = [
            # 基本购买动词
            'buy', 'purchase', 'get', 'obtain', 'acquire', 'procure', 'secure', 'gain',
            # 获取动词
            'take', 'have', 'grab', 'pick up', 'pick', 'select', 'choose',
            # 装备相关
            'equip', 'wear', 'use', 'wield', 'don', 'put on', 'try on', 'carry', 'hold',
            # 请求获取
            'i want', 'i need', 'i would like', 'i\'d like', 'i desire', 'i wish'
        ]
        
        # 检查是否包含购买/装备动词
        for verb in purchase_verbs:
            if re.search(r'\b' + re.escape(verb) + r'\b', query_lower):
                return True
        
        # 3. 检查常见的购买/装备表达模式
        purchase_patterns = [
            r'i\s+want\s+to\s+(?:buy|purchase|get|have|take|equip|wear|use)',
            r'i\s+need\s+to\s+(?:buy|purchase|get|have|take|equip|wear|use)',
            r'(?:can|could|may|might)\s+i\s+(?:please\s+)?(?:have|get|buy|purchase|take|equip|wear|use)',
            r'(?:i\'ll|i will)\s+(?:take|buy|get|have|equip|wear|use)',
            r'(?:i\'d like|i would like)\s+to\s+(?:buy|purchase|get|have|take|equip|wear|use)',
            r'(?:give|sell|hand|pass)\s+me\b',
            r'let me (?:have|get|buy|purchase|take|equip|wear|use)'
        ]
        
        for pattern in purchase_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # 4. 检查是否包含金钱相关词汇
        money_terms = ['price', 'cost', 'gold', 'coin', 'money', 'how much', 'pay', 'afford']
        if any(term in query_lower for term in money_terms):
            return True
        
        # 5. 检查是否包含确认或决定性词汇
        decision_terms = ['take it', 'buy it', 'get it', 'have it', 'this one', 'that one', 'i\'ll take', 'i\'ll buy', 'i\'ll get']
        if any(term in query_lower for term in decision_terms):
            return True
        
        return False
    
    def detect_intent_keywords(self, query: str) -> Dict[str, bool]:
        """
        检测查询中的关键词以确定意图
        
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
            'what does it cost', 'what is the cost', "what's the cost"
        ])
        
        # 攻击力查询意图
        is_attack_query = any(term in query_lower for term in [
            'attack', 'damage', 'power', 'strength', 'how strong', 'how powerful',
            'attack value', 'damage value', 'damage output', 'damage per second', 'dps',
            'how much damage', 'what is the damage', "what's the damage",
            'attack power', 'combat power', 'weapon power'
        ])
        
        # 类型查询意图
        is_type_query = any(term in query_lower for term in [
            'type', 'kind', 'category', 'class', 'what type', 'what kind', 'what category',
            'is it a', 'is this a', 'is that a', 'is it an', 'is this an', 'is that an',
            'is it the', 'is this the', 'is that the', 'is it', 'is this', 'is that'
        ])
        
        # 描述查询意图
        is_description_query = any(term in query_lower for term in [
            'describe', 'tell me about', 'what is', 'what are', 'what does it do',
            'what do they do', 'what can it do', 'what can they do',
            'tell me more about', 'information about', 'details about',
            'what do you know about', 'can you tell me about', 'explain', 'info on'
        ])
        
        # 装备意图
        is_equip_intent = any(term in query_lower for term in [
            'equip', 'use', 'wear', 'wield', 'hold', 'carry', 'take', 'get',
            'i want to use', 'i want to equip', 'i want to wear',
            'i would like to use', 'i would like to equip', 'i would like to wear',
            'can i use', 'can i equip', 'can i wear', 'may i use', 'may i equip', 'may i wear'
        ])
        
        # 任务查询意图
        is_quest_query = any(term in query_lower for term in [
            'quest', 'mission', 'task', 'objective', 'assignment', 'adventure', 'expedition', 'hunt', 'journey', 'campaign',
            'about the quest', 'about the mission', 'about the task',
            'what is the quest', 'what is the mission', 'what is the task',
            'tell me about the quest', 'tell me about the mission', 'tell me about the task',
            'quest info', 'mission info', 'task info'
        ])
        
        # 确认意图
        is_confirmation = any(term in query_lower for term in [
            'yes', 'confirm', 'correct', 'right', 'that\'s right', 'sure', 'okay', 'ok',
            'yep', 'yeah', 'yup', 'affirmative', 'roger', 'i confirm', 'confirmed',
            'definitely', 'absolutely', 'certainly', 'indeed', 'of course', 'by all means'
        ])
        
        # 开始/接受任务意图
        is_start_query = any(term in query_lower for term in [
            'start', 'begin', 'accept', 'take', 'go', 'now', 'let\'s go', 'i\'m ready',
            'i am ready', 'ready', 'proceed', 'continue', 'let\'s start', 'let\'s begin',
            'i accept', 'i\'ll take it', 'i want to start', 'i\'d like to start'
        ])
        
        # 搜索意图
        is_search_query = any(term in query_lower for term in [
            'search', 'find', 'look', 'show', 'list', 'available', 'what', 'which',
            'can you find', 'can you show', 'can you list', 'i\'m looking for',
            'i am looking for', 'do you have', 'is there', 'are there'
        ])
        
        return {
            'is_price_query': is_price_query,
            'is_attack_query': is_attack_query,
            'is_type_query': is_type_query,
            'is_description_query': is_description_query,
            'is_equip_intent': is_equip_intent,
            'is_quest_query': is_quest_query,
            'is_confirmation': is_confirmation,
            'is_start_query': is_start_query,
            'is_search_query': is_search_query
        }

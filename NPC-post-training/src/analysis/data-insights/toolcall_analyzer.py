import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any

class ToolCallAnalyzer:
    """分析对话和工具调用模式的工具"""
    
    def __init__(self):
        self.stats = {
            'dialogue_patterns': [],  # 存储对话和对应的工具调用
            'function_types': Counter(),  # 函数类型统计
            'parameter_patterns': defaultdict(list),  # 参数模式
            'dialogue_keywords': defaultdict(Counter),  # 对话关键词与函数的关联
            'target_item_patterns': [],  # 目标物品模式
            'dialogue_length': [],  # 对话长度
        }
        
    def analyze_data(self, data_file: str):
        """分析数据文件"""
        print(f"正在分析文件: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for conversation in data:
            # 遍历每个对话的每个回合
            for turn_key in [k for k in conversation.keys() if k.startswith('turn_')]:
                turn_data = conversation[turn_key]
                if 'dialogue' in turn_data and 'gold_functions' in turn_data:
                    self._analyze_turn(turn_data)
        
        self._calculate_statistics()
        return self.stats
    
    def _analyze_turn(self, turn_data: Dict):
        """分析单个回合的数据"""
        dialogue = turn_data['dialogue']
        gold_functions = turn_data['gold_functions']
        
        # 获取最后一个用户消息
        last_user_message = ""
        target_items = []
        for msg in reversed(dialogue):
            if msg['speaker'] == 'player':
                last_user_message = msg['text']
                target_items = msg.get('target_item', [])
                break
        
        # 分析函数调用
        for func in gold_functions:
            func_name = func['name']
            parameters = func['parameters']
            
            # 统计函数类型
            self.stats['function_types'][func_name] += 1
            
            # 收集对话模式
            dialogue_pattern = {
                'user_message': last_user_message,
                'function_name': func_name,
                'parameters': parameters,
                'target_items': target_items
            }
            self.stats['dialogue_patterns'].append(dialogue_pattern)
            
            # 分析参数模式
            for param_name, param_value in parameters.items():
                self.stats['parameter_patterns'][func_name].append({
                    'param_name': param_name,
                    'param_value': param_value,
                    'in_target_items': self._check_in_target_items(param_value, target_items)
                })
            
            # 分析对话关键词
            keywords = self._extract_keywords(last_user_message)
            for keyword in keywords:
                self.stats['dialogue_keywords'][func_name][keyword] += 1
            
            # 分析目标物品模式
            if target_items:
                self.stats['target_item_patterns'].append({
                    'target_items': target_items,
                    'function_name': func_name,
                    'parameters': parameters
                })
            
            # 记录对话长度
            self.stats['dialogue_length'].append(len(last_user_message.split()))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单的关键词提取，可以根据需要改进
        text = text.lower()
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = text.split()
        # 过滤停用词
        stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                     'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords
    
    def _check_in_target_items(self, param_value: str, target_items: List[Dict]) -> bool:
        """检查参数值是否在目标物品中"""
        if not target_items:
            return False
        
        for item in target_items:
            if 'name' in item and item['name'].lower() == param_value.lower():
                return True
        return False
    
    def _calculate_statistics(self):
        """计算统计信息"""
        # 计算函数类型分布
        total_functions = sum(self.stats['function_types'].values())
        function_distribution = {}
        for func_name, count in self.stats['function_types'].items():
            function_distribution[func_name] = {
                'count': count,
                'percentage': round(count / total_functions * 100, 2) if total_functions > 0 else 0
            }
        self.stats['function_distribution'] = function_distribution
        
        # 计算参数来源分布
        param_source_distribution = {}
        for func_name, params in self.stats['parameter_patterns'].items():
            from_target_item = sum(1 for p in params if p['in_target_items'])
            total_params = len(params)
            param_source_distribution[func_name] = {
                'from_target_item': from_target_item,
                'from_dialogue': total_params - from_target_item,
                'percentage_from_target': round(from_target_item / total_params * 100, 2) if total_params > 0 else 0
            }
        self.stats['param_source_distribution'] = param_source_distribution
        
        # 计算关键词-函数关联强度
        keyword_function_association = {}
        for func_name, keywords in self.stats['dialogue_keywords'].items():
            top_keywords = keywords.most_common(10)
            keyword_function_association[func_name] = [
                {'keyword': kw, 'count': cnt} for kw, cnt in top_keywords
            ]
        self.stats['keyword_function_association'] = keyword_function_association
    
    def save_results(self, output_dir: str):
        """保存分析结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存总体统计结果
        with open(os.path.join(output_dir, 'toolcall_stats.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'function_distribution': self.stats['function_distribution'],
                'param_source_distribution': self.stats['param_source_distribution'],
                'keyword_function_association': self.stats['keyword_function_association']
            }, f, indent=4, ensure_ascii=False)
        
        # 保存对话-函数调用模式
        with open(os.path.join(output_dir, 'dialogue_function_patterns.json'), 'w', encoding='utf-8') as f:
            json.dump(self.stats['dialogue_patterns'], f, indent=4, ensure_ascii=False)
        
        # 保存目标物品模式
        with open(os.path.join(output_dir, 'target_item_patterns.json'), 'w', encoding='utf-8') as f:
            json.dump(self.stats['target_item_patterns'], f, indent=4, ensure_ascii=False)
        
        print(f"分析结果已保存到: {output_dir}")

if __name__ == '__main__':
    # 设置文件路径
    data_file = '/path/to/npc-rl/data/task1_sample.json'
    output_dir = '/path/to/npc-rl/src/cpdc-boost/data-insights/results/toolcall'
    
    # 创建分析器并分析数据
    analyzer = ToolCallAnalyzer()
    analyzer.analyze_data(data_file)
    analyzer.save_results(output_dir)
    
    print("分析完成！")

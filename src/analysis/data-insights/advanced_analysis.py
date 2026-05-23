import json
import os
from collections import defaultdict, Counter
import statistics
from typing import Dict, List, Any, Tuple
import re
from datetime import datetime

class AdvancedDataAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = self._get_data_files()
        self.stats = {
            'total_files': 0,
            'turn_stats': [],
            'gold_response_length': [],
            'gold_functions_count': [],
            'function_types': set(),
            'dialogue_patterns': defaultdict(int),
            'knowledge_info_count': [],
            'knowledge_types': set(),
            'worldview_keywords': defaultdict(int),
            'quest_difficulty': Counter(),
            'time_analysis': defaultdict(int),
            'emotion_analysis': defaultdict(int),
            'item_distribution': defaultdict(int),
            'dialogue_act_analysis': defaultdict(int)
        }

    def _get_data_files(self) -> List[str]:
        """获取数据目录中的所有JSON文件"""
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                if f.endswith('.json') and os.path.isfile(os.path.join(self.data_dir, f))]

    def analyze_files(self):
        """分析所有数据文件"""
        for file_path in self.files:
            print(f"分析文件: {file_path}...")
            self._analyze_file(file_path)
        
        self._calculate_statistics()
        self._save_insights()
        self._print_results()

    def _analyze_file(self, file_path: str):
        """分析单个文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"解析错误 {file_path}: {e}")
                return

        for item in data:
            self.stats['total_files'] += 1
            
            # 分析对话轮次
            self.stats['turn_stats'].append(item['total_turn'])
            
            # 分析世界观关键词
            self._analyze_worldview(item.get('worldview', ''))
            
            # 分析任务难度
            self._analyze_quest_difficulty(item)
            
            # 分析时间模式
            self._analyze_time_patterns(item)
            
            # 分析对话行为
            self._analyze_dialogue_acts(item)
            
            # 分析黄金回复和函数
            self._analyze_gold_responses(item)
            
            # 分析知识库
            self._analyze_knowledge(item)
            
            # 分析物品分布
            self._analyze_item_distribution(item)

    def _analyze_worldview(self, text: str):
        """分析世界观关键词"""
        keywords = [
            'monster', 'guild', 'quest', 'adventurer', 'weapon',
            'magic', 'gold', 'battle', 'town', 'village',
            'kingdom', 'dungeon', 'treasure', 'dragon', 'spell'
        ]
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                self.stats['worldview_keywords'][kw] += 1

    def _analyze_quest_difficulty(self, item: Dict):
        """分析任务难度分布"""
        difficulty = item.get('state', {}).get('quest_difficulty', 'unknown')
        if difficulty != 'unknown':
            self.stats['quest_difficulty'][difficulty] += 1

    def _analyze_time_patterns(self, item: Dict):
        """分析时间模式"""
        time_info = item.get('state', {}).get('time', '').lower()
        if 'morning' in time_info:
            self.stats['time_analysis']['morning'] += 1
        elif 'afternoon' in time_info:
            self.stats['time_analysis']['afternoon'] += 1
        elif 'evening' in time_info or 'night' in time_info:
            self.stats['time_analysis']['evening_night'] += 1

    def _analyze_dialogue_acts(self, item: Dict):
        """分析对话行为"""
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'can you', 'could you']
        
        for turn in range(item['total_turn']):
            turn_key = f'turn_{turn}'
            if turn_key in item:
                for msg in item[turn_key]['dialogue']:
                    text = msg['text'].lower()
                    if any(q in text for q in question_words):
                        self.stats['dialogue_act_analysis']['question'] += 1
                    if '!' in text:
                        self.stats['dialogue_act_analysis']['exclamation'] += 1
                    if '?' in text:
                        self.stats['dialogue_act_analysis']['question_mark'] += 1
                    if len(text.split()) > 20:  # 长回复
                        self.stats['dialogue_act_analysis']['long_response'] += 1

    def _analyze_gold_responses(self, item: Dict):
        """分析黄金回复"""
        if 'gold_response' in item:
            response = item['gold_response']
            self.stats['gold_response_length'].append(len(response.split()))
            
            # 情感分析
            positive_words = ['great', 'good', 'excellent', 'wonderful', 'happy', 'thanks', 'thank you']
            negative_words = ['bad', 'terrible', 'awful', 'sad', 'angry', 'hate']
            
            response_lower = response.lower()
            if any(word in response_lower for word in positive_words):
                self.stats['emotion_analysis']['positive'] += 1
            if any(word in response_lower for word in negative_words):
                self.stats['emotion_analysis']['negative'] += 1
                
        if 'gold_functions' in item:
            self.stats['gold_functions_count'].append(len(item['gold_functions']))
            for func in item['gold_functions']:
                self.stats['function_types'].add(func['name'])

    def _analyze_knowledge(self, item: Dict):
        """分析知识库"""
        if 'knowledge' in item and 'knowledge_info' in item['knowledge']:
            info = item['knowledge']['knowledge_info']
            self.stats['knowledge_info_count'].append(len(info))
            for item in info:
                if 'type' in item:
                    self.stats['knowledge_types'].add(item['type'])

    def _analyze_item_distribution(self, item: Dict):
        """分析物品分布"""
        if 'knowledge' in item and 'knowledge_info' in item['knowledge']:
            for item_info in item['knowledge']['knowledge_info']:
                if 'type' in item_info:
                    self.stats['item_distribution'][item_info['type']] += 1

    def _calculate_statistics(self):
        """计算统计信息"""
        # 基本统计
        self.stats['turns_mean'] = statistics.mean(self.stats['turn_stats']) if self.stats['turn_stats'] else 0
        self.stats['turns_median'] = statistics.median(self.stats['turn_stats']) if self.stats['turn_stats'] else 0
        
        # 回复长度统计
        if self.stats['gold_response_length']:
            self.stats['response_length_mean'] = statistics.mean(self.stats['gold_response_length'])
            self.stats['response_length_median'] = statistics.median(self.stats['gold_response_length'])
        
        # 函数统计
        if self.stats['gold_functions_count']:
            self.stats['functions_mean'] = statistics.mean(self.stats['gold_functions_count'])
            self.stats['functions_median'] = statistics.median(self.stats['gold_functions_count'])
        
        # 知识库统计
        if self.stats['knowledge_info_count']:
            self.stats['knowledge_info_mean'] = statistics.mean(self.stats['knowledge_info_count'])
            self.stats['knowledge_info_median'] = statistics.median(self.stats['knowledge_info_count'])

    def _save_insights(self):
        """保存洞察结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 'insights')
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备可序列化的统计信息
        serializable_stats = {}
        for key, value in self.stats.items():
            if isinstance(value, (set,)):
                serializable_stats[key] = list(value)
            elif isinstance(value, (defaultdict, Counter)):
                serializable_stats[key] = dict(value)
            else:
                serializable_stats[key] = value
        
        # 保存完整统计信息
        with open(os.path.join(output_dir, f'full_analysis_{timestamp}.json'), 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        # 保存关键洞察
        key_insights = {
            'total_conversations': self.stats['total_files'],
            'avg_turns': round(self.stats['turns_mean'], 2),
            'most_common_dialogue_pattern': max(self.stats['dialogue_patterns'].items(), key=lambda x: x[1])[0] if self.stats['dialogue_patterns'] else 'N/A',
            'most_common_item_type': max(self.stats['item_distribution'].items(), key=lambda x: x[1])[0] if self.stats['item_distribution'] else 'N/A',
            'quest_difficulty_distribution': dict(self.stats['quest_difficulty']),
            'time_distribution': dict(self.stats['time_analysis']),
            'emotion_distribution': dict(self.stats['emotion_analysis']),
            'top_worldview_keywords': dict(sorted(self.stats['worldview_keywords'].items(), key=lambda x: x[1], reverse=True)[:5]) if self.stats['worldview_keywords'] else {}
        }
        
        with open(os.path.join(output_dir, f'key_insights_{timestamp}.json'), 'w', encoding='utf-8') as f:
            json.dump(key_insights, f, indent=2, ensure_ascii=False)

    def _print_results(self):
        """打印分析结果"""
        print("\n=== 数据分析结果 ===")
        print(f"分析文件总数: {self.stats['total_files']}")
        
        print("\n=== 对话轮次统计 ===")
        print(f"平均对话轮次: {self.stats['turns_mean']:.2f}")
        print(f"中位数对话轮次: {self.stats['turns_median']}")
        
        if 'response_length_mean' in self.stats:
            print("\n=== 黄金回复统计 ===")
            print(f"平均回复长度(词): {self.stats['response_length_mean']:.2f}")
            print(f"中位数回复长度(词): {self.stats['response_length_median']}")
        
        if self.stats['quest_difficulty']:
            print("\n=== 任务难度分布 ===")
            for diff, count in self.stats['quest_difficulty'].most_common():
                print(f"  {diff}: {count} 次")
        
        if self.stats['time_analysis']:
            print("\n=== 时间分布 ===")
            for time, count in self.stats['time_analysis'].items():
                print(f"  {time}: {count} 次")
        
        if self.stats['emotion_analysis']:
            print("\n=== 情感分析 ===")
            for emotion, count in self.stats['emotion_analysis'].items():
                print(f"  {emotion}: {count} 次")
        
        if self.stats['item_distribution']:
            print("\n=== 物品类型分布 ===")
            for item_type, count in sorted(self.stats['item_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {item_type}: {count} 次")
        
        if self.stats['worldview_keywords']:
            print("\n=== 世界观关键词 ===")
            for kw, count in sorted(self.stats['worldview_keywords'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {kw}: {count} 次")

if __name__ == "__main__":
    analyzer = AdvancedDataAnalyzer('/path/to/npc-rl/data')
    analyzer.analyze_files()

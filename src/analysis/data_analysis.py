import json
import os
from collections import defaultdict
import statistics
from typing import Dict, List, Any

class DataAnalyzer:
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
            'knowledge_types': set()
        }

    def _get_data_files(self) -> List[str]:
        """Get all JSON files in the data directory"""
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                if f.endswith('.json')]

    def analyze_files(self):
        """Analyze all data files"""
        for file_path in self.files:
            print(f"Analyzing {file_path}...")
            self._analyze_file(file_path)
        
        self._calculate_statistics()
        self._print_results()

    def _analyze_file(self, file_path: str):
        """Analyze a single file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error parsing {file_path}: {e}")
                return

        for item in data:
            self.stats['total_files'] += 1
            
            # Analyze turns
            self.stats['turn_stats'].append(item['total_turn'])
            
            # Analyze gold responses and functions
            if 'gold_response' in item:
                self.stats['gold_response_length'].append(len(item['gold_response']))
            if 'gold_functions' in item:
                self.stats['gold_functions_count'].append(len(item['gold_functions']))
                for func in item['gold_functions']:
                    self.stats['function_types'].add(func['name'])

            # Analyze dialogue patterns
            for turn in range(item['total_turn']):
                turn_key = f'turn_{turn}'
                if turn_key in item:
                    dialogue = item[turn_key]['dialogue']
                    pattern = self._get_dialogue_pattern(dialogue)
                    self.stats['dialogue_patterns'][pattern] += 1

            # Analyze knowledge
            if 'knowledge' in item and 'knowledge_info' in item['knowledge']:
                info = item['knowledge']['knowledge_info']
                self.stats['knowledge_info_count'].append(len(info))
                for item in info:
                    if 'type' in item:
                        self.stats['knowledge_types'].add(item['type'])

    def _get_dialogue_pattern(self, dialogue: List[Dict[str, str]]) -> str:
        """Get dialogue pattern (e.g., player-npc-player-npc)"""
        speakers = [d['speaker'] for d in dialogue]
        return '-'.join(speakers)

    def _calculate_statistics(self):
        """Calculate statistics from collected data"""
        self.stats['turns_mean'] = statistics.mean(self.stats['turn_stats'])
        self.stats['turns_median'] = statistics.median(self.stats['turn_stats'])
        
        if self.stats['gold_response_length']:
            self.stats['response_length_mean'] = statistics.mean(self.stats['gold_response_length'])
            self.stats['response_length_median'] = statistics.median(self.stats['gold_response_length'])
        
        if self.stats['gold_functions_count']:
            self.stats['functions_mean'] = statistics.mean(self.stats['gold_functions_count'])
            self.stats['functions_median'] = statistics.median(self.stats['gold_functions_count'])
        
        if self.stats['knowledge_info_count']:
            self.stats['knowledge_info_mean'] = statistics.mean(self.stats['knowledge_info_count'])
            self.stats['knowledge_info_median'] = statistics.median(self.stats['knowledge_info_count'])

    def _print_results(self):
        """Print analysis results"""
        print("\n=== Data Analysis Results ===")
        print(f"Total files analyzed: {self.stats['total_files']}")
        print(f"\nTurn Statistics:")
        print(f"  Mean turns per conversation: {self.stats['turns_mean']:.2f}")
        print(f"  Median turns per conversation: {self.stats['turns_median']}")
        
        if 'response_length_mean' in self.stats:
            print(f"\nGold Response Statistics:")
            print(f"  Mean response length: {self.stats['response_length_mean']:.2f}")
            print(f"  Median response length: {self.stats['response_length_median']}")
        
        if 'functions_mean' in self.stats:
            print(f"\nGold Functions Statistics:")
            print(f"  Mean functions per response: {self.stats['functions_mean']:.2f}")
            print(f"  Median functions per response: {self.stats['functions_median']}")
            print(f"  Unique function types: {len(self.stats['function_types'])}")
            print("  Function types:")
            for func_type in sorted(self.stats['function_types']):
                print(f"    - {func_type}")

        print(f"\nDialogue Patterns:")
        for pattern, count in sorted(self.stats['dialogue_patterns'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count} occurrences")

        if 'knowledge_info_mean' in self.stats:
            print(f"\nKnowledge Statistics:")
            print(f"  Mean knowledge items per conversation: {self.stats['knowledge_info_mean']:.2f}")
            print(f"  Median knowledge items per conversation: {self.stats['knowledge_info_median']}")
            print(f"  Unique knowledge types: {len(self.stats['knowledge_types'])}")
            print("  Knowledge types:")
            for knowledge_type in sorted(self.stats['knowledge_types']):
                print(f"    - {knowledge_type}")

if __name__ == "__main__":
    analyzer = DataAnalyzer('/path/to/npc-rl/data')
    analyzer.analyze_files()

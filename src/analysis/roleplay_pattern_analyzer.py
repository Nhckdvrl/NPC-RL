import json
import os
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
import torch

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class RoleplayAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.sentiment_analyzer = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize Qwen tokenizer and sentiment analyzer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct", 
                trust_remote_code=True
            )
            print("Successfully loaded Qwen tokenizer")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
        
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("Successfully loaded sentiment analyzer")
        except Exception as e:
            print(f"Error loading sentiment analyzer: {e}")
    
    @staticmethod
    def load_data(file_path: str) -> List[Dict]:
        """Load conversation data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def extract_messages(data: List[Dict], role: str = 'all') -> List[str]:
        """Extract messages by role (all, human, gpt, system)."""
        messages = []
        for conv in data:
            if not conv.get('conversations'):
                continue
            for msg in conv['conversations']:
                if role == 'all' or msg.get('from') == role:
                    messages.append(msg.get('value', ''))
        return messages
    
    def save_unique_responses(self, messages: List[str], output_path: str) -> None:
        """Save unique GPT responses to a file."""
        if not messages:
            print("No messages to save")
            return
            
        # Get unique messages (case-insensitive)
        unique_messages = sorted(list({msg.lower(): msg for msg in messages if msg.strip()}.values()))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, msg in enumerate(unique_messages, 1):
                f.write(f"--- Response {i} ---\n")
                f.write(f"{msg}\n\n")
        
        print(f"\nSaved {len(unique_messages)} unique responses to {output_path}")
        return unique_messages
    
    def analyze_response_patterns(self, messages: List[str]) -> Dict:
        """Analyze patterns in roleplay responses."""
        if not messages:
            return {}
            
        patterns = {
            'question': r'\?$',
            'exclamation': r'!$',
            'ellipsis': r'\.{3,}$',
            'greeting': r'^(Hi|Hello|Hey|Greetings)',
            'acknowledgment': r'^(I see|Got it|Understood|Noted|Thanks)',
            'suggestion': r'\b(can\s+try|suggest|recommend|how\s+about|why\s+not)\b',
            'emotion': r'\b(wow|great|amazing|wonderful|terrible|unfortunately|sadly)\b'
        }
        
        pattern_counts = {name: 0 for name in patterns}
        
        for msg in messages:
            for name, pattern in patterns.items():
                if re.search(pattern, msg, re.IGNORECASE):
                    pattern_counts[name] += 1
        
        # Calculate percentages
        total = len(messages)
        pattern_percentages = {
            name: (count / total) * 100 
            for name, count in pattern_counts.items()
        }
        
        return {
            'total_responses': total,
            'pattern_counts': pattern_counts,
            'pattern_percentages': pattern_percentages
        }
    
    def analyze_sentiment(self, messages: List[str], sample_size: int = 100) -> Dict:
        """Analyze sentiment of messages."""
        if not self.sentiment_analyzer or not messages:
            return {}
            
        sample = messages[:sample_size]
        try:
            sentiments = self.sentiment_analyzer(sample)
            sentiment_counts = Counter([s['label'] for s in sentiments])
            return {
                'samples_analyzed': len(sentiments),
                'sentiment_distribution': dict(sentiment_counts)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_token_stats(self, messages: List[str]) -> Dict:
        """Analyze token statistics."""
        if not self.tokenizer or not messages:
            return {}
            
        all_tokens = []
        token_lengths = []
        
        for msg in messages:
            tokens = self.tokenizer.tokenize(msg)
            all_tokens.extend(tokens)
            token_lengths.append(len(tokens))
        
        if not token_lengths:
            return {}
            
        return {
            'total_tokens': len(all_tokens),
            'avg_tokens_per_message': sum(token_lengths) / len(token_lengths),
            'max_tokens': max(token_lengths),
            'min_tokens': min(token_lengths),
            'vocab_size': len(set(all_tokens)),
            'token_length_distribution': dict(Counter(token_lengths))
        }
    
    def get_common_phrases(self, messages: List[str], min_length: int = 2, max_length: int = 4) -> List[Tuple[str, int]]:
        """Find common n-grams in the messages."""
        if not messages:
            return []
            
        phrases = []
        for msg in messages:
            words = msg.lower().split()
            for n in range(min_length, min(max_length + 1, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    if len(phrase) > 10:  # Filter out very short phrases
                        phrases.append(phrase)
        
        return Counter(phrases).most_common(20)

def main():
    # Configuration
    input_file = '/path/to/npc-rl/data/sft/task1/stage_1.json'
    output_dir = '/path/to/npc-rl/src/cpdc-boost/data-insights'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = RoleplayAnalyzer()
    
    # Load and process data
    print("\nLoading data...")
    data = analyzer.load_data(input_file)
    
    # Extract different types of messages
    all_messages = analyzer.extract_messages(data, 'all')
    gpt_responses = analyzer.extract_messages(data, 'gpt')
    human_messages = analyzer.extract_messages(data, 'human')
    
    print(f"\nTotal conversations: {len(data)}")
    print(f"Total messages: {len(all_messages)}")
    print(f"AI responses: {len(gpt_responses)}")
    print(f"Human messages: {len(human_messages)}")
    
    # Analyze response patterns
    print("\n=== Response Pattern Analysis ===")
    pattern_analysis = analyzer.analyze_response_patterns(gpt_responses)
    
    print("\nCommon Patterns in AI Responses:")
    for pattern, count in pattern_analysis['pattern_counts'].items():
        percentage = pattern_analysis['pattern_percentages'][pattern]
        print(f"- {pattern.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Analyze sentiment
    print("\n=== Sentiment Analysis ===")
    sentiment = analyzer.analyze_sentiment(gpt_responses)
    if 'error' in sentiment:
        print(f"Sentiment analysis error: {sentiment['error']}")
    else:
        print("Sentiment distribution in AI responses (sample):")
        for sentiment_type, count in sentiment['sentiment_distribution'].items():
            print(f"- {sentiment_type}: {count} ({(count/sentiment['samples_analyzed'])*100:.1f}%)")
    
    # Token statistics
    if analyzer.tokenizer:
        print("\n=== Token Analysis ===")
        token_stats = analyzer.analyze_token_stats(gpt_responses)
        print(f"Average tokens per response: {token_stats['avg_tokens_per_message']:.1f}")
        print(f"Vocabulary size: {token_stats['vocab_size']}")
        print(f"Response length (tokens): {token_stats['min_tokens']}-{token_stats['max_tokens']}")
    
    # Common phrases
    print("\n=== Common Phrases in AI Responses ===")
    common_phrases = analyzer.get_common_phrases(gpt_responses)
    for phrase, count in common_phrases:
        print(f"- '{phrase}': {count} occurrences")
    
    # Save unique responses
    output_path = '/path/to/npc-rl/src/cpdc-boost/data-insights/roleplay-v1-response.txt'
    unique_responses = analyzer.save_unique_responses(gpt_responses, output_path)
    
    print("\nAnalysis complete!")
    print(f"Total unique responses saved: {len(unique_responses) if unique_responses else 0}")

if __name__ == "__main__":
    main()

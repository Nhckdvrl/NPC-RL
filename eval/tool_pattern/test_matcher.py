import json
import sys
import os

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/path/to/npc-rl/eval/tool_pattern')
sys.path.append('/path/to/npc-rl/agents')

# 导入原始的ToolPatternMatcher
from tool_pattern_matcher import ToolPatternMatcher as OriginalMatcher

# 导入重构后的ToolPatternMatcher
from tool_pattern_matcher_new import ToolPatternMatcher as NewMatcher

def test_with_sample_queries():
    """
    使用样例查询测试两个匹配器的结果
    """
    # 初始化匹配器
    original_matcher = OriginalMatcher()
    new_matcher = NewMatcher()
    
    # 测试查询列表
    test_queries = [
        "I want to buy a sword",
        "Tell me about the Short Sword",
        "How much does the Short Sword cost?",
        "What's the attack power of the Short Sword?",
        "What type of weapon is the Short Sword?",
        "I'll take the Short Sword",
        "I want to equip the Short Sword",
        "Show me available quests",
        "Tell me about the Collecting Medical Herbs quest",
        "What's the reward for Collecting Medical Herbs?",
        "How long does Collecting Medical Herbs take?",
        "I want to select Collecting Medical Herbs",
        "Let's start the Collecting Medical Herbs quest",
        "Yes, I confirm"
    ]
    
    # 测试目标物品
    target_items = [{"name": "Short Sword"}]
    
    print("\n===== 测试原始匹配器与重构匹配器 =====\n")
    
    for i, query in enumerate(test_queries):
        print(f"\n查询 {i+1}: '{query}'")
        
        # 获取原始匹配器结果
        original_result = original_matcher.match_function(query, target_items)
        print(f"原始匹配器结果: {json.dumps(original_result, ensure_ascii=False, indent=2)}")
        
        # 获取新匹配器结果
        new_result = new_matcher.match_function(query, target_items)
        print(f"新匹配器结果: {json.dumps(new_result, ensure_ascii=False, indent=2)}")
        
        # 比较结果
        if original_result.get('name') == new_result.get('name'):
            match_status = "✓ 函数名匹配"
        else:
            match_status = "✗ 函数名不匹配"
        
        # 比较参数
        original_params = original_result.get('parameters', {})
        new_params = new_result.get('parameters', {})
        
        if original_params == new_params:
            param_status = "✓ 参数完全匹配"
        else:
            param_status = "✗ 参数不完全匹配"
            
        print(f"比较结果: {match_status}, {param_status}")

def test_with_real_data():
    """
    使用实际数据集测试匹配器
    """
    # 加载训练数据
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'task1_train.json')
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法加载训练数据: {e}")
        return
    
    # 初始化匹配器
    original_matcher = OriginalMatcher()
    new_matcher = NewMatcher()
    
    print("\n===== 使用实际数据测试匹配器 =====\n")
    print(f"加载了 {len(data)} 条训练数据")
    
    # 统计匹配情况
    total_samples = 0
    original_function_matches = 0
    original_param_matches = 0
    new_function_matches = 0
    new_param_matches = 0
    
    # 按函数类型统计
    function_stats = {}
    
    # 随机选择20条数据进行测试
    import random
    samples = random.sample(data, min(20, len(data)))
    
    for i, sample in enumerate(samples):
        # 获取用户查询和目标函数
        dialogue = sample.get('dialogue', [])
        if not dialogue:
            continue
            
        # 获取最后一个用户查询
        user_queries = [turn['content'] for turn in dialogue if turn['role'] == 'user']
        if not user_queries:
            continue
            
        query = user_queries[-1]
        
        # 获取目标函数
        gold_functions = sample.get('gold_functions', [])
        if not gold_functions:
            continue
            
        gold_function = gold_functions[0] if gold_functions else None
        if not gold_function:
            continue
            
        total_samples += 1
        
        # 更新函数类型统计
        function_name = gold_function.get('name', '')
        if function_name not in function_stats:
            function_stats[function_name] = {
                'total': 0,
                'original_matches': 0,
                'new_matches': 0
            }
        function_stats[function_name]['total'] += 1
        
        print(f"\n样本 {i+1}:")
        print(f"用户查询: '{query}'")
        print(f"目标函数: {json.dumps(gold_function, ensure_ascii=False)}")
        
        # 获取原始匹配器结果
        original_result = original_matcher.match_function(query)
        print(f"原始匹配器结果: {json.dumps(original_result, ensure_ascii=False)}")
        
        # 获取新匹配器结果
        new_result = new_matcher.match_function(query)
        print(f"新匹配器结果: {json.dumps(new_result, ensure_ascii=False)}")
        
        # 比较原始匹配器结果与目标函数
        if original_result.get('name') == gold_function.get('name'):
            original_function_matches += 1
            print("✓ 原始匹配器函数名匹配目标")
            
            # 比较参数
            gold_params = gold_function.get('parameters', {})
            original_params = original_result.get('parameters', {})
            
            # 简单比较参数键是否匹配
            if set(gold_params.keys()) == set(original_params.keys()):
                original_param_matches += 1
                print("✓ 原始匹配器参数键匹配目标")
            else:
                print("✗ 原始匹配器参数键不匹配目标")
        else:
            print("✗ 原始匹配器函数名不匹配目标")
        
        # 比较新匹配器结果与目标函数
        if new_result.get('name') == gold_function.get('name'):
            new_function_matches += 1
            function_stats[function_name]['new_matches'] += 1
            print("✓ 新匹配器函数名匹配目标")
            
            # 比较参数
            gold_params = gold_function.get('parameters', {})
            new_params = new_result.get('parameters', {})
            
            # 简单比较参数键是否匹配
            if set(gold_params.keys()) == set(new_params.keys()):
                new_param_matches += 1
                print("✓ 新匹配器参数键匹配目标")
            else:
                print("✗ 新匹配器参数键不匹配目标")
        else:
            print("✗ 新匹配器函数名不匹配目标")
    
    # 打印统计结果
    if total_samples > 0:
        print(f"\n统计结果 (基于 {total_samples} 个样本):")
        print("原始匹配器:\n-------------------")
        print(f"函数名匹配率: {original_function_matches/total_samples:.2%}")
        print(f"参数键匹配率: {original_param_matches/total_samples:.2%}")
        
        print("\n新匹配器:\n-------------------")
        print(f"函数名匹配率: {new_function_matches/total_samples:.2%}")
        print(f"参数键匹配率: {new_param_matches/total_samples:.2%}")
        
        # 按函数类型打印统计结果
        print("\n按函数类型统计:\n-------------------")
        for func_name, stats in function_stats.items():
            if stats['total'] > 0:
                match_rate = stats['new_matches'] / stats['total']
                print(f"{func_name}: {match_rate:.2%} ({stats['new_matches']}/{stats['total']})")
        
        # 比较两个匹配器的性能
        if original_function_matches > 0 and new_function_matches > 0:
            improvement = (new_function_matches - original_function_matches) / original_function_matches * 100
            print(f"\n函数名匹配改进: {improvement:.2f}%")

def main():
    """
    主函数
    """
    # 测试样例查询
    test_with_sample_queries()
    
    # 测试实际数据
    test_with_real_data()

if __name__ == "__main__":
    main()

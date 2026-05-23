import unittest
import json
import os
import sys
import pathlib

# 添加项目根目录到 Python 路径
current_dir = pathlib.Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# 导入需要测试的函数
from src.reward_score.cpdc_toolcall_executor import calculate_score, Executor

class TestSftDataMatching(unittest.TestCase):
    """测试 SFT 数据中的工具调用匹配情况"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 设置环境变量
        os.environ["NO_THINK"] = "true"
        
        # 加载 SFT 数据
        self.sft_data_path = os.path.join(project_root, "data", "sft", "task1", "stage_0_test.json")
        with open(self.sft_data_path, 'r', encoding='utf-8') as f:
            self.sft_data = json.load(f)
    
    def test_sft_data_example(self):
        """测试 SFT 数据中的工具调用匹配"""
        # 从 SFT 数据中提取第二个对话（包含工具调用）
        conversation = self.sft_data[1]["conversations"]
        
        # 提取 gold 标准（从 function_call 中）
        gold_function_str = None
        for msg in conversation:
            if msg["from"] == "function_call":
                gold_function_str = msg["value"]
                break
        
        self.assertIsNotNone(gold_function_str, "未找到 gold 标准工具调用")
        
        # 解析 gold 标准
        gold_function = json.loads(gold_function_str)
        gold_standard = [{
            "name": gold_function["name"],
            "parameters": gold_function["arguments"]
        }]
        
        # 创建预测工具调用
        prediction_exact = f"""<tool_call>
        {{
            "name": "{gold_function["name"]}",
            "parameters": {json.dumps(gold_function["arguments"])}
        }}
        </tool_call>
        """
        
        # 创建参数不同的预测
        different_args = gold_function["arguments"].copy()
        if "quest_reward" in different_args:
            different_args["quest_reward"] = "200 gold"  # 修改参数值
        
        prediction_different = f"""<tool_call>
        {{
            "name": "{gold_function["name"]}",
            "parameters": {json.dumps(different_args)}
        }}
        </tool_call>
        """
        
        # 计算得分
        score_exact = calculate_score(prediction_exact, gold_standard)
        score_different = calculate_score(prediction_different, gold_standard)
        
        print(f"SFT 数据完全匹配得分: {score_exact}")
        print(f"SFT 数据参数不同得分: {score_different}")
        
        # 打印详细信息
        print(f"\nSFT 数据测试详细信息:")
        print(f"- Gold 标准: {gold_standard}")
        print(f"- 完全匹配预测: {prediction_exact}")
        print(f"- 参数不同预测: {prediction_different}")
        
        # 断言
        self.assertEqual(score_exact, 1.0, "SFT 数据完全匹配应得分 1.0")
        self.assertEqual(score_different, 0.0, "SFT 数据参数不同应得分 0.0")
    
    def test_multiple_sft_examples(self):
        """测试多个 SFT 数据示例"""
        # 这里我们可以添加更多的 SFT 数据测试，如果有需要的话
        pass

if __name__ == '__main__':
    unittest.main()

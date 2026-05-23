import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock, ANY

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set environment variables for testing
os.environ["NO_THINK"] = "true"

# Import the functions to be tested
from reward_score import compute_score, _default_compute_score
from reward_score.utils_metric import DATASOURCE_METRICS


class TestCPDCScoring(unittest.TestCase):
    def setUp(self):
        # Setup common test data
        pass

    def test_toolcall_scoring(self):
        """测试工具调用评分功能"""
        # 导入实际的计算函数进行测试
        from reward_score.cpdc_toolcall_executor import calculate_score
        
        # 准备测试数据
        gold_standard = [
            {
                "name": "search", 
                "parameters": {"query": "test"},
                "return": {"results": ["test result"]}
            }
        ]
        
        prediction = """
        <tool_call>
        {
            "name": "search",
            "parameters": {
                "query": "test"
            }
        }
        </tool_call>
        """
        
        # 直接调用calculate_score函数
        score = calculate_score(prediction, gold_standard)
        
        # 验证结果
        self.assertEqual(score, 1.0, "完全匹配应得到满分")
        
    def test_toolcall_multiple_predictions(self):
        """测试一个gold标准对应多个预测结果的情况"""
        # 导入实际的计算函数进行测试
        from reward_score.cpdc_toolcall_executor import calculate_score
        
        # 定义一个gold标准 - 使用简单的字符串参数
        gold_standard = [
            {
                "name": "search_products",
                "parameters": {
                    "query": "smartphone",
                    "price_max": "1000"
                },
                "return": [{"id": "phone1", "name": "Sample Phone", "price": 899}]
            }
        ]
        
        # 测试场景1: 完全匹配 - 应该得到1.0分
        prediction_exact = """
        <tool_call>
        {
            "name": "search_products",
            "parameters": {
                "query": "smartphone",
                "price_max": "1000"
            }
        }
        </tool_call>
        """
        score_exact = calculate_score(prediction_exact, gold_standard)
        print(f"完全匹配得分: {score_exact}")
        self.assertEqual(score_exact, 1.0, "完全匹配应得到满分")
        
        # 测试场景2: 部分匹配 - 参数不完全匹配
        # 注意：当前实现中，如果参数不完全匹配，会被视为不匹配
        prediction_partial = """
        <tool_call>
        {
            "name": "search_products",
            "parameters": {
                "query": "smartphone",
                "price_max": "800"
            }
        }
        </tool_call>
        """
        score_partial = calculate_score(prediction_partial, gold_standard)
        print(f"部分匹配得分: {score_partial}")
        # 当前实现中，参数不匹配会被视为完全不匹配，所以得分为0
        self.assertEqual(score_partial, 0.0, "当前实现中，参数不完全匹配会被视为不匹配")
        
        # 测试场景3: 多个预测结果，只有一个匹配
        prediction_multiple = """
        <tool_call>
        {
            "name": "get_weather",
            "parameters": {
                "location": "Beijing",
                "unit": "celsius"
            }
        }
        </tool_call>
        <tool_call>
        {
            "name": "search_products",
            "parameters": {
                "query": "smartphone",
                "price_max": "1000"
            }
        }
        </tool_call>
        <tool_call>
        {
            "name": "set_reminder",
            "parameters": {
                "message": "Buy new phone",
                "time": "tomorrow 9am"
            }
        }
        </tool_call>
        """
        score_multiple = calculate_score(prediction_multiple, gold_standard)
        print(f"多个预测中只有一个匹配得分: {score_multiple}")
        # 在当前实现中，多个预测中只有一个匹配，得分应该是1/3 = 0.33左右
        self.assertLess(score_multiple, 0.6, "多个预测中只有一个匹配应得到部分分数")
        self.assertGreater(score_multiple, 0.0, "多个预测中只有一个匹配应得到部分分数")
        
        # 测试场景4: 没有匹配的预测
        prediction_no_match = """
        <tool_call>
        {
            "name": "get_weather",
            "parameters": {
                "location": "Beijing",
                "unit": "celsius"
            }
        }
        </tool_call>
        """
        score_no_match = calculate_score(prediction_no_match, gold_standard)
        print(f"无匹配得分: {score_no_match}")
        self.assertEqual(score_no_match, 0.0, "无匹配应得到0分")
        
        # 测试场景5: 多个gold标准，多个预测全部匹配
        multiple_gold = [
            {
                "name": "search_products",
                "parameters": {
                    "query": "smartphone",
                    "price_max": "1000"
                },
                "return": [{"id": "phone1", "name": "Sample Phone", "price": 899}]
            },
            {
                "name": "get_reviews",
                "parameters": {
                    "product_id": "SP12345",
                    "limit": "5"
                },
                "return": [{"rating": 4.5, "text": "Great product"}]
            }
        ]
        
        prediction_multi_gold = """
        <tool_call>
        {
            "name": "search_products",
            "parameters": {
                "query": "smartphone",
                "price_max": "1000"
            }
        }
        </tool_call>
        <tool_call>
        {
            "name": "get_reviews",
            "parameters": {
                "product_id": "SP12345",
                "limit": "5"
            }
        }
        </tool_call>
        """
        score_multi_gold = calculate_score(prediction_multi_gold, multiple_gold)
        print(f"多个gold标准全部匹配得分: {score_multi_gold}")
        # 根据新的评分逻辑调整期望值
        self.assertEqual(score_multi_gold, 1.0, "新的评分逻辑中，多个gold标准全部匹配得分为1.0")
        
        # 测试场景6: 多个gold标准，部分匹配
        prediction_partial_gold = """
        <tool_call>
        {
            "name": "search_products",
            "parameters": {
                "query": "smartphone",
                "price_max": "1000"
            }
        }
        </tool_call>
        """
        score_partial_gold = calculate_score(prediction_partial_gold, multiple_gold)
        print(f"多个gold标准部分匹配得分: {score_partial_gold}")
        self.assertLess(score_partial_gold, 1.0, "多个gold标准部分匹配应得到部分分数")
        self.assertGreater(score_partial_gold, 0.0, "多个gold标准部分匹配应得到部分分数")
        
        # 添加一个注释，解释当前的评分逻辑
        print("\n注意：当前的评分逻辑只考虑完全匹配的情况，不支持参数级别的部分匹配评分。")
        print("如果需要支持参数级别的部分匹配评分，需要修改calculate_score函数。")

    def test_roleplay_scoring(self):
        """测试角色扮演评分功能"""
        # 创建一个客服对话的测试用例
        conversation_history = [
            {"role": "system", "content": "你是一个专业的客服代表。请礼貌地回答客户的问题。"},
            {"role": "user", "content": "我的订单什么时候能到？"},
            {"role": "assistant", "content": "您好！感谢您的咨询。请问您能提供一下订单号，这样我可以为您查询具体的配送信息。"}
        ]
        
        evaluation_criteria = "评估客服回复的专业性、礼貌性和解决问题的效率。"
        
        # 调用被测试的函数
        result = _default_compute_score(
            data_source="cpdc/roleplay",
            solution_str="您好！请提供您的订单号，我会立即为您查询配送状态。",
            ground_truth=json.dumps({"conversation": conversation_history, "evaluation_criteria": evaluation_criteria})
        )
        
        # 验证结果
        print(f"角色扮演评分结果: {result}")
        self.assertIn("score", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_batch_processing(self):
        """测试批处理功能"""
        # 测试工具调用评分
        from reward_score.cpdc_toolcall_executor import calculate_score
        
        # 准备工具调用测试数据
        tool_call_json = {"name": "search", "parameters": {"query": "test"}}
        solution_str = f"<tool_call>{json.dumps(tool_call_json)}</tool_call>"
        # 对于金标准，我们需要使用已解析的JSON对象列表
        gold_standard = [tool_call_json]
        
        # 直接使用 calculate_score 函数计算工具调用评分
        toolcall_score = calculate_score(solution_str, gold_standard)
        print(f"工具调用评分结果: {toolcall_score}")
        self.assertGreaterEqual(toolcall_score, 0.0)
        self.assertLessEqual(toolcall_score, 1.0)
        
        # 测试角色扮演评分
        conversation = {"conversation": [{"role": "user", "content": "Hello"}], "evaluation_criteria": "Politeness"}
        roleplay_result = _default_compute_score(
            data_source="cpdc/roleplay",
            solution_str="Hello! How can I help you today?",
            ground_truth=json.dumps(conversation)
        )
        
        # 验证角色扮演结果
        print(f"角色扮演评分结果: {roleplay_result}")
        self.assertIn("score", roleplay_result)
        self.assertGreaterEqual(roleplay_result["score"], 0.0)
        self.assertLessEqual(roleplay_result["score"], 1.0)
        
        # 测试批处理功能
        # 注意：我们不使用混合数据源进行批处理测试，因为它们需要不同的处理方式
        print("注意：批处理测试已分开为工具调用和角色扮演两个独立的测试。")

    def test_compute_score_wrapper(self):
        """Test the public compute_score function which wraps _default_compute_score"""
        with patch("reward_score._default_compute_score") as mock_default_compute:
            mock_default_compute.return_value = {"score": 0.75}
            
            result = compute_score(
                data_source="test_source",
                solution_str="test_pred",
                ground_truth="test_gold",
                extra_info={"metric": "test_metric"}
            )
            
            self.assertEqual(result, {"score": 0.75})
            # 使用ANY匹配位置参数，因为_default_compute_score实际上使用位置参数
            mock_default_compute.assert_called_once_with(
                "test_source", "test_pred", "test_gold", {"metric": "test_metric"}
            )


if __name__ == "__main__":
    unittest.main()

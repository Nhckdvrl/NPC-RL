import json
from typing import Dict, List, Any, Optional
import os
class LLMLogger:
    def __init__(self):
        self.logs: Dict[int, Dict] = {}
        self.current_index = -1

    def create_new_log(self) -> int:
        """Create a new log entry and return its index"""
        self.current_index += 1
        self.logs[self.current_index] = {
            "stage_0": {
                "messages": [],
                "tools": [],
                "responses": [],
                "toolcall": []
            },
            "stage_1": {
                "messages": [],
                "responses": []
            },
            "gold_response": None,
            "gold_toolcall": None
        }
        return self.current_index

    def log_stage_0(self, index: int, messages: List[Dict], tools: List[Dict], 
                   responses: List[Dict], toolcall: List[Dict]) -> None:
        """Log function calling stage (stage_0) data"""
        if index in self.logs:
            self.logs[index]["stage_0"]["messages"] = messages
            self.logs[index]["stage_0"]["tools"] = tools
            self.logs[index]["stage_0"]["responses"] = responses
            self.logs[index]["stage_0"]["toolcall"] = toolcall

    def log_stage_1(self, index: int, messages: List[Dict], responses: List[Dict]) -> None:
        """Log dialogue generation stage (stage_1) data"""
        if index in self.logs:
            self.logs[index]["stage_1"]["messages"] = messages
            self.logs[index]["stage_1"]["responses"] = responses

    def log_response(self, index: int, response: Any, stage: int) -> None:
        """Log response for a specific stage"""
        if index in self.logs:
            stage_key = f"stage_{stage}"
            if stage_key in self.logs[index]:
                self.logs[index][stage_key]["responses"] = response

    def log_tool(self, index: int, toolcall: List[Dict]) -> None:
        """Log response for a specific stage 0"""
        if index in self.logs:
            stage_key = f"stage_0"
            if stage_key in self.logs[index]:
                self.logs[index][stage_key]["toolcall"].append(toolcall)

    def log_gold_data(self, index: int, gold_response: Optional[Any] = None, 
                     gold_toolcall: Optional[Any] = None) -> None:
        """Log gold (ground truth) data"""
        if index in self.logs:
            if gold_response is not None:
                self.logs[index]["gold_response"] = gold_response
            if gold_toolcall is not None:
                self.logs[index]["gold_toolcall"] = gold_toolcall

    def get_logs(self) -> Dict[int, Dict]:
        """Get all logs"""
        return self.logs

    def save_to_file(self, filepath: str) -> None:
        """Save logs to a JSON file"""
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)

# Create a global logger instance
llm_logger = LLMLogger()

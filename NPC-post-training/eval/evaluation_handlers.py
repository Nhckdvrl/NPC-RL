"""
This module contains handlers for different evaluation tasks.
"""
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_eval import LLMJudge
from coser_eval import evaluate_game_roleplay_turn
from evaluation_metrics import evaluate_tool_calls


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metrics': self.metrics,
            'details': self.details
        }


class BaseEvaluator:
    """Base class for evaluators."""
    
    def evaluate_turn(
        self,
        gold_data: Any,
        model_response: Any,
        turn_idx: int,
        llm_judge: Optional[LLMJudge] = None
    ) -> EvaluationResult:
        """Evaluate a single turn."""
        raise NotImplementedError
    
    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> EvaluationResult:
        """Aggregate results from multiple turns."""
        raise NotImplementedError


class Task1Evaluator(BaseEvaluator):
    """Evaluator for Task 1 (tool calls + response quality)."""
    
    def evaluate_turn(
        self,
        gold_data: Dict[str, Any],
        model_response: Dict[str, Any],
        turn_idx: int,
        llm_judge: Optional[LLMJudge] = None
    ) -> EvaluationResult:
        """Evaluate a single turn for Task 1."""
        # Evaluate tool calls
        gold_functions = gold_data.get('gold_functions', [])
        predicted_functions = model_response.get('functions', [])
        tool_metrics = evaluate_tool_calls(gold_functions, predicted_functions)
        
        # Prepare response
        metrics = {
            'tool_precision': tool_metrics['precision'],
            'tool_recall': tool_metrics['recall'],
            'tool_f1': tool_metrics['f1'],
            'tool_tp': tool_metrics['tp'],
            'tool_fp': tool_metrics['fp'],
            'tool_fn': tool_metrics['fn'],
            'coser_score': 0.0  # Placeholder, would be updated by LLM judge
        }
        
        details = {
            'gold_functions': gold_functions,
            'predicted_functions': predicted_functions,
            'tool_metrics': tool_metrics
        }
        
        return EvaluationResult(metrics=metrics, details=details)
    
    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> EvaluationResult:
        """Aggregate results from multiple turns."""
        if not results:
            return EvaluationResult(metrics={}, details={})
            
        # Initialize sums
        metrics_keys = results[0].metrics.keys()
        sums = {k: 0.0 for k in metrics_keys}
        
        # Sum up all metrics
        for result in results:
            for k, v in result.metrics.items():
                if k in sums:
                    sums[k] += v
        
        # Calculate averages
        num_results = len(results)
        avg_metrics = {k: v / num_results for k, v in sums.items()}
        
        return EvaluationResult(metrics=avg_metrics, details={"num_turns": num_results})


class Task2Evaluator(BaseEvaluator):
    """Evaluator for Task 2 (direct response quality)."""
    
    def evaluate_turn(
        self,
        gold_data: Dict[str, Any],
        model_response: Dict[str, Any],
        turn_idx: int,
        llm_judge: Optional[LLMJudge] = None
    ) -> EvaluationResult:
        """Evaluate a single turn for Task 2."""
        if llm_judge is None:
            raise ValueError("LLMJudge is required for Task 2 evaluation")
        
        # Extract question and answers
        question = gold_data.get('dialogue', [{}])[-1].get('text', '')
        gold_answer = gold_data.get('gold_response', '')
        model_answer = model_response.get('response', model_response)
        
        if not question or not gold_answer:
            return EvaluationResult(
                metrics={'coser_score': 0.0},
                details={'error': 'Missing question or gold answer'}
            )
        
        # Evaluate with CoSER
        eval_result = evaluate_game_roleplay_turn(
            turn_number=turn_idx,
            worldview_details="Task 2 Evaluation - Direct question answering task",
            current_turn_state_knowledge=json.dumps({
                "question": question,
                "expected_answer": gold_answer,
                "task_type": "direct_qa"
            }, indent=2),
            player_profile={
                "name": "User",
                "role": "User asking questions",
                "persona": "User is asking questions to test the AI assistant's capabilities.",
                "goals": ["Get accurate information"],
                "inventory": "None"
            },
            npc_profile={
                "name": "Assistant",
                "role": "AI Assistant",
                "persona": "Helpful, knowledgeable AI assistant that provides accurate and concise answers.",
                "attitude": "Helpful and professional"
            },
            full_dialogue_history=[
                {"speaker": "Player", "text": question},
                {"speaker": "Assistant", "text": model_answer}
            ],
            player_query_for_current_turn=question,
            model_response_text=model_answer,
            gold_standard_npc_response_text=gold_answer,
            gold_standard_npc_functions=[],
            dimension_to_evaluate="all",
            llm_judge_instance=llm_judge,
            game_title="Task 2 Evaluation",
            additional_instructions_text=(
                "Please evaluate the response's accuracy, helpfulness, and naturalness. "
                "Focus on whether the response correctly answers the question and provides "
                "useful information."
            )
        )
        
        # Extract score and reasoning
        coser_score = float(eval_result.get('score', 0))
        
        return EvaluationResult(
            metrics={'coser_score': coser_score},
            details={
                'question': question,
                'model_answer': model_answer,
                'gold_answer': gold_answer,
                'evaluation': eval_result
            }
        )
    
    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> EvaluationResult:
        """Aggregate results from multiple turns."""
        if not results:
            return EvaluationResult(metrics={}, details={})
            
        # Calculate average CoSER score
        total_score = sum(r.metrics.get('coser_score', 0) for r in results)
        avg_score = total_score / len(results)
        
        return EvaluationResult(
            metrics={'avg_coser_score': avg_score},
            details={'num_turns': len(results)}
        )


class GameEvaluator(BaseEvaluator):
    """Evaluator for Game Role-Playing format."""
    
    def evaluate_turn(
        self,
        gold_data: Dict[str, Any],
        model_response: Dict[str, Any],
        turn_idx: int,
        llm_judge: Optional[LLMJudge] = None
    ) -> EvaluationResult:
        """Evaluate a single turn for Game Role-Playing."""
        if llm_judge is None:
            raise ValueError("LLMJudge is required for Game Role-Playing evaluation")
            
        # Get dialogue and current turn
        dialogue = gold_data.get('dialogue', [])
        if turn_idx >= len(dialogue):
            return EvaluationResult(
                metrics={'coser_score': 0.0},
                details={'error': f'Turn index {turn_idx} out of range'}
            )
            
        turn = dialogue[turn_idx]
        if turn.get('speaker') != 'NPC':
            # Skip non-NPC turns
            return EvaluationResult(
                metrics={'coser_score': 0.0},
                details={'skipped': 'Not an NPC turn'}
            )
            
        # Get player's query from previous turn
        player_query = dialogue[turn_idx-1]['text'] if turn_idx > 0 else "N/A"
        
        # Prepare dialogue history
        dialogue_history = []
        for i in range(turn_idx):
            if i < len(dialogue):
                speaker = dialogue[i].get('speaker', 'Unknown')
                text = dialogue[i].get('text', '')
                dialogue_history.append({'speaker': speaker, 'text': text})
        
        # Evaluate with CoSER
        eval_result = evaluate_game_roleplay_turn(
            turn_number=turn_idx,
            worldview_details=gold_data.get('worldview', 'No specific worldview provided.'),
            current_turn_state_knowledge=json.dumps({
                'current_state': gold_data.get('current_state', {}),
                'active_quests': gold_data.get('active_quests', []),
                'inventory': gold_data.get('inventory', [])
            }, indent=2),
            player_profile={
                'name': 'Player',
                'role': 'Player',
                'persona': gold_data.get('player_persona', 'No specific persona provided.')
            },
            npc_profile={
                'name': turn.get('npc_name', 'NPC'),
                'role': turn.get('npc_role', 'NPC'),
                'persona': turn.get('npc_persona', 'No specific NPC persona provided.'),
                'goals': turn.get('npc_goals', ['Engage in conversation'])
            },
            full_dialogue_history=dialogue_history,
            player_query_for_current_turn=player_query,
            model_response_text=model_response.get('content', ''),
            gold_standard_npc_response_text=turn.get('text', ''),
            gold_standard_npc_functions=turn.get('gold_functions', []),
            dimension_to_evaluate="NPC/Player Believability & Engagement",
            llm_judge_instance=llm_judge,
            game_title=gold_data.get('game_title', 'Unknown Game'),
            additional_instructions_text=(
                "Please evaluate the NPC's response for believability, "
                "engagement, and adherence to the game's context and the NPC's character."
            )
        )
        
        # Process evaluation results
        gca_score = float(eval_result.get('overall_score', 0.0))
        dimension_scores = eval_result.get('dimension_scores', {})
        
        # Evaluate tool calls if any
        gold_functions = turn.get('gold_functions', [])
        predicted_functions = model_response.get('functions', [])
        tool_metrics = evaluate_tool_calls(gold_functions, predicted_functions)
        
        # Prepare metrics
        metrics = {
            'coser_score': gca_score,
            'tool_precision': tool_metrics['precision'],
            'tool_recall': tool_metrics['recall'],
            'tool_f1': tool_metrics['f1'],
            'tool_tp': tool_metrics['tp'],
            'tool_fp': tool_metrics['fp'],
            'tool_fn': tool_metrics['fn']
        }
        
        # Add dimension scores
        for dim, score in dimension_scores.items():
            metrics[f'dim_{dim.lower().replace(" ", "_")}'] = score
        
        return EvaluationResult(
            metrics=metrics,
            details={
                'evaluation': eval_result,
                'tool_metrics': tool_metrics
            }
        )
    
    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> EvaluationResult:
        """Aggregate results from multiple turns."""
        if not results:
            return EvaluationResult(metrics={}, details={})
            
        # Initialize sums
        metrics_keys = set()
        for r in results:
            metrics_keys.update(r.metrics.keys())
            
        sums = {k: 0.0 for k in metrics_keys}
        counts = {k: 0 for k in metrics_keys}
        
        # Sum up all metrics
        for result in results:
            for k, v in result.metrics.items():
                if k in sums:
                    sums[k] += v
                    counts[k] += 1
        
        # Calculate averages
        avg_metrics = {}
        for k in metrics_keys:
            if counts[k] > 0:
                avg_metrics[k] = sums[k] / counts[k]
        
        return EvaluationResult(
            metrics=avg_metrics,
            details={'num_turns': len(results)}
        )


def get_evaluator(task_type: int) -> BaseEvaluator:
    """Factory function to get the appropriate evaluator for the task type."""
    if task_type == 1:
        return Task1Evaluator()
    elif task_type == 2:
        return Task2Evaluator()
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def get_game_evaluator() -> BaseEvaluator:
    """Get the evaluator for game role-playing format."""
    return GameEvaluator()

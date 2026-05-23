# Evaluation Module

This module provides a flexible and extensible framework for evaluating dialogue system responses across different tasks.

## Structure

The module is organized into the following components:

### Core Modules

1. **evaluator.py** - Main entry point for running evaluations
2. **evaluation_handlers.py** - Handles different evaluation tasks
3. **evaluation_metrics.py** - Implements evaluation metrics and utilities
4. **data_io.py** - Handles data loading and saving operations

## Usage

### Basic Evaluation

```python
from eval import get_evaluator, load_json_file

# Load data
gold_data = load_json_file("path/to/gold_data.json")
model_responses = load_json_file("path/to/model_responses.json")

# Get appropriate evaluator (task_type=1 for tool calls, task_type=2 for direct responses)
evaluator = get_evaluator(task_type=1)

# Evaluate a single conversation
result = evaluator.evaluate_turn(
    gold_data=gold_data[0],
    model_response=model_responses[0],
    turn_idx=0
)
```

### Game Role-Playing Evaluation

```python
from eval import get_game_evaluator, load_json_file

# Load game data
game_data = load_json_file("path/to/game_data.json")
model_responses = load_json_file("path/to/model_responses.json")

# Get game evaluator
evaluator = get_game_evaluator()

# Evaluate a game conversation
result = evaluator.evaluate_turn(
    gold_data=game_data[0],
    model_response=model_responses[0],
    turn_idx=0
)
```

## Command Line Interface

The main evaluator can be run from the command line:

```bash
python -m eval.evaluator \
    --data_file path/to/gold_data.json \
    --response_file path/to/model_responses.json \
    --model_name "MyModel" \
    --task_type 1  # Optional, will auto-detect if not provided
```

## Adding New Evaluators

To add a new evaluation task:

1. Create a new evaluator class that inherits from `BaseEvaluator`
2. Implement the required methods (`evaluate_turn`, `aggregate_results`)
3. Update the `get_evaluator` factory function in `evaluation_handlers.py`

## Dependencies

- Python 3.7+
- pandas
- llm_eval (custom module)
- coser_eval (custom module)

"""
This module handles data loading and saving operations.
"""
import json
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def load_json_file(file_path: Union[str, Path]) -> Any:
    """
    Loads a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {file_path}")
        raise


def save_results_to_csv(
    results_dir: Union[str, Path],
    metrics: Dict[str, Any],
    task_type: str,
    model_name: str,
    use_rag: bool = False
) -> str:
    """
    Save evaluation results to a CSV file, appending to existing file if it exists.
    
    Args:
        results_dir: Directory to save results
        metrics: Dictionary containing evaluation metrics
        task_type: Type of task (e.g., 'Task 1', 'Game Role-Playing')
        model_name: Name of the model being evaluated
        use_rag: Whether RAG was used
        
    Returns:
        Path to the saved CSV file
    """
    # Create results directory if it doesn't exist
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define CSV path
    csv_path = results_dir / "report.csv"
    
    # Prepare the new row
    rag_status = "with RAG" if use_rag else "no RAG"
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare base row with common fields
    new_row = {
        "Task Type": task_type,
        "Model": f"{model_name} ({rag_status})",
        "Timestamp": timestamp
    }
    
    # Add metrics to the row
    new_row.update(metrics)
    
    # Define all possible columns for consistent ordering
    all_columns = [
        "Task Type", "Model", "CoSER Score", "Turns Evaluated", "Conversations",
        "Tool Precision", "Tool Recall", "Tool F1", "Tool TP", "Tool FP", "Tool FN",
        "Timestamp"
    ]
    
    # Read existing CSV or create new DataFrame with appropriate columns
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Ensure all expected columns exist
        for col in all_columns:
            if col not in df.columns:
                df[col] = None
    else:
        df = pd.DataFrame(columns=all_columns)
    
    # Add new row at the beginning
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    
    # Save to CSV with consistent column order
    df[all_columns].to_csv(csv_path, index=False)
    
    return str(csv_path)


def is_game_format(conv_data: Any) -> bool:
    """
    Check if the conversation data follows the game role-play format.
    
    Args:
        conv_data: Conversation data to check
        
    Returns:
        bool: True if data is in game format, False otherwise
    """
    return isinstance(conv_data, dict) and 'worldview' in conv_data and 'dialogue' in conv_data


def is_task2_format(response_data: Any) -> bool:
    """
    Check if the response data matches task2 format (direct string responses).
    
    Args:
        response_data: Response data to check
        
    Returns:
        bool: True if data is in task2 format, False otherwise
    """
    if not isinstance(response_data, list) or not response_data:
        return False
    first_item = response_data[0]
    if not isinstance(first_item, dict):
        return False
    # Check if any turn is a direct string (task2) vs dict with 'response' key (task1)
    for key, value in first_item.items():
        if key.startswith('turn_') and isinstance(value, str):
            return True
    return False

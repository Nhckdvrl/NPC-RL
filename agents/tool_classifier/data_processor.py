# agents/tool_classifier/data_processor.py
import json
from typing import List, Tuple, Any, Optional, Dict
from sklearn.model_selection import train_test_split
import os # For config path if needed

# Assuming config.py might have paths or parameters
# from .config import YOUR_CONFIG_SETTINGS_IF_ANY

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []

def preprocess_text_data(texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """
    Basic text preprocessing.
    For now, it's a placeholder. Implementations might include:
    - Lowercasing
    - Removing punctuation
    - Stop word removal
    - Stemming/Lemmatization
    """
    # Example: simple lowercasing
    processed_texts = [text.lower() for text in texts]
    # Labels might also need preprocessing depending on their nature
    processed_labels = labels
    print(f"Data preprocessed. {len(processed_texts)} texts, {len(processed_labels)} labels.")
    return processed_texts, processed_labels

def load_and_preprocess_data(
    data_path: str,
    text_key: str = 'utterance', # Key in JSON for text
    label_key: str = 'tool_id', # Key in JSON for label
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Loads data from a JSON file, preprocesses it, and splits into train/test sets.
    Assumes JSON data is a list of dictionaries, e.g., [{'utterance': '...', 'tool_id': '...'}, ...]
    """
    raw_data = load_json_data(data_path)
    if not raw_data:
        return [], [], [], []

    texts = [item.get(text_key, "") for item in raw_data]
    labels = [item.get(label_key, "") for item in raw_data]

    # Filter out entries where text or label might be missing
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        if text and label: # Ensure both text and label are present
            filtered_texts.append(text)
            filtered_labels.append(label)
        else:
            print(f"Warning: Skipping entry with missing text ('{text}') or label ('{label}').")
    
    if not filtered_texts or not filtered_labels:
        print("Error: No valid data to process after filtering.")
        return [], [], [], []

    texts, labels = preprocess_text_data(filtered_texts, filtered_labels)

    if not texts: # Check if texts became empty after preprocessing (unlikely with current simple preprocess)
        print("Error: No data left after preprocessing.")
        return [], [], [], []

    # Stratify only if there's more than one class and enough samples per class for splitting
    can_stratify = len(set(labels)) > 1
    if can_stratify:
        label_counts = {label: labels.count(label) for label in set(labels)}
        min_samples_for_stratify = sum(1 for count in label_counts.values() if count < (2 if test_size > 0 else 1) ) # at least 2 for split, 1 if no split
        if min_samples_for_stratify > 0 and test_size > 0 : # if any class has < 2 samples, stratify might fail for splits
             # Check if any class has fewer samples than required for n_splits in StratifiedShuffleSplit (default is 1 for test_split)
            if any(count < 2 for count in label_counts.values()): # Stricter check for scikit-learn's train_test_split stratify
                print(f"Warning: Cannot stratify due to some classes having < 2 samples. Proceeding without stratification. Label counts: {label_counts}")
                can_stratify = False

    if test_size > 0 and len(texts) > 1 : # Need at least 2 samples to split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, 
            stratify=labels if can_stratify else None
        )
        print(f"Data split: {len(train_texts)} train, {len(test_texts)} test samples.")
        return train_texts, test_texts, train_labels, test_labels
    else:
        # If test_size is 0 or only 1 sample, return all data as training data
        print(f"Data loaded: {len(texts)} training samples (no test split performed).")
        return texts, [], labels, []

if __name__ == '__main__':
    # Example usage (assuming you have a sample data file)
    # Create a dummy data file for testing
    dummy_data = [
        {"utterance": "Book a flight to New York", "tool_id": "BookFlight"},
        {"utterance": "What's the weather like in London?", "tool_id": "GetWeather"},
        {"utterance": "Find a hotel in Paris for next week", "tool_id": "FindHotel"},
        {"utterance": "book a flight to tokyo", "tool_id": "BookFlight"},
        {"utterance": "weather in san francisco", "tool_id": "GetWeather"},
        {"utterance": "I need a hotel in Berlin", "tool_id": "FindHotel"}
    ]
    dummy_file_path = "dummy_data.json"
    with open(dummy_file_path, 'w') as f:
        json.dump(dummy_data, f)

    train_texts, test_texts, train_labels, test_labels = load_and_preprocess_data(dummy_file_path)
    if train_texts:
        print("\nSample Training Data:")
        for i in range(min(2, len(train_texts))):
            print(f"Text: {train_texts[i]}, Label: {train_labels[i]}")
    if test_texts:
        print("\nSample Test Data:")
        for i in range(min(2, len(test_texts))):
            print(f"Text: {test_texts[i]}, Label: {test_labels[i]}")
    
    # Clean up dummy file
    os.remove(dummy_file_path)

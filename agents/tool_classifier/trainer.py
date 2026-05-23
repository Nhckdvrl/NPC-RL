# agents/tool_classifier/trainer.py
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .data_processor import load_and_preprocess_data
from .algorithms.base_classifier import BaseClassifier
from .algorithms.svm_classifier import SVMClassifier 
# Import other classifiers here as they are created, e.g.:
# from .algorithms.naive_bayes_classifier import NaiveBayesClassifier
# from .algorithms.logistic_regression_classifier import LogisticRegressionClassifier

# Could be moved to a config file or constants.py
DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/tool_classifier"))
DEFAULT_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/tool_classifier"))

SUPPORTED_CLASSIFIERS = {
    "svm": SVMClassifier,
    # "naive_bayes": NaiveBayesClassifier, # Add when implemented
    # "logistic_regression": LogisticRegressionClassifier, # Add when implemented
}

def train_and_evaluate_model(
    data_path: str,
    classifier_type: str, 
    model_name: Optional[str] = None,
    save_dir: str = DEFAULT_MODELS_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    classifier_params: Optional[Dict[str, Any]] = None,
    text_key: str = 'utterance',
    label_key: str = 'tool_id'
) -> Tuple[Optional[BaseClassifier], Optional[Dict[str, Any]]]:
    """
    Trains a specified classifier, evaluates it, and saves the model and results.

    Args:
        data_path (str): Path to the JSON data file.
        classifier_type (str): Type of classifier to train (e.g., 'svm').
        model_name (Optional[str]): Name for the model. If None, generated from type and timestamp.
        save_dir (str): Directory to save the trained model.
        results_dir (str): Directory to save training and evaluation reports.
        test_size (float): Fraction of data to use for the test set.
        random_state (Optional[int]): Random seed for reproducibility.
        classifier_params (Optional[Dict[str, Any]]): Parameters to pass to the classifier's constructor.
        text_key (str): Key in JSON for text data.
        label_key (str): Key in JSON for label data.

    Returns:
        Tuple[Optional[BaseClassifier], Optional[Dict[str, Any]]]: 
            The trained classifier instance and a dictionary containing evaluation metrics.
            Returns (None, None) if training fails.
    """
    print(f"--- Starting training and evaluation for {classifier_type} ---")

    # 1. Load and preprocess data
    print(f"Loading data from: {data_path}")
    train_texts, test_texts, train_labels, test_labels = load_and_preprocess_data(
        data_path=data_path,
        text_key=text_key,
        label_key=label_key,
        test_size=test_size,
        random_state=random_state
    )

    if not train_texts or not train_labels:
        print("Error: No training data available after loading/preprocessing.")
        return None, None

    # 2. Initialize Classifier
    if classifier_type.lower() not in SUPPORTED_CLASSIFIERS:
        print(f"Error: Classifier type '{classifier_type}' is not supported.")
        print(f"Supported types are: {list(SUPPORTED_CLASSIFIERS.keys())}")
        return None, None
    
    ClassifierClass = SUPPORTED_CLASSIFIERS[classifier_type.lower()]
    try:
        classifier_params = classifier_params or {}
        classifier_params.setdefault('random_state', random_state) # Ensure random_state is passed
        classifier = ClassifierClass(**classifier_params)
        print(f"Initialized {classifier_type} classifier.")
    except Exception as e:
        print(f"Error initializing {classifier_type} classifier: {e}")
        return None, None

    # 3. Train Model
    print(f"Training {classifier_type} model...")
    try:
        training_metrics = classifier.train(train_texts, train_labels)
        print("Training completed.")
        print("Training Metrics:", json.dumps(training_metrics, indent=2))
    except Exception as e:
        print(f"Error during model training: {e}")
        # import traceback
        # traceback.print_exc()
        return None, None

    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{classifier_type}_{timestamp}"
    
    # 4. Save Model
    try:
        model_save_path_base = os.path.join(save_dir, model_name)
        # The save method in BaseClassifier expects directory and a base name for components
        classifier.save(directory=save_dir, model_name=model_name)
        print(f"Model saved with base name '{model_name}' in directory '{save_dir}'.")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Proceed with evaluation even if saving fails, but log the error.

    # 5. Evaluate Model
    evaluation_metrics = None
    if test_texts and test_labels:
        print(f"\nEvaluating {classifier_type} model on the test set...")
        try:
            evaluation_metrics = classifier.evaluate(test_texts, test_labels)
            print("Evaluation completed.")
            print("Evaluation Metrics:", json.dumps(evaluation_metrics, indent=2))
        except Exception as e:
            print(f"Error during model evaluation: {e}")
    else:
        print("\nNo test data available for evaluation. Skipping evaluation.")

    # 6. Save Reports (Training and Evaluation)
    os.makedirs(results_dir, exist_ok=True)
    report_filename_base = os.path.join(results_dir, f"{model_name}_report")

    try:
        full_report = {
            "model_name": model_name,
            "classifier_type": classifier_type,
            "data_path": data_path,
            "training_timestamp": datetime.now().isoformat(),
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics if evaluation_metrics else "Not performed",
            "classifier_parameters": classifier_params,
            "data_split": {
                "test_size": test_size,
                "random_state": random_state,
                "num_train_samples": len(train_texts),
                "num_test_samples": len(test_texts) if test_texts else 0
            }
        }
        with open(f"{report_filename_base}.json", 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=4, default=str) # Use default=str for non-serializable types
        print(f"Full training and evaluation report saved to {report_filename_base}.json")
    except Exception as e:
        print(f"Error saving report: {e}")

    return classifier, evaluation_metrics

if __name__ == '__main__':
    print("Tool Classifier Trainer Module")
    # This is an example of how to run the trainer.
    # You would typically call this from a script with command-line arguments.

    # Create a dummy data file for testing
    dummy_data_content = [
        {"utterance": "Book a flight to New York for tomorrow", "tool_id": "BookFlight"},
        {"utterance": "What's the weather like in London?", "tool_id": "GetWeather"},
        {"utterance": "Find a hotel in Paris for next week", "tool_id": "FindHotel"},
        {"utterance": "I want to book a flight to Tokyo", "tool_id": "BookFlight"},
        {"utterance": "Current weather in San Francisco please", "tool_id": "GetWeather"},
        {"utterance": "Need a hotel in Berlin from 10th to 12th", "tool_id": "FindHotel"},
        {"utterance": "Reserve a table for two at an Italian restaurant", "tool_id": "BookRestaurant"},
        {"utterance": "Play some jazz music", "tool_id": "PlayMusic"},
        {"utterance": "Flight booking for three people to London", "tool_id": "BookFlight"},
        {"utterance": "Tell me the weather in Berlin", "tool_id": "GetWeather"},
        {"utterance": "I need a room in a hotel in Madrid", "tool_id": "FindHotel"},
        {"utterance": "Book a flight to Rome", "tool_id": "BookFlight"}
    ]
    dummy_data_file = os.path.join(DEFAULT_RESULTS_DIR, "dummy_training_data.json")
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    with open(dummy_data_file, 'w') as f:
        json.dump(dummy_data_content, f)

    print(f"\n--- Example: Training SVM Classifier ---")
    svm_classifier_instance, svm_eval_metrics = train_and_evaluate_model(
        data_path=dummy_data_file,
        classifier_type='svm',
        model_name='my_svm_example_model',
        classifier_params={'C': 1.0, 'kernel': 'linear'},
        save_dir=os.path.join(DEFAULT_MODELS_DIR, "trainer_examples"),
        results_dir=os.path.join(DEFAULT_RESULTS_DIR, "trainer_examples")
    )

    if svm_classifier_instance:
        print(f"\nSVM Model trained. Evaluation metrics: {svm_eval_metrics}")
        # Example prediction with the trained model
        sample_queries = [
            "book a flight to Paris", 
            "what is the weather in tokyo",
            "find a hotel near me"
        ]
        if svm_classifier_instance.is_fitted():
            predictions = svm_classifier_instance.predict(sample_queries)
            print("Sample predictions:")
            for q, p in zip(sample_queries, predictions):
                print(f"  Query: '{q}' -> Predicted Tool: '{p}'")
        else:
            print("SVM classifier instance was not fitted correctly.")
    else:
        print("SVM model training failed.")
    
    # Clean up dummy file (optional)
    # os.remove(dummy_data_file)
    # print(f"Cleaned up {dummy_data_file}")

# agents/tool_classifier/algorithms/svm_classifier.py
from typing import List, Any, Dict, Optional
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
try:
    from base_classifier import BaseClassifier
except ImportError:
    from .base_classifier import BaseClassifier

# Define default paths for model components if not provided during save/load
# These could also be managed via a config file or passed around more explicitly.
DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/tool_classifier"))

class SVMClassifier(BaseClassifier):
    """
    A Support Vector Machine (SVM) classifier using TF-IDF for text vectorization.
    """
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'linear',
        gamma: str = 'scale',
        random_state: Optional[int] = 42,
        # No model_path in __init__ for this concrete class, 
        # it's handled by save/load and BaseClassifier's model_path for the main pipeline.
        **kwargs
    ):
        super().__init__(**kwargs) # model_path from BaseClassifier will be None initially
        
        # Components are initialized here. They will be fitted during train() or loaded by load().
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.svm = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state, probability=True)
        
        # The scikit-learn pipeline bundles vectorizer and SVM.
        # This pipeline object is what self.model (from BaseClassifier) will store.
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('svm', self.svm)
        ])
        # self._is_fitted is inherited from BaseClassifier, default False

    def train(self, texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
        """Train the SVM classifier. The pipeline (vectorizer and SVM) and LabelEncoder are fitted."""
        print(f"Starting SVM training with {len(texts)} samples.")
        
        # Fit LabelEncoder separately as it's not part of the scikit-learn pipeline by default
        # for transforming target labels. It's crucial that this instance is saved.
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Fit the pipeline (which includes TF-IDF vectorizer and SVM)
        self.pipeline.fit(texts, encoded_labels)
        self.model = self.pipeline # Store the fitted pipeline in self.model
        self._is_fitted = True
        
        # Evaluate on training data (optional, but good for a baseline)
        train_predictions_encoded = self.model.predict(texts)
        
        accuracy = accuracy_score(encoded_labels, train_predictions_encoded)
        # Use target_names for a more readable report if available
        class_names = list(self.label_encoder.classes_)
        report = classification_report(encoded_labels, train_predictions_encoded, target_names=class_names, output_dict=True, zero_division=0)
        
        print(f"SVM training completed. Training Accuracy: {accuracy:.4f}")
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "num_samples": len(texts),
            "num_classes": len(self.label_encoder.classes_)
        }

    def predict(self, texts: List[str]) -> List[str]:
        """Predict labels for a list of texts."""
        if not self.is_fitted() or self.model is None:
            raise RuntimeError("Classifier has not been trained or loaded. Call train() or load() first.")
        
        encoded_predictions = self.model.predict(texts)
        string_labels = self.label_encoder.inverse_transform(encoded_predictions)
        return list(string_labels)

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict probabilities for each class for a list of texts."""
        if not self.is_fitted() or self.model is None:
            raise RuntimeError("Classifier has not been trained or loaded. Call train() or load() first.")
        if not hasattr(self.model.named_steps['svm'], 'probability') or not self.model.named_steps['svm'].probability:
            raise RuntimeError("SVM model was not trained with probability=True.")

        probabilities_array = self.model.predict_proba(texts)
        results = []
        for prob_array_for_sample in probabilities_array:
            class_probs = {self.label_encoder.classes_[i]: prob_array_for_sample[i] for i in range(len(prob_array_for_sample))}
            results.append(class_probs)
        return results

    def evaluate(self, texts: List[str], true_labels: List[str], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the classifier on the given texts and true labels.
        
        Args:
            texts: List of input texts to predict.
            true_labels: List of true labels corresponding to the texts.
            **kwargs: Additional arguments to pass to the prediction method.
            
        Returns:
            Dictionary containing evaluation metrics including accuracy and classification report.
        """
        if not self.is_fitted() or self.model is None:
            raise RuntimeError("Classifier has not been trained or loaded. Call train() or load() first.")
        
        # Get predictions for the input texts
        predicted_labels = self.predict(texts)
        
        # Get the unique classes present in the true and predicted labels
        unique_true = set(true_labels)
        unique_pred = set(predicted_labels)
        
        # Find the intersection of classes between true and predicted labels
        common_classes = sorted(list(unique_true.union(unique_pred)))
        
        # Get all possible classes from the label encoder (training classes)
        all_classes = list(self.label_encoder.classes_)
        
        # Find any missing classes (present in test but not in training)
        missing_in_training = unique_true - set(all_classes)
        if missing_in_training:
            print(f"Warning: Found {len(missing_in_training)} classes in test data that were not in training: {missing_in_training}")
        
        # Only include classes that are in the label encoder (training classes)
        # This ensures we don't try to report on classes the model has never seen
        valid_classes = [c for c in common_classes if c in all_classes]
        
        # If no valid classes, return a basic accuracy score
        if not valid_classes:
            print("Warning: No valid classes found in both true and predicted labels.")
            accuracy = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels) if true_labels else 0.0
            return {
                "accuracy": accuracy,
                "classification_report": {
                    "accuracy": accuracy,
                    "warning": "No valid classes found in both true and predicted labels"
                }
            }
            
        # Generate classification report with valid classes only
        report_dict = classification_report(
            true_labels, 
            predicted_labels,
            labels=valid_classes,
            target_names=valid_classes,
            output_dict=True,
            zero_division=0
        )
        
        # Calculate overall accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        print(f"Evaluation completed with {len(valid_classes)} classes.")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Add information about missing classes to the report
        if missing_in_training:
            report_dict["missing_classes_in_training"] = list(missing_in_training)
        
        return {
            "accuracy": accuracy,
            "classification_report": report_dict,
            "num_classes_evaluated": len(valid_classes)
        }

    def save(self, directory: str, model_name: str):
        """Save the trained model (pipeline) and the label encoder."""
        if not self.is_fitted() or self.model is None:
            raise RuntimeError("Cannot save model: Classifier has not been trained yet.")

        os.makedirs(directory, exist_ok=True)
        
        pipeline_path = os.path.join(directory, f"{model_name}_pipeline.joblib")
        label_encoder_path = os.path.join(directory, f"{model_name}_label_encoder.joblib")
        
        joblib.dump(self.model, pipeline_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        
        # Store paths for potential reloading by this instance if needed, though load is a classmethod
        self.model_path = pipeline_path # BaseClassifier's model_path for the main component
        # self.vectorizer_path and self.label_encoder_path are instance attributes for clarity
        # but the vectorizer is part of the pipeline, so only label_encoder is saved separately.
        self.label_encoder_path_saved = label_encoder_path 
        print(f"SVM model (pipeline) saved to {pipeline_path}")
        print(f"Label encoder saved to {label_encoder_path}")

    @classmethod
    def load(cls, directory: str, model_name: str, **kwargs) -> 'SVMClassifier':
        """Load a trained SVM model (pipeline) and its label encoder."""
        pipeline_path = os.path.join(directory, f"{model_name}_pipeline.joblib")
        label_encoder_path = os.path.join(directory, f"{model_name}_label_encoder.joblib")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

        # Create an instance of the class. Pass any relevant kwargs from the original __init__ if needed.
        # For SVMClassifier, C, kernel, etc., are part of the saved pipeline, so not strictly needed here
        # unless we want to allow overriding them, which is not typical for load.
        classifier = cls(**kwargs) 
        
        classifier.model = joblib.load(pipeline_path)
        classifier.label_encoder = joblib.load(label_encoder_path)
        
        # The pipeline loaded into classifier.model already contains the fitted vectorizer and SVM.
        # We can re-assign them to instance attributes for direct access if desired, but it's optional.
        classifier.pipeline = classifier.model 
        classifier.vectorizer = classifier.model.named_steps['tfidf']
        classifier.svm = classifier.model.named_steps['svm']
        
        classifier._is_fitted = True
        classifier.model_path = pipeline_path # Store the path of the loaded main component
        classifier.label_encoder_path_saved = label_encoder_path

        print(f"SVM model (pipeline) loaded from {pipeline_path}")
        print(f"Label encoder loaded from {label_encoder_path}")
        return classifier

if __name__ == '__main__':
    # Example Usage
    print("SVMClassifier Example Usage:")
    texts_train = [
        "Book a flight to New York for tomorrow",
        "What's the weather like in London?",
        "Find a hotel in Paris for next week",
        "I want to book a flight to Tokyo",
        "Current weather in San Francisco please",
        "Need a hotel in Berlin from 10th to 12th",
        "Reserve a table for two at an Italian restaurant",
        "Play some jazz music",
        "Flight booking for three people to London"
    ]
    labels_train = [
        "BookFlight", "GetWeather", "FindHotel", 
        "BookFlight", "GetWeather", "FindHotel",
        "BookRestaurant", "PlayMusic", "BookFlight"
    ]

    texts_test = [
        "I need a flight to Berlin", 
        "show me the weather in Paris",
        "find a place to stay in Rome"
    ]
    labels_test = ["BookFlight", "GetWeather", "FindHotel"]

    # --- Training --- 
    svm_clf = SVMClassifier(random_state=42)
    train_metrics = svm_clf.train(texts_train, labels_train)
    print("Training Metrics:", train_metrics)

    # --- Prediction ---
    predictions = svm_clf.predict(texts_test)
    print("\nTest Predictions:", predictions)

    # --- Probability Prediction ---
    if svm_clf.is_fitted() and svm_clf.pipeline.named_steps['svm'].probability:
        probas = svm_clf.predict_proba(texts_test)
        print("\nTest Probabilities:")
        for i, text in enumerate(texts_test):
            print(f"  Text: '{text}' -> Probs: {probas[i]}")

    # --- Evaluation ---
    eval_metrics = svm_clf.evaluate(texts_test, labels_test)
    print("\nEvaluation Metrics on Test Set:", eval_metrics)

    # --- Saving and Loading ---
    save_dir = os.path.join(DEFAULT_MODEL_DIR, "svm_example")
    model_file_name = "my_svm_model"
    
    print(f"\nSaving model to directory: {save_dir} with name: {model_file_name}")
    svm_clf.save(directory=save_dir, model_name=model_file_name)

    print("\nLoading model...")
    loaded_svm_clf = SVMClassifier.load(directory=save_dir, model_name=model_file_name)

    loaded_predictions = loaded_svm_clf.predict(texts_test)
    print("Predictions from loaded model:", loaded_predictions)
    assert all(p1 == p2 for p1, p2 in zip(predictions, loaded_predictions)), "Predictions from loaded model do not match!"
    print("Successfully saved and loaded SVM model.")

    # Clean up dummy model files (optional)
    # import shutil
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    #     print(f"Cleaned up {save_dir}")

# agents/tool_classifier/algorithms/base_classifier.py
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import joblib
import os

class BaseClassifier(ABC):
    """
    Abstract base class for all classifier models.
    """
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        self.model = None
        self.model_path = model_path # Generic path for the primary model component
        self._is_fitted = False

    @abstractmethod
    def train(self, texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            texts (List[str]): List of input texts.
            labels (List[str]): List of corresponding labels.
            **kwargs: Additional training parameters.
            
        Returns:
            Dict[str, Any]: A dictionary containing training metrics (e.g., accuracy, loss).
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for a list of texts.
        
        Args:
            texts (List[str]): List of input texts.
            
        Returns:
            List[str]: List of predicted labels.
        """
        pass

    @abstractmethod
    def evaluate(self, texts: List[str], true_labels: List[str], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the classifier on a test set.
        
        Args:
            texts (List[str]): List of input texts.
            true_labels (List[str]): List of true labels.
            **kwargs: Additional evaluation parameters.
            
        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics (e.g., accuracy, precision, recall, f1-score).
        """
        pass

    @abstractmethod
    def save(self, directory: str, model_name: str):
        """
        Save the trained model and any associated components (e.g., vectorizer, label encoder).
        
        Args:
            directory (str): The directory to save the model files.
            model_name (str): A base name for the model files.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, directory: str, model_name: str) -> 'BaseClassifier':
        """
        Load a trained model and its components.
        
        Args:
            directory (str): The directory from which to load the model files.
            model_name (str): The base name of the model files.
            
        Returns:
            BaseClassifier: An instance of the loaded classifier.
        """
        pass
    
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

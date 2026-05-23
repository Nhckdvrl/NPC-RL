# agents/tool_classifier/config.py
import os

# Base directories
CLASSIFIER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CLASSIFIER_DIR, "../../"))

# Data paths
TRIGGERED_UTTERANCES_JSON = os.path.join(
    PROJECT_ROOT, 
    "agents/tool_classifier/references_data/triggered_utterances.json"
)
TASK1_TEST_JSON = os.path.join(PROJECT_ROOT, "data/task1_test.json")
TRAIN_TXT_FILE = os.path.join(CLASSIFIER_DIR, "train.txt")

# Model paths
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/tool_classifier/")
VECTORIZER_PATH = os.path.join(SAVED_MODEL_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "tool_classifier_model.joblib")
LABEL_ENCODER_PATH = os.path.join(SAVED_MODEL_DIR, "label_encoder.joblib")

# Ensure saved_model directory exists
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# Model parameters
TEXT_FEATURE_MODEL = 'tfidf'  # or 'bow', 'word2vec'
CLASSIFIER_TYPE = 'logistic_regression'  # or 'naive_bayes', 'svm'

# Preprocessing parameters
REMOVE_STOP_WORDS = True
LEMMATIZE = True

# Classification parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2  # For train/test split

# Performance settings
TFIDF_MAX_FEATURES = 5000  # Limit vocabulary size
N_JOBS = -1  # Use all CPU cores

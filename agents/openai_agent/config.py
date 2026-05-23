import os

# OpenAI API Configuration
def get_openai_api_key():
    return os.environ.get("OPENAI_API_KEY", "dummy_api_key_for_local_testing")

def get_openai_base_url():
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

def get_if_roleplay_only():
    return os.getenv("ROLEPLAY_ONLY", "no").lower() in ["1", "true", "yes"]

def get_openai_model():
    """Get the current OpenAI model from environment variable"""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
TOP_P = float(os.getenv("TOP_P", 0.8))
MAX_TOKENS_FUNCTION_CALL = int(os.getenv("MAX_TOKENS_FUNCTION_CALL", 200))
MAX_TOKENS_RESPONSE_GENERATION = int(os.getenv("MAX_TOKENS_RESPONSE_GENERATION", 200))
MAX_TOKENS_FUNCTION_CALL_LIMIT = 2000
MAX_TOKENS_RESPONSE_GENERATION_LIMIT = 200
USE_GOLD_FUNCTIONS_FOR_ROLEPLAY = os.getenv("USE_GOLD_FUNCTIONS_FOR_ROLEPLAY", "0").lower() in ["1", "true", "yes"]
# RAG System Configuration
USE_RAG = os.getenv("USE_RAG", "0").lower() in ["1", "true", "yes"]
RAG_TOP_N = int(os.getenv("RAG_TOP_N", 3))
RAG_SIMILARITY_THRESHOLD = int(os.getenv("RAG_SIMILARITY_THRESHOLD", 55))

# Debugging and Logging Configuration
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
INTERACTION_LOG_FILE = os.getenv("INTERACTION_LOG_FILE", "results/interaction_log.json")

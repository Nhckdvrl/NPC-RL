import json
import os
import sys
import logging
import math
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from huggingface_hub import HfApi, login

# --- Configuration ---
# !!! WARNING: Hardcoding tokens is a security risk. Prefer environment variables. !!!
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", "")
EXPORT_HUB_MODEL_ID = "yinita/cpdc-emb-MiniLM-L6-v2-finetune-v1"
BASE_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Project paths (assuming this script is in agents/tool_classify/)
BASE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH_TEMPLATE = os.path.join(BASE_PROJECT_PATH, "data", "toolenv", "tools_json", "{}.json")
OUTPUT_MODEL_PATH = os.path.join(BASE_PROJECT_PATH, "models", "tool_classify_finetuned", BASE_MODEL_NAME.split('/')[-1] + "-finetuned")

# Training parameters
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 1
EVALUATION_STEPS = 1000 # Evaluate every N training steps
WARMUP_STEPS_RATIO = 0.1 # 10% of training steps for warmup

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def load_dataset(dataset_name: str, for_evaluation: bool = False):
    """Loads a dataset and converts it to InputExample format."""
    file_path = DATA_PATH_TEMPLATE.format(dataset_name)
    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {file_path}")
        return []

    for entry in data:
        query = entry.get('query')
        gold_tool_calls = entry.get('gold_tool_calls', [])

        if not query or not gold_tool_calls:
            continue

        for tool_name in gold_tool_calls:
            if for_evaluation:
                examples.append(InputExample(texts=[query, tool_name], label=1.0)) # Positive pair for similarity evaluation
            else:
                examples.append(InputExample(texts=[query, tool_name])) # Pair for MultipleNegativesRankingLoss
    logger.info(f"Loaded {len(examples)} examples from {dataset_name}.json")
    return examples

def main():
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

    # 1. Login to Hugging Face Hub
    logger.info(f"Logging into Hugging Face Hub...")
    try:
        login(token=HF_HUB_TOKEN)
        logger.info("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {e}")
        # Decide if you want to exit or continue without uploading
        # return

    # 2. Load base model
    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)

    # 3. Load datasets
    logger.info("Loading training data...")
    train_samples = load_dataset("train")
    if not train_samples:
        logger.error("No training data loaded. Exiting.")
        return

    logger.info("Loading development (sample) data for evaluation...")
    dev_samples = load_dataset("sample", for_evaluation=True)
    if not dev_samples:
        logger.warning("No development (sample) data loaded for evaluation.")

    logger.info("Loading test data for evaluation...")
    test_samples = load_dataset("test", for_evaluation=True)
    if not test_samples:
        logger.warning("No test data loaded for evaluation.")

    # 4. Prepare DataLoader and Loss
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 5. Prepare Evaluator
    evaluators = []
    if dev_samples:
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev-sample', batch_size=TRAIN_BATCH_SIZE)
        evaluators.append(dev_evaluator)
    if test_samples:
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test', batch_size=TRAIN_BATCH_SIZE)
        evaluators.append(test_evaluator)
    
    # Use the first evaluator for checkpoint saving if multiple, or None
    main_evaluator = evaluators[0] if evaluators else None
    
    # Calculate warmup steps
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = math.ceil(num_training_steps * WARMUP_STEPS_RATIO)

    # 6. Fine-tune the model
    logger.info(f"Starting fine-tuning for {NUM_EPOCHS} epochs...")
    logger.info(f"Training steps per epoch: {len(train_dataloader)}")
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Evaluation steps: {EVALUATION_STEPS}")

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=main_evaluator, # Pass the main evaluator for checkpointing
              epochs=NUM_EPOCHS,
              evaluation_steps=EVALUATION_STEPS,
              warmup_steps=warmup_steps,
              output_path=OUTPUT_MODEL_PATH,
              save_best_model=True, # Saves the best model based on the evaluator
              show_progress_bar=True)
    
    logger.info(f"Fine-tuning completed. Model saved to {OUTPUT_MODEL_PATH}")

    # If multiple evaluators were used, run them on the final model
    if len(evaluators) > 1:
        logger.info("Evaluating final model on all specified datasets:")
        for evaluator in evaluators:
            logger.info(f"Running evaluator: {evaluator.name}")
            model.evaluate(evaluator, output_path=OUTPUT_MODEL_PATH)

    # 7. Upload model to Hugging Face Hub
    logger.info(f"Uploading model to Hugging Face Hub: {EXPORT_HUB_MODEL_ID}")
    try:
        # Ensure the model is saved before uploading if fit() didn't save the final state as desired
        # model.save(OUTPUT_MODEL_PATH) # Usually fit() with output_path handles this
        
        model.save_to_hub(
            repo_id=EXPORT_HUB_MODEL_ID,
            organization=None, # Set if it's under an org
            private=False, # Set to True if you want a private model
            commit_message="feat: Add fine-tuned embedding model v1",
            token=HF_HUB_TOKEN, # Pass token if login() wasn't sufficient or for specific repo rights
            exist_ok=True # Overwrite if model already exists
        )
        logger.info(f"Model successfully uploaded to {EXPORT_HUB_MODEL_ID}")
    except Exception as e:
        logger.error(f"Failed to upload model to Hugging Face Hub: {e}")

if __name__ == '__main__':
    main()


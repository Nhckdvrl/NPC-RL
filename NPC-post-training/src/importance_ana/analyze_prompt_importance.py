import json
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.toolcall-syn.llm_client import LLMJudge, LLMJudgeConfig

def create_importance_rating_prompt(conversation_data):
    """
    Creates a prompt for the LLM to rate the importance of each context section.
    """
    worldview = conversation_data.get('worldview', '')
    player_info = json.dumps(conversation_data.get('player', {}), indent=2)
    npc_info = json.dumps(conversation_data.get('npc', {}), indent=2)
    state_info = json.dumps(conversation_data.get('state', {}), indent=2)
    # Assuming 'dialogue' contains the conversation history and the golden response is the last one from the assistant
    dialogue = conversation_data.get('dialogue', [])
    golden_response = ""
    if dialogue and isinstance(dialogue, list):
        for turn in reversed(dialogue):
            if turn.get('speaker') == 'npc':
                golden_response = turn.get('text', '')
                break

    if not golden_response:
        return None # Cannot create a prompt without a golden response to evaluate against

    prompt = f"""
    As an expert in prompt engineering for conversational AI in RPGs, your task is to evaluate the importance of different context sections for generating a specific NPC response.

    **Context Sections:**

    1.  **Worldview**:
        ```json
        {worldview}
        ```

    2.  **Player Info**:
        ```json
        {player_info}
        ```

    3.  **NPC Info**:
        ```json
        {npc_info}
        ```

    4.  **State Info**:
        ```json
        {state_info}
        ```

    **Golden NPC Response:**
    `{golden_response}`

    **Instructions:**
    Rate the importance of each context section in generating the golden NPC response on a scale of 0 to 10, where 0 is completely irrelevant and 10 is absolutely essential. Provide your answer in a single JSON object format.

    Example response format: {{"worldview": 7, "player": 4, "npc": 9, "state": 5}}
    """
    return prompt.strip()

def main():
    """Main function to run the analysis."""
    # 1. Setup
    print("Initializing LLM Judge...")
    # Make sure your OPENAI_API_KEY and other env variables are set correctly
    llm_config = LLMJudgeConfig(model_name="gpt-4o-mini") # Using a smaller model for cost-effectiveness
    llm_judge = LLMJudge(config=llm_config)
    input_file = '/path/to/npc-rl/data/task1_train.json'
    output_csv = '/path/to/npc-rl/results/prompt_importance_analysis.csv'

    # 2. Load Data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 3. Create Prompts
    print("Creating importance rating prompts...")
    prompts = []
    valid_data_indices = []
    for i, data_point in enumerate(all_data):
        prompt = create_importance_rating_prompt(data_point)
        if prompt:
            prompts.append(prompt)
            valid_data_indices.append(i)

    # 4. Batch Inference
    print(f"Sending {len(prompts)} prompts for batch prediction...")
    # The batch_predict method from the user's description is used here.
    # Assuming it handles batching, retries, and returns a list of string responses.
    responses = llm_judge.batch_predict(prompts)

    # 5. Aggregate and Analyze Results
    print("Aggregating and analyzing results...")
    importance_scores = []
    for i, response_str in enumerate(tqdm(responses, desc="Parsing responses")):
        try:
            # Clean the response to ensure it's valid JSON
            cleaned_response = response_str.strip().replace('`', '')
            if cleaned_response.startswith('json'):
                cleaned_response = cleaned_response[4:]
            
            scores = json.loads(cleaned_response)
            if isinstance(scores, dict):
                scores['data_id'] = all_data[valid_data_indices[i]]['data_id']
                importance_scores.append(scores)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not parse JSON response for prompt {i}: {response_str}. Error: {e}")

    if not importance_scores:
        print("No valid importance scores were parsed. Exiting.")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(importance_scores)
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to {output_csv}")

    # Calculate and display average scores
    avg_scores = df.mean(numeric_only=True)
    print("\n--- Average Importance Scores (0-10) ---")
    print(avg_scores.round(2))
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

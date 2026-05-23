import json
import os
import sys
import pandas as pd
from tqdm import tqdm
import time
from openai import OpenAI
from llm_client import LLMJudge, LLMJudgeConfig
config = LLMJudgeConfig(
    temperature=0.7,  # 使用较高的温度以增加多样性
    max_tokens=1024,
    model_name="gpt-4o"
)
llm = LLMJudge(config)
def create_importance_rating_prompt(base_data, turn_data):
    """Creates a prompt to rate the importance of each context section for a specific turn."""
    # Extract base context (same for all turns in a conversation)
    worldview = base_data.get('worldview', '')
    player_info = json.dumps(base_data.get('player', {}), indent=2)
    npc_info = json.dumps(base_data.get('npc', {}), indent=2)
    state_info = json.dumps(base_data.get('state', {}), indent=2)
    knowledge = base_data.get('knowledge', {})
    knowledge_info = json.dumps(knowledge.get('knowledge_info', []), indent=2)
    general_info = knowledge.get('general_info', '')

    # Extract turn-specific context
    dialogue_history = json.dumps(turn_data.get('dialogue', []), indent=2)
    golden_response = turn_data.get('gold_response', '')

    if not golden_response:
        return None

    prompt = f"""
    As an expert in prompt engineering for conversational AI in RPGs, evaluate the importance of the following context sections for generating the golden NPC response.

    **Context Sections:**
    1. Worldview: {worldview}
    2. Player Info: {player_info}
    3. NPC Info: {npc_info}
    4. State Info: {state_info}
    5. Knowledge Info (Specific Items/Facts): {knowledge_info}
    6. General Info (Broader Knowledge): {general_info}
    7. Dialogue History: {dialogue_history}

    **Golden NPC Response:** `{golden_response}`

    **Instructions:**
    Rate the importance of each section on a scale of 0-10 (0=irrelevant, 10=essential). Provide your answer in a single, minified JSON object with the same structure as the example.

    Example: <json>{{"worldview": X,"player": X,"npc": X,"state": X,"knowledge_info": X,"general_info": X,"dialogue": X}}</json>
    Answer:
    """
    return prompt.strip()

def main():
    output_csv = '/path/to/npc-rl/results/full_prompt_importance_analysis/task1_train.csv'
    input_file = '/path/to/npc-rl/data/task1_train.json'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    print("Creating importance rating prompts for each turn...")
    prompts = []
    prompt_metadata = []
    for data_point in tqdm(all_data, desc="Processing conversations"):
        total_turns = data_point.get('total_turn', 0)
        for i in range(total_turns):
            turn_key = f'turn_{i}'
            if turn_key in data_point:
                turn_data = data_point[turn_key]
                prompt = create_importance_rating_prompt(data_point, turn_data)
                if prompt:
                    prompts.append(prompt)
                    prompt_metadata.append({'data_id': data_point['data_id'], 'turn': i})

    print(f"Sending {len(prompts)} prompts for batch prediction...")
    responses = llm.batch_predict(prompts)

    print("Aggregating and analyzing results...")
    importance_scores = []
    for i, response_str in enumerate(tqdm(responses, desc="Parsing responses")):
        try:
            cleaned_response = response_str.split('<json>')[-1].split('</json>')[0]
            scores = json.loads(cleaned_response)
            if isinstance(scores, dict):
                scores.update(prompt_metadata[i])
                importance_scores.append(scores)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not parse JSON response for prompt {i}: {response_str}. Error: {e}")
        except IndexError as e:
            print(f"Warning: Could not parse JSON response for prompt {i}: {response_str}. Error: {e}")
    if not importance_scores:
        print("No valid scores were parsed. Exiting.")
        return

    df = pd.DataFrame(importance_scores)
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to {output_csv}")

    # Calculate and display average scores
    avg_scores = df.drop(columns=['data_id', 'turn']).mean(numeric_only=True)
    print("\n--- Average Importance Scores (0-10) ---")
    print(avg_scores.round(2))
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

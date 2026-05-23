import json
from datasets import load_from_disk
import os

def main():
    dataset_path = "/path/to/npc-rl/data/chat_format/Coser_tmp/test_set_custom_format/train/"
    print(f"Attempting to load dataset from: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    try:
        # Load the dataset from disk
        dataset = load_from_disk(dataset_path)
        print(f"Successfully loaded dataset from {dataset_path}")
        print(f"Dataset features: {dataset.features}")
        print(f"Number of examples in dataset: {len(dataset)}")

        # Get the first few examples (e.g., 1 or 2)
        num_examples_to_show = 2
        examples_to_show = []
        
        if len(dataset) > 0:
            for i in range(min(num_examples_to_show, len(dataset))):
                examples_to_show.append(dataset[i])
        
        if examples_to_show:
            print(f"\nShowing first {len(examples_to_show)} example(s) as JSON:")
            # Convert to JSON string with indentation for readability
            try:
                json_output = json.dumps(examples_to_show, indent=4, ensure_ascii=False)
                print(json_output)
            except TypeError as te:
                print(f"TypeError during JSON serialization: {te}")
                print("Attempting to print examples directly (may not be pure JSON if complex objects are present):")
                for i, example in enumerate(examples_to_show):
                    print(f"\n--- Example {i+1} ---")
                    print(example)
        else:
            print("Dataset is empty or no examples could be retrieved.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

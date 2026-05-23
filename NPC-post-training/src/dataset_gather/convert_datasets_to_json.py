import os
import json
from datasets import load_from_disk

from datasets import load_dataset

def convert_single_arrow_to_json(arrow_file_path):
    """Loads a single .arrow file and saves its content as a .json file."""
    print(f"--- Processing file: {arrow_file_path} ---")
    
    if not os.path.isfile(arrow_file_path):
        print(f"Error: File does not exist: {arrow_file_path}")
        return

    output_json_path = os.path.splitext(arrow_file_path)[0] + ".json"

    try:
        print(f"Loading dataset from {arrow_file_path}...")
        # Use load_dataset with the 'arrow' format for a single file
        dataset_dict = load_dataset("arrow", data_files={"train": arrow_file_path})
        dataset = dataset_dict["train"]
        
        print(f"Dataset loaded. Contains {len(dataset)} examples.")

        print("Converting dataset to list of dictionaries...")
        all_examples = dataset.to_list()
        print(f"Conversion complete. {len(all_examples)} examples in list.")

        print(f"Writing JSON to: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved JSON to {output_json_path}")

    except Exception as e:
        print(f"An error occurred while processing {arrow_file_path}: {e}")

def main():
    specific_arrow_file = "/path/to/npc-rl/data/chat_format/Coser_tmp/sft_sharegpt_format/train/sample-data-00000-of-00005.arrow"
    
    print(f"\n--- Converting specific Arrow file to JSON: {specific_arrow_file} ---")
    convert_single_arrow_to_json(specific_arrow_file)
    print("-" * 50)

if __name__ == "__main__":
    main()

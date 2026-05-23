import os
from datasets import load_dataset

def download_and_inspect_coser_dataset():
    dataset_name = "Neph0s/CoSER"
    # Corrected base path for saving the dataset
    save_base_path = "/path/to/npc-rl/data/chat_format/Coser_tmp/"

    # Ensure the base save directory exists
    os.makedirs(save_base_path, exist_ok=True)
    print(f"Base save path: {save_base_path}")

    try:
        print(f"Loading dataset '{dataset_name}'...")
        # Load the dataset. This might download it if not cached.
        # The `load_dataset` function returns a DatasetDict, which contains all splits.
        # Define the specific files we want to load as separate datasets
        # to handle their distinct schemas.
        dataset_configs = {
            "sft_sharegpt": {
                "data_files": "train/sft_conversations_sharegpt.json",
                "save_subdir": "sft_sharegpt_format"
            },
            "test_set_custom": {
                "data_files": "test/test_set.json",
                "save_subdir": "test_set_custom_format"
            }
        }

        loaded_datasets_info = {}

        for config_name, config_details in dataset_configs.items():
            print(f"\nProcessing configuration: '{config_name}'")
            try:
                # Load each configuration as a separate dataset
                # The load_dataset will return a DatasetDict, even if only one file/split is specified.
                # We expect a 'train' split by default if not further specified by the file structure.
                current_dataset_dict = load_dataset(dataset_name, data_files=config_details["data_files"])
                print(f"Successfully loaded data for '{config_name}'.")
                
                # Save each split found in this loaded configuration
                for split_name, dataset_split in current_dataset_dict.items():
                    # Use a more descriptive save path including the original split name
                    specific_save_subdir = os.path.join(save_base_path, config_details["save_subdir"], split_name)
                    os.makedirs(specific_save_subdir, exist_ok=True)
                    print(f"Saving '{config_name} - {split_name}' split to {specific_save_subdir}...")
                    dataset_split.save_to_disk(specific_save_subdir)
                    print(f"'{config_name} - {split_name}' split saved successfully.")
                
                loaded_datasets_info[config_name] = current_dataset_dict

            except Exception as e:
                print(f"Error processing configuration '{config_name}': {e}")
                continue # Continue to the next configuration if one fails

        print("\n--- Inspection of Loaded Datasets ---")
        if not loaded_datasets_info:
            print("No datasets were successfully loaded and saved.")
            return

        for config_name, dataset_dict in loaded_datasets_info.items():
            print(f"\n--- Configuration: {config_name} ---")
            print(dataset_dict)
            for split_name, dataset_split in dataset_dict.items():
                print(f"\n  --- {split_name.upper()} SPLIT (from {config_name}) --- ")
                print(f"  Number of examples: {len(dataset_split)}")
                print(f"  Features: {dataset_split.features}")
                if len(dataset_split) > 0:
                    print("  First example:")
                    print(dataset_split[0])
                if len(dataset_split) > 1:
                    print("  Second example:")
                    print(dataset_split[1])

        print("\nDownload and inspection complete.")
        print(f"Dataset saved in subdirectories under: {save_base_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_inspect_coser_dataset()

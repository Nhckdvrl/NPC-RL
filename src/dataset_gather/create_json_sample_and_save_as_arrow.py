import json
import random
import pyarrow as pa
import os

def sample_json_and_save_as_arrow(input_json_path, output_arrow_path, num_samples=5):
    print(f"Processing JSON file: {input_json_path}")
    print(f"Outputting {num_samples} samples to Arrow file: {output_arrow_path}")

    if not os.path.isfile(input_json_path):
        print(f"Error: Input JSON file not found: {input_json_path}")
        return

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {input_json_path}")

        if len(data) < num_samples:
            print(f"Warning: Requested {num_samples} samples, but only {len(data)} records available. Sampling all available records.")
            samples = data
        else:
            samples = random.sample(data, num_samples)
        print(f"Successfully sampled {len(samples)} records.")

        if not samples:
            print("Error: No samples were selected (original data might be empty).")
            return

        # Convert list of dictionaries to PyArrow Table
        # This infers the schema from the first sample, assuming all samples have the same structure.
        print("Converting samples to PyArrow Table...")
        try:
            table = pa.Table.from_pylist(samples)
            print("Successfully converted to PyArrow Table.")
        except Exception as e:
            print(f"Error converting list of dicts to PyArrow Table: {e}")
            print("This might be due to inconsistent data types or structures within the samples.")
            # Attempt to infer schema more robustly if simple from_pylist fails
            # For complex or nested data, a more explicit schema definition might be needed.
            # Trying with schema inference from all samples (can be slower)
            try:
                print("Attempting schema inference from all samples...")
                # Infer schema from the first record to define fields
                # This is a simplified approach; for highly variable structures, more robust schema handling is needed.
                schema_fields = [(key, pa.infer_type(value)) for key, value in samples[0].items()]
                schema = pa.schema(schema_fields)
                table = pa.Table.from_pylist(samples, schema=schema)
                print("Successfully converted to PyArrow Table with explicit schema inference.")
            except Exception as e_schema:
                print(f"Error converting to PyArrow Table even with schema inference: {e_schema}")
                return

        # Write the Table to an Arrow file (IPC format)
        print(f"Writing Arrow Table to {output_arrow_path}...")
        with pa.OSFile(output_arrow_path, 'wb') as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        
        print(f"Successfully saved {len(samples)} samples to {output_arrow_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}. File might be corrupted or not valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    input_file = "/path/to/npc-rl/data/chat_format/Coser_tmp/sft_sharegpt_format/train/data-00000-of-00005.json"
    output_file = "/path/to/npc-rl/data/chat_format/Coser_tmp/sft_sharegpt_format/train/sample-data-00000-of-00005.arrow"
    num_to_sample = 5

    sample_json_and_save_as_arrow(input_file, output_file, num_to_sample)

if __name__ == "__main__":
    # Ensure pyarrow is installed
    try:
        import pyarrow
    except ImportError:
        print("Error: pyarrow library is not installed. Please install it with 'pip install pyarrow'")
        exit(1)
    main()

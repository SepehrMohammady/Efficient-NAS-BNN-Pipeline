# test_load_hf.py (Refined for minimal loading tests)
from datasets import load_dataset, get_dataset_infos
import time
import traceback

DATASET_NAME = "Harvard-Edge/Wake-Vision"
DATASET_CONFIG = None 
DATASET_SPLIT_TO_USE = 'train_quality'
NUM_EXAMPLES_TO_CHECK = 3

print(f"--- Test 1: Get Dataset Info (Metadata only) for '{DATASET_NAME}' ---")
try:
    start_time_info = time.time()
    # This should only fetch metadata, not the actual data files.
    infos = get_dataset_infos(DATASET_NAME) # Pass dataset_name directly
    end_time_info = time.time()
    
    print(f"SUCCESS: get_dataset_infos completed in {end_time_info - start_time_info:.2f}s.")
    if DATASET_CONFIG: # If a specific config was needed and exists
        if DATASET_CONFIG in infos:
            print(f"Info for config '{DATASET_CONFIG}':")
            print(infos[DATASET_CONFIG])
            if DATASET_SPLIT_TO_USE in infos[DATASET_CONFIG].splits:
                 print(f"Split '{DATASET_SPLIT_TO_USE}' found in config '{DATASET_CONFIG}'. Size: {infos[DATASET_CONFIG].splits[DATASET_SPLIT_TO_USE].num_examples} examples.")
            else:
                 print(f"Split '{DATASET_SPLIT_TO_USE}' NOT found in config '{DATASET_CONFIG}'. Available: {list(infos[DATASET_CONFIG].splits.keys())}")
        else:
            print(f"Config '{DATASET_CONFIG}' not found. Available configs: {list(infos.keys())}")
    else: # No specific config, assume it's the main one
        # Heuristic: try to find the main config or the one that matches dataset name parts
        main_config_name = DATASET_NAME.split("/")[-1].lower() # e.g. "wake-vision"
        found_info = None
        if main_config_name in infos:
            found_info = infos[main_config_name]
        elif 'default' in infos: # Common default config name
            found_info = infos['default']
        elif infos: # Just take the first one if others not found
            found_info = next(iter(infos.values()))

        if found_info:
            print(f"Info for a primary config (e.g., '{found_info.config_name}'):")
            print(f"  Description: {found_info.description[:100]}...") # Print start of description
            print(f"  Features: {found_info.features}")
            print(f"  Size in Bytes: {found_info.size_in_bytes}")
            if DATASET_SPLIT_TO_USE in found_info.splits:
                print(f"  Split '{DATASET_SPLIT_TO_USE}' found. Size: {found_info.splits[DATASET_SPLIT_TO_USE].num_examples} examples.")
            else:
                print(f"  Split '{DATASET_SPLIT_TO_USE}' NOT found. Available in this config: {list(found_info.splits.keys())}")
        else:
            print("Could not determine primary config info.")


except Exception as e:
    print(f"ERROR in Test 1 (get_dataset_infos): {e}")
    traceback.print_exc()


print(f"\n--- Test 2: Attempting to stream first {NUM_EXAMPLES_TO_CHECK} examples from '{DATASET_SPLIT_TO_USE}' ---")
try:
    start_time_stream = time.time()
    # Use streaming=True and iterate only a few times
    ds_stream = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split=DATASET_SPLIT_TO_USE, streaming=True)
    print("SUCCESS: load_dataset (streaming) initiated.")
    count_stream = 0
    print("Iterating through stream...")
    for example in ds_stream:
        if count_stream >= NUM_EXAMPLES_TO_CHECK:
            break
        print(f"  Streamed example {count_stream + 1} keys: {list(example.keys())}")
        # print(f"    Label ({LABEL_COLUMN_NAME}): {example.get(LABEL_COLUMN_NAME)}") # Requires LABEL_COLUMN_NAME
        count_stream += 1
    end_time_stream = time.time()
    if count_stream == NUM_EXAMPLES_TO_CHECK:
        print(f"SUCCESS: Streamed {count_stream} examples in {end_time_stream - start_time_stream:.2f}s.")
    else:
        print(f"WARNING: Only streamed {count_stream}/{NUM_EXAMPLES_TO_CHECK} examples in {end_time_stream - start_time_stream:.2f}s.")


except Exception as e:
    print(f"ERROR in Test 2 (streaming data): {e}")
    traceback.print_exc()

print(f"\n--- Test 3: Attempting to load first {NUM_EXAMPLES_TO_CHECK} examples (NON-STREAMING SLICE) ---")
try:
    start_time_nonstream = time.time()
    # This syntax directly fetches a slice. It will download the required data parts for these few examples.
    # This is often more robust for small checks than full streaming if streaming init is problematic.
    ds_slice = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split=f'{DATASET_SPLIT_TO_USE}[:{NUM_EXAMPLES_TO_CHECK}]')
    print(f"SUCCESS: load_dataset (non-streaming slice) loaded {len(ds_slice)} examples.")
    count_nonstream = 0
    for example in ds_slice:
        print(f"  Non-streamed example {count_nonstream + 1} keys: {list(example.keys())}")
        # print(f"    Label ({LABEL_COLUMN_NAME}): {example.get(LABEL_COLUMN_NAME)}")
        count_nonstream += 1
    end_time_nonstream = time.time()
    print(f"SUCCESS: Processed {count_nonstream} non-streamed examples in {end_time_nonstream - start_time_nonstream:.2f}s.")

except Exception as e:
    print(f"ERROR in Test 3 (non-streaming slice data): {e}")
    traceback.print_exc()

print("\n--- Test Script Finished ---")
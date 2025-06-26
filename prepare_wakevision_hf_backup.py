import os
import shutil
from datasets import load_dataset, Image as HFImage # For Image feature
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
import numpy as np
import time # For potential sleeps
import traceback # For detailed error printing

# --- Configuration ---
DATASET_NAME = "Harvard-Edge/Wake-Vision"
DATASET_CONFIG = None 
DATASET_SPLIT_TO_USE = 'train_quality' 

TARGET_BASE_DIR = "./data/WakeVision_Prepared_V3" # Using a new versioned dir name
TARGET_IMAGE_SIZE = (128, 128) 
TOTAL_SUBSET_SIZE = 1000     # KEEP SMALL for initial full pipeline testing
VAL_SPLIT_SIZE = 0.1 
SEED = 42

# === VERIFIED LABEL CONFIGURATION (based on Hugging Face dataset viewer & error logs) ===
LABEL_COLUMN_NAME = 'person' 
POSITIVE_LABEL_VALUE = 1     # Assuming 'Yes' for person maps to 1
CLASS_MAP = {
    1: "person_present",    
    0: "no_person_present"  
}
# --- End Configuration ---

def inspect_dataset_features(): # Keep this function for future reference or debugging
    print("--- Inspecting Dataset Features (Harvard-Edge/Wake-Vision) ---")
    # ... (content of inspect_dataset_features from previous response, if you want to keep it) ...
    # For now, this function is not called by default in the main execution block.
    pass


def prepare_wake_vision():
    print(f"\n--- Starting Data Preparation for Wake Vision ---")
    print(f"Targeting a subset of {TOTAL_SUBSET_SIZE} images for '{TARGET_BASE_DIR}'")
    print(f"Images will be resized to: {TARGET_IMAGE_SIZE}")
    print(f"Using label column: '{LABEL_COLUMN_NAME}', Positive class mapped from value: {POSITIVE_LABEL_VALUE}")
    print(f"Class mapping for folders: {CLASS_MAP}")

    train_dir_output = os.path.join(TARGET_BASE_DIR, 'train')
    val_dir_output = os.path.join(TARGET_BASE_DIR, 'val')

    if os.path.exists(TARGET_BASE_DIR):
        print(f"Removing existing directory: {TARGET_BASE_DIR}")
        shutil.rmtree(TARGET_BASE_DIR)
    os.makedirs(train_dir_output, exist_ok=True)
    os.makedirs(val_dir_output, exist_ok=True)

    for class_name_folder in CLASS_MAP.values():
        os.makedirs(os.path.join(train_dir_output, class_name_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir_output, class_name_folder), exist_ok=True)

    print(f"\nLoading '{DATASET_NAME}' (split: '{DATASET_SPLIT_TO_USE}') from Hugging Face using streaming...")
    try:
        full_dataset_stream = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split=DATASET_SPLIT_TO_USE, streaming=True)
        print("Dataset stream initiated.")
    except Exception as e:
        print(f"FATAL: Error loading dataset stream: {e}"); traceback.print_exc(); return

    images_data = [] 
    print(f"\nFetching and processing up to {TOTAL_SUBSET_SIZE} images from the stream...")
    processed_image_count = 0
    iteration_attempts = 0
    # Allow more attempts than TOTAL_SUBSET_SIZE to account for skips due to errors or missing data
    max_iteration_attempts = TOTAL_SUBSET_SIZE * 5 if TOTAL_SUBSET_SIZE > 100 else TOTAL_SUBSET_SIZE * 10

    for example in full_dataset_stream:
        if processed_image_count >= TOTAL_SUBSET_SIZE: break
        if iteration_attempts >= max_iteration_attempts:
            print(f"Warning: Reached max iteration attempts ({max_iteration_attempts}) while trying to get {TOTAL_SUBSET_SIZE} images.")
            break
        
        iteration_attempts += 1
        current_raw_label_for_log = "N/A"
        
        try:
            # time.sleep(0.01) # Very small delay, might not be needed if connection is stable for small N

            img_pil = example.get('image')
            current_raw_label = example.get(LABEL_COLUMN_NAME)
            current_raw_label_for_log = current_raw_label # For logging in except block

            if not isinstance(img_pil, PILImage.Image): continue
            if current_raw_label is None: continue 
            if not isinstance(current_raw_label, int):
                print(f"Warning: Label for item approx {iteration_attempts} is not int: {current_raw_label} (type: {type(current_raw_label)}). Skipping.")
                continue

            mapped_binary_label = 1 if current_raw_label == POSITIVE_LABEL_VALUE else 0
            
            img_resized = img_pil.resize(TARGET_IMAGE_SIZE, PILImage.Resampling.LANCZOS)
            images_data.append({'image': img_resized, 'label': mapped_binary_label, 'original_idx_approx': iteration_attempts})
            processed_image_count += 1

            if processed_image_count % (max(1, TOTAL_SUBSET_SIZE // 20)) == 0 or processed_image_count == TOTAL_SUBSET_SIZE:
                 print(f"  Fetched and processed {processed_image_count}/{TOTAL_SUBSET_SIZE} images (attempt {iteration_attempts})...")
        
        except (KeyboardInterrupt): print("INFO: Processing interrupted by user."); break
        except Exception as e: # Catch generic exceptions during processing one example
            print(f"ERROR processing example approx attempt {iteration_attempts} (raw label value: {current_raw_label_for_log}): {e}. Skipping this example.")
            # traceback.print_exc() # Uncomment for full traceback for these errors
            
    print(f"\nFinished fetching. Total images collected for subset: {len(images_data)}")
    
    # Calculate raw label distribution from the successfully processed images
    # This was missing in your last version; it should count the mapped labels (0s and 1s)
    if images_data:
        mapped_labels_collected = [item['label'] for item in images_data]
        unique_mapped, counts_mapped = np.unique(mapped_labels_collected, return_counts=True)
        print(f"Mapped binary label distribution collected (0 for '{CLASS_MAP[0]}', 1 for '{CLASS_MAP[1]}'): {dict(zip(unique_mapped, counts_mapped))}")
    else:
        print("No images were successfully processed into images_data list.")
        return # Cannot proceed

    if len(images_data) < TOTAL_SUBSET_SIZE * 0.1 and TOTAL_SUBSET_SIZE > 100: 
        print(f"Warning: Only collected {len(images_data)} images, significantly less than target {TOTAL_SUBSET_SIZE}.")
    if not images_data: print("No images were successfully processed to split. Aborting."); return

    labels_for_split = [item['label'] for item in images_data] 
    indices = np.arange(len(images_data))
    
    unique_final_mapped_labels, counts_final_mapped_labels = np.unique(labels_for_split, return_counts=True)
    mapped_label_dist_str = dict(zip([CLASS_MAP.get(l, f"Unknown_mapped_{l}") for l in unique_final_mapped_labels], counts_final_mapped_labels))
    print(f"Final mapped binary label distribution in subset to be saved: {mapped_label_dist_str}")

    can_stratify = True
    if len(unique_final_mapped_labels) < 2:
        if len(images_data) > 1 : print(f"Warning: Subset contains only {len(unique_final_mapped_labels)} class after mapping. Cannot stratify.")
        can_stratify = False
    else: 
        for label_val, count_val in zip(unique_final_mapped_labels, counts_final_mapped_labels):
             if count_val < 2 : # For train_test_split, stratification needs at least 2 samples per class.
                print(f"Warning: Mapped class {CLASS_MAP.get(label_val, label_val)} has only {count_val} sample(s). Stratification disabled.")
                can_stratify = False; break
    
    effective_val_split_size = VAL_SPLIT_SIZE
    if not can_stratify and len(unique_final_mapped_labels) < 2 and VAL_SPLIT_SIZE > 0 :
        print("Disabling validation split as only one class is present or insufficient samples for stratification.")
        effective_val_split_size = 0.0
    elif VAL_SPLIT_SIZE > 0 and len(images_data) * VAL_SPLIT_SIZE < 1 and len(images_data) > 1 : # If requested val is less than 1 image
        print(f"Warning: Requested validation split {VAL_SPLIT_SIZE*100}% of {len(images_data)} is less than 1 image. Setting validation size to 0.")
        effective_val_split_size = 0.0
        can_stratify = False
    elif VAL_SPLIT_SIZE == 0:
        can_stratify = False # No stratification if no validation split

    train_indices, val_indices = train_test_split(
        indices,
        test_size=effective_val_split_size, 
        random_state=SEED,
        stratify=(labels_for_split if can_stratify and effective_val_split_size > 0 else None) # Only stratify if possible and val split exists
    )
    
    print(f"\nSplitting into {len(train_indices)} train and {len(val_indices)} validation images.")

    print("Saving training images...")
    for tr_idx_list_pos, master_idx in enumerate(train_indices):
        item = images_data[master_idx]
        img_pil_to_save = item['image']
        mapped_binary_label_to_save = item['label']
        class_folder_name = CLASS_MAP[mapped_binary_label_to_save]
        class_path = os.path.join(train_dir_output, class_folder_name)
        os.makedirs(class_path, exist_ok=True) 
        img_pil_to_save.save(os.path.join(class_path, f"train_img_{master_idx}_id{tr_idx_list_pos}.png")) # More unique name

    print("Saving validation images...")
    for val_idx_list_pos, master_idx in enumerate(val_indices):
        item = images_data[master_idx]
        img_pil_to_save = item['image']
        mapped_binary_label_to_save = item['label']
        class_folder_name = CLASS_MAP[mapped_binary_label_to_save]
        class_path = os.path.join(val_dir_output, class_folder_name)
        os.makedirs(class_path, exist_ok=True)
        img_pil_to_save.save(os.path.join(class_path, f"val_img_{master_idx}_id{val_idx_list_pos}.png"))

    print("\nWake Vision data preparation complete for the subset.")
    print(f"Train data: {train_dir_output} ({len(train_indices)} images)")
    print(f"Validation data (for NAS): {val_dir_output} ({len(val_indices)} images)")

if __name__ == '__main__':
    # To inspect features first (run this script directly from terminal):
    # print("Running feature inspection first...")
    # inspect_dataset_features()
    # print("\nIf label configuration at the top of the script needs changes, edit them now.")
    # input("Press Enter to continue with data preparation, or Ctrl+C to edit script...")
    
    prepare_wake_vision()
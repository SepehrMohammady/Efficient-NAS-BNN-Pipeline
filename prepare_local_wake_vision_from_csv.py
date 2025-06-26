import os
import shutil
import pandas as pd
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm 
import traceback
import random # Ensure random is imported

# --- Configuration ---
LOCAL_WAKE_VISION_BASE_DIR = "./WakeVision" 
EXTRACTED_IMAGES_SUBDIR = "extracted_train_images"
TRAIN_METADATA_CSV_FILENAME = "wake_vision_train_large.csv"

TARGET_OUTPUT_BASE_DIR = "./data/WakeVision_From_Local_SSD_V3" # Incremented version for new run
TARGET_IMAGE_SIZE = (128, 128) 
TOTAL_SUBSET_SIZE = 5000      # Keep small (e.g., 50k) until all issues are resolved
VAL_SPLIT_SIZE = 0.1          
SEED = 42                      
IMAGE_FILENAME_COLUMN_IN_CSV = 'filename' 
LABEL_COLUMN_IN_CSV = 'person'            
POSITIVE_LABEL_VALUE_IN_CSV = 1           
CLASS_MAP_FOR_SAVING = {1: "person_present", 0: "no_person_present"}
# --- End Configuration ---

def prepare_local_wake_vision_from_csv():
    print(f"--- Starting Data Preparation from Local CSV and Extracted Images ---")
    
    local_extracted_images_full_path = os.path.abspath(os.path.join(LOCAL_WAKE_VISION_BASE_DIR, EXTRACTED_IMAGES_SUBDIR))
    train_metadata_csv_full_path = os.path.abspath(os.path.join(LOCAL_WAKE_VISION_BASE_DIR, TRAIN_METADATA_CSV_FILENAME))
    target_output_base_dir_abs = os.path.abspath(TARGET_OUTPUT_BASE_DIR)

    print(f"Source Extracted Images Path: {local_extracted_images_full_path}")
    print(f"Source Metadata CSV: {train_metadata_csv_full_path}")
    print(f"Target Output Base Directory: {target_output_base_dir_abs}")
    print(f"Target Subset Size: {TOTAL_SUBSET_SIZE}, Target Image Size: {TARGET_IMAGE_SIZE}")
    print(f"Labeling: CSV Column '{LABEL_COLUMN_IN_CSV}', Positive Value '{POSITIVE_LABEL_VALUE_IN_CSV}' maps to class '1' ({CLASS_MAP_FOR_SAVING.get(1, 'N/A')})")

    if os.path.exists(target_output_base_dir_abs):
        print(f"Removing existing target directory: {target_output_base_dir_abs}")
        shutil.rmtree(target_output_base_dir_abs)
    os.makedirs(target_output_base_dir_abs, exist_ok=True)
    train_dir_output = os.path.join(target_output_base_dir_abs, 'train')
    val_dir_output = os.path.join(target_output_base_dir_abs, 'val')
    os.makedirs(train_dir_output, exist_ok=True)
    os.makedirs(val_dir_output, exist_ok=True)

    for class_name_folder in CLASS_MAP_FOR_SAVING.values():
        os.makedirs(os.path.join(train_dir_output, class_name_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir_output, class_name_folder), exist_ok=True)

    print(f"\nLoading metadata from {train_metadata_csv_full_path}...")
    if not os.path.exists(train_metadata_csv_full_path):
        print(f"FATAL: Metadata CSV file not found: {train_metadata_csv_full_path}"); return False
    try:
        df_metadata = pd.read_csv(train_metadata_csv_full_path)
        print(f"Successfully loaded metadata: {len(df_metadata)} rows.")
    except Exception as e:
        print(f"FATAL: Error loading metadata CSV: {e}"); traceback.print_exc(); return False

    all_potential_images = []
    print(f"\nProcessing metadata and matching with existing image files...")
    if IMAGE_FILENAME_COLUMN_IN_CSV not in df_metadata.columns:
        print(f"FATAL: Image filename column '{IMAGE_FILENAME_COLUMN_IN_CSV}' not found in CSV."); return False
    if LABEL_COLUMN_IN_CSV not in df_metadata.columns:
        print(f"FATAL: Label column '{LABEL_COLUMN_IN_CSV}' not found in CSV."); return False

    raw_label_distribution_csv = df_metadata[LABEL_COLUMN_IN_CSV].value_counts(dropna=False).to_dict()
    print(f"Raw label distribution in full CSV (column '{LABEL_COLUMN_IN_CSV}'): {raw_label_distribution_csv}")

    missing_files_count = 0
    for index, row in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc="Scanning CSV & matching images"):
        try:
            image_filename_from_csv = row[IMAGE_FILENAME_COLUMN_IN_CSV]
            raw_label_val_from_csv = row[LABEL_COLUMN_IN_CSV]
            if not isinstance(image_filename_from_csv, str): missing_files_count += 1; continue
            if pd.isna(raw_label_val_from_csv): missing_files_count += 1; continue
            full_image_path = os.path.join(local_extracted_images_full_path, image_filename_from_csv)
            if os.path.exists(full_image_path):
                current_label_val_for_comp = type(POSITIVE_LABEL_VALUE_IN_CSV)(raw_label_val_from_csv)
                mapped_binary_label = 1 if current_label_val_for_comp == POSITIVE_LABEL_VALUE_IN_CSV else 0
                all_potential_images.append({'filepath': full_image_path, 'label': mapped_binary_label})
            else: missing_files_count += 1
        except KeyError as ke: print(f"KeyError at CSV row {index}: {ke}. Skipping."); missing_files_count += 1; continue
        except Exception as e_loop: print(f"Unexpected error processing CSV row {index}: {e_loop}"); missing_files_count += 1; continue
    
    if missing_files_count > 0:
        print(f"Warning: {missing_files_count} (out of {len(df_metadata)}) image files listed in CSV were not found or had processing errors during scan.")
    print(f"Found {len(all_potential_images)} existing and processable image files with labels from CSV.")
    if not all_potential_images:
        print("FATAL: No usable image files found. Check paths, CSV content, and extracted image names."); return False

    random.seed(SEED)
    subset_to_process = []
    df_all_potential = pd.DataFrame(all_potential_images)

    if len(all_potential_images) > TOTAL_SUBSET_SIZE:
        print(f"DEBUG: Attempting to sample {TOTAL_SUBSET_SIZE} images from {len(all_potential_images)} available valid images.")
        unique_labels_in_potential, counts_in_potential = np.unique(df_all_potential['label'], return_counts=True)
        temp_dist_for_print = dict(zip([CLASS_MAP_FOR_SAVING.get(l, f"UnknownLabel_{l}") for l in unique_labels_in_potential], counts_in_potential))
        print(f"DEBUG: Distribution of {len(all_potential_images)} matched files before subsetting: {temp_dist_for_print}")

        can_stratify_sample = len(unique_labels_in_potential) >= 2
        n_samples_per_group_ideal = TOTAL_SUBSET_SIZE // len(unique_labels_in_potential) if can_stratify_sample else TOTAL_SUBSET_SIZE

        if can_stratify_sample:
            print(f"DEBUG: Attempting balanced stratified sampling (ideal {n_samples_per_group_ideal} per class)...")
            try:
                selected_df_groups = []
                for label_val, group_df in df_all_potential.groupby('label'):
                    n_to_sample = min(len(group_df), n_samples_per_group_ideal)
                    selected_df_groups.append(group_df.sample(n=n_to_sample, random_state=SEED, replace=False))
                selected_df = pd.concat(selected_df_groups).reset_index(drop=True)
                print(f"DEBUG: After initial group sampling, got {len(selected_df)} samples.")
                
                if len(selected_df) < TOTAL_SUBSET_SIZE:
                    remaining_needed = TOTAL_SUBSET_SIZE - len(selected_df)
                    print(f"DEBUG: Need {remaining_needed} more samples.")
                    selected_filepaths = set(selected_df['filepath'])
                    remaining_items_df = df_all_potential[~df_all_potential['filepath'].isin(selected_filepaths)]
                    if len(remaining_items_df) > 0:
                        additional_samples_count = min(remaining_needed, len(remaining_items_df))
                        print(f"DEBUG: Sampling {additional_samples_count} additional items randomly.")
                        additional_samples_df = remaining_items_df.sample(n=additional_samples_count, random_state=SEED, replace=False)
                        selected_df = pd.concat([selected_df, additional_samples_df]).reset_index(drop=True)
                subset_to_process = selected_df.sample(frac=1, random_state=SEED).reset_index(drop=True).head(TOTAL_SUBSET_SIZE).to_dict('records') # Shuffle and ensure exact size
                print(f"DEBUG: Selected a subset of {len(subset_to_process)} images via stratified attempt.")
            except Exception as e: 
                print(f"WARNING: An unexpected error during stratified sampling ({e}). Falling back to random sample.")
                traceback.print_exc()
                random.shuffle(all_potential_images)
                subset_to_process = all_potential_images[:TOTAL_SUBSET_SIZE]
        else: 
             print("DEBUG: Not enough class diversity for initial stratified sample, or only one class. Taking random sample.")
             random.shuffle(all_potential_images)
             subset_to_process = all_potential_images[:TOTAL_SUBSET_SIZE]
    else: 
        subset_to_process = all_potential_images 
    
    print(f"DEBUG: Actual subset size to process into train/val: {len(subset_to_process)}")
    if not subset_to_process: 
        print("FATAL: Subset to process is empty after sampling attempt."); return False

    labels_for_split = [item['label'] for item in subset_to_process]
    indices = np.arange(len(subset_to_process))
    unique_mapped_labels_final, counts_mapped_labels_final = np.unique(labels_for_split, return_counts=True)
    final_mapped_label_dist_str = dict(zip([CLASS_MAP_FOR_SAVING.get(l, f"Unknown_mapped_{l}") for l in unique_mapped_labels_final], counts_mapped_labels_final))
    print(f"DEBUG: Final mapped binary label distribution in subset to be split: {final_mapped_label_dist_str}")

    can_stratify_tts = len(unique_mapped_labels_final) >= 2 and all(c >= 2 for c in counts_mapped_labels_final)
    if not can_stratify_tts: print("Warning: Cannot stratify train/val split robustly due to insufficient samples in one class in the final subset.")

    actual_test_size_for_split = VAL_SPLIT_SIZE
    if len(subset_to_process) * VAL_SPLIT_SIZE < 1 and len(subset_to_process) > 1: # if val split would be < 1 image
        actual_test_size_for_split = 1 # Ensure at least 1 image if possible
    elif len(subset_to_process) <=1 : # Not enough to split
         actual_test_size_for_split = 0


    if actual_test_size_for_split == 0 and len(subset_to_process) > 0 : 
        print("Warning: Not enough data for a validation split or VAL_SPLIT_SIZE is 0. Using all data for training.")
        train_indices = indices
        val_indices = np.array([]) # Ensure it's an array
    elif not subset_to_process: 
        train_indices, val_indices = np.array([]), np.array([])
    else:
        try:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=actual_test_size_for_split,
                random_state=SEED,
                stratify=labels_for_split if can_stratify_tts else None
            )
        except ValueError as ve_split: # Catch error if stratify fails due to too few samples
            print(f"Warning: Stratified train_test_split failed ({ve_split}). Splitting without stratification.")
            train_indices, val_indices = train_test_split(
                indices, test_size=actual_test_size_for_split, random_state=SEED, stratify=None
            )

    print(f"\nSplitting into {len(train_indices)} train and {len(val_indices)} validation images.")

    if not train_indices.size > 0 and len(subset_to_process) > 0 :
        print(f"FATAL: No training images after split. This should not happen if subset_to_process was populated."); return False
    
    def process_and_save_local(img_indices_list, output_dir_base, split_name_label):
        print(f"Processing and saving {split_name_label} images...")
        if not isinstance(img_indices_list, (np.ndarray, list)) or len(img_indices_list) == 0 :
            print(f"No images to save for {split_name_label} split (indices list is empty or invalid type).")
            return
        
        for list_idx, master_idx in enumerate(tqdm(img_indices_list, desc=f"Saving {split_name_label}")):
            item = subset_to_process[master_idx]
            original_filepath = item['filepath']
            mapped_binary_label = item['label']
            try:
                img = PILImage.open(original_filepath)
                if img.mode != 'RGB': img = img.convert('RGB')
                img_resized = img.resize(TARGET_IMAGE_SIZE, PILImage.Resampling.LANCZOS)
                class_folder_name = CLASS_MAP_FOR_SAVING[mapped_binary_label]
                output_class_path = os.path.join(output_dir_base, class_folder_name)
                img_resized.save(os.path.join(output_class_path, f"{split_name_label}_img_{list_idx}.png"))
            except FileNotFoundError: print(f"ERROR during save: File not found: {original_filepath}")
            except PILImage.UnidentifiedImageError: print(f"ERROR during save: Cannot identify image file: {original_filepath}")
            except Exception as e_save: print(f"ERROR processing/saving image {original_filepath}: {e_save}"); traceback.print_exc()
        print(f"Finished saving {split_name_label} images.")

    process_and_save_local(train_indices, train_dir_output, "train")
    process_and_save_local(val_indices, val_dir_output, "val")

    print("\nLocal Wake Vision data preparation complete for the subset.")
    print(f"Train data: {train_dir_output} (approx. {len(train_indices)} images)")
    print(f"Validation data (for NAS): {val_dir_output} (approx. {len(val_indices)} images)")
    return True

if __name__ == '__main__':
    if not prepare_local_wake_vision_from_csv():
        print("FATAL: Data preparation script failed overall.")
        exit(1)
    else:
        print("Data preparation script completed successfully.")
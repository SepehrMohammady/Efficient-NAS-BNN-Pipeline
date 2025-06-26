# Example: WakeVision Dataset Configuration
# Copy this section to Cell 2 of run_all.ipynb for WakeVision dataset

# --- Dataset and Architecture Configuration ---
dataset_name = "WakeVision"
architecture_name = "superbnn_wakevision_large" 
wakevision_img_size = 128

# --- Paths ---
data_path = "./data/WakeVision_From_Local_SSD_V3"
base_work_dir = "./work_dirs/wakevision_nasbnn_run"

# --- Training Parameters ---
train_supernet_epochs = 10
train_supernet_batch_size = 64
train_supernet_lr = "2.5e-3"
train_supernet_wd = "5e-6"

# --- Search Parameters ---
search_max_epochs = 10
search_population_num = 50
search_ops_min = 3.8  # Adjust based on check_ops.py output
search_ops_max = 6.2  # Adjust based on check_ops.py output
search_step = 0.2

# --- Test Parameters ---
ops_key_to_test1 = 5
ops_key_to_test2 = 6

# --- Fine-tuning Parameters ---
finetune_batch_size = 64
finetune_lr = "5e-5"
finetune_epochs = 50

# Example: ImageNet Dataset Configuration
# Copy this section to Cell 2 of run_all.ipynb for ImageNet dataset

# --- Dataset and Architecture Configuration ---
dataset_name = "imagenet"
architecture_name = "superbnn" # Original NAS-BNN architecture for ImageNet
imagenet_img_size = 224

# --- Paths ---
# Ensure this path points to your actual ImageNet data (train/ and val/ folders)
data_path = "./data/imagenet" 
base_work_dir = "./work_dirs/imagenet_nasbnn_run"

# --- Training Parameters ---
# NOTE: Original paper uses 120 epochs. On a generic laptop, this is extremely slow.
# We set 20 here for a "test run". For full reproduction, set to 120.
train_supernet_epochs = 20
train_supernet_batch_size = 32 # Reduced for 6GB VRAM (RTX 4050)
train_supernet_lr = "2.5e-3" 
train_supernet_wd = "1e-5"

# --- Search Parameters ---
search_max_epochs = 20
search_population_num = 1024 # Original paper setting
# OPs range based on ImageNet results (ReActNet-A is ~87M, NAS-BNN ~57M)
search_ops_min = 3.0
search_ops_max = 8.0
search_step = 0.5

# --- Test Parameters ---
# Keys approximately matching 57M and 80M OPs
ops_key_to_test1 = 5
ops_key_to_test2 = 8

# --- Fine-tuning Parameters ---
finetune_batch_size = 32 # Reduced for VRAM
finetune_lr = "5e-5"
finetune_epochs = 20 # Full reproduction typically 300-512

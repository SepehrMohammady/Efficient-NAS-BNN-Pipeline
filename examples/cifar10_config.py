# Example: CIFAR-10 Dataset Configuration
# Copy this section to Cell 2 of run_all.ipynb for CIFAR-10 dataset

# --- Dataset and Architecture Configuration ---
dataset_name = "cifar10"
architecture_name = "superbnn_cifar10_large" 
cifar_img_size = 32

# --- Paths ---
data_path = "./data/CIFAR10"
base_work_dir = "./work_dirs/cifar10_nasbnn_run"

# --- Training Parameters ---
train_supernet_epochs = 50
train_supernet_batch_size = 2048
train_supernet_lr = "2.5e-3"
train_supernet_wd = "5e-6"

# --- Search Parameters ---
search_max_epochs = 20
search_population_num = 512
search_ops_min = 0.03  # Adjust based on check_ops.py output
search_ops_max = 1.1   # Adjust based on check_ops.py output
search_step = 0.05

# --- Test Parameters ---
ops_key_to_test1 = 0
ops_key_to_test2 = 1

# --- Fine-tuning Parameters ---
finetune_batch_size = 1024
finetune_lr = "1e-5"
finetune_epochs = 25

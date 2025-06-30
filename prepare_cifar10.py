import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

print("Downloading CIFAR-10 dataset...")
# Download training data
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
# Download test data (which we'll use as our final validation/test set)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

print("Preparing ImageFolder structure for CIFAR-10...")

base_dir = './data/CIFAR10'
train_dir_output = os.path.join(base_dir, 'train')
val_dir_output = os.path.join(base_dir, 'val') # For NAS validation
test_dir_output = os.path.join(base_dir, 'test') # Original CIFAR-10 test set

# Clean up old directories if they exist
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

os.makedirs(train_dir_output, exist_ok=True)
os.makedirs(val_dir_output, exist_ok=True)
os.makedirs(test_dir_output, exist_ok=True)

classes = train_set.classes

# Create class directories
for class_name in classes:
    os.makedirs(os.path.join(train_dir_output, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir_output, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir_output, class_name), exist_ok=True)

# Split original training set into new train and val sets (e.g., 45k for train, 5k for val)
# Stratified split to maintain class proportions
train_indices, val_indices = train_test_split(
    np.arange(len(train_set)),
    test_size=0.1,  # 10% for validation -> 5k images
    random_state=42, # for reproducibility
    stratify=train_set.targets
)

print("Processing new training set...")
for idx in train_indices:
    image, label_idx = train_set[idx]
    class_name = classes[label_idx]
    # CIFAR-10 images are PIL Images, can be saved directly
    image.save(os.path.join(train_dir_output, class_name, f"{idx}.png"))

print("Processing new validation set...")
for idx in val_indices:
    image, label_idx = train_set[idx]
    class_name = classes[label_idx]
    image.save(os.path.join(val_dir_output, class_name, f"{idx}.png"))
    
print("Processing original test set (for final evaluation)...")
for idx in range(len(test_set)):
    image, label_idx = test_set[idx]
    class_name = classes[label_idx]
    image.save(os.path.join(test_dir_output, class_name, f"test_{idx}.png"))

print("CIFAR-10 data preparation complete.")
print(f"Train data: {train_dir_output}")
print(f"Validation data (for NAS): {val_dir_output}")
print(f"Test data (original test): {test_dir_output}")

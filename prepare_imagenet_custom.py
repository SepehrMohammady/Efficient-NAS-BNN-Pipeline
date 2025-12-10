import os
import tarfile
import shutil
from tqdm import tqdm
import urllib.request

def prepare_imagenet(root_dir):
    """
    Prepares ImageNet dataset from .tar files.
    Expected structure:
    root_dir/
        ILSVRC2012_img_train.tar
        ILSVRC2012_img_val.tar
    """
    train_tar = os.path.join(root_dir, 'ILSVRC2012_img_train.tar')
    val_tar = os.path.join(root_dir, 'ILSVRC2012_img_val.tar')
    
    if not os.path.exists(train_tar) or not os.path.exists(val_tar):
        print(f"❌ Error: Please ensure both tar files are in {root_dir}:")
        print(f"   - {train_tar}")
        print(f"   - {val_tar}")
        return

    # --- 1. Prepare Training Data ---
    print("\n📦 1/2: Processing Training Data (this can take 30-60 mins)...")
    train_dir = os.path.join(root_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    # Extract main train tar
    print("   Extracting main training tar...")
    with tarfile.open(train_tar) as tar:
        tar.extractall(train_dir)
        
    # Extract sub-tars (each synset is a .tar)
    sub_tars = [f for f in os.listdir(train_dir) if f.endswith('.tar')]
    print(f"   Extracting {len(sub_tars)} class sub-tars...")
    
    for tar_file in tqdm(sub_tars):
        full_path = os.path.join(train_dir, tar_file)
        class_name = tar_file.split('.')[0]
        class_dir = os.path.join(train_dir, class_name)
        
        os.makedirs(class_dir, exist_ok=True)
        with tarfile.open(full_path) as tar:
            tar.extractall(class_dir)
        
        # Remove the sub-tar to save space
        os.remove(full_path)

    # --- 2. Prepare Validation Data ---
    print("\n📦 2/2: Processing Validation Data...")
    val_dir = os.path.join(root_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    
    # Extract val tar
    print("   Extracting validation tar...")
    with tarfile.open(val_tar) as tar:
        tar.extractall(val_dir)
        
    # Download helper script for validation mapping
    # This standard script maps the 50k images to the 1000 folders
    print("   Downloading validation mapping script...")
    valprep_url = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
    valprep_path = os.path.join(val_dir, 'valprep.sh')
    
    try:
        urllib.request.urlretrieve(valprep_url, valprep_path)
        
        # We need to execute the logic of valprep.sh in Python since we are on Windows
        # Reading the shell script to get the mapping logic would be complex,
        # but the logic is simple: move images to folders.
        # However, valprep.sh relies on `mkdir` and `mv`.
        # A more robust way on Windows is to download the SYNSETS list and ground truth.
        
        # Alternative: Use pre-defined mapping logic.
        # The validation images are alphabetical ILSVRC2012_val_00000001.JPEG ...
        # We need the ground truth mapping.
        
        print("   Organizing validation images into class folders...")
        # Download the synset mapping
        map_url = "https://raw.githubusercontent.com/raghakot/keras-resnet/master/data/imagenet_classes.txt" # Simple list
        # Actually, let's use the explicit LOC mapping file to be safe
        
        # Better approach: The 'valprep.sh' essentially unzips and runs a massive list of 'mv' commands.
        # Let's just Read the valprep.sh file we downloaded and parse it!
        
        with open(valprep_path, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Moving files"):
            parts = line.strip().split(' ')
            if len(parts) >= 2 and line.startswith('mv'): 
                # Format: mv ILSVRC2012_val_00000293.JPEG n01440764/
                img_file = parts[1]
                target_folder = parts[2].replace('/', '')
                
                # Create folder if not exists
                target_path = os.path.join(val_dir, target_folder)
                os.makedirs(target_path, exist_ok=True)
                
                # Move file
                src = os.path.join(val_dir, img_file)
                dst = os.path.join(target_path, img_file)
                if os.path.exists(src):
                    shutil.move(src, dst)

        # Cleanup
        if os.path.exists(valprep_path):
            os.remove(valprep_path)
            
    except Exception as e:
        print(f"❌ Error organizing validation data: {e}")

    print("\n✅ ImageNet Preparation Complete!")
    print(f"   Train: {train_dir}")
    print(f"   Val:   {val_dir}")

if __name__ == "__main__":
    # Hardcoded default path or user input
    default_path = "./data/imagenet"
    print(f"Default path: {default_path}")
    user_path = input("Enter path to folder containing .tar files (press Enter for default): ").strip()
    target_path = user_path if user_path else default_path
    
    prepare_imagenet(target_path)

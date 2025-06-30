import argparse
import os
import os.path as osp
import random
import shutil

import tqdm

random.seed(0)


def move_file(src_dir, dst_dir):
    pathDir = os.listdir(src_dir)

    picknumber = 50
    sample = random.sample(pathDir, picknumber)
    for name in sample:
        shutil.move(osp.join(src_dir, name), osp.join(dst_dir, name))


def link_file(src_dir, dst_dir):
    pathDir = os.listdir(src_dir)
    for name in pathDir:
        os.symlink(osp.join(src_dir, name), osp.join(dst_dir, name))


parser = argparse.ArgumentParser()
parser.add_argument('src_dir', metavar='DIR', help='Source directory containing ImageNet data')
parser.add_argument('dst_dir', metavar='DIR', help='Destination directory for split data')
args = parser.parse_args()

if __name__ == '__main__':
    train_dir = osp.join(args.src_dir, 'train')
    dst_train_dir = osp.join(args.dst_dir, 'train')
    dst_val_dir = osp.join(args.dst_dir, 'val')

    if not osp.exists(dst_train_dir):
        os.makedirs(dst_train_dir)
    else:
        print('Error: Destination training directory already exists!')
        exit(1)

    if not osp.exists(dst_val_dir):
        os.makedirs(dst_val_dir)
    else:
        print('Error: Destination validation directory already exists!')
        exit(1)

    classes = os.listdir(train_dir)
    print(f"Found {len(classes)} classes in ImageNet training directory.")
    
    for c in tqdm.tqdm(classes, desc="Processing classes"):
        src_path = osp.join(train_dir, c)
        dst_train_path = osp.join(dst_train_dir, c)
        dst_val_path = osp.join(dst_val_dir, c)
        
        if not osp.exists(dst_train_path):
            os.makedirs(dst_train_path)
        else:
            print(f'Error: Destination training class directory {c} already exists!')
            exit(1)
            
        if not osp.exists(dst_val_path):
            os.makedirs(dst_val_path)
        else:
            print(f'Error: Destination validation class directory {c} already exists!')
            exit(1)
            
        # Move some files to val split
        move_file(src_path, dst_val_path)
        
        # Link remaining files to train split
        link_file(src_path, dst_train_path)
    
    print("ImageNet split complete.")
    print(f"Training data: {dst_train_dir}")
    print(f"Validation data: {dst_val_dir}")

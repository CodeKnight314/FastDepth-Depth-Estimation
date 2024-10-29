import os
import shutil
from tqdm import tqdm
from glob import glob
import argparse

def create_directories(destination_dir: str):
    paths = [
        os.path.join(destination_dir, "train", "RGB"),
        os.path.join(destination_dir, "train", "Depth"),
        os.path.join(destination_dir, "test", "RGB"),
        os.path.join(destination_dir, "test", "Depth"),
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    return paths

def copy_images(img_pairs, dest_dirs, description):
    for rgb, depth in tqdm(img_pairs, desc=description):
        shutil.copy(rgb, os.path.join(dest_dirs[0], os.path.basename(rgb)))
        shutil.copy(depth, os.path.join(dest_dirs[1], os.path.basename(depth)))

def preprocess_data(source_dir: str, destination_dir: str):
    train_rgb_dir, train_depth_dir, test_rgb_dir, test_depth_dir = create_directories(destination_dir)
    
    train_rgb_imgs = glob(os.path.join(source_dir, "nyu2_train", "*", "*.jpg"))
    train_depth_imgs = glob(os.path.join(source_dir, "nyu2_train", "*", "*.png"))
    test_rgb_imgs = glob(os.path.join(source_dir, "nyu2_test", "*_colors.png"))
    test_depth_imgs = glob(os.path.join(source_dir, "nyu2_test", "*_depth.png"))
    
    copy_images(zip(train_rgb_imgs, train_depth_imgs), (train_rgb_dir, train_depth_dir), "[Processing files for train directory]")
    copy_images(zip(test_rgb_imgs, test_depth_imgs), (test_rgb_dir, test_depth_dir), "[Processing files for test directory]")

def main():
    parser = argparse.ArgumentParser(description="Merge photos from a directory and its subdirectories")
    parser.add_argument("--source", type=str, required=True, help="Path to the source directory")
    parser.add_argument("--destination", type=str, required=True, help="Path to the destination directory")
    
    args = parser.parse_args()
    preprocess_data(args.source, args.destination)

if __name__ == "__main__":
    main()
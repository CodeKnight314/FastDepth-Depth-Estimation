import os
import torch
import argparse
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchvision import transforms as T
import numpy as np
from model import FastDepth

def inference(root_dir: str, output_dir: str, path: str):
    """
    Main function to perform depth estimation on images in a directory.
    
    Parameters:
    - root_dir: str : Directory containing input images.
    - output_dir: str : Directory to save depth images.
    - path: str : path to .pth file for FastDepth
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = FastDepth(input_channels=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_dir = sorted(glob(os.path.join(root_dir, "*")))
    
    for img_path in tqdm(img_dir, desc="[INFO] Depth-estimation: "):
        img = Image.open(img_path).convert("RGB")
        
        img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            depth_tensor = model(img_tensor)
            
        depth_image = depth_tensor.squeeze().cpu().numpy()
        depth_image = 255 * depth_image
        depth_image = depth_image.astype(np.uint8)
        
        depth_pil = Image.fromarray(depth_image)
        img_name = os.path.basename(img_path).split('.')[0]
        depth_pil.save(os.path.join(output_dir, f"{img_name}_depth.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth estimation on images using a pre-trained model.")
    
    parser.add_argument('--root_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save depth images.")
    parser.add_argument('--resize', action='store_true', help="Whether to resize depth images to the original image size.")
    
    args = parser.parse_args()
    
    inference(args.root_dir, args.output_dir, args.resize)

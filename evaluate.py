import torch
import torch.nn as nn
import argparse
from loss import ScaleInvariantLoss
from dataset import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model import FastDepth

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, val_dl, output_path, device="cuda"):
    criterion = ScaleInvariantLoss(weight=0.5)
    
    os.makedirs(output_path, exist_ok=True)

    model.eval()
    total_val_loss = 0.0
    
    for i, data in enumerate(tqdm(val_dl, desc="[Evaluating validation dataset]")):
        rgb, depth = data
        rgb, depth = rgb.to(device), depth.to(device)
        
        with torch.no_grad():
            outputs = model(rgb)
        
        loss = criterion(outputs, depth)
        total_val_loss += loss.item()
        
        rgb = rgb.cpu().squeeze().permute(1, 2, 0)
        pred_depth = outputs.cpu().squeeze()
        gt_depth = depth.cpu().squeeze()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")
        
        axes[1].imshow(pred_depth, cmap="viridis")
        axes[1].set_title("Predicted Depth")
        axes[1].axis("off")
        
        axes[2].imshow(gt_depth, cmap="viridis")
        axes[2].set_title("Ground Truth Depth")
        axes[2].axis("off")
        
        plt.savefig(os.path.join(output_path, f"sample_{i}.png"))
        plt.close(fig)

    avg_val_loss = total_val_loss / len(val_dl)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")

    return avg_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output files and logs")
    args = parser.parse_args()
    
    model = FastDepth(input_channels=3).to(device=device)    
    
    val_dl = load_dataset(args.root_dir, "test", 1)
    
    evaluate(model, val_dl, args.output_path)
import torch
import torch.nn as nn
import argparse
from loss import FastDepthLoss
from dataset import load_dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model import ResNetFastDepth

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model: ResNetFastDepth, val_dl: DataLoader, output_path: str, device: torch.device ="cuda"):
    criterion = FastDepthLoss()
    
    os.makedirs(output_path, exist_ok=True)

    model.eval()
    total_val_loss = 0.0
    
    for i, data in enumerate(tqdm(val_dl, desc="[Evaluating validation dataset]")):
        rgb, depth = data
        rgb, depth = rgb.to(device), depth.to(device)
        
        with torch.no_grad():
            outputs = model(rgb)
            outputs = nn.functional.interpolate(outputs, size=depth.shape[2:], mode='bilinear', align_corners=False)

        loss = criterion(outputs, depth)
        total_val_loss += loss.item()
        
        rgb = rgb.cpu().squeeze().permute(1, 2, 0)
        
        pred_depth = outputs.cpu().squeeze().numpy()
        pred_depth = pred_depth / pred_depth.max() if pred_depth.max() > 0 else 1.0
        
        gt_depth = depth.cpu().squeeze().numpy()
        gt_depth = gt_depth / gt_depth.max() if gt_depth.max() > 0 else 1.0
        
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
    parser.add_argument("--root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save output files and logs")
    parser.add_argument("--path", type=str, required=True, help="Path for model .pth file")
    args = parser.parse_args()
    
    model = ResNetFastDepth().to(device=device)    
    model.load_state_dict(torch.load(args.path))
    
    val_dl = load_dataset(args.root, "test", 1)
    
    evaluate(model, val_dl, args.output)
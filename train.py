import torch
from tqdm import tqdm
from model import ResNetFastDepth
import torch.optim as opt
import argparse
from dataset import load_dataset, DataLoader
from loss import FastDepthLoss, kl_divergence
import os
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def map_histogram(array: torch.Tensor, output_dir: str):
    """
    Mapping histograms of any given torch array 
    
    Args: 
        array (torch.Tensor): 1D or 2D tensors that contain the array of values within distribution 
        output_dir (str): output path that directly references where to store the png
    """
    if len(array.shape) > 2:
        array = array.view(array.shape[0], -1)
        
    np_array = array.numpy()
    plt.hist(np_array)
    plt.savefig(output_dir)
    plt.close()
    
def map_heatmap(pred: torch.Tensor, gt: torch.Tensor, output_dir: str): 
    """
    Mapping heatmap of l1-loss
    
    Args:
        array (torch.Tensor): 2D tensors of
    """
    diff = (pred - gt) ** 2
    plt.imshow(diff, cmap="hot", interpolation="nearest")
    plt.savefig(output_dir)
    plt.close()
    
def diagnose_gradient_flow(model: ResNetFastDepth):
    print("Gradient Flow Diagnostics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            if grad_norm > 1.0:
                print(f"WARNING: Large gradient in {name}")
            elif grad_norm < 1e-6:
                print(f"WARNING: Vanishing gradient in {name}")
    
def depth_estimation(model: ResNetFastDepth, 
                     optimizer: opt.SGD, 
                     train_dl: DataLoader, 
                     valid_dl: DataLoader, 
                     loss_fn: FastDepthLoss, 
                     epochs: int, 
                     output_path: str):
    
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-5)

    train_curve = [] 
    valid_curve = []

    for epoch in range(epochs):
        model.train()
                
        total_train_loss = 0.0
        for rgb, depth in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            rgb, depth = rgb.to(device), depth.to(device)
            optimizer.zero_grad()
            
            outputs = model(rgb)
            
            loss = loss_fn(outputs, depth)
                        
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        diagnose_gradient_flow(model)

        avg_train_loss = total_train_loss / len(train_dl)

        model.eval()
        total_val_loss = 0.0
        abs_rel_sum = 0.0
        rmse_sum = 0.0
        total_kl_divergence = 0.0
        
        with torch.no_grad():
            for i, (rgb, depth) in tqdm(enumerate(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}")):
                rgb, depth = rgb.to(device), depth.to(device)
                outputs = model(rgb)
                
                loss = loss_fn(outputs, depth)
                abs_rel = torch.mean(torch.abs(outputs - depth) / depth).item()
                rmse = torch.sqrt(torch.mean((outputs - depth) ** 2)).item()
                kl_loss = kl_divergence(outputs.view(outputs.shape[0], -1), depth.view(depth.shape[0], -1))
                
                total_val_loss += loss.item()
                abs_rel_sum += abs_rel 
                rmse_sum += rmse
                total_kl_divergence += kl_loss
                
                map_histogram(outputs[0], os.path.join(output_path, f"random_sample_pred_{i}.png"))
                map_histogram(depth[0],  os.path.join(output_path, f"random_sample_gt_{i}.png"))
                map_heatmap(outputs[0], depth[0], os.path.join(output_path, f"random_sample_heatmap_{i}.png"))

        avg_val_loss = total_val_loss / len(valid_dl)
        avg_abs_rel = abs_rel_sum / len(valid_dl)
        avg_rmse = rmse_sum / len(valid_dl)
        avg_kl = total_kl_divergence / len(valid_dl)
        
        train_curve.append(avg_train_loss)
        valid_curve.append(avg_val_loss)
            
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(output_path, "depth_model.pth"))
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  AbsRel: {avg_abs_rel:.4f}")
        print(f"  RMSE: {avg_rmse:.4f}")
        print(f"  KL Loss: {avg_kl:.4f}")

    data_range = range(1, len(train_curve) + 1)
    plt.plot(train_curve, data_range, label="training loss curve")
    plt.plot(valid_curve, data_range, label="validation loss curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validaiton Loss on graph")
    plt.savefig(os.path.join(output_path, f"Curve_plot_{time.time()}.png"))
    plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save output files and logs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--validation", action='store_true', help="Flag to disable validation metrics")
    args = parser.parse_args()
    
    model = ResNetFastDepth().to(device=device)
    optimizer = opt.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    
    directory = os.path.join(args.output, "experiment_log")
    os.makedirs(directory, exist_ok=True)
    
    criterion = FastDepthLoss()
    
    train_dl = load_dataset(args.root, "train", args.batch)
    val_dl = load_dataset(args.root, "test", args.batch)
    
    depth_estimation(model, optimizer, train_dl, val_dl, criterion, args.epochs, args.output)
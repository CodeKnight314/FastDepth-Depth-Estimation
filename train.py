import torch
from utils.early_stop import EarlyStopMechanism
from utils.log_writer import LOGWRITER
from tqdm import tqdm
from model import ResNetDepth
import torch.optim as opt
import argparse
from dataset import load_dataset, DataLoader
from loss import FastDepthLoss
import os
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def depth_estimation(model: ResNetDepth, 
                     optimizer: opt.SGD, 
                     train_dl: DataLoader, 
                     valid_dl: DataLoader, 
                     loss_fn: FastDepthLoss, 
                     epochs: int, 
                     output_path: str):
    
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=1e-5)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for rgb, depth in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            rgb, depth = rgb.to(device), depth.to(device)
            optimizer.zero_grad()
            
            outputs = model(rgb)
            
            loss = loss_fn(outputs, depth)
            
            total_val_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)

        model.eval()
        total_val_loss = 0.0
        abs_rel_sum = 0.0
        rmse_sum = 0.0
        with torch.no_grad():
            for rgb, depth in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                rgb, depth = rgb.to(device), depth.to(device)
                outputs = model(rgb)
                
                loss = loss_fn(outputs, depth)
                abs_rel = torch.mean(torch.abs(outputs - depth) / depth).item()
                rmse = torch.sqrt(torch.mean((outputs - depth) ** 2)).item()
                
                total_val_loss += loss.item()
                abs_rel_sum += abs_rel 
                rmse_sum += rmse

        avg_val_loss = total_val_loss / len(valid_dl)
        avg_abs_rel = abs_rel_sum / len(valid_dl)
        avg_rmse = rmse_sum / len(valid_dl)
            
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(output_path, "depth_model.pth"))
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  AbsRel: {avg_abs_rel:.4f}")
        print(f"  RMSE: {avg_rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save output files and logs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--validation", action='store_true', help="Flag to disable validation metrics")
    args = parser.parse_args()
    
    model = ResNetDepth().to(device=device)
    optimizer = opt.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    
    directory = os.path.join(args.output, "experiment_log")
    os.makedirs(directory, exist_ok=True)
    
    criterion = FastDepthLoss()
    
    train_dl = load_dataset(args.root, "train", args.batch)
    val_dl = load_dataset(args.root, "test", args.batch)
    
    depth_estimation(model, optimizer, train_dl, val_dl, criterion, args.epochs, args.output)
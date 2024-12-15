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

def map_histogram(pred: torch.Tensor, gt: torch.Tensor, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pred_np = pred.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    
    ax.hist(pred_np, bins=50, alpha=0.5, label='Predicted Depth', color='blue')
    ax.hist(gt_np, bins=50, alpha=0.5, label='Ground Truth', color='red')
    
    ax.set_xlabel('Depth Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Depth Value Distribution')
    ax.legend()
    
    plt.savefig(output_path)
    plt.close()
    
def map_heatmap(pred: torch.Tensor, gt: torch.Tensor, output_dir: str): 
    """
    Mapping heatmap of l1-loss
    
    Args:
        array (torch.Tensor): 2D tensors of
    """
    diff = pred - gt
    plt.imshow(diff)
    plt.savefig(output_dir)
    plt.close()

def plot_curves(data: list, labels: list, path: str, title: str = None, x_axis: str = None, y_axis: str = None): 
    if len(data) != len(labels):
        raise ValueError("The number of data lists must match the number of labels.")

    for d, label in zip(data, labels):
        data_range = range(1, len(d) + 1)
        plt.plot(data_range, d, label=label)
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
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
    kl_curve = [] 
    absRel_curve = [] 
    rmse_curve = []

    for epoch in range(epochs):
        model.train()
                
        total_train_loss = 0.0
        for rgb, depth in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            rgb, depth = rgb.to(device), depth.to(device)
            optimizer.zero_grad()
            
            outputs = model(rgb)
            
            loss = loss_fn(outputs, depth)
                        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
        
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

        avg_val_loss = total_val_loss / len(valid_dl)
        avg_abs_rel = abs_rel_sum / len(valid_dl)
        avg_rmse = rmse_sum / len(valid_dl)
        avg_kl = total_kl_divergence / len(valid_dl)
        
        train_curve.append(avg_train_loss)
        valid_curve.append(avg_val_loss)
        kl_curve.append(avg_kl)
        absRel_curve.append(avg_abs_rel)
        rmse_curve.append(avg_rmse)
            
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(output_path, f"Epoch_{epoch+1}_depth_model.pth"))
        
        rgb, depth = next(iter(val_dl))
        with torch.no_grad(): 
            pred = model(rgb)
        
        for i in range(rgb.shape[0]): 
            map_histogram(rgb[i].squeeze(0), pred[i].squeeze(0), os.path.join(output_path, f"Epoch_{epoch+1}/sample_{i}_histogram.png"))
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  AbsRel: {avg_abs_rel:.4f}")
        print(f"  RMSE: {avg_rmse:.4f}")
        print(f"  KL Loss: {avg_kl:.4f}")

    plot_curves([train_curve, valid_curve], labels=["Training Loss", "Validation Loss"], path=os.path.join(output_path, f"Loss function curve.png"), title="Loss function curve", x_axis="Epochs", y_axis="Loss")
    plot_curves([kl_curve], ["KL Divergence Loss"], path=os.path.join(output_path, "KL_Divergence_Curve.png"), title="KL Divergence Curve", x_axis="Epoch", y_axis="Loss")
    plot_curves([absRel_curve], ["ABS REL Loss"], path=os.path.join(output_path, "ABS_REL_Curve.png"), title="ABS REL loss curve", x_axis="Epochs", y_axis="Loss")
    plot_curves([rmse_curve], ["RMSE Loss"], path=os.path.join(output_path, "RMSE_Curve.png"), title="ABS REL loss curve", x_axis="Epochs", y_axis="Loss")

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
    
    criterion = FastDepthLoss()
    
    train_dl = load_dataset(args.root, "train", args.batch)
    val_dl = load_dataset(args.root, "test", args.batch)
    
    depth_estimation(model, optimizer, train_dl, val_dl, criterion, args.epochs, args.output)
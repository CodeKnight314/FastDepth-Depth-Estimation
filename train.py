import torch
from utils.early_stop import EarlyStopMechanism
from utils.log_writer import LOGWRITER
from tqdm import tqdm
from model import FastDepth
import torch.optim as opt
import argparse
from dataset import load_dataset, DataLoader
from loss import FastDepthLoss
import os
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def depth_estimation(model: FastDepth, 
                     optimizer: opt.SGD, 
                     train_dl: DataLoader, 
                     valid_dl: DataLoader, 
                     logger: LOGWRITER, 
                     loss_fn: FastDepthLoss, 
                     epochs: int, output_path: str):
    
    es_mech = EarlyStopMechanism(metric_threshold=0.015, 
                                 mode='min', 
                                 grace_threshold=50, 
                                 save_path=os.path.join(output_path, "saved_weights/"))
    
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=1e-5)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for rgb, depth in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            rgb, depth = rgb.to(device), depth.to(device)
            optimizer.zero_grad()
            
            outputs = model(rgb)
            
            outputs = nn.functional.interpolate(outputs, size=depth.shape[2:], mode='bilinear', align_corners=False)

            loss = loss_fn(outputs, depth)
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                outputs = nn.functional.interpolate(outputs, size=depth.shape[2:], mode='bilinear', align_corners=False)

                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_dl)

        es_mech.step(model=model, metric=avg_val_loss)
        scheduler.step()
        
        if es_mech.check():
            logger.write("[INFO] Early Stopping Mechanism Engaged. Training procedure ended early.")
            break

        logger.log_results(epoch=epoch+1, tr_loss=avg_train_loss, val_loss=avg_val_loss)
    es_mech.save_model(model=model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save output files and logs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training and validation")
    args = parser.parse_args()
    
    model = FastDepth(input_channels=3).to(device=device)
    optimizer = opt.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    
    directory = os.path.join(args.output, "experiment_log")
    os.makedirs(directory, exist_ok=True)
    logger = LOGWRITER(output_directory=directory, total_epochs=args.epochs)
    
    criterion = FastDepthLoss()
    
    train_dl = load_dataset(args.root, "train", args.batch)
    val_dl = load_dataset(args.root, "test", args.batch)
    
    depth_estimation(model, optimizer, train_dl, val_dl, logger, criterion, args.epochs, args.output)

import torch
from utils.early_stop import EarlyStopMechanism
from utils.log_writer import LOGWRITER
from tqdm import tqdm
from model import FastDepth
import torch.optim as opt
import argparse
from dataset import load_dataset
from loss import ScaleInvariantLoss
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def depth_estimation(model, optimizer, train_dl, valid_dl, logger, loss_fn, epochs, output_path):
    # Early Stop mechanism
    es_mech = EarlyStopMechanism(metric_threshold=0.015, 
                                 mode='min', 
                                 grace_threshold=10, 
                                 save_path=os.path.join(output_path, "saved_weights/"))

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for images, labels in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
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
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_dl)

        es_mech.step(model=model, metric=avg_val_loss)
        if es_mech.check():
            logger.write("[INFO] Early Stopping Mechanism Engaged. Training procedure ended early.")
            break

        logger.log_results(epoch=epoch+1, tr_loss=avg_train_loss, val_loss=avg_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir")
    parser.add_argument("--output_path")
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--batch_size", default=8)
    args = parser.parse_args()
    
    model = FastDepth(input_channels=3).to(device=device)
    optimizer = opt.SGD(args.lr, momentum=0.9, weight_decay=0.0001)
    
    logger = LOGWRITER(output_directory=os.path.join(args.output_path, "experiment_log"), total_epochs=args.epochs)
    
    criterion = ScaleInvariantLoss(weight=0.5)
    
    train_dl = load_dataset(args.root_dir, "train", args.batch_size)
    val_dl = load_dataset(args.root_dir, "test", args.batch_size)
    
    depth_estimation(model, optimizer, train_dl, val_dl, logger, criterion, args.epochs, args.output_path)
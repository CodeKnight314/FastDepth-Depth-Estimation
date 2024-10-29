import torch
import torch.nn as nn 
import torch.nn.functional as F

class ScaleInvariantLoss(nn.Module):
    def __init__(self, weight: float = 0.5): 
        super().__init__()
        self.weight = weight 

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        eps = 1e-8

        y_pred = torch.clamp(pred, min=eps)
        y_true = torch.clamp(gt, min=eps)

        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        y_true_flat = y_true.view(y_true.size(0), -1)

        log_pred = torch.log(y_pred_flat)
        log_gt = torch.log(y_true_flat)
        diff = log_pred - log_gt

        mse_loss = torch.mean(diff ** 2)

        variance_reduction = (torch.mean(diff)) ** 2

        loss = mse_loss - self.weight * variance_reduction

        return loss

if __name__ == "__main__":
    criterion = ScaleInvariantLoss()

    # Use absolute values to ensure valid depth ranges
    x = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))
    y = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))

    loss = criterion(x, y)
    print(f"Total calculated loss for x and y is: {loss.item()}")
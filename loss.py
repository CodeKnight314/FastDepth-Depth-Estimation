import torch
import torch.nn as nn

class FastDepthLoss(nn.Module):
    def __init__(self, weight_scale_invariant: float = 0.2, weight_l1: float = 0.5): 
        super(FastDepthLoss, self).__init__()
        self.weight_scale_invariant = weight_scale_invariant
        self.weight_l1 = weight_l1

    def scale_invariant_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        eps = 1e-4
        y_pred = torch.clamp(pred, min=eps)
        y_true = torch.clamp(gt, min=eps)
        
        diff = torch.log(y_pred) - torch.log(y_true)
        mse_loss = torch.mean(diff ** 2)
        variance_reduction = torch.mean(diff, dim=(1, 2, 3)) ** 2
        variance_reduction = torch.mean(variance_reduction)

        return mse_loss - self.weight_scale_invariant * variance_reduction

    def l1_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        return torch.mean(torch.abs(pred - gt))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        scale_invariant = self.scale_invariant_loss(pred, gt)
        l1 = self.l1_loss(pred, gt)
        loss = self.weight_scale_invariant * scale_invariant + self.weight_l1 * l1
        return loss

if __name__ == "__main__":
    criterion = FastDepthLoss()

    x = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))
    y = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))

    loss = criterion(x, y)
    print(f"Total calculated loss for x and y is: {loss.item()}")
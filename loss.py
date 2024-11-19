import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FastDepthLoss(nn.Module):
    def __init__(self,
                 weight_scale_invariant: float = 1.0,
                 weight_l1: float = 1.0):

        super(FastDepthLoss, self).__init__()

        self.weight_scale_invariant = weight_scale_invariant
        self.weight_l1 = weight_l1

    def scale_invariant_loss(self, pred: torch.Tensor, target: torch.Tensor):
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        target = torch.clamp(target, min=eps)

        log_pred = torch.log(pred)
        log_target = torch.log(target)

        diff = log_pred - log_target
        mse_term = torch.mean(diff ** 2)
        bias_term = (torch.mean(diff) ** 2)

        loss = mse_term - 0.5 * bias_term
        return loss

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        si_loss = self.scale_invariant_loss(pred, gt)
        l1_loss = F.l1_loss(pred, gt)

        total_loss = si_loss + l1_loss

        return total_loss

if __name__ == "__main__":
    criterion = FastDepthLoss()

    x = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))
    y = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))

    loss = criterion(x, y)
    print(f"Total calculated loss for x and y is: {loss.item()}")
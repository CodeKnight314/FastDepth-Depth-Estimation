import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FastDepthLoss(nn.Module):
    def __init__(self,
                 weight_scale_invariant: float = 0.2,
                 weight_l1: float = 0.5,
                 weight_gradient=0.2,
                 weight_ssim = 0.1,
                 weight_edge=0.1):

        super(FastDepthLoss, self).__init__()

        self.weight_scale_invariant = weight_scale_invariant
        self.weight_l1 = weight_l1
        self.weight_gradient = weight_gradient
        self.weight_edge = weight_edge
        self.weight_ssim = weight_ssim

        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

    def scale_invariant_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        eps = 1e-6
        y_pred = torch.clamp(pred, min=eps)
        y_true = torch.clamp(gt, min=eps)

        diff = torch.log(y_pred) - torch.log(y_true)
        num_pixels = diff.shape[1] * diff.shape[2] * diff.shape[3]

        first_term = torch.mean(diff ** 2)

        second_term = (torch.sum(diff, dim=(1, 2, 3)) ** 2) / (num_pixels ** 2)
        second_term = torch.mean(second_term)

        return first_term - self.weight_scale_invariant * second_term

    def gradient_map(self, x):
        _, _, h_x, w_x = x.size()

        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]

        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad

    def gradient_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        eps = 1e-4
        y_pred = torch.clamp(pred, min=eps)
        y_true = torch.clamp(gt, min=eps)

        y_pred_map = self.gradient_map(y_pred)
        y_true_map = self.gradient_map(y_true)

        return F.l1_loss(y_pred_map, y_true_map)

    def ssim_loss(self, pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        sigma = 1.5
        gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2))
                     for x in range(window_size)], dtype=torch.float32)
        gauss = gauss/gauss.sum()

        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(gt, window, padding=window_size//2, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(gt * gt, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(pred * gt, window, padding=window_size//2, groups=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - torch.mean(ssim_map)

    def edge_aware_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)

        pred_edges = torch.sqrt(F.conv2d(pred, sobel_x, padding=1)**2 +
                              F.conv2d(pred, sobel_y, padding=1)**2)
        gt_edges = torch.sqrt(F.conv2d(gt, sobel_x, padding=1)**2 +
                            F.conv2d(gt, sobel_y, padding=1)**2)

        pred_edges = (pred_edges > pred_edges.mean()).float()
        gt_edges = (gt_edges > gt_edges.mean()).float()

        return F.binary_cross_entropy(pred_edges, gt_edges)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        si_loss = self.scale_invariant_loss(pred, gt)
        l1_loss = F.l1_loss(pred, gt)
        grad_loss = self.gradient_loss(pred, gt)
        ssim = self.ssim_loss(pred, gt)
        edge_loss = self.edge_aware_loss(pred, gt)

        total_loss = (self.weight_scale_invariant * si_loss +
                     self.weight_l1 * l1_loss +
                     self.weight_gradient * grad_loss +
                     self.weight_ssim * ssim +
                     self.weight_edge * edge_loss)

        return total_loss

if __name__ == "__main__":
    criterion = FastDepthLoss()

    x = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))
    y = torch.abs(torch.randn((16, 1, 256, 256), dtype=torch.float32))

    loss = criterion(x, y)
    print(f"Total calculated loss for x and y is: {loss.item()}")
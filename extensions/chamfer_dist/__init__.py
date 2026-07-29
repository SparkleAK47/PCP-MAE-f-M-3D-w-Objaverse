"""
Pure PyTorch implementations of Chamfer distance.

Uses only built-in PyTorch ops (torch.cdist + torch.min) so autograd
handles backward automatically — no hand-written backward pass needed.

This avoids the CUDA OOM (138 GiB allocate error) caused by dimension
mismatches in the previous custom ChamferFunction.backward.
"""
import torch
import torch.nn as nn


class ChamferDistanceL1(nn.Module):
    """
    Chamfer L1 (mean Euclidean distance).

    loss = mean(min_j ||x_i - y_j||_2) + mean(min_i ||y_j - x_i||_2)

    torch.cdist(xyz1, xyz2, p=2) returns the Euclidean (L2) distances,
    i.e. sqrt(sum((x-y)^2)). This is what we use for "L1" Chamfer.
    """
    def __init__(self, ignore_border=False):
        super().__init__()
        self.ignore_border = ignore_border

    def forward(self, xyz1, xyz2):
        # cdist(p=2) -> Euclidean distance  sqrt(sum((x-y)^2))
        dist = torch.cdist(xyz1.float(), xyz2.float(), p=2)  # (B, N, M)
        dist1 = dist.min(dim=2).values  # (B, N)
        dist2 = dist.min(dim=1).values  # (B, M)
        if self.ignore_border:
            dist1 = dist1[:, 1:-1]
            dist2 = dist2[:, 1:-1]
        return dist1.mean() + dist2.mean()


class ChamferDistanceL2(nn.Module):
    """
    Chamfer distance using Euclidean (L2) distance.

    NOTE: Despite the name "L2", the original implementation computes
    the actual Euclidean distance (sqrt of sum of squares), NOT squared
    Euclidean distance. This matches the original hand-written backward
    where gradient = grad_dist * (x - y) / (dist + eps), i.e. the
    derivative of sqrt(L2). We preserve this exact behavior by using
    torch.cdist(xyz1, xyz2, p=2) which returns sqrt'd L2, without
    further squaring.

    loss = mean(min_j ||x_i - y_j||_2) + mean(min_i ||y_j - x_i||_2)
    """
    def __init__(self, ignore_border=False):
        super().__init__()
        self.ignore_border = ignore_border

    def forward(self, xyz1, xyz2):
        dist = torch.cdist(xyz1.float(), xyz2.float(), p=2)  # (B, N, M) — sqrt'd L2
        dist1 = dist.min(dim=2).values  # (B, N)
        dist2 = dist.min(dim=1).values  # (B, M)
        if self.ignore_border:
            dist1 = dist1[:, 1:-1]
            dist2 = dist2[:, 1:-1]
        return dist1.mean() + dist2.mean()

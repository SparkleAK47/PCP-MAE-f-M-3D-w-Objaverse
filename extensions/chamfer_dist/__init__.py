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
    Chamfer distance using squared Euclidean distance.

    NOTE: The original CUDA kernel (chamfer.cu) computes
    dist = x² + y² + z²  (no sqrt), i.e. squared L2 distance.
    We match this behavior exactly:
      torch.cdist(xyz1, xyz2, p=2) returns sqrt'd L2, so we square it:
      dist_sq = (sqrt(∑(x-y)²))² = ∑(x-y)²  (squared L2)

    loss = mean(min_j ||x_i - y_j||²) + mean(min_i ||y_j - x_i||²)

    The PyTorch autograd backward for cdist(p=2)² produces:
      d(dist²)/d(x) = 2 * (x - y)
    which matches the original CUDA backward.
    """
    def __init__(self, ignore_border=False):
        super().__init__()
        self.ignore_border = ignore_border

    def forward(self, xyz1, xyz2):
        # cdist(p=2) -> sqrt'd L2, then square to recover squared L2
        dist = torch.cdist(xyz1.float(), xyz2.float(), p=2)  # (B, N, M)
        dist_sq = dist ** 2  # (B, N, M) — squared L2, matches original CUDA kernel
        dist1 = dist_sq.min(dim=2).values  # (B, N)
        dist2 = dist_sq.min(dim=1).values  # (B, M)
        if self.ignore_border:
            dist1 = dist1[:, 1:-1]
            dist2 = dist2[:, 1:-1]
        return dist1.mean() + dist2.mean()

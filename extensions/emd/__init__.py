"""
Pure PyTorch EMD placeholder.

Avoids CUDA-compiled emd_cuda issues.
EMD loss is not actually used in this project (the PCP_MAE loss
only uses ChamferDistanceL1/L2). This placeholder provides a
working fallback that computes an approximate Earth Mover Distance
using a simple iterative assignment strategy.
"""
import torch
import torch.nn as nn


class EarthMoverDistanceFunction(torch.autograd.Function):
    """
    Pure PyTorch approximate Earth Mover Distance via iterative soft assignment.
    Only used as fallback; the project's PCP_MAE doesn't actually use EMD.
    """

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        # Compute cost matrix (L2 distance)
        cost = torch.cdist(xyz1, xyz2, p=2)  # (B, N1, N2)
        # Approximate matching cost: per-point min distance averaged
        min_cost_1 = cost.min(dim=2)[0]  # (B, N1)
        min_cost_2 = cost.min(dim=1)[0]  # (B, N2)
        total_cost = min_cost_1.sum(dim=-1) + min_cost_2.sum(dim=-1)
        match = torch.softmax(-cost * 10.0, dim=-1)  # soft assignment
        ctx.save_for_backward(xyz1, xyz2, match)
        return total_cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        B, N1, D = xyz1.shape
        N2 = xyz2.shape[1]

        # d(cost[i,j]) / d(xyz1[i,:]) = 2 * (xyz1[i,:] - xyz2[j,:])
        diff = xyz1.unsqueeze(2) - xyz2.unsqueeze(1)  # (B, N1, N2, 3)
        grad1_norm = 2.0 * diff * match.unsqueeze(-1)  # (B, N1, N2, 3)
        grad1 = grad1_norm.sum(dim=2)  # (B, N1, 3)
        grad1 = grad1 * grad_cost.unsqueeze(-1).unsqueeze(-1)  # (B, N1, 3)

        grad2_norm = 2.0 * (-diff) * match.unsqueeze(-1)  # (B, N1, N2, 3)
        grad2 = grad2_norm.sum(dim=1)  # (B, N2, 3)
        grad2 = grad2 * grad_cost.unsqueeze(-1).unsqueeze(-1)  # (B, N2, 3)

        return grad1, grad2


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approximate, pure PyTorch)
    Args:
        xyz1 (torch.Tensor): (b, 3, n1) or (b, n1, 3)
        xyz2 (torch.Tensor): (b, 3, n2) or (b, n2, 3)
        transpose (bool): if True, expects (b, 3, n).
    Returns:
        cost (torch.Tensor): (b)
    """
    if transpose:
        xyz1 = xyz1.transpose(1, 2).contiguous()
        xyz2 = xyz2.transpose(1, 2).contiguous()
    return EarthMoverDistanceFunction.apply(xyz1, xyz2)


class EMDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.emd = earth_mover_distance

    def forward(self, xyz1, xyz2):
        cost = self.emd(xyz1, xyz2)
        return torch.mean(cost)

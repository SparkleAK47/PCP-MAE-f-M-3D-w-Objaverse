"""
Pure PyTorch implementation of pointnet2_utils.
Replaces the C++/CUDA pointnet2_ops package to avoid RTX 5090 (CUDA 12.8)
compilation compatibility issues.

Exports the same API as the original pointnet2_ops.pointnet2_utils:
    - furthest_point_sample(xyz, npoint)  -> (B, npoint)  LongTensor indices
    - gather_operation(features, idx)     -> gathered features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint] (LongTensor)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


# Alias for compatibility
farthest_point_sample = furthest_point_sample


def gather_operation(features, idx):
    """
    Input:
        features: input features, [B, C, N]
        idx: sample index data, [B, npoint]
    Return:
        gathered features, [B, C, npoint]
    """
    B, C, N = features.shape
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, npoint]
    gathered = torch.gather(features, 2, idx_expanded)
    return gathered


# Keep API compatibility: have same functions referenced elsewhere
three_nn = None
three_interpolate = None
ball_query = None
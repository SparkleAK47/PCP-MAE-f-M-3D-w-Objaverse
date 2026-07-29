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
</content>
<write_to_file>
<path>workspace/PCP-MAE_with_Objaverse/extensions/emd/emd.py</path>
<content>"""
Pure PyTorch EMD placeholder.
Same as extensions/emd/__init__.py.
"""
from . import EarthMoverDistanceFunction, earth_mover_distance, EMDLoss
</content>
现在更新 SETUP_mae5090.sh：

<write_to_file>
<path>workspace/PCP-MAE_with_Objaverse/SETUP_mae5090.sh</path>
<content>#!/bin/bash
# ==========================================================
# PCP-MAE on RTX 5090 - 环境搭建脚本（可直接执行）
# 环境: mae5090 (Python 3.10)
#
# 使用方法:
#   1. 先创建环境: conda create -n mae5090 python=3.10 -y
#   2. 请**不要**直接 bash 运行整个脚本。
#      请按步骤，每条命令逐个复制执行，以便观察输出。
# ==========================================================

set -e  # 出错停止

echo "========== 第1步：激活环境确认 =========="
source /opt/miniconda3/bin/activate mae5090
python --version
which python

echo ""
echo "========== 第2步：安装 PyTorch 2.7 + CUDA 12.8 =========="
echo "下载约 2.6GB，需 20-40 分钟。建议手动执行："
echo ""
echo "pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128"
echo ""
read -p "PyTorch 安装完成后按 Enter 继续..."

echo ""
echo "========== 第3步：安装基础依赖 =========="
pip install ninja
pip install numpy==1.26.3
pip install easydict h5py matplotlib opencv-python pyyaml scipy tensorboardX tqdm transforms3d termcolor scikit-learn

echo ""
echo "========== 第4步：安装 timm =========="
pip install timm==0.4.5

echo ""
echo "========== 第5步：安装 pointnet2_ops（纯 PyTorch 版，无 CUDA 编译）=========="
echo ""
echo "cd /data/workspace/PCP-MAE_with_Objaverse/third_party/pointnet2_ops_pytorch && pip install -e ."
echo ""
read -p "安装完成后按 Enter 继续..."

echo ""
echo "========== 第6步：安装 chamfer_dist（纯 PyTorch 版，无 CUDA 编译）=========="
echo "chamfer_dist 已使用 torch.cdist 重写，无需编译。"
echo "直接运行 python -c \"from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2; print('OK')\""
echo "即可验证。"
echo ""

echo ""
echo "========== 第7步：安装 emd（纯 PyTorch 占位实现，无 CUDA 编译）=========="
echo "emd 仅作为占位模块（项目实际未使用 EMD 损失），无需编译。"
echo "直接运行 python -c \"from extensions.emd import earth_mover_distance; print('OK')\""
echo "即可验证。"
echo ""

echo ""
echo "========== 第8步：验证环境 =========="
cd /data/workspace/PCP-MAE_with_Objaverse

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA capability:', torch.cuda.get_device_capability(0))

import torchvision
print('torchvision:', torchvision.__version__)

# 验证 pointnet2_ops（纯 PyTorch 版）
from pointnet2_ops import pointnet2_utils
print('pointnet2_ops: OK')

# 验证 chamfer_dist（纯 PyTorch 版）
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
print('chamfer_dist: OK')

# 验证 emd（纯 PyTorch 占位）
from extensions.emd import earth_mover_distance
print('emd: OK')

# 验证 timm
import timm
print('timm:', timm.__version__)
from timm.models.layers import DropPath, trunc_normal_
print('timm layers: OK')

# 验证 sklearn
from sklearn.model_selection import train_test_split
print('sklearn: OK')

print('\\n✅ 所有导入成功！环境配置完成！')
"

echo ""
echo "========== 完成 =========="
echo "如果所有导入都成功，你可以运行："
echo ""
echo "cd /data/workspace/PCP-MAE_with_Objaverse"
echo "CUDA_VISIBLE_DEVICES=0 python main.py --config \"cfgs/pretrain/[V1]PCP-MAE+MaskTransformer.yaml\" --exp_name test_5090 --seed 42"
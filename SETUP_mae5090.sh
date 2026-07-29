#!/bin/bash
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
echo "========== 第2步：安装 PyTorch 2.13 + CUDA 12.8（RTX 5090 Blackwell 支持）=========="
echo "RTX 5090 (Blackwell sm_120) 需要 PyTorch 2.13+。"
echo "下载约 2.6GB，需 20-40 分钟。建议手动执行："
echo ""
echo "pip install torch==2.13.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128"
echo ""
echo "如果仍报 'no kernel image' 错误，改用 cu124 试试："
echo "pip install torch==2.13.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu124"
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
echo "========== 第6步：验证 chamfer_dist（纯 PyTorch 版，无 CUDA 编译）=========="
echo "chamfer_dist 已使用 torch.cdist 重写，无需编译。"
echo "运行验证："
echo ""
echo "python -c \"from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2; print('OK')\""
echo ""
read -p "验证完成后按 Enter 继续..."

echo ""
echo "========== 第7步：验证 emd（纯 PyTorch 占位实现，无 CUDA 编译）=========="
echo "emd 仅作为占位模块（项目实际未使用 EMD 损失），无需编译。"
echo "运行验证："
echo ""
echo "python -c \"from extensions.emd import earth_mover_distance; print('OK')\""
echo ""
read -p "验证完成后按 Enter 继续..."

echo ""
echo "========== 第8步：完整环境验证 =========="
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
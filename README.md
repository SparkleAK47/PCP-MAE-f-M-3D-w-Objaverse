# PCP-MAE（MiniGPT-3D 适配版 · Objaverse 分支）

本仓库在 [PCP-MAE 官方实现](https://github.com/aHapBean/PCP-MAE) 基础上做了大量修改，用于在 **Objaverse 660K 点云** 上预训练点云编码器，并接入 [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D) 作为唯一可行的下游评测框架。

> **主文档**：全部实验流程（含 ShapeNet55-34 对照）、编码器变体总览、MiniGPT-3D 四阶段训练与评测的完整文档请参阅 **[MiniGPT-3D 主 README](../MiniGPT-3D/README.md)**。

原始 PCP-MAE 论文说明、ShapeNet 分类/分割评测流程见 [`origin_readme.md`](origin_readme.md)。

---

## 1. 环境安装

### 1.1 当前环境（RTX 5090 / `mae5090`）

训练显卡已迁移至 **RTX 5090（Blackwell，sm_120，24GB）**，conda 环境为 **`mae5090`**（Python 3.10）。实测已安装版本：

| 项目 | 版本 |
|------|------|
| PyTorch | `2.13.0+cu130`（RTX 5090 Blackwell 需要 2.13+） |
| torchvision | `0.28.0+cu130` |
| timm | `0.4.5` |
| numpy | `1.26.3` |
| pointnet2_ops | 纯 PyTorch 版（从 `third_party/pointnet2_ops_pytorch` 本地安装，**无需编译 CUDA 扩展**） |
| chamfer_dist | 纯 PyTorch 实现（`torch.cdist` + autograd，**无需编译**） |
| emd | 纯 PyTorch 占位实现（本项目 loss 为 `cdl2`，实际不使用 EMD，**无需编译**） |

快速验证：

```bash
conda activate mae5090
cd /data/workspace/PCP-MAE_with_Objaverse

# 验证环境可用（无需编译）
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.emd import earth_mover_distance
import timm
print('extensions + timm OK')
PY

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/[V1]PCP‑MAE+MaskTransformer.yaml" \
  --exp_name test_5090 --seed 42
```

分步安装脚本见 [`SETUP_mae5090.sh`](SETUP_mae5090.sh)（请逐条复制命令执行以观察输出，**不要**直接 `bash` 整个脚本）。

说明：
- 开启 bfloat16 AMP，单卡 24GB 可支撑有效 batch=64。
- config 中 `val/test.bs` 已从 256 下调为 **128**，防止 24G 显存 OOM。
- 训练日志每 epoch 记录 `val_total_loss`，刷新最优时保存 `ckpt-best.pth`（详见 §5）。

### 1.2 历史环境（RTX 3090 / `pcpmae`，保留供参考）

早期实验在 `pcpmae` 环境（Python 3.10，PyTorch 2.0.1+cu118，单卡 RTX 3090）上完成。如需复现历史环境：

```bash
conda create -n pcpmae python=3.10 -y
conda activate pcpmae

# Install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install required packages
pip install -r requirements.txt

# Install the extensions（仅历史环境需要编译）
cd ./extensions/chamfer_dist && python setup.py install --user
cd ../emd && python setup.py install --user
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

`mae5090` 环境下以上扩展均**不需要编译**（见 §1.1）。
---

## 2. 相对官方仓库的主要改动

| 项目 | 说明 |
|------|------|
| 模型配置 | 与 MiniGPT-3D 对齐：8192 点、6 维输入（xyz+rgb）、group_size=32、num_group=512、trans_dim=384、depth=12、num_heads=6 |
| 数据集 | 新增 `ObjaverseNPY` 数据加载器，读取 MiniGPT-3D 的 `{id}_8192.npy` |
| 编码器实现 | `models/pointbert_mg/` 从 `MiniGPT-3D/minigpt4/models/pointbert` 复制，保证 Group/Encoder/PointTransformer 与下游一致 |
| 编码器变体 | 通过 `encoder_type` 切换：`mask_transformer`（V1）或 `point_transformer`（V2） |
| 训练 | bfloat16 AMP，有效 batch=64（`total_bs=64, step_per_update=8`）；单卡 RTX 3090（`pcpmae`）或 RTX 5090（`mae5090`）均可训练 |
| 训练策略 | **已移除早停**：固定 `max_epoch` + 余弦退火；`tools/runner_pretrain.py` 仅保留验证 loss 监控与 `ckpt-best` 保存（见 §5.3） |
| 调度策略 | CosineAnnealingLR（timm `CosineLRScheduler`）+ 3 epoch warmup，`T_max = scheduler.epochs = max_epoch`，`lr_min=1e-6`；**Objaverse 建议 ≥100 epoch，ShapeNet55-34 建议 ≥300 epoch** |
| 学习率 | 基础学习率 `0.0005`（AdamW，weight_decay 0.05） |
| 评测 | 官方 ShapeNet/ScanObjectNN 微调流程**不再作为本项目的最终评测**；编码器质量统一通过 MiniGPT-3D 四阶段训练 + GPT/Qwen 主观评测衡量 |
| ShapeNet55-34 编码器 | 独立的 ShapeNet55-34（PCP-MAE + MaskTransformer）训练代码位于 **[PCP-MAE_with_ShapeNet](../PCP-MAE_with_ShapeNet)** 仓库 |

---

## 3. 编码器变体一览

本仓库预训练了 5 种编码器，命名与导出权重对应关系如下（权重位于 `MiniGPT-3D/params_weight/pc_encoder/`）：

| 代号 | 权重文件名（MiniGPT-3D 侧） | 预训练方式 | 配置文件 | 说明 |
|------|------------------------------|------------|----------|------|
| **V1 random** | `point_model_V1+random-cls.pth` | PCP-MAE + `MaskTransformer`（cross-attn） | `cfgs/pretrain/[V1]PCP‑MAE+MaskTransformer.yaml` | MaskTransformer 无原生 cls，导出时自动随机初始化（trunc_normal，std=0.02） |
| **V2** | `point_model_v2.pth` | PCP-MAE + `PointTransformerMAEEncoder` | `cfgs/pretrain/[V2]PCP‑MAE+PointTransformer.yaml` | 与 MiniGPT-3D 推理使用一致的 PointTransformer，含原生 cls |
| **Point-MAE** | `old/point_model_pointmae.pth`（已归档） | 纯 Point-MAE（`ita=0`），PointTransformer 编码器 | `cfgs/pretrain/Point‑MAE+PointTransformer.yaml` | 关闭中心预测，其余与 V2 相同 |
| **Mask Point-MAE** | `old/point_model_maskmae.pth`（已归档） | 纯 Point-MAE（`ita=0`），MaskTransformer 编码器 | `cfgs/pretrain/Point‑MAE+MaskTransformer.yaml` | 交叉注意力架构 + 无中心预测；导出时自动随机生成 cls |
| **ShapeNet55-34** | `old/pcpmae_ShapeNet_fixed.pth`（已归档；修复前原始权重 `old/pcpmae_ShapeNet.pth`） | [PCP-MAE_with_ShapeNet](../PCP-MAE_with_ShapeNet) 训练的 PCP-MAE（MaskTransformer） | 该仓库的 `[ShapeNet55-34]PCP‑MAE+MaskTransformer.yaml` | 3 维输入，导出后需 `fix_pcpmae_shapenet.py` 做 3ch→6ch 修复 |

> **上述 5 个编码器均已通过 MiniGPT-3D 的 freeze 与 unfreeze 两种训练**（见 §7.2）。
>
> 已废弃/淘汰：V1（旧权重 `old/point_model_pcpmae.pth`，无 cls）与 V1 hybrid（旧权重 `old/point_model_hybrid.pth`，从 Baseline 复制 cls）已被 **V1 random** 取代；V2 旧导出名 `point_model_pcp_v2.pth` 已改为 `point_model_v2.pth`；Mask Point-MAE 历史导出名 `old/mask-pcpmae.pth` 已归档。
>
> Baseline（ULIP-2）与 Random（随机初始化）不在本仓库训练，对应 `params_weight/pc_encoder/` 下的 `point_model.pth` 与 `point_model_random.pth`。

### 架构差异说明

- **V1 random / Mask Point-MAE（MaskTransformer）**：预训练时 patch tokens 经 cross‑attention（visible + mask 双分支），无 cls token；点云全局表征由 512 个 mask token 聚合得到。导出权重时由 `ckpt_extract.py` 自动随机初始化 `cls_token`/`cls_pos`（trunc_normal，std=0.02）。
- **V2 / Point-MAE（PointTransformerMAEEncoder）**：预训练直接采用与 MiniGPT-3D 完全相同的 PointTransformer 结构（self‑attention + cls token），点云全局表征即 `cls_token`，权重可无缝导出，无需额外合并。

### cls_token 初始化方式对比

| 编码器 | cls_token 来源 | 说明 |
|--------|---------------|------|
| V1 random | 随机初始化（trunc_normal，std=0.02） | `ckpt_extract.py` 导出时自动生成，无需人工干预 |
| Mask Point-MAE | 随机初始化（trunc_normal，std=0.02） | `ckpt_extract.py` 导出时自动生成 |
| V2 | 预训练原生权重 | 预训练 checkpoint 自带 |
| Point-MAE | 预训练原生权重 | 预训练 checkpoint 自带 |

> “随机初始化”两种变体的 `cls_token` 未经预训练，但在 MiniGPT-3D Stage 1 微调时会与其他参数一起训练，差距有限。
>
> V1 hybrid 方式已弃用：V1 预训练权重导出时通过 `ckpt_extract.py` 自动随机补 `cls_token`/`cls_pos`（与 Mask Point-MAE 一致的流程），这样所有 MaskTransformer 编码器在初始化方式上保持一致，消除了混淆变量。
---

## 4. 数据准备

### 4.1 Objaverse 660K 点云

数据目录与 MiniGPT-3D 共用：`/data/workspace/MiniGPT-3D/data/objaverse_data/`，每个物体一个 `{objaverse_id}_8192.npy`（8192 点，xyz+rgb 共 6 维），train/test 划分由 `train.txt` / `test.txt`（每行一个点云 id，不含扩展名）指定。

```
/data/workspace/MiniGPT-3D/data/objaverse_data/
├── train.txt                 # 训练集 id 列表
├── test.txt                  # 测试集 id 列表
└── {objaverse_id}_8192.npy   # 每物体一个，8192×6
```

`cfgs/dataset_configs/ObjaverseNPY.yaml` 的 `data_root` 指向上列目录（`npoints=8192`）；预训练 config 的 train/val/test 均复用该加载器。

### 4.2 ShapeNet55-34 点云（§6.5 对照编码器用）

数据位于 ShapeNet 仓库内：`PCP-MAE_with_ShapeNet/data/ShapeNet55-34/ShapeNet55/`（官方 ShapeNet 加载器格式：`shapenet_pc/` 点云文件 + `train.txt`/`test.txt`），路径配置见该仓库 `cfgs/dataset_configs/ShapeNet-55.yaml`。注意其输入为 **3 维 xyz**（无 RGB）。

> ⚠️ ShapeNet55-34 是 3 维输入，预训练出的编码器需要 **3ch→6ch 维度修复** 才能接入 MiniGPT-3D（6 通道输入为 xyz+rgb，RGB 部分补零，等价于黑色输入）。详见 §6.5。

---

## 5. 实验目录结构与日志

### 5.1 目录命名规则

`experiments/{config文件名stem}/{config父目录}/{exp_name}`，代码位置 `utils/parser.py`（约第 99 行）。以 `[V1]PCP‑MAE+MaskTransformer.yaml` + `--exp_name pcpmae_minigpt3d` 为例：

```
experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/
```

| config 文件名（stem） | 实验目录 |
|------|------|
| `[V1]PCP‑MAE+MaskTransformer` | `experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/` |
| `[V2]PCP‑MAE+PointTransformer` | `experiments/[V2]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/` |
| `Point‑MAE+PointTransformer` | `experiments/Point‑MAE+PointTransformer/pretrain/point_mae_objaverse/` |
| `Point‑MAE+MaskTransformer` | `experiments/Point‑MAE+MaskTransformer/pretrain/mask_mae_objaverse/` |

> 注意：`experiments/` 下的一级目录名由 **config 文件名**（stem）决定，重命名 config 后新实验的输出路径随之改变（旧实验目录不会自动改名）。
>
> 本地 `experiments/` 被 `.gitignore` 忽略；所有编码器的预训练权重与日志统一归档在 `/data/datasets/PCP_checkpoint/`，归档目录对照见 §5.4。

### 5.2 检查点与日志

```
experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/
├── YYYYMMDD_HHMMSS.log    # 训练日志（每次启动生成新文件）
├── config.yaml            # 本次运行的配置快照
├── ckpt-last.pth          # 最新 checkpoint（同时保存当前 best_val_loss）
├── ckpt-best.pth          # 验证集 total_loss 最低的 checkpoint
└── ckpt-epoch-XXX.pth     # 自 epoch 0 起每 10 个 epoch 保存一次
```

**日志格式**（早停逻辑已移除，见 §5.3）：

```
[Training] EPOCH: 0 loss = 0.011000 val_total_loss = 0.003787
[Validation] EPOCH: 0 val_total_loss = 0.003787
[Validation] New best val_loss=0.003787 @ epoch 0. Saved ckpt-best.
[Training] EPOCH: 10 loss = 0.002945 val_total_loss = 0.002113
[Validation] EPOCH: 10 val_total_loss = 0.002113
[Validation] New best val_loss=0.002113 @ epoch 10. Saved ckpt-best.
```

**关于 loss 数值**

- `loss1`（重建 Chamfer L2）：前期快速下降（0.02 → 0.002），是总 loss 的主要构成。
- `loss2`（中心预测 Chamfer L2）：下降较慢，稳定在 ~0.0009–0.0011。
- `val_total_loss` = 验证集上的 `loss1 + loss2`。**`val_total_loss` 刷新最优时保存 `ckpt-best.pth`**；当训练计划足够长（余弦退火跑完）时，`ckpt-best` 通常落在接近最后一个 epoch 的位置。

**TensorBoard**

```
experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/
└── TFBoard/
    ├── train/            # train_loss, val_loss
    └── test/             # test_loss
```

```bash
tensorboard --logdir experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/TFBoard
```

### 5.3 训练策略：不建议使用早停

**观察到的现象**：旧策略（早停 patience=5 + `max_epoch=75`）下，V1/V2 均在 epoch 21 前后因验证 loss 连续 5 个 epoch 无改进而提前终止。改为固定计划 + 余弦退火后，在 5090 上重训 V1 的 30 epoch 中，`val_total_loss` 一直下降到最后一个 epoch，**日志中最后一次 `New best val_loss` 出现在 epoch 30（best == last，共刷新 25 次）**——说明训练结束时模型仍有优化空间，此前的“收敛”实际上是早停把学习率计划截断造成的假象。

**为什么不适合早停**：

1. 余弦退火需要跑完 `T_max` 整个周期，学习率才能降到 `lr_min`。早停把退火过程截断在 LR 仍很高的阶段，模型没有机会在低 LR 下进一步收敛。
2. 训练前期 `val_total_loss` 受高学习率主导，连续几个 epoch 不改进不代表真正收敛。
3. 早停使不同计划之间的对比不可控（终止 epoch 由运气决定）。

**建议（当前规范）**：

- **不使用早停**，固定 `max_epoch`，并令 `scheduler.kwargs.epochs == max_epoch`（余弦曲线正好在最后一个 epoch 结束）。
- **Objaverse** 数据集：`max_epoch` **≥ 100**。
- **ShapeNet55-34** 数据集：`max_epoch` **≥ 300**（ShapeNet 仓库现有 `cfgs/pretrain/[ShapeNet55-34]PCP‑MAE+MaskTransformer.yaml` 即为 300）。
- 其余训练参数（batch、lr、warmup、mask ratio 等）参考 `cfgs/pretrain/[V1]PCP‑MAE+MaskTransformer.yaml`。

**代码现状**：`tools/runner_pretrain.py` 已移除早停逻辑（commit `2b63814`），每 epoch 只做验证 loss 记录 + 刷新最优时保存 `ckpt-best.pth`；`ckpt-epoch-XXX.pth` 自 epoch 0 起每 10 个 epoch 保存一次；`ckpt-last.pth` 每 epoch 保存。

### 5.4 预训练权重与日志归档位置

本地 `experiments/` 只保留最近一次运行；各编码器历次预训练的权重与日志已统一拷贝至 `/data/datasets/PCP_checkpoint/`：

| 编码器 | 训练策略 | 归档目录（含全部权重/日志） |
|------|----------|------------------|
| V1（当前 30 epoch） | Standard CosineAnnealing（无早停） | `/data/datasets/PCP_checkpoint/Standard CosineAnnealing/[V1]PCP‑MAE+MaskTransformer/` |
| V2（当前 30 epoch） | Standard CosineAnnealing（无早停） | `/data/datasets/PCP_checkpoint/Standard CosineAnnealing/[V2]PCP‑MAE+PointTransformer/` |
| V1 / V2 / Point‑MAE / Mask Point‑MAE / ShapeNet55-34（历史） | 早停时期 | `/data/datasets/PCP_checkpoint/Early-Stopping/…`（子目录与 config stem 同名） |
| 冗余训练测试（已废弃） | — | `/data/datasets/PCP_checkpoint/Redundant training tests/Early-Stopping/…`（含各 config 早停时期重跑与 `Objaverse_bad_loss`） |
| 官方预训练权重 | — | `/data/datasets/PCP_checkpoint/official/PCP-MAE-275.pth`、`PCP-MAE-300.pth` |
---

## 6. 各编码器训练流程

> 以下命令均默认 `conda activate mae5090` + 单卡（`CUDA_VISIBLE_DEVICES=0`）；历史 RTX 3090 环境见 §1.2。

### 6.1 V1：PCP-MAE + MaskTransformer

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/[V1]PCP‑MAE+MaskTransformer.yaml" \
  --exp_name pcpmae_minigpt3d --seed 42
```

- 当前 config：`epochs/max_epoch = 30`、`T_max = 30`；按 §5.3 建议 **以 ≥100 epoch 重训**。
- 输出：`experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/`
- 权重归档：`/data/datasets/PCP_checkpoint/Standard CosineAnnealing/[V1]PCP‑MAE+MaskTransformer/`（30 epoch 运行的 `ckpt-best.pth` 的 val loss 与最后一个 epoch 相同）
- 导出（自动补 cls）：

```bash
python ckpt_extract.py \
  --ckpt "experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/ckpt-best.pth" \
  --out point_model_V1+random-cls.pth
cp "point_model_V1+random-cls.pth" /data/workspace/MiniGPT-3D/params_weight/pc_encoder/
```

### 6.2 V2：PCP-MAE + PointTransformer

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/[V2]PCP‑MAE+PointTransformer.yaml" \
  --exp_name pcp_minigpt_encoder_objaverse --seed 42
```

- 输出：`experiments/[V2]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/`
- 权重归档：`/data/datasets/PCP_checkpoint/Standard CosineAnnealing/[V2]PCP‑MAE+PointTransformer/`
- 导出：

```bash
python ckpt_extract.py \
  --ckpt "experiments/[V2]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/ckpt-best.pth" \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_v2.pth
```

### 6.3 Point-MAE + PointTransformer

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/Point‑MAE+PointTransformer.yaml" \
  --exp_name point_mae_objaverse --seed 42
```

- 输出：`experiments/Point‑MAE+PointTransformer/pretrain/point_mae_objaverse/`
- 权重归档：`/data/datasets/PCP_checkpoint/Early-Stopping/Point-MAE+PointTransformer/`
- 已导出权重：`params_weight/pc_encoder/old/point_model_pointmae.pth`（已归档）

```bash
python ckpt_extract.py \
  --ckpt "experiments/Point‑MAE+PointTransformer/pretrain/point_mae_objaverse/ckpt-best.pth" \
  --out point_model_pointmae.pth
```

### 6.4 Point-MAE + MaskTransformer

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/Point‑MAE+MaskTransformer.yaml" \
  --exp_name mask_mae_objaverse --seed 42
```

- 输出：`experiments/Point‑MAE+MaskTransformer/pretrain/mask_mae_objaverse/`
- 权重归档：`/data/datasets/PCP_checkpoint/Early-Stopping/Point‑MAE+MaskTransformer/`
- 已导出权重：`params_weight/pc_encoder/old/point_model_maskmae.pth`（已归档；导出时自动生成随机 cls_token）

```bash
python ckpt_extract.py \
  --ckpt "experiments/Point‑MAE+MaskTransformer/pretrain/mask_mae_objaverse/ckpt-best.pth" \
  --out point_model_maskmae.pth
```

### 6.5 ShapeNet55-34 权重（对照实验）

训练代码位于 **`PCP-MAE_with_ShapeNet`** 仓库（独立代码库，3 维输入，`total_bs=32`），完整训练命令见该仓库 README 的 Pre-training 章节。接入 MiniGPT-3D 的完整流程：

```bash
# ① 预训练（ShapeNet 仓库内；建议 max_epoch >= 300，不使用早停）
cd /data/workspace/PCP-MAE_with_ShapeNet
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/[ShapeNet55-34]PCP‑MAE+MaskTransformer.yaml" \
  --exp_name pcpmae_pretrain --seed 42

# ② 导出（ShapeNet 仓库内；输出 160 个键，自动补齐 cls_token/cls_pos）
python ckpt_extract.py \
  --ckpt "experiments/[ShapeNet55-34]PCP‑MAE+MaskTransformer/pretrain/pcpmae_pretrain/ckpt-last.pth" \
  --out pcpmae_ShapeNet.pth

# ③ 复制到 MiniGPT-3D
cp pcpmae_ShapeNet.pth /data/workspace/MiniGPT-3D/params_weight/pc_encoder/

# ④ 维度失配修复（3ch → 6ch；cls 已在 ② 中生成，此步为 no-op 校验）
cd /data/workspace/MiniGPT-3D
python fix_pcpmae_shapenet.py \
  --src params_weight/pc_encoder/pcpmae_ShapeNet.pth \
  --dst params_weight/pc_encoder/pcpmae_ShapeNet_fixed.pth
```

**修复细节（ShapeNet 权重与其他编码器的唯一差异）**：

| 项目 | 官方/其他编码器 | ShapeNet55-34 预训练导出 | 修复后 |
|------|------------------|----------------------|------|
| `encoder.first_conv.0.weight` | `(128, 6, 1)` | `(128, 3, 1)`（仅 xyz） | `(128, 6, 1)` |
| `encoder.cls_token` | 有 | **缺失** | ② 中自动生成（trunc_normal，std=0.02） |
| `encoder.cls_pos` | 有 | **缺失** | ② 中自动生成 |

- `first_conv` 为 1×1 卷积：RGB 3 通道补零，等价于黑色输入约定（MiniGPT-3D 对无 RGB 的 ShapeNet 样本的默认行为）。
- `fix_pcpmae_shapenet.py` 自带一致性检查：对照 `point_model.pth`（官方 ULIP-2）报告 missing / unexpected / 形状失配的键，修复后应全部通过。
- 完整流程与说明见 [PCP-MAE_with_ShapeNet README「权重导出与修复」](../PCP-MAE_with_ShapeNet/README.md)。
- 权重归档：`/data/datasets/PCP_checkpoint/Early-Stopping/[ShapeNet55-34]PCP‑MAE+MaskTransformer/`；已导出权重在 `params_weight/pc_encoder/old/`（`pcpmae_ShapeNet.pth` / `pcpmae_ShapeNet_fixed.pth`）。
---

## 7. 接入 MiniGPT-3D

### 7.1 权重放置

导出后的权重文件统一放在 `MiniGPT-3D/params_weight/pc_encoder/`（当前磁盘状态）：

```
params_weight/pc_encoder/
├── point_model.pth                 # Baseline（ULIP-2 / Point-BERT，官方）
├── point_model_random.pth          # Random（随机初始化，未预训练）
├── point_model_V1+random-cls.pth   # V1 random（自动随机 cls）
├── point_model_v2.pth              # V2
└── old/                            # 归档权重
    ├── point_model_pointmae.pth            # Point-MAE
    ├── point_model_maskmae.pth             # Mask Point-MAE
    ├── pcpmae_ShapeNet.pth                 # ShapeNet55-34（修复前）
    ├── pcpmae_ShapeNet_fixed.pth           # ShapeNet55-34（3ch→6ch 修复后）
    ├── point_model_pcpmae.pth              # （已废弃）V1，无 cls
    ├── point_model_hybrid.pth              # （已废弃）V1 hybrid
    ├── point_model_pcp_v2.pth              # （已废弃）V2 旧导出名
    └── mask-pcpmae.pth                     # （历史）Mask Point-MAE 旧导出名
```

### 7.2 Freeze / Unfreeze 验证状态

**5 个编码器（V1 random、V2、Point-MAE、Mask Point-MAE、ShapeNet55-34）均已通过 MiniGPT-3D 的 freeze 与 unfreeze 两种训练**。各变体推荐 `freeze_pc` 取值如下（详见 MiniGPT-3D 主 README §6）：

| 变体 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | 推荐 `freeze_pc` |
|------|---------|---------|---------|---------|------------------|
| Baseline（ULIP-2） | unfreeze | unfreeze | unfreeze | unfreeze | `False` |
| V1 random | unfreeze | freeze | freeze | freeze | `True` |
| V2 | unfreeze | freeze | freeze | freeze | `True` |
| Point-MAE | unfreeze | freeze | freeze | freeze | `True` |
| Mask Point-MAE | unfreeze | freeze | freeze | freeze | `True` |
| ShapeNet55-34 | unfreeze | freeze | freeze | freeze | `True` |
| Random | —（消融） | — | — | — | — |

配置文件位于 `MiniGPT-3D/configs_stage4/`，命名规范为 `{variant}_freeze_pc_{stage}.yaml`。

> ⚠️ PCP-MAE 编码器在“group masking”任务上预训练，接入 MiniGPT-3D 时建议 Stage 1 微调后冻结（Stage 1 是唯一 pc_encoder 输出直连 `pc_encoder_adapter`、真正参与训练的阶段）；否则后续阶段会破坏已冻结 LLM 的注意力模式，生成质量退化。Baseline（ULIP-2）在 MiniGPT-3D 中从头训练，全程可不冻结。

---

## 8. 诊断脚本

### 8.1 验证模型可构建、权重可加载

```bash
python test_v2.py \
  --model_path /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_v2.pth \
  --num_points 8192 --device cuda
```

预期输出：

```
[V2 Model Check]
  trans_dim=384 depth=12 num_heads=6
  point_feat: (1, 8192, 384)  ✓
  point_feat_std: 0.6738
  point_feat_mean: -0.0031
  ✓ Model built and weights loaded successfully
```

### 8.2 多编码器权重分布对比

```bash
python compare_encoders.py \
  --models "V1,/data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_V1+random-cls.pth" \
          "V2,/data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_v2.pth" \
          "Baseline,/data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model.pth" \
  --num_points 8192 --device cuda
```

输出 point_feat 分布对比表（std / mean / min / max）+ 各模型参数量统计，用于判断各编码器是否“在同一分布量级”。

---

## 9. 权重导出脚本说明

| 脚本 | 适用编码器 | 说明 |
|------|-----------|------|
| [`ckpt_extract.py`](ckpt_extract.py)（仓库根目录） | **全部编码器**（V1 / V2 / Point‑MAE×2 / ShapeNet55-34） | 从预训练 checkpoint 提取 MAE_encoder 骨干，输出 `{'base_model': ...}`（共 160 个 tensor）；MaskTransformer 自动补齐缺失的 `cls_token`/`cls_pos`（trunc_normal，std=0.02）；内置 9 键完整性检查 |
| [`MiniGPT-3D/fix_pcpmae_shapenet.py`](../MiniGPT-3D/fix_pcpmae_shapenet.py) | ShapeNet55-34 权重 | 修复 ShapeNet55-34 权重的维度失配：3ch→6ch（+cls 校验）；需在 `ckpt_extract.py` 之后运行 |

导出后的文件格式统一为：

```python
{'base_model': {
    'encoder.*', 'reduce_dim.*', 'pos_embed.*',
    'blocks.*', 'norm.*',
    'cls_token', 'cls_pos'   # V2/Point-MAE 含原生 cls；MaskTransformer 为自动生成
}}
```

与 `MiniGPT-3D/minigpt4/models/pointbert/point_encoder.py` 的 `load_checkpoint()` 直接兼容（`torch.load` + `load_state_dict` 即可，无需转换）。

> `ckpt_extract.py` 已对全部编码器类型验证通过：导出的键集合与 `point_model.pth` / `point_model_V1+random-cls.pth` 完全一致（均为 160 键）；ShapeNet55-34 额外需要的修复仅为 `first_conv` 的 3ch→6ch。

---

## 10. 文件索引

| 路径 | 用途 |
|------|------|
| [`note.txt`](note.txt) | 各编码器单卡/多卡训练与导出命令速查 |
| [`origin_readme.md`](origin_readme.md) | 官方 PCP-MAE 说明（论文与原始评测流程） |
| [`SETUP_mae5090.sh`](SETUP_mae5090.sh) | RTX 5090（`mae5090`）环境分步安装脚本 |
| [`cfgs/pretrain/[V1]PCP‑MAE+MaskTransformer.yaml`](cfgs/pretrain/[V1]PCP‑MAE+MaskTransformer.yaml) | V1 配置（MaskTransformer + PCP-MAE，Objaverse 参数基准） |
| [`cfgs/pretrain/[V2]PCP‑MAE+PointTransformer.yaml`](cfgs/pretrain/[V2]PCP‑MAE+PointTransformer.yaml) | V2 配置（PointTransformer + PCP-MAE） |
| [`cfgs/pretrain/Point‑MAE+PointTransformer.yaml`](cfgs/pretrain/Point‑MAE+PointTransformer.yaml) | Point-MAE 消融（PointTransformer 编码器，ita=0） |
| [`cfgs/pretrain/Point‑MAE+MaskTransformer.yaml`](cfgs/pretrain/Point‑MAE+MaskTransformer.yaml) | Point-MAE 消融（MaskTransformer 编码器，ita=0） |
| [`models/PCP_MAE.py`](models/PCP_MAE.py) | 模型定义（含 `PointTransformerMAEEncoder` + `MaskTransformer`） |
| [`models/pointbert_mg/`](models/pointbert_mg/) | MiniGPT-3D 兼容的 PointTransformer 实现 |
| [`ckpt_extract.py`](ckpt_extract.py) | 通用权重提取（支持所有编码器，带完整性校验 + 自动补 cls） |
| [`tools/runner_pretrain.py`](tools/runner_pretrain.py) | 预训练 runner（验证 loss 监控 + `ckpt-best` 保存，**无早停**） |
| [`tools/builder.py`](tools/builder.py) | 模型/数据集构建器、优化器/调度器配置（timm `CosineLRScheduler`） |
| `extensions/` | `chamfer_dist` / `emd` 纯 PyTorch 实现（`mae5090` 环境无需编译） |
| `/data/datasets/PCP_checkpoint/` | 预训练权重与日志归档目录（对照见 §5.4） |
---

## Contact

If you have any questions related to the code or the paper, feel free to email Xiangdong (`zhangxiangdong@sjtu.edu.cn`) or Shaofeng (`sherrylone@sjtu.edu.cn`).

## License

PCP-MAE is released under MIT License. See the [LICENSE](./LICENSE) file for more details. Besides, the licensing information for `pointnet2` modules is available [here](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/UNLICENSE).

## Acknowledgements

This codebase is built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [ReCon](https://github.com/qizekun/ReCon), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhang2024pcp,
  title={PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders},
  author={Zhang, Xiangdong and Zhang, Shaofeng and Yan, Junchi},
  journal={arXiv preprint arXiv:2408.08753},
  year={2024}
}
```

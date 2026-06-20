# PCP-MAE（MiniGPT-3D 适配版）

本仓库在 [PCP-MAE 官方实现](https://github.com/aHapBean/PCP-MAE) 基础上做了大量修改，用于在 **Objaverse 660K 点云** 上预训练点云编码器，并接入 [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D) 作为唯一可行的下游评测框架。

原始 PCP-MAE 论文说明、ShapeNet 分类/分割评测流程见 [`origin_readme.md`](origin_readme.md)。

---

## 1. 相对官方仓库的主要改动

| 项目 | 说明 |
|------|------|
| 模型配置 | 与 MiniGPT-3D 对齐：8192 点、6 维输入（xyz+rgb）、group_size=32、num_group=512、trans_dim=384、depth=12、num_heads=6 |
| 数据集 | 新增 `ObjaverseNPY` 数据加载器，读取 MiniGPT-3D 的 `*_8192.npy` |
| 编码器实现 | `models/pointbert_mg/` 从 `MiniGPT-3D/minigpt4/models/pointbert` 复制，保证 Group/Encoder/PointTransformer 与下游一致 |
| 编码器变体 | 通过 `encoder_type` 切换：`mask_transformer`（V1 旧版 cross-attn）或 `point_transformer`（V2 兼容版 self-attn+cls） |
| 训练 | 开启 bfloat16 AMP，单卡 RTX 3090 可训练；有效 batch=64（`total_bs=64, step_per_update=8`） |
| 评测 | 官方 ShapeNet/ScanObjectNN 微调流程**不再作为本项目的最终评测**；编码器质量统一通过 MiniGPT-3D 四阶段训练 + GPT/Qwen 主观评测衡量 |

---

## 2. 编码器变体一览

实验中使用了以下几种编码器权重，命名与实验代号对应关系如下：

| 代号 | 权重文件名（MiniGPT-3D 侧） | 预训练方式 | 配置文件 | 说明 |
|------|------------------------------|------------|----------|------|
| **Baseline** | `point_model.pth` | ULIP-2 / Point-BERT（官方） | — | MiniGPT-3D 原始编码器，非本仓库训练 |
| **V1** | `point_model_pcpmae.pth` | PCP-MAE + `MaskTransformer`（cross-attn） | `cfgs/pretrain/base.yaml` | 无 `cls_token`/`cls_pos` |
| **V1 hybrid** | `point_model_hybrid.pth` | V1 骨干 + 从 Baseline 拷贝 cls | 手动合并 | 弥补 V1 缺失的 cls 参数 |
| **V2** | `point_model_pcp_v2.pth` | PCP-MAE + `PointTransformerMAEEncoder` | `cfgs/pretrain/base_minigpt_encoder.yaml` | 预训练前向与 MiniGPT 推理一致，含 cls |
| **Point-MAE** | `point_model_pointmae.pth` | 纯 Point-MAE（`ita=0`） | `cfgs/pretrain/ablation_point_mae.yaml` | 关闭中心预测，其余与 V2 相同 |
| **ShapeNet55-34** | `pcpmae_ShapeNet.pth` / `pcpmae_ShapeNet_fixed.pth` | 官方 ShapeNet 预训练权重 | 官方 `base.yaml` | 需 `fix_pcpmae_shapenet.py` 适配 6 维输入 |

### 架构差异（关键）

```
V1 (MaskTransformer):
  patch tokens → cross-attention（visible + mask 双分支）→ 无 cls
  权重导出到 MiniGPT PointTransformer（纯 self-attn + cls）→ 计算图不匹配

V2 / Point-MAE (PointTransformerMAEEncoder):
  patch tokens → cls + 全长 self-attn（与 MiniGPT 相同）→ mask 后再送 PCP decoder
  权重可直接导出为 MiniGPT 的 point_model 格式
```

---

## 3. 数据准备

### 3.1 Objaverse（主实验数据）

数据目录与 MiniGPT-3D 共用：

```
/data/workspace/MiniGPT-3D/data/objaverse_data/
├── train.txt          # 训练集 ID 列表
├── test.txt           # 验证/测试 ID 列表
└── {id}_8192.npy      # shape: (8192, 6)，xyz + rgb ∈ [0,1]
```

配置文件 [`cfgs/dataset_configs/ObjaverseNPY.yaml`](cfgs/dataset_configs/ObjaverseNPY.yaml) 中 `data_root` 指向上述路径。

### 3.2 ShapeNet55-34（对照实验）

使用官方 PCP-MAE 预训练权重（[Google Drive](https://drive.google.com/drive/folders/1smQMWBBEdMOXVAzIBs3xCBrcyQDg8_GS)），本地路径示例：

```
/data/workspace/PCP-MAE/data/ShapeNet55-34/
```

ShapeNet 点云仅 3 维 xyz，接入 MiniGPT-3D 前需运行修复脚本（见 §5.5）。

---

## 4. 实验目录与日志位置

所有预训练实验的输出目录由以下规则自动生成：

```
./experiments/{config文件名}/{config父目录}/{exp_name}/
```

例如 `--config cfgs/pretrain/base.yaml --exp_name pcpmae_minigpt3d` 对应：

```
experiments/base/pretrain/pcpmae_minigpt3d/
├── YYYYMMDD_HHMMSS.log    # 训练日志（每次启动一个新时间戳文件）
├── config.yaml            # 本次实验配置快照
├── ckpt-last.pth          # 最新 checkpoint
```

TensorBoard 日志：

```
experiments/{config文件名}/{config父目录}/TFBoard/{exp_name}/
```

训练日志中关注：

```
[Epoch X/300][Batch Y/Z] ... Losses = ['loss1', 'loss2'] lr = ...
```

- `loss1`：Chamfer 重建损失（点云几何）
- `loss2`：中心预测损失（`ita * loss2`，Point-MAE 消融中 `ita=0` 时恒为 0）

---

## 5. 各编码器训练流程

### 5.1 V1：PCP-MAE + MaskTransformer（Objaverse）

**代号**：`hybrid-with-objaverse`（V1）

**配置**：[`cfgs/pretrain/base.yaml`](cfgs/pretrain/base.yaml)  
**特点**：默认 `encoder_type=mask_transformer`，使用 cross-attention 的 `MaskTransformer`，与 MiniGPT 推理图不一致。

```bash
cd /data/workspace/PCP-MAE

# 从头训练
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base.yaml \
  --exp_name pcpmae_minigpt3d \
  --seed 42

# 从 checkpoint 继续（换学习率等）
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base.yaml \
  --exp_name pcpmae_minigpt3d \
  --start_ckpts experiments/base/pretrain/pcpmae_minigpt3d/ckpt-last.pth

# 中断后续训（自动找 ckpt-last.pth）
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base.yaml \
  --exp_name pcpmae_minigpt3d \
  --resume
```

**日志 / checkpoint**：

```
experiments/base/pretrain/pcpmae_minigpt3d/
```

**权重导出（手动，无 cls）**：

从 `ckpt-last.pth` 中提取 `MAE_encoder.{encoder,reduce_dim,pos_embed,blocks,norm.*}`，去掉 `pred_head`、`ita` 等预测头键，保存为 MiniGPT 格式的 `base_model` 字典。

导出结果示例：`point_model_pcpmae.pth`（**不含** `cls_token`、`cls_pos`，加载时会出现 `missing_keys`）。

**Hybrid 合并（V1 hybrid）**：

将官方 `point_model.pth` 中的 `cls_token`、`cls_pos` 并入 V1 骨干：

```python
import torch
original = torch.load('./params_weight/pc_encoder/point_model.pth', map_location='cpu')
new = torch.load('point_model_pcpmae.pth', map_location='cpu')
new['base_model']['cls_token'] = original['base_model']['cls_token']
new['base_model']['cls_pos'] = original['base_model']['cls_pos']
torch.save(new, 'point_model_hybrid.pth')
```

> **注意**：V1 方案因预训练计算图与 MiniGPT 推理图不一致，不推荐继续使用。保留此流程仅为实验记录。

---

### 5.2 V2：PCP-MAE + PointTransformerMAEEncoder（Objaverse）

**代号**：`objaverse V2`

**配置**：[`cfgs/pretrain/base_minigpt_encoder.yaml`](cfgs/pretrain/base_minigpt_encoder.yaml)  
**特点**：`encoder_type: point_transformer`，使用 `PointTransformerMAEEncoder`（`pointbert_mg` 的标准 self-attn + cls），与 MiniGPT `PointTransformer` 前向一致。

```bash
cd /data/workspace/PCP-MAE

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base_minigpt_encoder.yaml \
  --exp_name pcp_minigpt_encoder_objaverse \
  --seed 42

# 继续训练
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base_minigpt_encoder.yaml \
  --exp_name pcp_minigpt_encoder_objaverse \
  --seed 42 \
  --resume
```

**日志 / checkpoint**：

```
experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/
```

**权重导出**：

```bash
# 方式一：专用导出脚本
python tools/export_minigpt_encoder.py \
  --pcp-ckpt experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/ckpt-last.pth \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_pcp_v2.pth

# 方式二：通用提取脚本（V2 / Point-MAE 均适用）
python ckpt_extract.py \
  --ckpt experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/ckpt-last.pth \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_pcp_v2.pth
```

导出后应包含 `cls_token`、`cls_pos` 及全部 backbone 键；复制到 MiniGPT-3D：

```
MiniGPT-3D/params_weight/pc_encoder/point_model_pcp_v2.pth
```

---

### 5.3 Point-MAE 消融（Objaverse）

**配置**：[`cfgs/pretrain/ablation_point_mae.yaml`](cfgs/pretrain/ablation_point_mae.yaml)  
**特点**：与 V2 完全相同，唯一区别是 `ita: 0.0`（关闭 PCP 中心预测分支，纯 Point-MAE）。

```bash
cd /data/workspace/PCP-MAE

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/ablation_point_mae.yaml \
  --exp_name point_mae_objaverse \
  --seed 42
```

**日志 / checkpoint**：

```
experiments/ablation_point_mae/pretrain/point_mae_objaverse/
├── YYYYMMDD_HHMMSS.log
├── ckpt-last.pth
└── ckpt-best.pth
```

**权重导出**：

```bash
python ckpt_extract.py \
  --ckpt experiments/ablation_point_mae/pretrain/point_mae_objaverse/ckpt-last.pth \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_pointmae.pth
```

---

### 5.4 Baseline：官方 Point-BERT（非本仓库训练）

MiniGPT-3D 自带的 ULIP-2 预训练权重：

```
MiniGPT-3D/params_weight/pc_encoder/point_model.pth
```

作为所有实验的对照基线，无需在本仓库中训练。

---

### 5.5 ShapeNet55-34 权重（对照实验）



```
MiniGPT-3D/params_weight/pc_encoder/pcpmae_ShapeNet.pth
```

ShapeNet 仅 3 维 xyz，需修复后才能被 MiniGPT-3D（6 维输入）加载：

```bash
cd /data/workspace/MiniGPT-3D

python fix_pcpmae_shapenet.py \
  --src params_weight/pc_encoder/pcpmae_ShapeNet.pth \
  --dst params_weight/pc_encoder/pcpmae_ShapeNet_fixed.pth
```

修复内容：
1. `encoder.first_conv.0.weight`：3 通道 → 6 通道（RGB 补零，等价于黑色输入）
2. 缺失的 `cls_token` / `cls_pos`：随机初始化

---

## 6. 接入 MiniGPT-3D

### 6.1 放置权重

将导出的 `.pth` 文件放入：

```
/data/workspace/MiniGPT-3D/params_weight/pc_encoder/
```

### 6.2 修改训练配置

在 `MiniGPT-3D/train_configs/MiniGPT_3D/stage_{1,2,3,4,5}.yaml` 中设置：

```yaml
model:
  pc_encoder_ckpt: "./params_weight/pc_encoder/point_model_pcp_v2.pth"  # 换成你的权重
  freeze_pc: True    # 冻结编码器（默认）
  # freeze_pc: False  # 解冻编码器（见 stage_5.yaml）
```

各实验对应权重示例：

| 实验 | `pc_encoder_ckpt` | `freeze_pc` |
|------|-------------------|-------------|
| Baseline | `point_model.pth` | `True` |
| V1 / V1 hybrid | `point_model_hybrid.pth` | `True` / `False`（stage_5） |
| V2 | `point_model_pcp_v2.pth` | `True` / `False`（stage_5） |
| Point-MAE | `point_model_pointmae.pth` | `True` |
| ShapeNet55-34 | `pcpmae_ShapeNet_fixed.pth` | `True` / `False`（stage_5） |

### 6.3 四阶段训练

```bash
cd /data/workspace/MiniGPT-3D

export PYTHONPATH=$PWD
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_1.yaml > log_pcpmae_stage_1.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_2.yaml > log_pcpmae_stage_2.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_3.yaml > log_pcpmae_stage_3.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_4.yaml > log_pcpmae_stage_4.txt
```

MiniGPT-3D 训练日志位于项目根目录：`log_pcpmae_stage_{1,2,3,4}.txt`（或 `log_stage_*.txt`）。

各 stage 输出目录在 yaml 的 `run.output_dir`，例如 `./output/pcpmae/stage_{1,2,3,4}/`。



### 6.5 主观评测

评测配置：[`MiniGPT-3D/eval_configs/benchmark_evaluation_paper.yaml`](../MiniGPT-3D/eval_configs/benchmark_evaluation_paper.yaml)

```bash
cd /data/workspace/MiniGPT-3D

# 开放词汇分类
python pointllm/eval/eval_objaverse.py \
  --out_path ./evaluate/<实验名> \
  --task_type classification \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 0

# ModelNet40 闭集分类
python pointllm/eval/eval_modelnet_cls.py \
  --out_path ./evaluate/<实验名> \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 0

# 物体描述生成
python pointllm/eval/eval_objaverse.py \
  --out_path ./evaluate/<实验名> \
  --task_type captioning \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 2

# Qwen API 主观打分
python pointllm/eval/evaluator_opensource_llm_QwenAPI.py \
  --results_path ./evaluate/<实验名>/evaluation/<结果json> \
  --eval_type <任务类型> \
  --model_type qwen-flash \
  --parallel --num_workers 4
```

> 若所有 stage 均为 `freeze_pc: True`，评测 yaml 中 **必须** 设置 `pc_encoder_ckpt` 指向对应预训练权重；否则加载的是错误/默认编码器。

---

## 7. 编码器质量诊断（MiniGPT-3D 侧）

在接入四阶段训练之前，可用特征对比脚本快速检查权重是否合理（注意，该诊断结果不决定最终评估质量）：

```bash
cd /data/workspace/MiniGPT-3D

python point_model_VS_hybrid.py \
  --ckpt-a ./params_weight/pc_encoder/point_model.pth \
  --ckpt-b ./params_weight/pc_encoder/point_model_pcp_v2.pth \
  --data-path ./data/modelnet40_data/modelnet40_test_8192pts_fps.dat \
  --max-samples 2468
```

关注指标：
- **Same-sample cosine (A vs B)**：cls / router 应接近 1.0；若接近 0 说明表征空间完全不同
- **kNN accuracy (ModelNet40)**：baseline 的 router 特征通常 75%+；候选权重若接近随机（~2.5%）则不应进入四阶段训练
- **Intra − Inter margin**：baseline cls margin ~0.11；margin 接近 0 说明类间不可分

---

## 8. 权重导出脚本说明

| 脚本 | 适用编码器 | 输出 |
|------|-----------|------|
| [`tools/export_minigpt_encoder.py`](tools/export_minigpt_encoder.py) | V2（`encoder_type=point_transformer`） | 含 cls 的 `base_model` 字典 |
| [`ckpt_extract.py`](ckpt_extract.py) | V2、Point-MAE | 同上，带完整性校验 |
| 手动提取（V1） | V1（`MaskTransformer`） | 无 cls，需 hybrid 合并 |
| [`MiniGPT-3D/fix_pcpmae_shapenet.py`](../MiniGPT-3D/fix_pcpmae_shapenet.py) | ShapeNet 官方权重 | 3ch→6ch + 补 cls |

导出后的文件格式统一为：

```python
{'base_model': {
    'encoder.*', 'reduce_dim.*', 'pos_embed.*',
    'blocks.*', 'norm.*',
    'cls_token', 'cls_pos'   # V2/Point-MAE/修复后的 ShapeNet 权重包含
}}
```

与 `MiniGPT-3D/minigpt4/models/pointbert/point_encoder.py` 的 `load_checkpoint()` 直接兼容。

---



## 11. 文件索引

| 路径 | 用途 |
|------|------|
| [`note.txt`](note.txt) | 实验命令速查 |
| [`origin_readme.md`](origin_readme.md) | 官方 PCP-MAE 说明 |
| [`cfgs/pretrain/base.yaml`](cfgs/pretrain/base.yaml) | V1 配置 |
| [`cfgs/pretrain/base_minigpt_encoder.yaml`](cfgs/pretrain/base_minigpt_encoder.yaml) | V2 配置 |
| [`cfgs/pretrain/ablation_point_mae.yaml`](cfgs/pretrain/ablation_point_mae.yaml) | Point-MAE 消融配置 |
| [`models/PCP_MAE.py`](models/PCP_MAE.py) | 模型定义（含 `PointTransformerMAEEncoder`） |
| [`models/pointbert_mg/`](models/pointbert_mg/) | MiniGPT-3D 兼容的 PointTransformer 实现 |

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

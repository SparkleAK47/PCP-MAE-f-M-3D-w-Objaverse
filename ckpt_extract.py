#!/usr/bin/env python3
"""
Extract MAE_encoder backbone weights from a PCP-MAE / Point-MAE checkpoint
and save in MiniGPT-3D point_model format (key: 'base_model').

Supports both encoder architectures:
  - PointTransformer (self-attn only, has cls_token/cls_pos)
  - MaskTransformer  (cross-attn, lacks cls_token/cls_pos -> auto-generated)

Usage:
    # [V1] PCP-MAE + MaskTransformer (auto-generates cls_token/cls_pos):
    python ckpt_extract.py --ckpt experiments/\[V1\]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/ckpt-best.pth \
                           --out point_model_V1+random-cls.pth

    # [V2] PCP-MAE + PointTransformer (native cls_token/cls_pos):
    python ckpt_extract.py --ckpt experiments/\[V2\]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/ckpt-best.pth \
                           --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_v2.pth

    # Point-MAE + PointTransformer:
    python ckpt_extract.py --ckpt experiments/Point‑MAE+PointTransformer/pretrain/point_mae_objaverse/ckpt-best.pth \
                           --out point_model_pointmae.pth

    # Point-MAE + MaskTransformer (auto-generates cls_token/cls_pos):
    python ckpt_extract.py --ckpt experiments/Point‑MAE+MaskTransformer/pretrain/mask_mae_objaverse/ckpt-best.pth \
                           --out point_model_maskmae.pth

The output file can be placed directly into:
    MiniGPT-3D/params_weight/pc_encoder/
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict


def extract_encoder_weights(ckpt_path, out_path, trans_dim=384):
    """Extract backbone weights, auto-generating cls_token/cls_pos if needed."""
    src = torch.load(ckpt_path, map_location='cpu')

    # Locate model state dict
    if 'base_model' in src:
        state_dict = src['base_model']
    elif 'model' in src:
        state_dict = src['model']
    elif 'state_dict' in src:
        state_dict = src['state_dict']
    else:
        raise KeyError(
            f"No recognized model state key found in checkpoint. "
            f"Available keys: {list(src.keys())}"
        )

    # Report checkpoint metadata
    for meta_key in ('epoch', 'metrics', 'best_metrics'):
        if meta_key in src:
            print(f"[info] checkpoint {meta_key}: {src[meta_key]}")

    new_state = OrderedDict()
    skipped = []

    for k, v in state_dict.items():
        # Strip DDP 'module.' prefix
        if k.startswith('module.'):
            k = k[7:]

        # Only process MAE_encoder keys (the PointTransformer backbone)
        if not k.startswith('MAE_encoder.'):
            continue

        k_body = k[len('MAE_encoder.'):]  # remove prefix

        # --- Map to MiniGPT-3D PointTransformer keys ---
        if k_body.startswith('encoder.'):
            new_state[k_body] = v
        elif k_body.startswith('reduce_dim.'):
            new_state[k_body] = v
        elif k_body.startswith('pos_embed.'):
            new_state[k_body] = v
        elif k_body.startswith('blocks.'):
            new_state[k_body] = v
        elif k_body.startswith('norm.'):
            new_state[k_body] = v
        elif k_body in ('cls_token', 'cls_pos'):
            new_state[k_body] = v
        else:
            skipped.append(k_body)

    # --- Handle missing cls_token / cls_pos ---
    # MaskTransformer lacks these; PointTransformerMAEEncoder has them.
    # MiniGPT-3D PointTransformer requires both to load successfully.
    missing_cls = []
    if 'cls_token' not in new_state:
        t = torch.zeros(1, 1, trans_dim)
        nn.init.trunc_normal_(t, std=0.02)
        new_state['cls_token'] = t
        missing_cls.append('cls_token')
    if 'cls_pos' not in new_state:
        t = torch.zeros(1, 1, trans_dim)
        nn.init.trunc_normal_(t, std=0.02)
        new_state['cls_pos'] = t
        missing_cls.append('cls_pos')

    if missing_cls:
        print(f"[info] Auto-generated (trunc_normal init): {', '.join(missing_cls)}")
        print(f"       This is expected for MaskTransformer checkpoints.")

    # --- Validation ---
    structural_required = [
        'cls_token',
        'cls_pos',
        'encoder.first_conv.0.weight',
        'encoder.second_conv.0.weight',
        'reduce_dim.weight',
        'pos_embed.0.weight',
        'blocks.blocks.0.attn.qkv.weight',
        'blocks.blocks.11.attn.qkv.weight',
        'norm.weight',
    ]
    missing = [k for k in structural_required if k not in new_state]
    if missing:
        print(f"[ERROR] Missing required keys: {missing}")
        raise RuntimeError(
            f"Export incomplete — {len(missing)} required keys missing. "
        )

    print(f"[ok] Extracted {len(new_state)} parameter tensors")
    print(f"[ok] All {len(structural_required)} required keys present")
    if skipped:
        print(f"[info] Skipped {len(skipped)} non-backbone keys "
              f"(decoder, pred_head, mask_token, etc.)")

    # Save in MiniGPT-3D format
    torch.save({'base_model': new_state}, out_path)
    print(f"[ok] Saved -> {out_path}")
    print(f"      Copy to: MiniGPT-3D/params_weight/pc_encoder/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract PointTransformer backbone from PCP-MAE/Point-MAE checkpoint'
    )
    parser.add_argument('--ckpt', required=True,
                        help='Path to PCP-MAE/Point-MAE checkpoint (e.g., ckpt-best.pth)')
    parser.add_argument('--out', required=True,
                        help='Output path for MiniGPT-3D compatible weight file')
    parser.add_argument('--trans-dim', type=int, default=384,
                        help='Transformer hidden dim for generated cls tokens (default: 384)')
    args = parser.parse_args()
    extract_encoder_weights(args.ckpt, args.out, trans_dim=args.trans_dim)

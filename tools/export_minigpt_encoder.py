#!/usr/bin/env python3
"""Export PCP MAE_encoder weights to MiniGPT point_model.pth format."""
import argparse
import torch
from collections import OrderedDict


def export(pcp_ckpt, out_path):
    ckpt = torch.load(pcp_ckpt, map_location='cpu')
    src = ckpt.get('base_model', ckpt.get('model', ckpt))

    dst = OrderedDict()
    prefix = 'MAE_encoder.'
    required = ('encoder.', 'reduce_dim.', 'cls_token', 'cls_pos', 'pos_embed.', 'blocks.', 'norm.')

    for k, v in src.items():
        k = k[7:] if k.startswith('module.') else k
        if not k.startswith(prefix):
            continue
        body = k[len(prefix):]
        if body.startswith(required) or body in ('cls_token', 'cls_pos'):
            dst[body] = v

    missing = [x for x in ['cls_token', 'cls_pos', 'encoder.first_conv.0.weight', 'blocks.blocks.0.attn.qkv.weight']
               if x not in dst]
    if missing:
        raise RuntimeError(f'Export incomplete, missing: {missing}')

    torch.save({'base_model': dst}, out_path)
    print(f'Saved {len(dst)} tensors -> {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcp-ckpt', required=True)
    parser.add_argument('--out', required=True)
    export(parser.parse_args().pcp_ckpt, parser.parse_args().out)
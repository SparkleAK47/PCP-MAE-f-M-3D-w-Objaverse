import torch
import numpy as np
import os
from torch.utils.data import Dataset
from utils.registry import DATASETS

@DATASETS.register_module()
class ObjaverseNPY(Dataset):
    def __init__(self, data_root='', npoints=8192, subset='train', **kwargs):
        self.data_root = data_root
        self.npoints = npoints
        
        # 假设 data_root 下有 train.txt / test.txt 列出文件名（不带后缀）
        split_file = os.path.join(data_root, f'{subset}.txt')
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_root, file_name + '.npy')
        data = np.load(file_path)  # (8192, 6)
        
        # 如果点数 > npoints，随机采样；如果 < npoints，重复或填充（这里简单随机采样）
        if data.shape[0] >= self.npoints:
            idxs = np.random.choice(data.shape[0], self.npoints, replace=False)
        else:
            idxs = np.random.choice(data.shape[0], self.npoints, replace=True)
        pts = data[idxs, :].astype(np.float32)
        
        # 返回坐标 + 特征（前3为xyz，后3为rgb）
        xyz = pts[:, :3]
        features = pts[:, 3:]   # 如果有rgb
        return xyz, features
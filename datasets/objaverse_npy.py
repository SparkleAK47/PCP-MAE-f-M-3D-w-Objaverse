import torch
import numpy as np
import os
from torch.utils.data import Dataset
from .build import DATASETS

@DATASETS.register_module()
class ObjaverseNPY(Dataset):
    def __init__(self, config, **kwargs):
        self.data_root = getattr(config, 'data_root', None) or getattr(config, 'DATA_PATH', None)
        self.npoints = getattr(config, 'npoints', None) or getattr(config, 'N_POINTS', 8192)
        self.subset = getattr(config, 'subset', 'train')

        list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        with open(list_file, 'r') as f:
            self.file_names = [line.strip() for line in f]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_root, f'{file_name}.npy')
        data = np.load(file_path)   # (8192, 6)

        if data.shape[0] >= self.npoints:
            idxs = np.random.choice(data.shape[0], self.npoints, replace=False)
        else:
            idxs = np.random.choice(data.shape[0], self.npoints, replace=True)
        pts = data[idxs, :].astype(np.float32)


        # 返回3元组，匹配训练循环的解包格式
        return 'objaverse', file_name, pts
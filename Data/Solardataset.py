import torch
from torch.utils.data import Dataset

import os

class MagnentDataset(Dataset):
    def __init__(self,mag_dir):
        self.dir = mag_dir
        self.all_files = os.listdir(mag_dir)

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self,idx):
        file_path = self.all_files[idx]
        return file_path
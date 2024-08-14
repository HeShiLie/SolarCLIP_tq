import torch
from astropy.io import fits
import numpy as np

import os
from pathlib import Path

import argparse

def fits_to_tensor(fits_path):
    # 读取 FITS 文件
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        # 替换 NaN 值为 0
        data = np.nan_to_num(data, nan=0.0)
        # 将 NumPy 数组转换为 PyTorch 张量
        tensor = torch.tensor(data, dtype=torch.float32)
    return tensor

def convert_fits_directory_to_pt(source_dir, target_dir):
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录中的所有 FITS 文件
    for fits_file in Path(source_dir).rglob('*.fits'):
        # 将 FITS 文件转换为 PyTorch 张量
        tensor = fits_to_tensor(fits_file)
        # 生成目标文件路径
        target_file = Path(target_dir) / (fits_file.stem + '.pt')
        # 保存张量为 .pt 文件
        torch.save(tensor, target_file)
        print(f"Converted {fits_file} to {target_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert FITS files to PyTorch tensors')
    parser.add_argument('--source_dir', default = "/home/huxing/202407/jsoc.stanford.edu/data/hmi/fits/2010", type=str, help='Source directory containing FITS files')
    parser.add_argument('--target_dir', default = "/home/huxing/202407/jsoc.stanford.edu/data/hmi/pt/2010", type=str, help='Target directory to save PyTorch tensors')

    args = parser.parse_args()

    source_dir = args.source_dir
    target_dir = args.target_dir

    convert_fits_directory_to_pt(source_dir, target_dir)
from tqdm import tqdm
from utils_tq import get_modal_dir, save_list
import numpy as np
import os

def update_exist_list(modal, save_dir = './Data/idx_list',time_interval = [0,7452000]):
    exist_idx = np.zeros(time_interval[1], dtype=np.bool)
    if modal == 'magnet':
        for i in tqdm(range(time_interval[0], time_interval[1])):
            dir_fits, dir_pt = get_modal_dir(modal, i)
            if os.path.exists(dir_pt):
                exist_idx[i] = True
        save_list(exist_idx, f'{save_dir}/{modal}_exist_idx.pkl')
    elif modal == '0094':
        for i in tqdm(range(time_interval[0], time_interval[1])):
            dir_fits, dir_pt = get_modal_dir(modal, i)
            if os.path.exists(dir_pt):
                exist_idx[i] = True
        save_list(exist_idx, f'{save_dir}/{modal}_exist_idx.pkl')

if __name__ == '__main__':
    update_exist_list('magnet')
    update_exist_list('0094')

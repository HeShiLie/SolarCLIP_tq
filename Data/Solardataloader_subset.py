from .Solardataloader import *
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def get_loader_by_time(time_step=60,time_interval = [0,6000],modal_list = ['magnet','0094'], load_imgs = True, enhance_list = [['log1p',224,1],['log1p',224,1]], batch_size = 32, shuffle=True, num_workers=4, sampler=None):
    dataset = multimodal_dataset(modal_list, load_imgs, enhance_list, time_interval,time_step)
    if sampler == 'distributed':
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=42
        )
    elif sampler == "None":
        sampler = None
    else:
        raise ValueError('sampler should be None or distributed')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)

    # return [Batchsize, modal_num, channel, height, width]
    return dataloader
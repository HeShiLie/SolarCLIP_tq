from .Solardataloader_tq import *

def get_loader_by_time(time_step=60,time_interval = [0,6000],modal_list = ['magnet','0094'], load_imgs = True, enhance_list = [['log1p',224,1],['log1p',224,1]], batch_size = 32, shuffle=True, num_workers=4):
    dataset = multimodal_dataset(modal_list, load_imgs, enhance_list, time_interval,time_step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # return [Batchsize, modal_num, channel, height, width]
    return dataloader
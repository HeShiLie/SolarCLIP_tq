import pickle
import wget
import os 

import numpy as np
import torch

from tqdm import tqdm

from utils import transfer_id_to_date, get_modal_dir,read_fits_image


def verify_and_download(modal,exist_idx_list, time_interval = [0,1e+32]):
    download_num = 0
    error_url = []
    
    for i in tqdm(range(len(exist_idx_list))):

        if i<time_interval[0] or i>time_interval[1]:
            continue

        if exist_idx_list[i] == 0: # no pt file now
            date_time = transfer_id_to_date(i)
            year = date_time.year
            month = date_time.month
            day = date_time.day
            hour = date_time.hour
            minute = date_time.minute

            if modal == 'magnet':
                if minute == 0:
                    url = f'http://jsoc.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits'
                    # http://jsoc.stanford.edu/data/hmi/fits/2011/02/02/hmi.M_720s.20110202_000001_TAI.fits
                    dir_fits = get_modal_dir('magnet',i)[0]
                    try:
                        wget.download(url, dir_fits) # download fits file
                        download_num += 1
                    except Exception as e:
                        error_url.append(url)
            else:
                raise ValueError('modal not supported')

    print(f'{download_num} files downloaded, {len(error_url)} files failed:')
    for url in error_url:
        print(url)

def dl_and_convert(modal,exist_idx_list, time_interval = [0,1e+32]):
    download_num = 0
    error_url = []
    
    for i in tqdm(range(len(exist_idx_list))):

        if i<time_interval[0] or i>time_interval[1]:
            continue

        if exist_idx_list[i] == 0: # no pt file now
            date_time = transfer_id_to_date(i)
            year = date_time.year
            month = date_time.month
            day = date_time.day
            hour = date_time.hour
            minute = date_time.minute

            if modal == 'magnet':
                if minute == 0:
                    url = f'http://jsoc.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits'
                    # http://jsoc.stanford.edu/data/hmi/fits/2011/02/02/hmi.M_720s.20110202_000001_TAI.fits
                    dir_fits, dir_pt = get_modal_dir('magnet',i)
                    try:
                        wget.download(url, dir_fits) # download fits file
                        download_num += 1
                        try:
                            fits_img = read_fits_image(dir_fits)
                            fits_img = np.nan_to_num(fits_img, nan=0.0)
                            pt_img = torch.tensor(fits_img,dtype=torch.float32)
                            pt_dir = os.path.dirname(dir_pt)
                            if not os.path.exists(pt_dir):
                                os.makedirs(pt_dir)
                            torch.save(pt_img, dir_pt)
                            # exist_idx_list[i] = True
                        except Exception as e:
                            print(f"Error occured : {e}, delete {dir_pt} if exists")
                            if os.path.exists(dir_pt):
                                os.remove(dir_pt)                           
                    except Exception as e:
                        error_url.append(url)
            else:
                raise ValueError('modal not supported')

    print(f'{download_num} files downloaded, {len(error_url)} files failed:')
    for url in error_url:
        print(url)
            

if __name__ == '__main__' :

    with open('/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/Data/idx_list/magnet_exist_idx.pkl','rb') as f:
        exist_idx_list = pickle.load(f)
    i = 13
    print(500000*i,500000*(i+1))
    print('start date :', transfer_id_to_date(500000*i))
    print('end date :', transfer_id_to_date(500000*(i+1)))

    dl_and_convert('magnet',exist_idx_list,[500000*i,500000*(i+1)])

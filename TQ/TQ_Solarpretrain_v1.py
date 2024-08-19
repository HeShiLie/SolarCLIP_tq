import torch
import numpy as np
import pickle
import json
import argparse
from types import SimpleNamespace

import os
import random
import time

from Model.encoder import Embeddings,PretrainModel
from Model.decoder import LinearDecoder

from TQ_Solarclip_train import load_args_from_json
from TQ_Solarclip_test import calculate_loss_pretrain
from Data.TQ_Solardataloader import enhance_funciton
from Data.utils_tq import transfer_date_to_id
import Data.TQ_Solardataloader_subset as Solardataloader_subset

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain encoder and decoder SolarCLIP model.')

    parser.add_argument('--modal', type=str, default='magnet', help='Modal for Pretrain.')

    parser.add_argument('--enhance_list', type=list, nargs="+", default=
                        ['log1p', 1], help='Enhance list for training')

    return parser.parse_args()

def train(train_loader, val_loader, 
          epochs, test_freq, save_freq, 
          modal, enhance_list):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Embeddings(
        in_channels=1,
        input_resolution=1024,
        patch_size=64,
        width=768,
        hidden_dropout_prob=1e-1
    )
    decoder = LinearDecoder(
        embed_dim=768,
        out_dim=1024*4
    )
    pretrainModel = PretrainModel(encoder, decoder).to(device)
    
    checkpoint_path = f'/mnt/tianwen-tianqing-nas/tianwen/home/202407/ctf/SolarCLIP/checkpoints/pretrain_modify/{modal}/{enhance_list[0]}/'
    logger_checkpoint_path = checkpoint_path + 'logger/'
    model_checkpoint_path = checkpoint_path + 'model/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(f'{model_checkpoint_path}'):
        os.makedirs(f'{model_checkpoint_path}')
    if not os.path.exists(f'{logger_checkpoint_path}'):
        os.makedirs(f'{logger_checkpoint_path}')

    
    optimizer = torch.optim.SGD(
        pretrainModel.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)

    test_epoch = epochs//test_freq
    save_epoch = epochs//save_freq

    logger_train_loss = []
    logger_lr = []
    logger_val_loss = []

    start_time = time.time()
    iter_time = time.time()
    print(f'Start training Modal: {modal}')
    for epoch in range(epochs):
        pretrainModel.train()
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            if modal == 'magnet':
                image = data[:,0,:,:,:]
            elif modal == '0094':
                image = data[:,1,:,:,:]
            image = enhance_funciton(image, enhance_list[0], enhance_list[1])
            iteration_txt = f"Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            loss = calculate_loss_pretrain(modal, pretrainModel.encoder, pretrainModel.decoder, image)
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            logger_train_loss.append(loss.item())
            logger_lr.append(scheduler.get_last_lr()[0])
            iteration_txt += f" Iteration time: {(time.time()-iter_time)/60:.2f} min |"
            iter_time = time.time()
            print(iteration_txt)

        scheduler.step()

        print(f'Epoch {epoch+1:>6}/{epochs:<6} | Train loss {logger_train_loss[-1]:<10.8f} | LR {logger_lr[-1]:<10.8f} | Cost {(time.time()-start_time)/60:.2f} min')

        if (epoch+1) % test_epoch == 0:
            with torch.no_grad():
                pretrainModel.eval()
                loss_results = []
                
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    if modal == 'magnet':
                        image = data[:,0,:,:,:]
                    elif modal == '0094':
                        image = data[:,1,:,:,:]
                    image = enhance_funciton(image, enhance_list[0], enhance_list[1])
                    loss = calculate_loss_pretrain(modal, encoder, decoder, image)
                    
                    logger_train_loss.append(loss.item())
                    logger_lr.append(scheduler.get_last_lr()[0])

                logger_val_loss.append(np.mean(loss_results))
                
                result_txt = f'Epoch {epoch+1:>6}/{epochs:<6} | '

                result_txt += f'Train loss {logger_train_loss[-1]:<10.8f} | '
                result_txt += f'Val loss {logger_val_loss[-1]:<10.8f} | '
                result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
                print(result_txt)

                with open(f'{logger_checkpoint_path}logger_train_loss.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss, f)
                with open(f'{logger_checkpoint_path}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr, f)
                with open(f'{logger_checkpoint_path}logger_val_loss.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss, f)

        if (epoch+1) % save_epoch == 0:
            torch.save({'model': PretrainModel.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch},
                       f'{model_checkpoint_path}epoch_{epoch+1}.pt')
            print(f'Model saved {(epoch+1)/epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')


def main():
    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=3*60,time_interval=[
                                                             start_date, end_date], modal_list=["magnet","0094"], load_imgs = True, enhance_list=[1024,0.5,90], batch_size=400, shuffle=True, num_workers=16)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=3*60,time_interval=[
                                                           start_date, end_date], modal_list=["magnet","0094"], load_imgs = True, enhance_list=[1024,0,0], batch_size=400, shuffle=True, num_workers=16)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

        # args = parse_args()
    config_path = ['./configs/pretrain/args1.json',
                   './configs/pretrain/args2.json',
                   './configs/pretrain/args3.json',
                   './configs/pretrain/args4.json']

    for i in range(4):
        args = load_args_from_json(config_path[i])
        modal = args.modal
        enhance_ls = args.enhance_list
        epochs = args.epochs
        test_freq = args.test_freq
        save_freq = args.save_freq

        train(train_loader, val_loader, epochs, test_freq, save_freq, modal, enhance_ls)

if __name__ == '__main__':
    main()  
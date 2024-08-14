import torch
import torch.multiprocessing as mp
import numpy as np
import pickle
import argparse

import os
import time

from Model.encoder import Embeddings,PretrainModel
from Model.decoder import LinearDecoder

from Solarclip_train_tq import load_args_from_json
from Data.Solardataloader import enhance_funciton
from Data.utils import transfer_date_to_id
import Data.Solardataloader_subset as Solardataloader_subset

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain encoder and decoder SolarCLIP model.')

    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')

    parser.add_argument('--enhance_list', type=list, nargs="+", default=
                        ['log1p', 1], help='Enhance list for training')

    return parser.parse_args()

def train(train_loader, val_loader, 
          epochs, test_freq, save_freq, 
          modal_list, enhance_list,device):

    encoder = Embeddings(
        in_channels=1,
        input_resolution=1024,
        patch_size=64,
        width=768,
        hidden_dropout_prob=1e-1
    )
    decoder = LinearDecoder(image_size = 1024, patch_size = 64, embed_dim = 768)
    pretrainModel_mag = PretrainModel(encoder, decoder).to(device)
    pretrainModel_0094 = PretrainModel(encoder, decoder).to(device)

    checkpoint_path_mag = f'/mnt/nas/home/huxing/202407/ctf/SolarCLIP/checkpoints/pretrain_5/{enhance_list[0]}/magnet/'
    logger_checkpoint_path_mag = checkpoint_path_mag + 'logger/'
    model_checkpoint_path_mag = checkpoint_path_mag + 'model/'
    if not os.path.exists(checkpoint_path_mag):
        os.makedirs(checkpoint_path_mag)
    if not os.path.exists(f'{model_checkpoint_path_mag}'):
        os.makedirs(f'{model_checkpoint_path_mag}')
    if not os.path.exists(f'{logger_checkpoint_path_mag}'):
        os.makedirs(f'{logger_checkpoint_path_mag}')

    checkpoint_path_0094 = f'/mnt/nas/home/huxing/202407/ctf/SolarCLIP/checkpoints/pretrain_5/{enhance_list[0]}/0094/'
    logger_checkpoint_path_0094 = checkpoint_path_0094 + 'logger/'
    model_checkpoint_path_0094 = checkpoint_path_0094 + 'model/'
    if not os.path.exists(checkpoint_path_0094):
        os.makedirs(checkpoint_path_0094)
    if not os.path.exists(f'{model_checkpoint_path_0094}'):
        os.makedirs(f'{model_checkpoint_path_0094}')
    if not os.path.exists(f'{logger_checkpoint_path_0094}'):
        os.makedirs(f'{logger_checkpoint_path_0094}')

    optimizer_mag = torch.optim.AdamW(
        pretrainModel_mag.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler_mag = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_mag, T_max=epochs)
    
    optimizer_0094 = torch.optim.AdamW(
        pretrainModel_0094.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler_0094 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_0094, T_max=epochs)

    test_epoch = epochs//test_freq
    save_epoch = epochs//save_freq

    logger_train_loss_mag = []
    logger_train_loss_mse_mag = []  
    logger_lr_mag = []
    logger_val_loss_mag = []
    logger_val_loss_mse_mag = []
    logger_train_loss_0094 = []
    logger_train_loss_mse_0094 = []
    logger_lr_0094 = []
    logger_val_loss_0094 = []
    logger_val_loss_mse_0094 = []

    start_time = time.time()
    iter_time = time.time()
    print(f'Start training Modal: {modal_list}')
    for epoch in range(epochs):
        pretrainModel_mag.train()
        pretrainModel_0094.train()
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            image_mag = data[:,0,:,:,:]
            image_0094 = data[:,1,:,:,:]
            image_mag = enhance_funciton(image_mag, enhance_list[0], enhance_list[1])
            image_0094 = enhance_funciton(image_0094, enhance_list[0], enhance_list[1])
            iteration_txt = f"Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            loss_mag, loss_mse_mag = pretrainModel_mag.calculate_loss(image_mag)
            loss_0094, loss_mse_0094 = pretrainModel_0094.calculate_loss(image_0094)
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer_mag.zero_grad()
            optimizer_0094.zero_grad()
            loss_mag.backward()
            loss_0094.backward()      
            optimizer_mag.step()
            optimizer_0094.step()
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            logger_train_loss_mag.append(loss_mag.item())
            logger_train_loss_mse_mag.append(loss_mse_mag.item())
            logger_lr_mag.append(scheduler_mag.get_last_lr()[0])
            logger_train_loss_0094.append(loss_0094.item())
            logger_train_loss_mse_0094.append(loss_mse_0094.item())
            logger_lr_0094.append(scheduler_0094.get_last_lr()[0])
            iteration_txt += f" Iteration time: {(time.time()-iter_time)/60:.2f} min |"
            iter_time = time.time()
            print(iteration_txt)

        scheduler_mag.step()
        scheduler_0094.step()

        print(f'Epoch {epoch+1:>6}/{epochs:<6} | Train loss magnet {logger_train_loss_mag[-1]:<10.8f} | LR of magnet {logger_lr_mag[-1]:<10.8f} | Train loss 0094 {logger_train_loss_0094[-1]:<10.8f} | LR of 0094 {logger_lr_0094[-1]:<10.8f} | Cost {(time.time()-start_time)/60:.2f} min')

        if (epoch+1) % test_epoch == 0:
            with torch.no_grad():
                pretrainModel_mag.eval()
                pretrainModel_0094.eval()
                loss_results_mag = []
                loss_mse_results_mag = []   
                loss_results_0094 = []
                loss_mse_results_0094 = []

                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    image_mag = data[:,0,:,:,:]
                    image_0094 = data[:,1,:,:,:]
                    image_mag = enhance_funciton(image_mag, enhance_list[0], enhance_list[1])
                    image_0094 = enhance_funciton(image_0094, enhance_list[0], enhance_list[1])
                    
                    loss_mag, loss_mse_mag = pretrainModel_mag.calculate_loss(image_mag,weights_type='cv-rdbu')
                    loss_0094, loss_mse_0094 = pretrainModel_0094.calculate_loss(image_0094,weights_type='3sgm-continous')
                    loss_results_mag.append(loss_mag.item())
                    loss_mse_results_mag.append(loss_mse_mag.item())
                    logger_lr_mag.append(scheduler_mag.get_last_lr()[0])
                    loss_results_0094.append(loss_0094.item())
                    loss_mse_results_0094.append(loss_mse_0094.item())
                    logger_lr_0094.append(scheduler_0094.get_last_lr()[0])

                logger_val_loss_mag.append(np.mean(loss_results_mag))
                logger_val_loss_mse_mag.append(np.mean(loss_mse_results_mag))
                logger_val_loss_0094.append(np.mean(loss_results_0094))
                logger_val_loss_mse_0094.append(np.mean(loss_mse_results_0094))
                
                result_txt = f'Epoch {epoch+1:>6}/{epochs:<6} | '
                result_txt += f'Magnet Train loss {logger_train_loss_mag[-1]:<10.8f} | '
                result_txt += f'Magnet Train loss mse {logger_train_loss_mse_mag[-1]:<10.8f} | '
                result_txt += f'Magnet Val loss {logger_val_loss_mag[-1]:<10.8f} | '
                result_txt += f'0094 Train loss {logger_train_loss_0094[-1]:<10.8f} | '
                result_txt += f'0094 Train loss mse {logger_train_loss_mse_0094[-1]:<10.8f} | '
                result_txt += f'0094 Val loss {logger_val_loss_0094[-1]:<10.8f} | '
                result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
                print(result_txt)

                with open(f'{logger_checkpoint_path_mag}logger_train_loss.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_train_loss_mse.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss_mse_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_val_loss.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_val_loss_mse.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss_mse_mag, f)
                with open(f'{logger_checkpoint_path_0094}logger_train_loss.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_train_loss_mse.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss_mse_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_val_loss.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_val_loss_mse.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss_mse_0094, f)


        if (epoch+1) % save_epoch == 0:
            torch.save({'model': pretrainModel_mag.state_dict(), 'optimizer': optimizer_mag.state_dict(),
                        'scheduler': scheduler_mag.state_dict(), 'epoch': epoch},
                       f'{model_checkpoint_path_mag}epoch_{epoch+1}.pt')
            torch.save({'model': pretrainModel_0094.state_dict(), 'optimizer': optimizer_0094.state_dict(),
                        'scheduler': scheduler_0094.state_dict(), 'epoch': epoch},
                       f'{model_checkpoint_path_0094}epoch_{epoch+1}.pt')
            print(f'Model saved {(epoch+1)/epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')



def main():
    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                             start_date, end_date], modal_list=["magnet","0094"], load_imgs = True, enhance_list=[1024,0.5,90], batch_size=400, shuffle=True, num_workers=0)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                           start_date, end_date], modal_list=["magnet","0094"], load_imgs = True, enhance_list=[1024,0,0], batch_size=400, shuffle=True, num_workers=0)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    config_path = './configs/pretrain/args3.json'

    args = parse_args()
    args = load_args_from_json(args,config_path)
    print(args)
    modal_list = args.modal_list
    enhance_ls = args.enhance_list
    epochs = args.epochs
    test_freq = args.test_freq
    save_freq = args.save_freq

    train(train_loader, val_loader, epochs, test_freq, save_freq, modal_list, enhance_ls, device='cuda:0')

    

if __name__ == '__main__':
    main()  
    # for i in range(3):
    #     args = parse_args()
    #     args = load_args_from_json(args,config_path[i])
    #     print(args)
    #     modal = args.modal
    #     enhance_ls = args.enhance_list
    #     epochs = args.epochs
    #     test_freq = args.test_freq
    #     save_freq = args.save_freq
    #     dev = args.device

    #     train(train_loader, val_loader, epochs, test_freq, save_freq, modal, enhance_ls,dev)
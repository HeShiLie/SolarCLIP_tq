import os
import time

import numpy as np

import torch
import torch.nn as nn

import argparse
import json
from types import SimpleNamespace

import pickle

from Data import Solardataloader_subset
from Data.utils import transfer_date_to_id
from Data.Solardataloader import enhance_funciton
from Model.get_weights import get_weights
from Model.SolarReconModel_VAE_pos_CLIP import get_SolarReconModel_VAE_pos_CLIP_from_args

import random
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP Reconstruction model like Unet.')

    parser.add_argument('--config_dir', type=str, default='None')

    # Pretrained model parameters
    parser.add_argument("--clip_model_path", type=str,
                        default="checkpoints/recon/0816_test1/SolarCLIP", help="The path to the SolarCLIP model.")
    parser.add_argument("--clip_model_id", type=int,
                        default=200, help="The id of the SolarCLIP model.")

    # Training parameters
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-1, help='Learning rate')
    parser.add_argument('--epochs', type=int, 
                        default=1000,help='Number of training epochs')
    parser.add_argument('--test_freq', type=int, 
                        default=100,help='Frequency of testing the model')
    parser.add_argument('--save_freq', type=int,
                        default=100,help='Frequency of saving the model')
    parser.add_argument('--early_stop', type=int, 
                        default=100, help='Early stop for training')
    parser.add_argument('--device', type=str,
                        default='cuda:3', help='Device for training')
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/VAE/test", help="The output path to save the model.")

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='Number of data loading workers')
    parser.add_argument('--machine', type=str, 
                        default='A40',help='Machine type for training')
    parser.add_argument('--sampler', type=str, 
                        default='distributed',help='Sampler for training')

    # SolarModel parameters
    parser.add_argument('--token_type', type=str,
                        default='all embedding', help='Token type for CLIP model')

    # SolarReconModel parameters
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--input_resolution', type=int, default=1024,
                        help='Input resolution')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch size')
    parser.add_argument('--width', type=int, default=768,
                        help='Width of the model')
    parser.add_argument('--vit_layers', type=int, default=12,
                        help='Number of vit layers')
    parser.add_argument('--transformer_layers', type=int, default=2,
                        help='Number of layers in ordinary transformer')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension')
    parser.add_argument('--norm_type', type=str, default='bn1d',
                        help='Normalization type')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='Output dimension')
    parser.add_argument('--output_size', type=int, default=1024,
                        help='Output size')
    parser.add_argument('--deprojection_type', type=str, default='linear',
                        help='Deprojection type')
    parser.add_argument('--with_bias', type=bool, default=True,
                        help='With bias')
    
    # Modal parameters
    parser.add_argument('--decoder_modal', type=str, 
                       default='magnet-0094', help='Modal for decoder')
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for load dataloader')
    parser.add_argument('--enhance_list', type=list, nargs="+", 
                        default=[['log1p', 1], ['log1p', 1]], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", 
                        default=[1024,0.5,90], help='Image preprocess list for training [resize, flip, rotate]')
    parser.add_argument('--weights_type_list', type=str, nargs="+",
                        default=['cv-rdbu', '3sgm-continous'], help='Weights type list for training')
    
    return parser.parse_args()

def save_args(args, checkpoint_dir):
    with open(f'{checkpoint_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f'args saved to {checkpoint_dir}/args.json')

def load_args_from_json(args,config_dir):
    with open(f'{config_dir}', 'r') as f:
        arg_json = json.load(f)
    for arg in arg_json:
        setattr(args, arg, arg_json[arg])
    return args

def main():
    args = parse_args()
    _ = args.config_dir
    if args.config_dir != 'None':
        args = load_args_from_json(args, args.config_dir)
    args.config_dir = _
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")

    checkpoint_path = args.checkpoint_path

    checkpoint_path = checkpoint_path + f'/{args.decoder_modal}/'
    logger_checkpoint_path = checkpoint_path + 'logger/'
    model_checkpoint_path = checkpoint_path + 'model/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(f'{model_checkpoint_path}'):
        os.makedirs(f'{model_checkpoint_path}')
    if not os.path.exists(f'{logger_checkpoint_path}'):
        os.makedirs(f'{logger_checkpoint_path}')


    save_args(args, checkpoint_path)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"Device: {device}")

    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12*10,time_interval=[
                                                             start_date, end_date], modal_list=args.modal_list, load_imgs= True, enhance_list=args.image_preprocess, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                             sampler=args.sampler)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12*10,time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list,  load_imgs= True, enhance_list=[args.image_preprocess[0],0,0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                           sampler=args.sampler)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    encoder_transformer_path = args.clip_model_path + '/'
    encoder_transformer_path = encoder_transformer_path + f'epoch_{args.clip_model_id}.pt'
    print('current_device', torch.cuda.current_device())  # 检查当前设备ID

    SolarCLIP_dict = torch.load(encoder_transformer_path)['model']
    if args.decoder_modal == '0094-magnet':
        visual_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_H.')}
        visual_state_dict = {k[len('visual_H.'):]: v for k, v in visual_state_dict.items()}
    elif args.decoder_modal == 'magnet-0094':
        visual_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_mag.')}
        visual_state_dict = {k[len('visual_mag.'):]: v for k, v in visual_state_dict.items()}
    elif args.decoder_modal == '0094-0094':
        visual_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_H.')}
        visual_state_dict = {k[len('visual_H.'):]: v for k, v in visual_state_dict.items()}
    elif args.decoder_modal == 'magnet-magnet':
        visual_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_mag.')}
        visual_state_dict = {k[len('visual_mag.'):]: v for k, v in visual_state_dict.items()}
    else:
        raise ValueError('Decoder modal not supported')

    SolarReconModel = get_SolarReconModel_VAE_pos_CLIP_from_args(args).to(device)
    SolarReconModel.vit_model.load_state_dict(visual_state_dict)
    # del SolarModel

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            SolarReconModel.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            SolarReconModel.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    epochs = args.epochs
    test_epoch = epochs//args.test_freq
    save_epoch = epochs//args.save_freq

    logger_train_loss = []
    logger_train_recon_loss = []
    logger_train_recon_loss_weighted = []
    logger_train_KLD = []
    logger_lr = []
    logger_val_loss = []
    logger_val_recon_loss = []
    logger_val_recon_loss_weighted = []
    logger_val_KLD = []

    start_time = time.time()
    iter_time = time.time()
    
    print(f'Start training SolarReconModel')
    
    for epoch in range(epochs):
        train_loss = []
        train_recon_loss = []
        train_recon_loss_weighted = []
        train_KLD = []

        SolarReconModel.train()
        epoch_time = time.time()
        for parm in SolarReconModel.vit_model.parameters():
            parm.requires_grad = False
        epoch_time = time.time()

        for i, data in enumerate(train_loader):
            data = data.to(device)
            if args.decoder_modal == 'magnet-0094':
                modal = data[:, 0, :, :, :] 
                modal = enhance_funciton(modal, args.enhance_list[0][0], args.enhance_list[0][1])
                modal_2 = data[:, 1, :, :, :]
                modal_2 = enhance_funciton(modal_2, args.enhance_list[1][0], args.enhance_list[1][1])
            elif args.decoder_modal == '0094-magnet':
                modal = data[:, 1, :, :, :] 
                modal = enhance_funciton(modal, args.enhance_list[1][0], args.enhance_list[1][1])
                modal_2 = data[:, 0, :, :, :]
                modal_2 = enhance_funciton(modal_2, args.enhance_list[0][0], args.enhance_list[0][1])
            elif args.decoder_modal == '0094-0094':
                modal = data[:, 1, :, :, :] 
                modal = enhance_funciton(modal, args.enhance_list[1][0], args.enhance_list[1][1])
                modal_2 = modal
            elif args.decoder_modal == 'magnet-magnet':
                modal = data[:, 0, :, :, :] 
                modal = enhance_funciton(modal, args.enhance_list[1][0], args.enhance_list[1][1])
                modal_2 = modal
            else:
                raise ValueError('Wrong decoder_modal')
            iteration_txt = f"Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            weights,_ = get_weights(args.weight_type, modal_2)
            recon, mu, logvar = SolarReconModel(modal)
            loss, recon_loss, recon_loss_weighted, KLD = SolarReconModel.loss_function(recon, modal_2, weights, mu, logvar)
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            train_loss.append(loss.item())
            train_recon_loss.append(recon_loss.item())
            train_recon_loss_weighted.append(recon_loss_weighted.item())
            train_KLD.append(KLD.item())
            logger_train_loss.append(loss.item())
            logger_train_recon_loss.append(recon_loss.item())
            logger_train_recon_loss_weighted.append(recon_loss_weighted.item())
            logger_train_KLD.append(KLD.item())
            logger_lr.append(scheduler.get_last_lr()[0])

            iteration_txt += f" Iteration time: {(time.time()-iter_time)/60:.2f} min |"
            iter_time = time.time()
            print(iteration_txt)
            # print(train_loss)

        scheduler.step()
        print(f'Epoch {epoch+1:>6}/{args.epochs:<6} | Train loss : {np.mean(train_loss):<10.8f} | Train recon loss : {np.mean(train_recon_loss):<10.8f} | Train recon loss weighted : {np.mean(train_recon_loss_weighted):<10.8f} | Train KLD : {np.mean(train_KLD):<10.8f} |')

        if (epoch+1) % test_epoch == 0: 
            with torch.no_grad():
                SolarReconModel.eval()

                val_loss = []
                val_recon_loss = []
                val_recon_loss_weighted = []
                val_KLD = []  

                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    if args.decoder_modal == 'magnet-0094':
                        modal = data[:, 0, :, :, :] 
                        modal = enhance_funciton(modal, args.enhance_list[0][0], args.enhance_list[0][1])
                        modal_2 = data[:, 1, :, :, :]
                        modal_2 = enhance_funciton(modal_2, args.enhance_list[1][0], args.enhance_list[1][1])
                    elif args.decoder_modal == '0094-magnet':
                        modal = data[:, 1, :, :, :] 
                        modal = enhance_funciton(modal, args.enhance_list[1][0], args.enhance_list[1][1])
                        modal_2 = data[:, 0, :, :, :]
                        modal_2 = enhance_funciton(modal_2, args.enhance_list[0][0], args.enhance_list[0][1])
                    elif args.decoder_modal == '0094-0094':
                        modal = data[:, 1, :, :, :] 
                        modal = enhance_funciton(modal, args.enhance_list[1][0], args.enhance_list[1][1])
                        modal_2 = modal
                    elif args.decoder_modal == 'magnet-magnet':
                        modal = data[:, 0, :, :, :] 
                        modal = enhance_funciton(modal, args.enhance_list[1][0], args.enhance_list[1][1])
                        modal_2 = modal
                    else:
                        raise ValueError('Wrong decoder_modal')

                    weights,_ = get_weights(args.weight_type, modal_2)
                    recon, mu, logvar = SolarReconModel(modal)
                    loss, recon_loss, recon_loss_weighted, KLD = SolarReconModel.loss_function(recon, modal_2, weights, mu, logvar)

                    val_loss.append(loss.item())
                    val_recon_loss.append(recon_loss.item())
                    val_recon_loss_weighted.append(recon_loss_weighted.item())
                    val_KLD.append(KLD.item())
                    
            logger_val_loss.append(np.mean(val_loss))
            logger_val_recon_loss.append(np.mean(val_recon_loss))
            logger_val_recon_loss_weighted.append(np.mean(val_recon_loss_weighted))
            logger_val_KLD.append(np.mean(val_KLD))
        
            result_txt = f'Epoch {epoch+1:>6}/{args.epochs:<6} | '
            result_txt += f'Train loss : {logger_train_loss[-1]:<10.8f} | '
            result_txt += f'Train recon loss : {logger_train_recon_loss[-1]:<10.8f} | '
            result_txt += f'Train recon loss weighted : {logger_train_recon_loss_weighted[-1]:<10.8f} | '
            result_txt += f'Train KLD : {logger_train_KLD[-1]:<10.8f} | '
            result_txt += f'Val loss : {logger_val_loss[-1]:<10.8f} | '
            result_txt += f'Val recon loss : {logger_val_recon_loss[-1]:<10.8f} | '
            result_txt += f'Val recon loss weighted : {logger_val_recon_loss_weighted[-1]:<10.8f} | '
            result_txt += f'Val KLD : {logger_val_KLD[-1]:<10.8f} | '
            
            print(result_txt)

            with open(f'{logger_checkpoint_path}logger_train_loss.pkl', 'wb') as f:
                pickle.dump(logger_train_loss, f)
            with open(f'{logger_checkpoint_path}logger_train_recon_loss.pkl', 'wb') as f:
                pickle.dump(logger_train_recon_loss, f)
            with open(f'{logger_checkpoint_path}logger_train_weight_mse.pkl', 'wb') as f:
                pickle.dump(logger_train_recon_loss_weighted, f)
            with open(f'{logger_checkpoint_path}logger_train_KLD.pkl', 'wb') as f:
                pickle.dump(logger_train_KLD, f)
            with open(f'{logger_checkpoint_path}logger_lr.pkl', 'wb') as f:
                pickle.dump(logger_lr, f)
            with open(f'{logger_checkpoint_path}logger_val_loss.pkl', 'wb') as f:
                pickle.dump(logger_val_loss, f)
            with open(f'{logger_checkpoint_path}logger_val_recon_loss.pkl', 'wb') as f:
                pickle.dump(logger_val_recon_loss, f)
            with open(f'{logger_checkpoint_path}logger_val_weight_mse.pkl', 'wb') as f:
                pickle.dump(logger_val_recon_loss_weighted, f)
            with open(f'{logger_checkpoint_path}logger_val_KLD.pkl', 'wb') as f:
                pickle.dump(logger_val_KLD, f)

        if (epoch+1) % save_epoch == 0:
            torch.save({'model': SolarReconModel.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch}, f'{model_checkpoint_path}epoch_{epoch}.pt')
            print(f'Model saved {(epoch+1)/args.epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')

if __name__ == '__main__':
    main()

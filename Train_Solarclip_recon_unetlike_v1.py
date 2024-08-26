import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import pickle
from types import SimpleNamespace
# import clip
# import clip.model

from Data import Solardataloader_subset
from Data.utils import transfer_date_to_id
from Data.Solardataloader import enhance_funciton
from Model.get_weights import get_weights

import os
import random
import time

from Model.SolarCLIP_modify import get_model_from_args, get_recon_unetlike_model_from_args
# from Model.vit import PretrainModel

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP Reconstruction model like Unet.')

    parser.add_argument('--config_dir', type=str, default='None')

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

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

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
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for load dataloader')
    parser.add_argument('--enhance_list', type=list, nargs="+", 
                        default=[['log1p', 1], ['log1p', 1]], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", 
                        default=[1024,0.5,90], help='Image preprocess list for training [resize, flip, rotate]')
    parser.add_argument('--weights_type_list', type=str, nargs="+",
                        default=['cv-rdbu', '3sgm-continous'], help='Weights type list for training')
    parser.add_argument('--embed_state_root', type=str, default='checkpoints/recon/0816_test1/SolarCLIP', help='Embedding state root path')
    parser.add_argument('--clip_model_id', type=int, required=True, help='select the model id for the first modal')

    # Save parameters
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/unetlike/train1", help="The output path to save the model.")

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

    checkpoint_path_mag = checkpoint_path + '/magnet/'
    logger_checkpoint_path_mag = checkpoint_path_mag + 'logger/'
    model_checkpoint_path_mag = checkpoint_path_mag + 'model/'
    if not os.path.exists(checkpoint_path_mag):
        os.makedirs(checkpoint_path_mag)
    if not os.path.exists(f'{model_checkpoint_path_mag}'):
        os.makedirs(f'{model_checkpoint_path_mag}')
    if not os.path.exists(f'{logger_checkpoint_path_mag}'):
        os.makedirs(f'{logger_checkpoint_path_mag}')

    checkpoint_path_0094 = checkpoint_path + '/0094/'
    logger_checkpoint_path_0094 = checkpoint_path_0094 + 'logger/'
    model_checkpoint_path_0094 = checkpoint_path_0094 + 'model/'
    if not os.path.exists(checkpoint_path_0094):
        os.makedirs(checkpoint_path_0094)
    if not os.path.exists(f'{model_checkpoint_path_0094}'):
        os.makedirs(f'{model_checkpoint_path_0094}')
    if not os.path.exists(f'{logger_checkpoint_path_0094}'):
        os.makedirs(f'{logger_checkpoint_path_0094}')

    save_args(args, checkpoint_path)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"Device: {device}")

    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                             start_date, end_date], modal_list=args.modal_list, load_imgs= True, enhance_list=args.image_preprocess, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list,  load_imgs= True, enhance_list=[args.image_preprocess[0],0,0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    # encoder_transformer_path = args.embed_state_root + '/model/'
    encoder_transformer_path = args.embed_state_root + '/'
    encoder_transformer_path = encoder_transformer_path + f'epoch_{args.clip_model_id}.pt'

    print('current_device', torch.cuda.current_device())  # 检查当前设备ID
    # SolarModel = get_model_from_args(args).to(device)
    SolarCLIP_dict = torch.load(encoder_transformer_path)['model']
    visual_mag_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_mag.')}
    visual_mag_state_dict = {k[len('visual_mag.'):]: v for k, v in visual_mag_state_dict.items()}
    # SolarModel.visual_mag.load_state_dict(torch.load(encoder_transformer_path)['model'])
    visual_0094_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_H.')}
    visual_0094_state_dict = {k[len('visual_H.'):]: v for k, v in visual_0094_state_dict.items()}

    SolarReconModel_mag = get_recon_unetlike_model_from_args(args).to(device)
    SolarReconModel_mag.vit_model.load_state_dict(visual_mag_state_dict)

    SolarReconModel_0094 = get_recon_unetlike_model_from_args(args).to(device)
    SolarReconModel_0094.vit_model.load_state_dict(visual_0094_state_dict)
    # SolarReconModel = nn.DataParallel(SolarReconModel, device_ids=[2,3])
    # del SolarModel

    optimizer_mag = torch.optim.SGD(
        SolarReconModel_mag.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler_mag = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_mag, T_max=args.epochs)
    
    optimizer_0094 = torch.optim.SGD(
        SolarReconModel_0094.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler_0094 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_0094, T_max=args.epochs)

    epochs = args.epochs
    test_epoch = epochs//args.test_freq
    save_epoch = epochs//args.save_freq

    logger_train_mse_mag = []
    logger_train_weight_mse_mag = []  
    logger_lr_mag = []
    logger_val_weight_mse_mag = []
    logger_val_mse_mag = []
    logger_train_mse_0094 = []
    logger_train_weight_mse_0094 = []
    logger_lr_0094 = []
    logger_val_weight_mse_0094 = []
    logger_val_mse_0094 = []

    start_time = time.time()
    iter_time = time.time()
    # min_weight_mse_val = 0
    # stop_count = 0
    print(f'Start training SolarReconModel')
    
    for epoch in range(epochs):
        SolarReconModel_mag.train()
        SolarReconModel_0094.train()
        epoch_time = time.time()
        for parm in SolarReconModel_mag.vit_model.parameters():
            parm.requires_grad = False
        for parm in SolarReconModel_0094.vit_model.parameters():
            parm.requires_grad = False
        for i, data in enumerate(train_loader):
            data = data.to(device)
            modal_mag = data[:, 0, :, :, :] 
            modal_0094 = data[:, 1, :, :, :]
            modal_mag = enhance_funciton(modal_mag, args.enhance_list[0][0], args.enhance_list[0][1])
            modal_0094 = enhance_funciton(modal_0094, args.enhance_list[1][0], args.enhance_list[1][1])
            iteration_txt = f"Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            weights_mag,_ = get_weights(args.weight_type_list[0], modal_mag)
            mse_weight_mag, mse_mag = SolarReconModel_mag.calculate_loss(modal_mag, weights_mag)
            weights_0094,_ = get_weights(args.weight_type_list[1], modal_0094)
            mse_weight_0094, mse_0094 = SolarReconModel_0094.calculate_loss(modal_0094, weights_0094)
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer_mag.zero_grad()
            mse_weight_mag.backward()
            optimizer_mag.step()
            optimizer_0094.zero_grad()
            mse_weight_0094.backward()
            optimizer_0094.step()
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            logger_train_weight_mse_mag.append(mse_weight_mag.item())
            logger_train_mse_mag.append(mse_mag.item())
            logger_lr_mag.append(scheduler_mag.get_last_lr()[0])
            logger_train_weight_mse_0094.append(mse_weight_0094.item())
            logger_train_mse_0094.append(mse_0094.item())
            logger_lr_0094.append(scheduler_0094.get_last_lr()[0])

            iteration_txt += f" Iteration time: {(time.time()-iter_time)/60:.2f} min |"
            iter_time = time.time()
            print(iteration_txt)

        scheduler_mag.step()
        scheduler_0094.step()

        if (epoch+1) % test_epoch == 0: 
            with torch.no_grad():
                SolarReconModel_mag.eval()
                SolarReconModel_0094.eval()
                weight_mse_results_mag = []
                mse_results_mag = []
                weight_mse_results_0094 = []
                mse_results_0094 = []  
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    modal_mag = data[:, 0, :, :, :] 
                    modal_0094 = data[:, 1, :, :, :]
                    modal_mag = enhance_funciton(modal_mag, args.enhance_list[0][0], args.enhance_list[0][1])
                    modal_0094 = enhance_funciton(modal_0094, args.enhance_list[1][0], args.enhance_list[1][1])

                    weights_mag,_ = get_weights(args.weight_type_list[0], modal_mag)
                    mse_weight_mag, mse_mag = SolarReconModel_mag.calculate_loss(modal_mag, weights_mag)
                    weights_0094,_ = get_weights(args.weight_type_list[1], modal_0094)
                    mse_weight_0094, mse_0094 = SolarReconModel_0094.calculate_loss(modal_0094, weights_0094)

                    weight_mse_results_mag.append(mse_weight_mag.item())
                    mse_results_mag.append(mse_mag.item())
                    logger_lr_mag.append(scheduler_mag.get_last_lr()[0])
                    weight_mse_results_0094.append(mse_weight_0094.item())
                    mse_results_0094.append(mse_0094.item())
                    logger_lr_0094.append(scheduler_0094.get_last_lr()[0])

            logger_val_weight_mse_mag.append(np.mean(weight_mse_results_mag))
            logger_val_mse_mag.append(np.mean(mse_results_mag))
            logger_val_weight_mse_0094.append(np.mean(weight_mse_results_0094))
            logger_val_mse_0094.append(np.mean(mse_results_0094))

            result_txt = f'Epoch {epoch+1:>6}/{args.epochs:<6} | '
            result_txt += f'Train weight mse magnet {logger_train_weight_mse_mag[-1]:<10.8f} | '
            result_txt += f'Train mse magnet {logger_train_mse_mag[-1]:<10.8f} | '
            result_txt += f'Val weight mse magnet {logger_val_weight_mse_mag[-1]:<10.8f} | '
            result_txt += f'Val mse magnet {logger_val_weight_mse_mag[-1]:<10.8f} | '
            result_txt += f'Train weight mse 0094 {logger_train_weight_mse_0094[-1]:<10.8f} | '
            result_txt += f'Train mse 0094 {logger_train_mse_0094[-1]:<10.8f} | '
            result_txt += f'Val weight mse 0094 {logger_val_weight_mse_0094[-1]:<10.8f} | '
            result_txt += f'Val mse 0094 {logger_val_weight_mse_0094[-1]:<10.8f} | '
            result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
            print(result_txt)

            with open(f'{logger_checkpoint_path_mag}logger_train_weight_mse.pkl', 'wb') as f:
                pickle.dump(logger_train_weight_mse_mag, f)
            with open(f'{logger_checkpoint_path_mag}logger_train_mse.pkl', 'wb') as f:
                pickle.dump(logger_train_mse_mag, f)
            with open(f'{logger_checkpoint_path_mag}logger_lr.pkl', 'wb') as f:
                pickle.dump(logger_lr_mag, f)
            with open(f'{logger_checkpoint_path_mag}logger_val_weight_mse.pkl', 'wb') as f:
                pickle.dump(logger_val_weight_mse_mag, f)
            with open(f'{logger_checkpoint_path_mag}logger_val_mse.pkl', 'wb') as f:
                pickle.dump(logger_val_mse_mag, f)

            with open(f'{logger_checkpoint_path_0094}logger_train_weight_mse.pkl', 'wb') as f:
                pickle.dump(logger_train_weight_mse_0094, f)
            with open(f'{logger_checkpoint_path_0094}logger_train_mse.pkl', 'wb') as f:
                pickle.dump(logger_train_mse_0094, f)
            with open(f'{logger_checkpoint_path_0094}logger_lr.pkl', 'wb') as f:
                pickle.dump(logger_lr_0094, f)
            with open(f'{logger_checkpoint_path_0094}logger_val_weight_mse.pkl', 'wb') as f:
                pickle.dump(logger_val_weight_mse_0094, f)
            with open(f'{logger_checkpoint_path_0094}logger_val_mse.pkl', 'wb') as f:
                pickle.dump(logger_val_mse_0094, f)


            # save best model, early stop
            # if logger_val_weight_mse[-1] < min_weight_mse_val:
            #     stop_count = 0
            #     min_weight_mse_val = logger_val_weight_mse[-1]
            #     torch.save({'model': SolarReconModel.state_dict(), 'optimizer': optimizer.state_dict(),
            #                 'scheduler': scheduler.state_dict(), 'epoch': epoch},
            #                 f'{model_checkpoint_path}best.pt')
            #     print(f'Model saved best, cost {(time.time()-start_time)/60:.2f} min\n')
            #     start_time = time.time()
            # else:
            #     stop_count += 1
            #     if stop_count > args.early_stop:
            #         print(f'Early stop at epoch {epoch+1}, best weighted mse {min_weight_mse_val:<7.2%}')
            #         break
            
        if (epoch+1) % save_epoch == 0:
            torch.save({'model': SolarReconModel_mag.state_dict(), 'optimizer': optimizer_mag.state_dict(),
                        'scheduler': scheduler_mag.state_dict(), 'epoch': epoch}, f'{model_checkpoint_path_mag}epoch_{epoch}.pt')
            torch.save({'model': SolarReconModel_0094.state_dict(), 'optimizer': optimizer_0094.state_dict(),
                        'scheduler': scheduler_0094.state_dict(), 'epoch': epoch}, f'{model_checkpoint_path_0094}epoch_{epoch}.pt')
            print(f'Model saved {(epoch+1)/args.epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')


if __name__ == '__main__':
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for using PyTorch DDP.
"""
import math

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
import time
import argparse
import os
import pickle
import json

from Model.SolarReconModels.without_CLIP.SolarReconModel_VAE_without_CLIP import get_SolarReconModel_VAE_without_CLIP_from_args
from Model.get_weights import get_weights
from Data import Solardataloader_subset
from Data.utils import transfer_date_to_id
from Data.Solardataloader import enhance_funciton

import matplotlib.pyplot as plt
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def get_lambda_kl(args, current_epoch):
        return 0.5 * args.lambda_ratio * (1 + math.cos(math.pi * current_epoch / args.epochs))

@torch.no_grad()
def hook_fn(module, input, output):
    rank = dist.get_rank()
    mem_allocated = torch.cuda.memory_allocated() / 1e6  # MB
    mem_reserved = torch.cuda.memory_reserved() / 1e6    # MB
    print(f"[Rank {rank}] Module: {module.__class__.__name__}, Allocated: {mem_allocated} MB, Reserved: {mem_reserved} MB")

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP model.')

    parser.add_argument('--config_dir', type=str, default='None')

    # Training parameters
    parser.add_argument('--global_batch_size', type=int,
                        default=6, help='Global batch size for training')
    parser.add_argument('--global_seed', type=int,
                        default=42, help='Global seed for training')
    parser.add_argument('--optimizer', type=str, 
                        default='AdamW',help='Optimizer for training')
    parser.add_argument('--learning_rate', type=float,
                        default=4e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, 
                        default=1000,help='Number of training epochs')
    parser.add_argument('--test_freq', type=int, default=100,
                        help='Frequency of testing the model')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Frequency of saving the model')
    # parser.add_argument('--device', type=str,
    #                     default='cuda:0', help='Device for training')
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/VAE_wituout_CLIP/bottleneck_80", help="The output path to save the model.")

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='Number of data loading workers')
    parser.add_argument('--machine', type=str, 
                        default='A40',help='Machine type for training')
    parser.add_argument('--sampler', type=str, 
                        default='distributed',help='Sampler for training')

    # Model parameters
    parser.add_argument('--loss_type', type=str, 
                        default='MSE', help='Type of VAE used in the model')
    parser.add_argument('--input_size', type=int, 
                        default=1024, help='Input size of the model')
    parser.add_argument('--image_channels', type=int, 
                        default=1, help='Number of input channels')
    parser.add_argument('--hidden_dim', type=int,
                        default=128, help='Dimension of the hidden layer')
    parser.add_argument('--layers', type=int,
                        default=3, help='Number of encoder or decoder layers in the model')
    parser.add_argument('--kernel_sizes', type=int, nargs="+",
                        default=[7, 7, 7], help='Kernel sizes for the encoder or decoder')
    parser.add_argument('--strides', type=int, nargs="+",
                        default=[4, 4, 4], help='Strides for the encoder or decoder')
    parser.add_argument('--num_groups', type=int,
                        default=16, help='Number of groups channels are divided into')
    parser.add_argument('--latent_dim', type=int,
                        default=3, help='Dimension of the latent space')
    parser.add_argument('--lambda_ratio', type=float,
                        default=0.1, help='Ratio of KLD loss to reconstruction loss')
    parser.add_argument('--Annealing', type=str,
                        default='None', help='Annealing method for KLD loss')

    # Modal parameters
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')
    parser.add_argument('--enhance_list', type=list, nargs="+", 
                        default=[['log1p', 1], ['log1p', 1]], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", 
                        default=[1024,0.5,90], help='Image preprocess list for training [resize, flip, rotate]')
    parser.add_argument('--not_use_weight', 
                        action='store_true', help='Use weight for training')
    parser.add_argument('--weight_type_list', type=list, nargs="+", 
                        default=['cv-rdbu', '3sgm-continous'], help='Weight type list for training')
    parser.add_argument('--decoder_modal', type=str, 
                        default='magnet-magnet', help='Decoder modal for training')

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
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a VAE pretrained model for DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    dist.init_process_group("nccl")
    print(f"global batch size: {args.global_batch_size}")
    print(f"World size: {dist.get_world_size()}")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    print(f"Rank={rank}, device={device}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.checkpoint_path, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        checkpoint_path = f"{args.checkpoint_path}/{args.decoder_modal}/" + "layers_" + str(args.layers) + "_kernel_" + str(args.kernel_sizes) + "_strides_" + str(args.strides)
        checkpoint_path = f"{checkpoint_path}/" + "hidden_dim_" + str(args.hidden_dim)
        checkpoint_path = f"{checkpoint_path}/" + "lambda_ratio_" + str(args.lambda_ratio) + args.Annealing + "weight_" + str(bool(1-args.not_use_weight))
        logger_checkpoint_path = checkpoint_path + "/logger"
        model_checkpoint_path = checkpoint_path + "/model"
        visulization_checkpoint_path = checkpoint_path + "/visulization"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(f'{model_checkpoint_path}'):
            os.makedirs(f'{model_checkpoint_path}')
        if not os.path.exists(f'{logger_checkpoint_path}'):
            os.makedirs(f'{logger_checkpoint_path}')
        if not os.path.exists(f'{visulization_checkpoint_path}'):
            os.makedirs(f'{visulization_checkpoint_path}')
        save_args(args, checkpoint_path)
        print(f"Saving checkpoints to {checkpoint_path}")

    # Setup data:
    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12*160,time_interval=[
                                                             start_date, end_date], modal_list=args.modal_list, load_imgs= True, enhance_list=args.image_preprocess, batch_size=args.global_batch_size//dist.get_world_size(), shuffle=False, num_workers=args.num_workers,
                                                             sampler=args.sampler)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12*160,time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list,  load_imgs= True, enhance_list=[args.image_preprocess[0],0,0], batch_size=args.global_batch_size//dist.get_world_size(), shuffle=False, num_workers=args.num_workers,
                                                           sampler=args.sampler)
    print(f"[Rank {rank}] DataLoader time: {(time.time()-start_time)/60:.2f} min")

    # Create model:
    assert args.input_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    cuda_memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
    model = get_SolarReconModel_VAE_without_CLIP_from_args(args)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    print(f"[Rank {rank}] Model cuda memory allocated: {torch.cuda.memory_allocated()/1e6-cuda_memory_allocated:.2f} MB")
    print(f"VAE Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"VAE Paramenters memory: {sum(p.numel() * p.element_size() for p in model.parameters())/ (1024*1024):.2f} MB")

    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not support.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

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

    test_epoch = args.epochs//args.test_freq
    save_epoch = args.epochs//args.save_freq

    print(f"[Rank {rank}] Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"[Rank {rank}] Beginning epoch {epoch+1}...")
        model.train()
        train_loss = []
        train_recon_loss = []
        train_recon_loss_weighted = []
        train_KLD = []
        for i, data in enumerate(train_loader):
            epoch_time = time.time()
            iter_time = time.time()
            data = data.to(device)
            if args.decoder_modal == 'magnet-magnet':
                x = data[:, 0, :, :, :]
                x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                y = x
                if not args.not_use_weight:
                    weights,_ = get_weights(args.weight_type_list[0], y)
                else:
                    weights = torch.ones_like(y)
                print('sum of weights:',torch.sum(weights))
                print('num of weights:',torch.numel(weights))
                print('elements of weights:',weights[0,0,0,:10])

            elif args.decoder_modal == 'magnet-0094': # todo
                x = data[:, 0, :, :, :]
                x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                y = data[:, 1, :, :, :]
                y = enhance_funciton(y, args.enhance_list[1][0], args.enhance_list[1][1])
                weights,_ = get_weights(args.weight_type_list[1], y)
            elif args.decoder_modal == '0094-0094':
                x = data[:, 1, :, :, :]
                x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                weights,_ = get_weights(args.weight_type_list[1], x)
            elif args.decoder_modal == '0094-magnet': # todo
                pass
            else:
                raise ValueError(f"Decoder modal {args.decoder_modal} not support.")
            iteration_txt = f"[Rank {rank}] Iteration {i} | Data time: {(time.time()-epoch_time)/60:.4f} min |"
            epoch_time = time.time()
            
            recon_x, mu, logvar = model(x)
            if args.Annealing == 'Cosine':
                lambda_kl = get_lambda_kl(args, epoch)
            elif args.Annealing == 'None':
                lambda_kl = args.lambda_ratio
            loss, recon_loss, recon_loss_weighted, KLD = model.module.loss_function(recon_x, y, weights, mu, logvar, lambda_kl)
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.4f} min |"
            epoch_time = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.4f} min |"
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
            iteration_txt += f" Logger time: {(time.time()-epoch_time)/60:.4f} min |"
            epoch_time = time.time()

            iteration_txt += f" Iteration time: {(time.time()-iter_time)/60:.4f} min |"
            iter_time = time.time()
            print(iteration_txt)

            del recon_x, mu, logvar, recon_loss, recon_loss_weighted, KLD
            del x, y, weights
            torch.cuda.empty_cache()

        scheduler.step()

        print(f"[Rank {rank}] | Epoch {epoch+1} | Train loss {np.mean(train_loss):.8f} | Train recon loss {np.mean(train_recon_loss):.8f} | Train recon loss weighted {np.mean(train_recon_loss_weighted):.8f} | Train KLD {np.mean(train_KLD):.8f} | KL weight {lambda_kl:.8f}")

        first_batch = True
        if (epoch+1) % test_epoch == 0:  
            with torch.no_grad():
                model.eval()
                val_loss = []
                val_recon_loss = []
                val_recon_loss_weighted = []
                val_KLD = []
                print(f"Beginning epoch {epoch}...")
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    if args.decoder_modal == 'magnet-magnet':
                        x = data[:, 0, :, :, :]
                        x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                        y = x
                        if not args.not_use_weight:
                            weights,_ = get_weights(args.weight_type_list[0], y)
                        else:
                            weights = torch.ones_like(y)
                        z = data[:, 1, :, :, :]
                        z = enhance_funciton(z, args.enhance_list[1][0], args.enhance_list[1][1])
                    elif args.decoder_modal == 'magnet-0094': # todo
                        x = data[:, 0, :, :, :]
                        x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                        y = data[:, 1, :, :, :]
                        y = enhance_funciton(y, args.enhance_list[1][0], args.enhance_list[1][1])
                        weights,_ = get_weights(args.weight_type_list[1], y)
                    elif args.decoder_modal == '0094-0094':
                        x = data[:, 1, :, :, :]
                        x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                        weights,_ = get_weights(args.weight_type_list[1], x)
                    elif args.decoder_modal == '0094-magnet': # todo
                        pass
                    else:
                        raise ValueError(f"Decoder modal {args.decoder_modal} not support.")
                        
                    recon_x, mu, logvar = model(x)
                    if args.Annealing == 'Cosine':
                        lambda_kl = get_lambda_kl(args, epoch)
                    elif args.Annealing == 'None':
                        lambda_kl = args.lambda_ratio
                    loss, recon_loss, recon_loss_weighted, KLD = model.module.loss_function(recon_x, y, weights, mu, logvar, lambda_kl)

                    if first_batch:
                        recon_image= recon_x.cpu().numpy()
                        modal = x.cpu().numpy()
                        modal_y = y.cpu().numpy()
                        modal_z = z.cpu().numpy()
                        if args.decoder_modal == 'magnet-0094':
                            mag_image = modal
                            h_image = modal_y
                        elif args.decoder_modal == 'magnet-magnet':
                            mag_image = modal
                            h_image = modal_z
                        else:
                            raise ValueError('Decoder modal not supported')
                        vmin_magnet = np.min(mag_image)
                        vmax_magnet = np.max(mag_image)
                        vmax_magnet = np.max([np.abs(vmin_magnet), np.abs(vmax_magnet)])/2
                        vmin_magnet = -vmax_magnet
                        vmin_0094 = 0
                        vmax_0094 = np.max(h_image)

                        plt.figure(figsize=(30, 12))
                        if args.decoder_modal == 'magnet-0094':
                            for i in range(5):
                                plt.subplot(3, 5, i+1)
                                plt.imshow(h_image[i,0,:,:], cmap='Reds',vmin=vmin_0094, vmax=vmax_0094)
                                plt.axis('off')
                                plt.title('0094 original')

                                plt.subplot(3, 5, i+6)
                                plt.imshow(recon_image[i,0,:,:], cmap='Reds',vmin=vmin_0094, vmax=vmax_0094)
                                plt.axis('off')
                                plt.title('0094 recon')

                                plt.subplot(3, 5, i+11)
                                plt.imshow(mag_image[i,0,:,:], cmap='RdBu_r',vmin=vmin_magnet, vmax=vmax_magnet)
                                plt.axis('off')
                                plt.title('Magnet orginal')
                        elif args.decoder_modal == 'magnet-magnet':
                            for i in range(5):
                                plt.subplot(3, 5, i+1)
                                plt.imshow(h_image[i,0,:,:], cmap='Reds',vmin=vmin_0094, vmax=vmax_0094)
                                plt.axis('off')
                                plt.title('0094 original')

                                plt.subplot(3, 5, i+6)
                                plt.imshow(recon_image[i,0,:,:], cmap='RdBu_r',vmin=vmin_magnet, vmax=vmax_magnet)
                                plt.axis('off')
                                plt.title('Magnet recon')

                                plt.subplot(3, 5, i+11)
                                plt.imshow(mag_image[i,0,:,:], cmap='RdBu_r',vmin=vmin_magnet, vmax=vmax_magnet)
                                plt.axis('off')
                                plt.title('Magnet orginal')
                        plt.savefig(f'{visulization_checkpoint_path}/reconstruction_comparison_epoch_{epoch+1}.png', bbox_inches='tight')
                        first_batch = False
                        
                    val_loss.append(loss.item())
                    val_recon_loss.append(recon_loss.item())
                    val_recon_loss_weighted.append(recon_loss_weighted.item())
                    val_KLD.append(KLD.item())
                
                logger_val_loss.append(np.mean(val_loss))
                logger_val_recon_loss.append(np.mean(val_recon_loss))
                logger_val_recon_loss_weighted.append(np.mean(val_recon_loss_weighted))
                logger_val_KLD.append(np.mean(val_KLD))

                result_txt = f'[Rank {rank}] Epoch {epoch+1:>6}/{args.epochs:<6} | '
                result_txt += f'Train loss {logger_train_loss[-1]:<10.8f} | '
                result_txt += f'Train recon loss {logger_train_recon_loss[-1]:<10.8f} | '
                result_txt += f'Train recon loss weighted {logger_train_recon_loss_weighted[-1]:<10.8f} | '
                result_txt += f'Train KLD {logger_train_KLD[-1]:<10.8f} | '
                result_txt += f'Val loss {logger_val_loss[-1]:<10.8f} | '
                result_txt += f'Val recon loss {logger_val_recon_loss[-1]:<10.8f} | '
                result_txt += f'Val recon loss weighted {logger_val_recon_loss_weighted[-1]:<10.8f} | '
                result_txt += f'Val KLD {logger_val_KLD[-1]:<10.8f} | '
                result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
                print(result_txt)
                if rank == 0:
                    with open(f'{logger_checkpoint_path}/logger_train_loss.pkl', 'wb') as f:
                        pickle.dump(logger_train_loss, f)
                    with open(f'{logger_checkpoint_path}/logger_train_recon_loss.pkl', 'wb') as f:
                        pickle.dump(logger_train_recon_loss, f)
                    with open(f'{logger_checkpoint_path}/logger_train_recon_loss_weighted.pkl', 'wb') as f:
                        pickle.dump(logger_train_recon_loss_weighted, f)
                    with open(f'{logger_checkpoint_path}/logger_train_KLD.pkl', 'wb') as f:
                        pickle.dump(logger_train_KLD, f)
                    with open(f'{logger_checkpoint_path}/logger_lr.pkl', 'wb') as f:
                        pickle.dump(logger_lr, f)
                    with open(f'{logger_checkpoint_path}/logger_val_loss.pkl', 'wb') as f:
                        pickle.dump(logger_val_loss, f)
                    with open(f'{logger_checkpoint_path}/logger_val_recon_loss.pkl', 'wb') as f:
                        pickle.dump(logger_val_recon_loss, f)
                    with open(f'{logger_checkpoint_path}/logger_val_recon_loss_weighted.pkl', 'wb') as f:
                        pickle.dump(logger_val_recon_loss_weighted, f)
                    with open(f'{logger_checkpoint_path}/logger_val_KLD.pkl', 'wb') as f:
                        pickle.dump(logger_val_KLD, f)

        if (epoch+1) % save_epoch == 0:  # todo
            # Save VAE checkpoint:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                }
                checkpoint_path = f"{model_checkpoint_path}/epoch_{epoch+1}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    print("Done!")
    cleanup()

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters). 
    args = parse_args()
    _ = args.config_dir
    if args.config_dir != 'None':
        args = load_args_from_json(args, args.config_dir)
    args.config_dir = _
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    main(args)

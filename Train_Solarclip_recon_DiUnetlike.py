# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
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

from diffusers.models import AutoencoderKL
import json

from Model.models import get_VAE_model_from_args
from Model.diffusion import create_diffusion
from Model.get_weights import get_weights
from Modules.diffusion_modules.denoisingmodels.unet.unet import get_DiUNet_model_from_args
from Data import Solardataloader_subset
from Data.utils import transfer_date_to_id
from Data.Solardataloader import enhance_funciton

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def get_last_layer_grad_norm(model):
    last_layer_params = list(model.parameters())[-1]
    if last_layer_params.grad is None:
        print("No gradients found for the last layer.")
        return None
    grad_norm = last_layer_params.grad.norm().item()
    return grad_norm

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP model.')

    parser.add_argument('--config_dir', type=str, 
                        default='None')
    parser.add_argument('--vae_config_dir', type=str,
                        default='/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/DiTlike/VAEpretrain/0094-0094/_0912_res_true_attn_flase/args.json', help='The path to the VAE model.')
    parser.add_argument('--vae_model_id', type=int,
                        default=240, help='The id of the VAE model.')
    parser.add_argument("--clip_model_path", type=str,
                        default="checkpoints/recon/0816_test1/SolarCLIP", help="The path to the SolarCLIP model.")
    parser.add_argument("--clip_model_id", type=int,
                        default=200, help="The id of the SolarCLIP model.")

    # Training parameters
    parser.add_argument('--global_batch_size', type=int,
                        default=90, help='Global batch size for training')
    parser.add_argument('--global_seed', type=int,
                        default=42, help='Global seed for training')
    parser.add_argument('--optimizer', type=str, 
                        default='AdamW', help='Optimizer for training')
    parser.add_argument('--learning_rate', type=float,
                        default=3e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, 
                        default=1000, help='Number of training epochs')
    parser.add_argument('--test_freq', type=int, 
                        default=100, help='Frequency of testing the model')
    parser.add_argument('--save_freq', type=int,
                        default=100, help='Frequency of saving the model')
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/DiUnetlike", help="The output path to save the model.")

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='Number of data loading workers')
    parser.add_argument('--machine', type=str, 
                        default='A40', help='Machine type for training')
    parser.add_argument('--sampler', type=str, 
                        default='distributed',help='Sampler for training')

    # UNet Model parameters
    parser.add_argument('--input_size', type=int, 
                        default=64, help='Input size of the model')
    parser.add_argument('--dim', type=int, 
                        default=64, help='Hidden channel dimension of the model')
    parser.add_argument('--init_dim', type=int,
                        default=None, help='Initial hidden channel dimension of the model')
    parser.add_argument('--out_dim', type=int,
                        default=None, help='Output channel dimension of the model')
    parser.add_argument('--dim_mults', type=tuple,
                        default=(1, 2, 4, 8), help='Hidden channel dimension multipliers of the model')
    parser.add_argument('--channels', type=int,
                        default=3, help='Number of input and output channels')
    parser.add_argument('--self_condition', type=bool,
                        default=True, help='Flag to indicate if the model is self-conditioned')
    parser.add_argument('--learned_variance', type=bool,
                        default=True, help='Flag to indicate if the model learns variance')
    parser.add_argument('--learned_sinusoidal_cond', type=bool,
                        default=False, help='Flag to indicate if the time embeddings learns sinusoidal condition')
    parser.add_argument('--random_fourier_features', type=bool,
                        default=False, help='Flag to indicate if the time embeddings uses random Fourier features')
    parser.add_argument('--learned_sinusoidal_dim', type=int,
                        default=64, help='Dimension of the time embeddings learned sinusoidal condition')
    parser.add_argument('--sinusoidal_pos_emb_theta', type=int,
                        default=10000, help='Theta of the sinusoidal position embeddings')
    parser.add_argument('--dropout', type=float,
                        default=0.1, help='Dropout rate for the model')
    parser.add_argument('--attn_dim_head', type=int,
                        default=32, help='Attention dimension head of the model')
    parser.add_argument('--attn_heads', type=int,
                        default=4, help='Number of attention heads in the model')
    parser.add_argument('--full_attn', type=bool,
                        default=None, help='Flag to indicate if the model uses full attention')
    parser.add_argument('--flash_attn', type=bool,
                        default=False, help='Flag to indicate if the model uses flash attention')
    
    # Modal parameters
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')
    parser.add_argument('--enhance_list', type=list, nargs="+", 
                        default=[['log1p', 1], ['log1p', 1]], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", 
                        default=[1024,0.5,90], help='Image preprocess list for training [resize, flip, rotate]')
    parser.add_argument('--weight_type_list', type=list, nargs="+", 
                        default=['cv-rdbu', '3sgm-continous'], help='Weight type list for training')
    parser.add_argument('--decoder_modal', type=str, 
                        default='0094-0094', help='Decoder modal for training')

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

def main(args, vae_args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.checkpoint_path, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        checkpoint_path = f"{args.checkpoint_path}/" + args.decoder_modal
        checkpoint_path = f"{checkpoint_path}/" + 'use_vae_lambda_1e-1_' + args.optimizer
        logger_checkpoint_path = checkpoint_path + "/logger"
        model_checkpoint_path = checkpoint_path + "/model"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(f'{model_checkpoint_path}'):
            os.makedirs(f'{model_checkpoint_path}')
        if not os.path.exists(f'{logger_checkpoint_path}'):
            os.makedirs(f'{logger_checkpoint_path}')
        for arg in vars(args):
            print(f"DIT args {arg:<30}: {getattr(args, arg)}")
        for arg in vars(vae_args):
            print(f"VAE args {arg:<30}: {getattr(vae_args, arg)}")
        save_args(args, checkpoint_path)
        print(f"Saving checkpoints to {checkpoint_path}")
    # Setup data:
    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12*10,time_interval=[
                                                             start_date, end_date], modal_list=args.modal_list, load_imgs= True, enhance_list=args.image_preprocess, batch_size=args.global_batch_size//dist.get_world_size(), shuffle=False, num_workers=args.num_workers,
                                                             sampler=args.sampler)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12*10,time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list,  load_imgs= True, enhance_list=[args.image_preprocess[0],0,0], batch_size=args.global_batch_size//dist.get_world_size(), shuffle=False, num_workers=args.num_workers,
                                                           sampler=args.sampler)
    print(f"[Rank {rank}] DataLoader time: {(time.time()-start_time)/60:.2f} min")

    # Create model:
    clip_model_path = f"{args.clip_model_path}/epoch_{args.clip_model_id}.pt"
    SolarCLIP_dict = torch.load(clip_model_path, weights_only=True)['model']
    visual_mag_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_mag.')}
    visual_mag_state_dict = {k[len('visual_mag.'):]: v for k, v in visual_mag_state_dict.items()}
    visual_0094_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_H.')}
    visual_0094_state_dict = {k[len('visual_H.'):]: v for k, v in visual_0094_state_dict.items()}

    assert args.input_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    model = get_DiUNet_model_from_args(args)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    vae = get_VAE_model_from_args(vae_args).to(device)
    vae.train()
    vae_checkpoint_path = os.path.dirname(args.vae_config_dir)
    vae_model_path = f"{vae_checkpoint_path}/model/epoch_{args.vae_model_id}.pt"
    vae.load_state_dict(torch.load(vae_model_path, map_location=torch.device(f'cuda:{device}'))['model'])
    requires_grad(vae, False)

    print(f"[Rank {rank}] | DiUnet Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not support.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    logger_train_loss = []
    logger_train_mse = []
    logger_train_vb = []
    logger_train_last_layer_norm = []
    logger_lr = []
    logger_val_loss = []
    logger_val_mse = []
    logger_val_vb = []

    start_time = time.time()

    test_epoch = args.epochs//args.test_freq
    save_epoch = args.epochs//args.save_freq

    print(f"[Rank {rank}] | Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"[Rank {rank}] | Beginning epoch {epoch}...")
        train_loss = []
        train_mse = []
        train_vb = []
        train_last_layer_norm = []
        for i, data in enumerate(train_loader):
            epoch_time = time.time()
            iter_time = time.time()
            data = data.to(device)
            if args.decoder_modal == 'magnet-magnet':
                x = data[:, 0, :, :, :]
                x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                # weights,_ = get_weights(args.weight_type_list[0], x)
                model.module.cond_vit.load_state_dict(visual_mag_state_dict)
                requires_grad(model.module.condition_pre_emb, False)
                y = model.module.condition_pre_emb(x)
            elif args.decoder_modal == 'magnet-0094': # todo
                pass
                # x = data[:, 0, :, :, :]
                # x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                # model.y_embedder.load_state_dict(visual_0094_state_dict)
                # requires_grad(model.y_embedder, False)
                # y = model.y_embedder(x)
            elif args.decoder_modal == '0094-0094':
                x = data[:, 1, :, :, :]
                x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                weights,_ = get_weights(args.weight_type_list[1], x)
                model.module.cond_vit.load_state_dict(visual_0094_state_dict)
                requires_grad(model.module.condition_pre_emb, False)
                y = model.module.condition_pre_emb(x)
            elif args.decoder_modal == '0094-magnet': # todo
                # x = data[:, 1, :, :, :]
                # x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                # model.y_embedder.load_state_dict(visual_mag_state_dict)
                # requires_grad(model.y_embedder, False)
                # y = model.y_embedder(x)
                pass
            else:
                raise ValueError(f"Decoder modal {args.decoder_modal} not support.")
            iteration_txt = f"[Rank {rank}] | Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()
                
            with torch.no_grad():
                x = vae.sample(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(x_self_cond=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer.zero_grad()
            loss.backward()
            last_layer_params = list(model.module.parameters())[-1:]
            torch.nn.utils.clip_grad_norm_(last_layer_params, 4e-3)
            optimizer.step()
            update_ema(ema, model.module)
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            grad_norm = get_last_layer_grad_norm(model.module)
            if grad_norm is not None:
                train_last_layer_norm.append(grad_norm)
                logger_train_last_layer_norm.append(grad_norm)
            train_loss.append(loss.item())
            train_mse.append(loss_dict["mse"].mean().item())
            train_vb.append(loss_dict["vb"].mean().item())
            logger_train_loss.append(loss.item())
            logger_train_mse.append(loss_dict["mse"].mean().item())
            logger_train_vb.append(loss_dict["vb"].mean().item())
            logger_lr.append(scheduler.get_last_lr()[0])
            iteration_txt += f" Logger time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            iteration_txt += f" Iteration time: {(time.time()-iter_time)/60:.2f} min |"
            iter_time = time.time()
            print(iteration_txt)

        scheduler.step()
        print(f"[Rank {rank}] | Epoch {epoch+1} | Train loss {np.mean(train_loss):<10.8f} | Train MSE {np.mean(train_mse):<10.8f} | Train VB {np.mean(train_vb):<10.8f} | Train last layer grad norm {np.mean(train_last_layer_norm):<10.8f}")

        if (epoch+1) % test_epoch == 0:  
            with torch.no_grad():
                model.eval()
                val_loss = []
                val_mse = []
                val_vb = []
                print(f"[Rank {rank}] | | Beginning Test epoch {epoch}...")
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    if args.decoder_modal == 'magnet-magnet':
                        x = data[:, 0, :, :, :]
                        x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                        model.module.cond_vit.load_state_dict(visual_mag_state_dict)
                        requires_grad(model.module.condition_pre_emb, False)
                        y = model.module.condition_pre_emb(x)
                    elif args.decoder_modal == 'magnet-0094': # todo
                        pass
                        # x = data[:, 0, :, :, :]
                        # x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                        # model.y_embedder.load_state_dict(visual_0094_state_dict)
                        # requires_grad(model.y_embedder, False)
                        # y = model.y_embedder(x)
                    elif args.decoder_modal == '0094-0094':
                        x = data[:, 1, :, :, :]
                        x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                        model.module.cond_vit.load_state_dict(visual_0094_state_dict)
                        requires_grad(model.module.condition_pre_emb, False)
                        y = model.module.condition_pre_emb(x)
                    elif args.decoder_modal == '0094-magnet': # todo
                        # x = data[:, 1, :, :, :]
                        # x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                        # model.y_embedder.load_state_dict(visual_mag_state_dict)
                        # requires_grad(model.y_embedder, False)
                        # y = model.y_embedder(x)
                        pass
                    else:
                        raise ValueError(f"Decoder modal {args.decoder_modal} not support.")
                    with torch.no_grad():
                        x = vae.sample(x)
                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                    model_kwargs = dict(x_self_cond=y)
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()
                    val_loss.append(loss.item())
                    val_mse.append(loss_dict["mse"].mean().item())
                    val_vb.append(loss_dict["vb"].mean().item())
                    
            logger_val_loss.append(np.mean(val_loss))
            logger_val_mse.append(np.mean(val_mse))
            logger_val_vb.append(np.mean(val_vb))

            result_txt = f'[Rank {rank}] | Epoch {epoch+1:>6}/{args.epochs:<6} | '
            result_txt += f'Train loss {logger_train_loss[-1]:<10.8f} | '
            result_txt += f'Train MSE {logger_train_mse[-1]:<10.8f} | '
            result_txt += f'Train VB {logger_train_vb[-1]:<10.8f} | '
            result_txt += f'Val loss {logger_val_loss[-1]:<10.8f} | '
            result_txt += f'Val MSE {logger_val_mse[-1]:<10.8f} | '
            result_txt += f'Val VB {logger_val_vb[-1]:<10.8f} | '
            result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
            print(result_txt)
            if rank == 0:
                with open(f'{logger_checkpoint_path}/logger_train_loss.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss, f)
                with open(f'{logger_checkpoint_path}/logger_train_mse.pkl', 'wb') as f:
                    pickle.dump(logger_train_mse, f)
                with open(f'{logger_checkpoint_path}/logger_train_vb.pkl', 'wb') as f:
                    pickle.dump(logger_train_vb, f)
                with open(f'{logger_checkpoint_path}/logger_train_last_layer_norm.pkl', 'wb') as f:
                    pickle.dump(logger_train_last_layer_norm, f)
                with open(f'{logger_checkpoint_path}/logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr, f)
                with open(f'{logger_checkpoint_path}/logger_val_loss.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss, f)
                with open(f'{logger_checkpoint_path}/logger_val_mse.pkl', 'wb') as f:
                    pickle.dump(logger_val_mse, f)
                with open(f'{logger_checkpoint_path}/logger_val_vb.pkl', 'wb') as f:
                    pickle.dump(logger_val_vb, f)

        if (epoch+1) % save_epoch == 0:  # todo
            # Save DiT checkpoint:
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

    vae_args = argparse.ArgumentParser().parse_args()
    vae_args = load_args_from_json(vae_args, args.vae_config_dir)

    main(args, vae_args)
